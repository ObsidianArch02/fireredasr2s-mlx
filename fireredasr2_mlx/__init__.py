"""FireRedASR2 in pure MLX: facade mirroring the official FireRedAsr2 API."""

from __future__ import annotations

import re
import sys
import time

import numpy as np
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .loader import load_weights
from .features import ASRFeatExtractor
from .tokenizer import ChineseCharEnglishSpmTokenizer
from .models.conformer_encoder import encoder_forward
from .models.transformer_decoder import batch_beam_search
from .llm import FireRedAsr2LlmMlx, FireRedAsr2LlmMlxConfig


@dataclass
class FireRedAsr2MlxConfig:
    beam_size: int = 3
    nbest: int = 1
    decode_max_len: int = 0
    softmax_smoothing: float = 1.25
    aed_length_penalty: float = 0.6
    eos_penalty: float = 1.0
    return_timestamp: bool = False


class FireRedAsr2Mlx:
    @classmethod
    def from_pretrained(cls, model_dir: str, config=FireRedAsr2MlxConfig()):
        model_dir = Path(model_dir)
        w = load_weights(str(model_dir))
        feat_extractor = ASRFeatExtractor(str(model_dir / "cmvn.ark"))
        tokenizer = ChineseCharEnglishSpmTokenizer(
            str(model_dir / "dict.txt"), str(model_dir / "train_bpe1000.model"))
        return cls(w, feat_extractor, tokenizer, config)

    def __init__(self, w, feat_extractor, tokenizer, config):
        self.w = w
        self.feat_extractor = feat_extractor
        self.tokenizer = tokenizer
        self.config = config

    def transcribe(self, batch_uttid, batch_wav_path):
        feats_np, lengths_np, durs, paths, uttids = self.feat_extractor(
            list(batch_wav_path), list(batch_uttid))
        if feats_np is None:
            return [{"uttid": u, "text": ""} for u in batch_uttid]

        feats = mx.array(np.asarray(feats_np, dtype=np.float32))
        lengths = mx.array(np.asarray(lengths_np, dtype=np.int32))
        t0 = time.time()
        enc_out, enc_lengths, enc_mask = encoder_forward(
            self.w, feats, lengths, n_layers=16)
        mx.eval(enc_out)
        hyps = batch_beam_search(
            self.w, enc_out, enc_mask, beam_size=self.config.beam_size,
            nbest=self.config.nbest, decode_max_len=self.config.decode_max_len,
            softmax_smoothing=self.config.softmax_smoothing,
            length_penalty=self.config.aed_length_penalty,
            eos_penalty=self.config.eos_penalty)
        elapsed = time.time() - t0
        total_dur = sum(durs)
        rtf = elapsed / total_dur if total_dur > 0 else 0.0

        results = []
        for ui, (uttid, wav, hyp, dur) in enumerate(zip(uttids, paths, hyps, durs)):
            hyp_ids = [int(i) for i in hyp[0]["yseq"]]
            text = self.tokenizer.detokenize(hyp_ids)
            text = re.sub(r"(<blank>)|(<sil>)", "", text)
            r = {"uttid": uttid, "text": text.lower(),
                 "confidence": round(hyp[0].get("confidence", 0.0), 3),
                 "dur_s": round(dur, 3), "rtf": f"{rtf:.4f}"}
            if isinstance(wav, str):
                r["wav"] = wav
            if self.config.return_timestamp:
                from .models.ctc import token_timestamps
                ts = token_timestamps(self.w, enc_out[ui], int(enc_lengths[ui]),
                                      hyp_ids)
                if ts is not None:
                    fixed = self._get_and_fix_timestamp(ts, hyp_ids, dur)
                    r["timestamp"] = fixed
            results.append(r)
        return results

    def _get_and_fix_timestamp(self, ts, hyp_ids, dur):
        """Verbatim port of FireRedAsr2._get_and_fix_timestamp (60ms shift)."""
        r3 = lambda x: round(x, 3)
        starts, ends = ts
        timestamp = []
        last_end = dur
        SHIFT = 0.06
        for hyp_id, start, end in zip(hyp_ids, starts, ends):
            token = self.tokenizer.detokenize([hyp_id], "", False)
            start = min(max(0, start - SHIFT), last_end)
            end = min(max(0, end - SHIFT), dur)
            last_end = end
            timestamp.append([token.lower(), r3(start), r3(end)])
        for i in range(len(timestamp)):
            idx = -(i + 1)
            _, start, end = timestamp[idx]
            if abs(dur - start) < 0.001:
                timestamp[idx][1] = dur - (i + 1) * 0.001
            if i != 0 and abs(dur - end) < 0.001:
                timestamp[idx][2] = dur - i * 0.001
        timestamp = self.tokenizer.merge_spm_timestamp(timestamp)
        return timestamp
