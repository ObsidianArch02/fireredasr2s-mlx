"""FireRedASR2-LLM facade in pure MLX: encoder -> adapter -> Qwen2 beam search.

Mirrors the official FireRedAsr2 LLM branch (fireredasr2/asr.py): same
feature extraction, same fixed decode prompt, same generate semantics
(transformers 5.x beam search, fp32). One deliberate addition: a
"confidence" field (mean per-token probability of the best hypothesis) so the
4-module system assembly works — the official LLM result dict has no
confidence at all, which crashes the official 4-module CLI (see
docs/PARITY_GAPS.md).
"""

from __future__ import annotations

import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mlx.core as mx

from .features import ASRFeatExtractor
from .llm_tokenizer import LlmTokenizerWrapperMlx
from .loader import load_llm_weights
from .models.fireredasr_llm import llm_transcribe


@dataclass
class FireRedAsr2LlmMlxConfig:
    beam_size: int = 3
    decode_max_len: int = 0
    decode_min_len: int = 0
    repetition_penalty: float = 1.0
    llm_length_penalty: float = 0.0


class FireRedAsr2LlmMlx:
    @classmethod
    def from_pretrained(cls, model_dir: str,
                        config: FireRedAsr2LlmMlxConfig | None = None):
        model_dir = Path(model_dir)
        w_enc, w_llm, cfg = load_llm_weights(str(model_dir))
        feat_extractor = ASRFeatExtractor(str(model_dir / "cmvn.ark"))
        tokenizer = LlmTokenizerWrapperMlx.build_llm_tokenizer(
            model_dir / "Qwen2-7B-Instruct")
        cfg["speech_token_id"] = tokenizer.speech_token_id
        cfg["eos_id"] = tokenizer.eos_token_id
        cfg["pad_id"] = tokenizer.pad_token_id
        input_ids, attention_mask = tokenizer.decode_prompt()
        return cls(w_enc, w_llm, cfg, feat_extractor, tokenizer,
                   mx.array(input_ids), mx.array(attention_mask),
                   config or FireRedAsr2LlmMlxConfig())

    def __init__(self, w_enc, w_llm, cfg, feat_extractor, tokenizer,
                 input_ids, attention_mask, config):
        self.w_enc = w_enc
        self.w_llm = w_llm
        self.cfg = cfg
        self.feat_extractor = feat_extractor
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.config = config

    def transcribe(self, batch_uttid, batch_wav_path):
        feats_np, lengths_np, durs, paths, uttids = self.feat_extractor(
            list(batch_wav_path), list(batch_uttid))
        if feats_np is None:
            return [{"uttid": u, "text": ""} for u in batch_uttid]

        results = []
        total_dur = sum(durs)
        t_start = time.time()
        for ui in range(feats_np.shape[0]):
            try:
                feats = mx.array(np.asarray(feats_np[ui:ui + 1],
                                            dtype=np.float32))
                lengths = mx.array(np.asarray(lengths_np[ui:ui + 1],
                                              dtype=np.int32))
                ids, confidence = llm_transcribe(
                    self.w_enc, self.w_llm, self.cfg, feats, lengths,
                    self.input_ids, self.attention_mask,
                    beam_size=self.config.beam_size,
                    decode_max_len=self.config.decode_max_len,
                    decode_min_len=self.config.decode_min_len,
                    repetition_penalty=self.config.repetition_penalty,
                    llm_length_penalty=self.config.llm_length_penalty)
                text = self.tokenizer.batch_decode([ids])[0]
            except Exception:
                traceback.print_exc()
                ids, confidence, text = [], 0.0, ""
            results.append({"ids": ids, "confidence": confidence,
                            "_text": text})
        elapsed = time.time() - t_start
        rtf = elapsed / total_dur if total_dur > 0 else 0.0

        out = []
        for uttid, wav, dur, r in zip(uttids, paths, durs, results):
            text = r["_text"].lower()
            text = re.sub(r"(<blank>)|(<sil>)", "", text)
            entry = {"uttid": uttid, "text": text,
                     "confidence": round(r["confidence"], 3),
                     "dur_s": round(dur, 3), "rtf": f"{rtf:.4f}"}
            if isinstance(wav, str):
                entry["wav"] = wav
            out.append(entry)
        return out
