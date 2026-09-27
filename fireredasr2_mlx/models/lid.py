"""FireRedLID in pure MLX: Conformer encoder + small AED language decoder.

Reuses the ASR Conformer blocks; the LID decoder is the same TransformerDecoder
class with 6 layers over a 120-language vocabulary (decode_max_len=2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..features import ASRFeatExtractor
from ..loader import load_safetensors_plain
from .conformer_encoder import encoder_forward
from .transformer_decoder import batch_beam_search


@dataclass
class FireRedLidMlxConfig:
    beam_size: int = 3
    nbest: int = 1
    decode_max_len: int = 2
    softmax_smoothing: float = 1.25
    aed_length_penalty: float = 0.6
    eos_penalty: float = 1.0


class FireRedLidMlx:
    @classmethod
    def from_pretrained(cls, model_dir, config=FireRedLidMlxConfig()):
        model_dir = Path(model_dir)
        w, args = load_safetensors_plain(model_dir)
        if "encoder.positional_encoding.pe" in w:
            pass  # loader handles PE buffers
        feat_extractor = ASRFeatExtractor(str(model_dir / "cmvn.ark"))
        from ..token_dict import TokenDict
        dict_ = TokenDict(str(model_dir / "dict.txt"), unk="<unk>")
        return cls(w, args, feat_extractor, dict_, config)

    def __init__(self, w, args, feat_extractor, dict_, config):
        self.w, self.args = w, args
        self.feat_extractor = feat_extractor
        self.dict = dict_
        self.config = config

    def process(self, batch_uttid, batch_wav_path):
        feats_np, lengths_np, durs, paths, uttids = self.feat_extractor(
            list(batch_wav_path), list(batch_uttid))
        if feats_np is None:
            return [{"uttid": u, "lang": ""} for u in batch_uttid]

        import mlx.core as mx
        feats = mx.array(np.asarray(feats_np, dtype=np.float32))
        lengths = mx.array(np.asarray(lengths_np, dtype=np.int32))
        enc_out, _, enc_mask = encoder_forward(
            self.w, feats, lengths, n_layers=int(self.args["n_layers_enc"]),
            n_head=int(self.args["n_head"]), d_model=int(self.args["d_model"]))
        hyps = batch_beam_search(
            self.w, enc_out, enc_mask,
            beam_size=self.config.beam_size, nbest=self.config.nbest,
            decode_max_len=self.config.decode_max_len,
            softmax_smoothing=self.config.softmax_smoothing,
            length_penalty=self.config.aed_length_penalty,
            eos_penalty=self.config.eos_penalty,
            sos_id=int(self.args["sos_id"]), eos_id=int(self.args["eos_id"]),
            pad_id=int(self.args["pad_id"]),
            n_layers=int(self.args["n_layers_lid_dec"]),
            n_head=int(self.args["n_head"]), d_model=int(self.args["d_model"]),
            dec_prefix="lid_decoder")

        results = []
        for uttid, wav, hyp, dur in zip(uttids, paths, hyps, durs):
            hyp_ids = [int(i) for i in hyp[0]["yseq"]]
            lang = " ".join(self.dict[i] for i in hyp_ids)
            results.append({"uttid": uttid, "lang": lang,
                            "confidence": round(hyp[0].get("confidence", 0.0), 3)})
            if isinstance(wav, str):
                results[-1]["wav"] = wav
        return results
