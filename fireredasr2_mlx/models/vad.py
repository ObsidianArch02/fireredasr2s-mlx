"""FireRedVAD in pure MLX: DFSMN speech detector + fbank/CMVN frontend.

Functional port of fireredvad/core/detect_model.py (DetectModel/DFSMN/FSMN).
Non-streaming path only (cache=None), which is what the official system uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..features import CMVN, KaldifeatFbank
from ..vad_postprocessor import VadPostprocessor
from .conformer_encoder import linear, relu


@dataclass
class FireRedVadMlxConfig:
    smooth_window_size: int = 5
    speech_threshold: float = 0.4
    min_speech_frame: int = 20
    max_speech_frame: int = 2000
    min_silence_frame: int = 20
    merge_silence_frame: int = 0
    extend_speech_frame: int = 0
    chunk_max_frame: int = 30000


def conv1d_groups(w: mx.array, x: mx.array, padding: int, dilation: int,
                  groups: int) -> mx.array:
    return mx.conv1d(x, w, stride=1, padding=padding, dilation=dilation,
                     groups=groups)


def fsmn(w: dict, prefix, x, N1: int, S1: int, N2: int, S2: int):
    """x: (N, T, P) channels-last. Mirrors torch FSMN.forward with cache=None."""
    T = x.shape[1]
    # torch keeps (N,P,T); mlx conv1d is channels-last (N,T,P) so time is axis 1
    lookback = conv1d_groups(w[prefix + ".lookback_filter.weight"], x,
                             padding=(N1 - 1) * S1, dilation=S1, groups=x.shape[-1])
    if N1 > 1:
        lookback = lookback[:, :-(N1 - 1) * S1, :]
    memory = x + lookback
    if N2 > 0 and T > 1:
        lookahead = conv1d_groups(w[prefix + ".lookahead_filter.weight"], x,
                                  padding=(N2 - 1) * S2, dilation=S2,
                                  groups=x.shape[-1])
        # torch: lookahead[:, :, N2*S2:] then F.pad(..., (0, S2))
        lookahead = mx.pad(lookahead[:, N2 * S2:, :], [(0, 0), (0, S2), (0, 0)])
        memory = memory + lookahead
    return memory


def dfsmn_block(w: dict, prefix, x, N1, S1, N2, S2):
    """DFSMNBlock: fc1(P->H,+ReLU) -> fc2(H->P, no bias) -> fsmn -> skip."""
    h = relu(linear(x, w[prefix + ".fc1.0.weight"], w[prefix + ".fc1.0.bias"]))
    p = linear(h, w[prefix + ".fc2.weight"])
    memory = fsmn(w, prefix + ".fsmn", p, N1, S1, N2, S2)
    return memory + x


def vad_forward(w: dict, feat: mx.array, args: dict) -> mx.array:
    """feat: (1, T, 80) -> speech probs (1, T, odim)."""
    D = w["dfsmn.fc1.0.weight"].shape[1]
    H = w["dfsmn.fc1.0.weight"].shape[0]
    P = w["dfsmn.fc2.0.weight"].shape[0]
    R = args["R"]
    M = args["M"]
    N1, S1, N2, S2 = args["N1"], args["S1"], args["N2"], args["S2"]

    h = relu(linear(feat, w["dfsmn.fc1.0.weight"], w["dfsmn.fc1.0.bias"]))
    p = relu(linear(h, w["dfsmn.fc2.0.weight"], w["dfsmn.fc2.0.bias"]))
    memory = fsmn(w, "dfsmn.fsmn1", p, N1, S1, N2, S2)
    for i in range(R - 1):
        memory = dfsmn_block(w, f"dfsmn.fsmns.{i}", memory, N1, S1, N2, S2)
    # M DNN layers: dnns = Sequential([Linear, ReLU, Dropout] * M) — ReLU
    # applies after EVERY linear (only Dropout follows the last one)
    out = memory
    for j in range(M):
        idx = j * 2
        out = linear(out, w[f"dfsmn.dnns.{idx}.weight"], w[f"dfsmn.dnns.{idx}.bias"])
        out = relu(out)
    logits = linear(out, w["out.weight"], w["out.bias"])
    return mx.sigmoid(logits)


class FireRedVadMlx:
    @classmethod
    def from_pretrained(cls, model_dir, config=FireRedVadMlxConfig()):
        model_dir = Path(model_dir)
        from ..loader import load_safetensors_plain
        w, args = load_safetensors_plain(model_dir)
        feat = KaldifeatFbank(num_mel_bins=80, frame_length=25, frame_shift=10,
                              dither=0)
        cmvn = CMVN(str(model_dir / "cmvn.ark"))
        post = VadPostprocessor(
            config.smooth_window_size, config.speech_threshold,
            config.min_speech_frame, config.max_speech_frame,
            config.min_silence_frame, config.merge_silence_frame,
            config.extend_speech_frame)
        return cls(w, args, feat, cmvn, post, config)

    def __init__(self, w, args, fbank, cmvn, post, config):
        self.w, self.args, self.fbank, self.cmvn, self.post = w, args, fbank, cmvn, post
        self.config = config

    def detect(self, audio, do_postprocess=True):
        import soundfile as sf
        if isinstance(audio, str):
            wav_np, sr = sf.read(audio, dtype="int16")
        elif isinstance(audio, (list, tuple)):
            wav_np, sr = audio
        else:
            wav_np, sr = audio, 16000
        assert sr == 16000
        dur = wav_np.shape[0] / sr
        fbank = self.fbank((sr, wav_np))
        fbank = self.cmvn(fbank)
        x = mx.array(fbank.astype(np.float32))[None]

        probs = []
        for s in range(0, x.shape[1], self.config.chunk_max_frame):
            chunk = x[:, s:s + self.config.chunk_max_frame, :]
            probs.append(vad_forward(self.w, chunk, self.args))
        prob = np.concatenate([np.asarray(p)[0, :, 0] for p in probs], axis=0)

        if not do_postprocess:
            return None, prob
        decisions = self.post.process(prob.tolist())
        starts_ends = self.post.decision_to_segment(decisions, dur)
        result = {"dur": round(dur, 3), "timestamps": starts_ends}
        if isinstance(audio, str):
            result["wav_path"] = audio
        return result, prob
