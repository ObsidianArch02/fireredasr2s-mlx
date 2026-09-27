#!/usr/bin/env python3
"""Smoke-test official FireRedASR2-AED on macOS (CPU), then dump golden tensors
for later MLX numeric-parity comparison."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "FireRedASR2S-main"
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config  # noqa: E402

MODEL_DIR = str(ROOT / "pretrained_models" / "FireRedASR2-AED")
WAVS = [
    str(REPO / "assets" / "hello_zh.wav"),
    str(REPO / "assets" / "hello_en.wav"),
    str(REPO / "assets" / "event.wav"),
]

cfg = FireRedAsr2Config(use_gpu=False, use_half=False, beam_size=3, nbest=1,
                        decode_max_len=0, softmax_smoothing=1.25,
                        aed_length_penalty=0.6, eos_penalty=1.0,
                        return_timestamp=True)
t0 = time.time()
model = FireRedAsr2.from_pretrained("aed", MODEL_DIR, cfg)
print(f"model loaded in {time.time()-t0:.1f}s")

results = model.transcribe([Path(w).stem for w in WAVS], WAVS)
for r in results:
    print(f"{Path(r.get('wav','?')).name}: {r['text']!r}  conf={r.get('confidence')}  "
          f"dur={r.get('dur_s')}s  rtf={r.get('rtf')}")
    if r.get("timestamp"):
        print("   ts:", json.dumps(r["timestamp"], ensure_ascii=False)[:200])

# golden tensors for MLX parity checks
print("\n=== dumping golden tensors ===")
feats, lengths, durs, paths, uttids = model.feat_extractor(WAVS[:2], [Path(w).stem for w in WAVS[:2]])
print("feats:", feats.shape, "lengths:", lengths.tolist())

captured = {}


def hook(name):
    def fn(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t):
            captured[name] = t.detach().clone()
    return fn


model.model.encoder.register_forward_hook(hook("encoder_out"))
with torch.no_grad():
    enc_out, enc_len, enc_mask = model.model.encoder(feats, lengths)
print("encoder_out:", tuple(enc_out.shape))
torch.save({
    "feats": feats, "lengths": lengths,
    "encoder_out": captured["encoder_out"], "enc_lengths": enc_len,
    "transcripts": results,
}, ROOT / "golden.pt")
print("saved golden.pt")
print("DONE")
