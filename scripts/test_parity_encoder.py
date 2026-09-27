#!/usr/bin/env python3
"""Numeric parity: MLX encoder vs PyTorch golden tensors.

Run AFTER test_aed.py has produced golden.pt (PyTorch side) and
loader.convert_checkpoint has produced model.safetensors.
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fireredasr2_mlx.loader import load_weights  # noqa: E402
from fireredasr2_mlx.models.conformer_encoder import encoder_forward  # noqa: E402

golden = torch.load(ROOT / "golden.pt", map_location="cpu", weights_only=False)
feats = golden["feats"].numpy()
lengths = golden["lengths"].numpy().astype(np.int32)
enc_ref = golden["encoder_out"].numpy()

w = load_weights(str(ROOT / "pretrained_models" / "FireRedASR2-AED"))
enc_out, enc_lengths, enc_mask = encoder_forward(
    w, mx.array(feats), mx.array(lengths), n_layers=16)
enc_out_np = np.asarray(enc_out.astype(mx.float32))

diff = np.abs(enc_out_np - enc_ref)
rel = diff / (np.abs(enc_ref) + 1e-8)
print(f"encoder_out shapes: mlx {enc_out_np.shape} vs torch {enc_ref.shape}")
print(f"max-abs-diff: {diff.max():.3e}   mean-abs-diff: {diff.mean():.3e}")
print(f"max-rel-diff: {rel.max():.3e}")
print(f"enc_lengths: mlx {enc_lengths.tolist()} vs torch {golden['enc_lengths'].tolist()}")

THRESH = 1e-4
if diff.max() < THRESH and enc_lengths.tolist() == list(golden["enc_lengths"].tolist()):
    print(f"PARITY PASS (< {THRESH})")
else:
    print(f"PARITY FAIL")
    sys.exit(1)
