"""FireRedASR2-LLM Adapter (encoder projector) in pure MLX.

Verbatim port of fireredasr2s/fireredasr2/models/module/adapter.py:
frames are folded pairwise (downsample_rate=2) then passed through
Linear -> ReLU -> Linear. Weight keys are the official state_dict names:
  encoder_projector.linear1.weight / .bias
  encoder_projector.linear2.weight / .bias
"""

from __future__ import annotations

import mlx.core as mx


def adapter_forward(w: dict, x, x_lens, ds: int = 2):
    """x: (1, T, encoder_dim) encoder output (padded).
    Returns (features (1, T//ds, llm_dim), valid_lengths (1,))."""
    B, T, D = x.shape
    discard = T % ds
    if discard > 0:
        x = x[:, : T - discard, :]
    T = x.shape[1]

    x = x.reshape(B, T // ds, D * ds)
    x = x @ w["encoder_projector.linear1.weight"].T \
        + w["encoder_projector.linear1.bias"]
    x = mx.maximum(x, mx.array(0.0, x.dtype))
    x = x @ w["encoder_projector.linear2.weight"].T \
        + w["encoder_projector.linear2.bias"]

    new_lens = mx.minimum(x_lens, mx.array(T, x_lens.dtype)) // ds
    return x, new_lens
