"""HF BERT encoder in pure MLX (post-LN BERT, LERT-compatible).

Mirrors HuggingFace BertModel forward: embeddings + N post-LN layers +
extended attention mask (masked = large negative). Keys follow HF names so
the converted safetensors load directly.
"""

from __future__ import annotations

import math

import mlx.core as mx

LN_EPS = 1e-12  # HF config layer_norm_eps for chinese-lert-base


def ln(x, w, b):
    return mx.fast.layer_norm(x, w, b, LN_EPS)


def linear(x, w, b=None):
    return x @ w.T if b is None else x @ w.T + b


def gelu(x):
    return 0.5 * x * (1.0 + mx.erf(x / mx.sqrt(mx.array(2.0, x.dtype))))


def bert_forward(w: dict, input_ids: mx.array, lengths) -> mx.array:
    """input_ids: (N, T) int. lengths: list[int] (valid length incl. CLS).
    Returns last hidden states (N, T, H)."""
    N, T = input_ids.shape
    H = w["bert.embeddings.word_embeddings.weight"].shape[1]
    n_layers = len({k.split(".")[3] for k in w
                    if k.startswith("bert.encoder.layer.")})

    ids = input_ids
    pos = mx.arange(T)
    x = (w["bert.embeddings.word_embeddings.weight"][ids]
         + w["bert.embeddings.position_embeddings.weight"][pos][None]
         + w["bert.embeddings.token_type_embeddings.weight"][0][None, None])
    x = ln(x, w["bert.embeddings.LayerNorm.weight"],
           w["bert.embeddings.LayerNorm.bias"])

    # HF extended mask: 0 for valid, large negative for pad, added to scores
    mask = mx.zeros((N, T), dtype=mx.float32)
    for i in range(N):
        if lengths[i] < T:
            mask[i, lengths[i]:] = -1e10
    mask_ext = mask[:, None, None, :]

    dk = H // 12
    for l in range(n_layers):
        p = f"bert.encoder.layer.{l}"
        q = linear(x, w[f"{p}.attention.self.query.weight"],
                   w[f"{p}.attention.self.query.bias"])
        k = linear(x, w[f"{p}.attention.self.key.weight"],
                   w[f"{p}.attention.self.key.bias"])
        v = linear(x, w[f"{p}.attention.self.value.weight"],
                   w[f"{p}.attention.self.value.bias"])
        q = q.reshape(N, T, 12, dk).transpose(0, 2, 1, 3)
        k = k.reshape(N, T, 12, dk).transpose(0, 2, 1, 3)
        v = v.reshape(N, T, 12, dk).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(dk) + mask_ext
        attn = mx.softmax(scores, axis=-1)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(N, T, H)
        attn_out = linear(ctx, w[f"{p}.attention.output.dense.weight"],
                          w[f"{p}.attention.output.dense.bias"])
        x = ln(x + attn_out, w[f"{p}.attention.output.LayerNorm.weight"],
               w[f"{p}.attention.output.LayerNorm.bias"])
        inter = gelu(linear(x, w[f"{p}.intermediate.dense.weight"],
                            w[f"{p}.intermediate.dense.bias"]))
        out = linear(inter, w[f"{p}.output.dense.weight"],
                     w[f"{p}.output.dense.bias"])
        x = ln(x + out, w[f"{p}.output.LayerNorm.weight"],
               w[f"{p}.output.LayerNorm.bias"])
    return x
