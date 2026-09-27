"""FireRedASR2 Conformer encoder in pure MLX.

Functional port of FireRedASR2S-main/fireredasr2s/fireredasr2/models/module/
conformer_encoder.py. All weights come from a flat dict of mx.arrays keyed by
the official torch state_dict names (minus the top-level "encoder." prefix,
kept here as "encoder.<name>" verbatim for 1:1 mapping from the checkpoint).

Shapes follow torch: activations (N, T, C); conv weights transposed at load
time (see loader.py), so this file is layout-agnostic.
"""

from __future__ import annotations

import math

import mlx.core as mx


def layer_norm(x, w, b):
    # torch nn.LayerNorm default eps=1e-5
    return mx.fast.layer_norm(x, w, b, 1e-5)


def linear(x, w, b=None):
    return x @ w.T + b if b is not None else x @ w.T


def swish(x):
    return x * mx.sigmoid(x)


def relu(x):
    return mx.maximum(x, mx.array(0.0, x.dtype))


# ---------------------------------------------------------------- U2: PE ---

def rel_positional_encoding(pe: mx.array, T: int) -> mx.array:
    """pe: precomputed (1, 2*max_len-1, D) buffer (torch buffer, same layout)."""
    Tmax = pe.shape[1]
    return pe[:, Tmax // 2 - T + 1: Tmax // 2 + T, :]


def build_rel_pe(d_model: int, max_len: int = 5000) -> mx.array:
    pe_pos = mx.zeros((max_len, d_model))
    pe_neg = mx.zeros((max_len, d_model))
    position = mx.arange(0, max_len, dtype=mx.float32).reshape(max_len, 1)
    div = mx.exp(mx.arange(0, d_model, 2, dtype=mx.float32)
                 * -(math.log(10000.0) / d_model))
    pe_pos[:, 0::2] = mx.sin(position * div)
    pe_pos[:, 1::2] = mx.cos(position * div)
    pe_neg[:, 0::2] = mx.sin(-position * div)
    pe_neg[:, 1::2] = mx.cos(-position * div)
    pe_pos = mx.flip(pe_pos, axis=0)[None]
    pe_neg = pe_neg[1:][None]
    return mx.concatenate([pe_pos, pe_neg], axis=1)


def abs_positional_encoding(d_model: int, max_len: int = 5000) -> mx.array:
    pe = mx.zeros((max_len, d_model))
    position = mx.arange(0, max_len, dtype=mx.float32).reshape(max_len, 1)
    div = mx.exp(mx.arange(0, d_model, 2, dtype=mx.float32)
                 * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = mx.sin(position * div)
    pe[:, 1::2] = mx.cos(position * div)
    return pe[None]


# ------------------------------------------------- U1: Conv2dSubsampling ---

def conv2d_subsampling(w: dict, x, x_mask):
    """x: (N, T, 80). Returns (embed (N, T', 1280), lengths, mask)."""
    # context padding: torch F.pad(x, (0,0,0,6)) on (N,T,80) -> pad T by 6 at end
    x = mx.pad(x, [(0, 0), (0, 6), (0, 0)])
    # torch conv input (N, C=1, H=T, W=80) == mlx NHWC (N, H=T, W=80, C=1)
    h = mx.expand_dims(x, -1)
    h = relu(mx.conv2d(h, w["encoder.input_preprocessor.conv.0.weight"], stride=2)
             + w["encoder.input_preprocessor.conv.0.bias"][None, None, None, :])
    h = relu(mx.conv2d(h, w["encoder.input_preprocessor.conv.2.weight"], stride=2)
             + w["encoder.input_preprocessor.conv.2.bias"][None, None, None, :])
    N, T2, W2, C = h.shape  # NHWC
    # torch flattens (C, D) channel-major: c*W2 + w
    h = h.transpose(0, 1, 3, 2).reshape(N, T2, C * W2)
    h = linear(h, w["encoder.input_preprocessor.out.weight"],
               w["encoder.input_preprocessor.out.bias"])
    # mask slice: torch x_mask[:, :, :-2:2][:, :, :-2:2] on (N,1,T) uint8
    m = x_mask[:, :, :-2:2][:, :, :-2:2]
    lengths = m[:, -1, :].sum(axis=-1).astype(mx.int32)
    return h, lengths, m


def padding_position_is_0(input_lengths, T: int) -> mx.array:
    # vectorized (compile-safe) form of: ones, then zero out positions >= L
    ar = mx.arange(T)
    return (ar[None, None, :] < input_lengths[:, None, None]).astype(mx.uint8)


# --------------------------------------------------- U3: RelPos MHSA -------

def _rel_shift(x):
    n, h, t1, t2 = x.shape
    zero_pad = mx.zeros((n, h, t1, 1), dtype=x.dtype)
    xp = mx.concatenate([zero_pad, x], axis=-1)
    xp = xp.reshape(n, h, t2 + 1, t1)
    x = xp[:, :, 1:, :].reshape(n, h, t1, t2)  # view_as back before slicing
    return x[:, :, :, : x.shape[-1] // 2 + 1]


def scaled_dot_attn(scores, v, mask):
    """mask: (N,1,T) uint8, 1=keep. scores: (N,H,T,T). Returns (out, attn)."""
    m = mask[:, None, :].astype(mx.bool_)  # (N,1,1,T) over keys
    scores = mx.where(m, scores, mx.array(-float("inf"), scores.dtype))
    attn = mx.softmax(scores, axis=-1)
    attn = mx.where(m, attn, mx.array(0.0, attn.dtype))
    return attn @ v, attn


def rel_pos_mhsa_cfg(w: dict, prefix, q_in, k_in, v_in, pos_emb, mask,
                     n_head: int, d_k: int):
    residual = q_in
    B, Tq, _ = q_in.shape
    Tk = k_in.shape[1]

    q = linear(layer_norm(q_in, w[prefix + ".layer_norm_q.weight"],
                          w[prefix + ".layer_norm_q.bias"]),
               w[prefix + ".w_qs.weight"]).reshape(B, Tq, n_head, d_k).transpose(0, 2, 1, 3)
    k = linear(layer_norm(k_in, w[prefix + ".layer_norm_k.weight"],
                          w[prefix + ".layer_norm_k.bias"]),
               w[prefix + ".w_ks.weight"]).reshape(B, Tk, n_head, d_k).transpose(0, 2, 1, 3)
    v = linear(layer_norm(v_in, w[prefix + ".layer_norm_v.weight"],
                          w[prefix + ".layer_norm_v.bias"]),
               w[prefix + ".w_vs.weight"]).reshape(B, Tk, n_head, d_k).transpose(0, 2, 1, 3)

    q_t = q.transpose(0, 2, 1, 3)  # (B, Tq, H, dk)
    n_pos = pos_emb.shape[0]
    p = linear(pos_emb, w[prefix + ".linear_pos.weight"])
    p = p.reshape(n_pos, -1, n_head, d_k).transpose(0, 2, 1, 3)  # (P, H, dk')

    q_u = (q_t + w[prefix + ".pos_bias_u"]).transpose(0, 2, 1, 3)  # (B,H,Tq,dk)
    q_v = (q_t + w[prefix + ".pos_bias_v"]).transpose(0, 2, 1, 3)

    matrix_ac = q_u @ k.transpose(0, 1, 3, 2)
    matrix_bd = q_v @ p.transpose(0, 1, 3, 2)  # (B,H,Tq,P)
    matrix_bd = _rel_shift(matrix_bd)

    scores = (matrix_ac + matrix_bd) * (1.0 / math.sqrt(d_k))
    out, _ = scaled_dot_attn(scores, v, mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, Tq, n_head * d_k)
    out = linear(out, w[prefix + ".fc.weight"])
    return out + residual


# ------------------------------------------------------- U4: Conv module ---

def conformer_conv(w: dict, prefix, x, mask):
    """x: (N, T, D); mask: (N,1,T) uint8. Returns (N, T, D).
    mlx conv1d is channels-last: input (N, T, C), weight (O, kW, C/groups)."""
    residual = x
    out = layer_norm(x, w[prefix + ".pre_layer_norm.weight"],
                     w[prefix + ".pre_layer_norm.bias"])
    keep = mask[:, 0, :].astype(mx.bool_)
    out = mx.where(keep[:, :, None], out, mx.array(0.0, out.dtype))
    out = mx.conv1d(out, w[prefix + ".pointwise_conv1.weight"],
                    stride=1, padding=0)  # (N, T, 2D)
    half = out.shape[-1] // 2
    out = out[..., :half] * mx.sigmoid(out[..., half:])  # GLU along channel
    out = mx.conv1d(out, w[prefix + ".depthwise_conv.weight"],
                    stride=1, padding=16, groups=out.shape[-1])
    out = swish(layer_norm(out, w[prefix + ".batch_norm.weight"],
                           w[prefix + ".batch_norm.bias"]))
    out = mx.conv1d(out, w[prefix + ".pointwise_conv2.weight"], stride=1, padding=0)
    out = mx.where(keep[:, :, None], out, mx.array(0.0, out.dtype))
    return out + residual


# --------------------------------------------------- U5: FFN + block -------

def conformer_ffn(w: dict, prefix, x):
    h = layer_norm(x, w[prefix + ".net.0.weight"],
                   w[prefix + ".net.0.bias"])
    h = linear(h, w[prefix + ".net.1.weight"],
               w[prefix + ".net.1.bias"])
    h = swish(h)
    h = linear(h, w[prefix + ".net.4.weight"],
               w[prefix + ".net.4.bias"])
    return h + x


def conformer_block(w: dict, prefix, x, pos_emb, mask, n_head: int, d_k: int):
    out = 0.5 * x + 0.5 * conformer_ffn(w, prefix + ".ffn1", x)
    out = rel_pos_mhsa_cfg(w, prefix + ".mhsa", out, out, out, pos_emb, mask,
                           n_head, d_k)
    out = conformer_conv(w, prefix + ".conv", out, mask)
    out = 0.5 * out + 0.5 * conformer_ffn(w, prefix + ".ffn2", out)
    out = layer_norm(out, w[prefix + ".layer_norm.weight"],
                     w[prefix + ".layer_norm.bias"])
    return out


def encoder_forward(w: dict, feats, input_lengths, n_layers: int,
                    n_head: int = 20, d_model: int = 1280):
    """feats: (N, T, 80) fbank+CMVN. Returns (enc_out, lengths, mask)."""
    src_mask = padding_position_is_0(input_lengths,
                                     feats.shape[1] + 6)
    embed, lengths, mask = conv2d_subsampling(w, feats, src_mask)
    pe_buf = w["pe_rel"]  # precomputed buffer
    pos_emb = rel_positional_encoding(pe_buf, embed.shape[1])
    x = embed
    d_k = d_model // n_head
    for l in range(n_layers):
        x = conformer_block(w, f"encoder.layer_stack.{l}", x, pos_emb, mask,
                            n_head, d_k)
    return x, lengths, mask
