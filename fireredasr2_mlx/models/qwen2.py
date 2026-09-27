"""FireRedASR2-LLM Qwen2 decoder in pure MLX.

Weight-dict driven port of the transformers 5.x Qwen2ForCausalLM forward
(eager attention path, fp32). The FireRedASR2-LLM checkpoint differs from
stock Qwen2-7B-Instruct: q/k/v projections carry biases and there is no
q/k RMSNorm, which is exactly what the official modeling file builds.

Weights are keyed by the HF state_dict names, verbatim:
  model.embed_tokens.weight
  model.layers.{i}.input_layernorm.weight
  model.layers.{i}.self_attn.{q,k,v,o}_proj.weight [+ {q,k,v}_proj.bias]
  model.layers.{i}.post_attention_layernorm.weight
  model.layers.{i}.mlp.{gate,up,down}_proj.weight
  model.norm.weight
  lm_head.weight

Two execution modes, numerically identical:
  qwen2_logits / qwen2_hidden — full forward, no cache (prefill, scoring)
  qwen2_prefill + qwen2_decode_step — KV-cached incremental decoding
"""

from __future__ import annotations

import mlx.core as mx

_NEG_INF = float(mx.finfo(mx.float32).min)


def rms_norm(x, weight, eps):
    # Qwen2RMSNorm: variance in fp32, result scaled by weight
    xf = x.astype(mx.float32)
    var = mx.mean(xf * xf, axis=-1, keepdims=True)
    return weight * (xf * mx.rsqrt(var + eps)).astype(x.dtype)


def _rope_cos_sin(T: int, head_dim: int, theta: float):
    # matches transformers Qwen2RotaryEmbedding: emb = cat(freqs, freqs)
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32)
                                / head_dim))
    pos = mx.arange(T, dtype=mx.float32)
    freqs = pos[:, None] * inv_freq[None, :]          # (T, D/2)
    emb = mx.concatenate([freqs, freqs], axis=-1)      # (T, D)
    return mx.cos(emb)[None, None], mx.sin(emb)[None, None]  # (1,1,T,D)


def _rotate_half(x):
    d = x.shape[-1] // 2
    return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def _apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (1,1,T,D)
    return x * cos + _rotate_half(x) * sin


def linear(x, w, b=None):
    return x @ w.T + b if b is not None else x @ w.T


def _attention(w, prefix, x, cos, sin, n_head, n_kv_head, head_dim,
               cache=None):
    """x: (B, T_new, D). cos/sin cover exactly T_new positions.

    cache: optional list [k (B,Hkv,S_prev,D), v (B,Hkv,S_prev,D)] updated
    in place when given. Attention is causal within the full sequence.
    """
    B, Tn, _ = x.shape
    q = linear(x, w[prefix + ".q_proj.weight"], w[prefix + ".q_proj.bias"])
    k = linear(x, w[prefix + ".k_proj.weight"], w[prefix + ".k_proj.bias"])
    v = linear(x, w[prefix + ".v_proj.weight"], w[prefix + ".v_proj.bias"])
    q = q.reshape(B, Tn, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, Tn, n_kv_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, Tn, n_kv_head, head_dim).transpose(0, 2, 1, 3)
    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)
    if cache is not None:
        if cache[0] is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        cache[0], cache[1] = k, v
    S = k.shape[2]
    # causal: new token at absolute position S-Tn+i attends keys <= that
    i = mx.arange(S)
    tq = mx.arange(S - Tn, S)
    mask = i[None, :] > tq[:, None]                        # (Tn, S_total)
    bias = mx.where(mask, _NEG_INF, 0.0).astype(x.dtype)[None, None]
    # GQA: repeat_interleave kv heads (head j uses kv head j // rep)
    rep = n_head // n_kv_head
    if rep > 1:
        k = mx.repeat(k, rep, axis=1)
        v = mx.repeat(v, rep, axis=1)
    scores = q @ k.transpose(0, 1, 3, 2) * (head_dim ** -0.5)
    scores = scores + bias
    attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
    out = attn @ v
    out = out.transpose(0, 2, 1, 3).reshape(B, Tn, n_head * head_dim)
    return linear(out, w[prefix + ".o_proj.weight"])


def _mlp(w, prefix, x):
    gate = linear(x, w[prefix + ".mlp.gate_proj.weight"])
    up = linear(x, w[prefix + ".mlp.up_proj.weight"])
    h = gate * mx.sigmoid(gate) * up
    return linear(h, w[prefix + ".mlp.down_proj.weight"])


def _decoder_layer(w, prefix, x, cos, sin, cfg, cache=None):
    h = rms_norm(x, w[prefix + ".input_layernorm.weight"], cfg["rms_eps"])
    h = _attention(w, prefix + ".self_attn", h, cos, sin,
                   cfg["n_head"], cfg["n_kv_head"], cfg["head_dim"],
                   cache=cache)
    x = x + h
    h = rms_norm(x, w[prefix + ".post_attention_layernorm.weight"],
                 cfg["rms_eps"])
    return x + _mlp(w, prefix, h)


def _run_layers(w, x, caches, cfg, T_total, T_new, start):
    cos, sin = _rope_cos_sin(T_total, cfg["head_dim"], cfg["rope_theta"])
    cos = cos[:, :, start:start + T_new, :]
    sin = sin[:, :, start:start + T_new, :]
    for i in range(cfg["n_layers"]):
        x = _decoder_layer(w, f"model.layers.{i}", x, cos, sin, cfg,
                           cache=caches[i] if caches is not None else None)
    return rms_norm(x, w["model.norm.weight"], cfg["rms_eps"])


def qwen2_hidden(w, embeds, cfg):
    """embeds: (B, T, D). Returns final-norm hidden states (B, T, D)."""
    return _run_layers(w, embeds, None, cfg, embeds.shape[1],
                       embeds.shape[1], 0)


def qwen2_logits(w, embeds, cfg):
    """embeds: (B, T, D). Returns next-token logits of the LAST position."""
    hidden = qwen2_hidden(w, embeds, cfg)
    return linear(hidden[:, -1, :], w["lm_head.weight"])


def qwen2_prefill(w, embeds, cfg):
    """embeds: (B, S, D). Returns (logits_last (B, V), caches)."""
    caches = [[None, None] for _ in range(cfg["n_layers"])]
    hidden = _run_layers(w, embeds, caches, cfg, embeds.shape[1],
                         embeds.shape[1], 0)
    return linear(hidden[:, -1, :], w["lm_head.weight"]), caches


def qwen2_decode_step(w, token_embeds, caches, cfg):
    """token_embeds: (B, 1, D). Returns (logits (B, V), S_total)."""
    T_total = caches[0][0].shape[2] + 1
    hidden = _run_layers(w, token_embeds, caches, cfg, T_total, 1,
                         T_total - 1)
    return linear(hidden[:, -1, :], w["lm_head.weight"]), T_total


def qwen2_reorder_cache(caches, indices):
    """Reorders the beam dimension of every cached k/v (HF _reorder_cache)."""
    idx = mx.array(indices, dtype=mx.int32)
    for c in caches:
        c[0] = c[0][idx]
        c[1] = c[1][idx]


def qwen2_logprobs_sequence(w, embeds, seq_ids, cfg):
    """Log-probs of tokens `seq_ids` (L,) following `embeds` prefix.

    One full forward over [prefix, seq]; returns per-token log-prob of the
    actual next token at each position of seq (L,).
    """
    emb_w = w["model.embed_tokens.weight"]
    T = embeds.shape[1] + len(seq_ids)
    cos, sin = _rope_cos_sin(T, cfg["head_dim"], cfg["rope_theta"])
    x = mx.concatenate(
        [embeds, emb_w[mx.array(seq_ids, dtype=mx.int32)][None]], axis=1)
    # fresh full pass (no cache): reuse the plain layer loop for clarity
    for i in range(cfg["n_layers"]):
        x = _decoder_layer(w, f"model.layers.{i}", x, cos, sin, cfg)
    hidden = rms_norm(x, w["model.norm.weight"], cfg["rms_eps"])
    # only positions that predict seq tokens need the lm_head
    logits = linear(hidden[:, embeds.shape[1] - 1: -1, :], w["lm_head.weight"])
    lp = logits[0] - mx.logsumexp(logits[0], axis=-1, keepdims=True)
    tgt = mx.array(seq_ids, dtype=mx.int32)
    return mx.take_along_axis(lp, tgt[:, None], axis=1)[:, 0]
