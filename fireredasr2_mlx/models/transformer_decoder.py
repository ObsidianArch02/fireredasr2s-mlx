"""FireRedASR2 AED Transformer decoder + batch beam search in pure MLX.

Functional port of transformer_decoder.py with a true KV cache:
- self-attention K/V are computed once per position (pre-norm input row) and
  appended to per-layer caches; on beam re-order the caches are row-gathered
  with the same rows_in_ys as ys;
- cross-attention K/V over the encoder outputs are computed exactly once.
Mathematically identical to the official prefix-recompute scheme (K/V inputs
are per-row LN + linear of layer inputs that never change for old positions),
so outputs stay bit-compatible with the torch golden results.
"""

from __future__ import annotations

import math

import mlx.core as mx

from .conformer_encoder import layer_norm, linear


def gelu(x):
    """Exact (erf) GELU — torch nn.GELU default."""
    return 0.5 * x * (1.0 + mx.erf(x / mx.sqrt(mx.array(2.0, x.dtype))))


def log_softmax(x, axis=-1):
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


def _heads(x, w, prefix, name, n_head: int, d_k: int, biased: bool):
    """Linear projection -> (B, H, T, dk). Mirrors DecoderMultiHeadAttention."""
    b = w[f"{prefix}.{name}.bias"] if biased else None
    h = linear(x, w[f"{prefix}.{name}.weight"], b)
    return h.reshape(x.shape[0], x.shape[1], n_head, d_k).transpose(0, 2, 1, 3)


def _attention(q, k, v, mask_b, d_k: int):
    """Scaled dot-product with the exact masked-softmax op order of the
    official DecoderScaledDotProductAttention. mask_b: (B,1,1,Tk) bool."""
    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
    scores = mx.where(mask_b, scores, mx.array(-float("inf"), scores.dtype))
    attn = mx.softmax(scores, axis=-1)
    attn = mx.where(mask_b, attn, mx.array(0.0, attn.dtype))
    return attn @ v


def decoder_ffn(w: dict, prefix, x):
    h = linear(x, w[prefix + ".w_1.weight"], w[prefix + ".w_1.bias"])
    h = gelu(h)
    return linear(h, w[prefix + ".w_2.weight"], w[prefix + ".w_2.bias"])


def topk_indices(x: mx.array, k: int, axis: int = -1):
    """Values + indices of top-k, sorted descending (mx.topk has no indices)."""
    idx = mx.argpartition(-x, kth=k - 1, axis=axis)
    idx = mx.take(idx, mx.arange(k), axis=axis)
    vals = mx.take_along_axis(x, idx, axis=axis)
    order = mx.argsort(-vals, axis=axis)
    idx = mx.take_along_axis(idx, order, axis=axis)
    vals = mx.take_along_axis(vals, order, axis=axis)
    return vals, idx


def batch_beam_search(w: dict, encoder_outputs, src_mask, beam_size: int = 3,
                      nbest: int = 1, decode_max_len: int = 0,
                      softmax_smoothing: float = 1.25,
                      length_penalty: float = 0.6, eos_penalty: float = 1.0,
                      sos_id: int = 3, eos_id: int = 4, pad_id: int = 2,
                      n_layers: int = 16, n_head: int = 20,
                      d_model: int = 1280, dec_prefix: str = "decoder"):
    B = beam_size
    N, Ti, H = encoder_outputs.shape
    NB = N * B
    d_k = d_model // n_head
    maxlen = decode_max_len if decode_max_len > 0 else Ti
    INF = 1e10

    enc = mx.repeat(encoder_outputs[:, None], B, axis=1).reshape(NB, Ti, H)
    smask = mx.repeat(src_mask[:, None], B, axis=1).reshape(NB, 1, Ti)
    smask_b = smask[:, None, :, :].astype(mx.bool_)  # (NB,1,1,Ti)

    # Cross-attn K/V depend only on encoder outputs: compute once per layer.
    cross_kvs = []
    for i in range(n_layers):
        p = f"{dec_prefix}.layer_stack.{i}.cross_attn"
        k = _heads(enc, w, p, "w_ks", n_head, d_k, biased=False)
        v = _heads(enc, w, p, "w_vs", n_head, d_k, biased=True)
        cross_kvs.append((k, v))

    ys = mx.full((NB, 1), sos_id, dtype=mx.int32)
    scores_list = [0.0] + [-INF] * (B - 1)
    scores = mx.array(scores_list * N, dtype=mx.float32).reshape(NB, 1)
    confidences = mx.zeros((NB, 1), dtype=mx.float32)
    is_finished = mx.zeros((NB, 1), dtype=mx.float32)
    # per-layer self-attn K/V caches: (K, V), each (NB, H, t, dk)
    self_kvs: list = [None] * n_layers

    # tie: prj weight == embedding weight
    emb_w = w[f"{dec_prefix}.tgt_word_emb.weight"]
    prj_w = w[f"{dec_prefix}.tgt_word_prj.weight"]
    scale = math.sqrt(d_model)
    pe = w["pe_abs"]

    for t in range(maxlen):
        T_seen = ys.shape[1]  # positions in the cache (incl. the new one)
        # decode only the newest row: embedding + PE at position T_seen-1
        h = emb_w[ys[:, -1:]] * scale + pe[:, T_seen - 1:T_seen, :]  # (NB,1,D)
        # newest query sees every non-pad key (causal row is all ones)
        q_mask = (ys != pad_id).astype(mx.bool_)[:, None, None, :]

        for i in range(n_layers):
            p = f"{dec_prefix}.layer_stack.{i}"
            # --- self-attn: K/V of the new row appended to the cache ---
            residual = h
            x = layer_norm(h, w[p + ".self_attn_norm.weight"],
                           w[p + ".self_attn_norm.bias"])
            q = _heads(x, w, p + ".self_attn", "w_qs", n_head, d_k, biased=True)
            k = _heads(x, w, p + ".self_attn", "w_ks", n_head, d_k, biased=False)
            v = _heads(x, w, p + ".self_attn", "w_vs", n_head, d_k, biased=True)
            prev = self_kvs[i]
            if prev is None:
                kc, vc = k, v
            else:
                kc = mx.concatenate([prev[0], k], axis=2)
                vc = mx.concatenate([prev[1], v], axis=2)
            self_kvs[i] = (kc, vc)
            o = _attention(q, kc, vc, q_mask, d_k)  # (NB,H,1,dk)
            o = o.transpose(0, 2, 1, 3).reshape(NB, 1, n_head * d_k)
            h = residual + linear(o, w[p + ".self_attn.fc.weight"],
                                  w[p + ".self_attn.fc.bias"])

            # --- cross-attn: K/V precomputed from encoder outputs ---
            residual = h
            x = layer_norm(h, w[p + ".cross_attn_norm.weight"],
                           w[p + ".cross_attn_norm.bias"])
            q = _heads(x, w, p + ".cross_attn", "w_qs", n_head, d_k, biased=True)
            ck, cv = cross_kvs[i]
            o = _attention(q, ck, cv, smask_b, d_k)
            o = o.transpose(0, 2, 1, 3).reshape(NB, 1, n_head * d_k)
            h = residual + linear(o, w[p + ".cross_attn.fc.weight"],
                                  w[p + ".cross_attn.fc.bias"])

            # --- ffn ---
            residual = h
            x = layer_norm(h, w[p + ".mlp_norm.weight"], w[p + ".mlp_norm.bias"])
            h = residual + decoder_ffn(w, p + ".mlp", x)

        h = layer_norm(h, w[f"{dec_prefix}.layer_norm_out.weight"],
                       w[f"{dec_prefix}.layer_norm_out.bias"])
        t_logit = h[:, -1, :] @ prj_w.T
        t_scores = log_softmax(t_logit / softmax_smoothing, axis=-1)

        if eos_penalty != 1.0:
            adj = mx.array(1.0, t_scores.dtype)
            t_scores = mx.concatenate(
                [t_scores[:, :eos_id],
                 t_scores[:, eos_id:eos_id + 1] * eos_penalty,
                 t_scores[:, eos_id + 1:]], axis=1)

        t_top_scores, t_top_ys = topk_indices(t_scores, B, axis=1)

        # finished beams keep only rank-0 score, force token to eos
        fin = is_finished.astype(mx.bool_)
        mask_score = mx.array(([0.0] + [-INF] * (B - 1)) * (N * B),
                              dtype=mx.float32).reshape(N * B, B)
        t_top_scores = t_top_scores * (1 - is_finished) + mask_score * is_finished
        t_top_ys = mx.where(fin, mx.full(t_top_ys.shape, eos_id, mx.int32),
                            t_top_ys)

        scores = scores + t_top_scores  # (NB, B)
        scores = scores.reshape(N, B * B)
        scores, top_ids = topk_indices(scores, B, axis=1)  # (N, B)
        scores = scores.reshape(-1, 1)

        rows = (top_ids // B).reshape(-1)  # (N*B,)
        stride = B * mx.arange(N).reshape(N, 1)
        rows_in_ys = (rows.reshape(N, B) + stride).reshape(-1)

        ys = ys[rows_in_ys]
        t_ys = mx.take_along_axis(
            t_top_ys.reshape(N, B * B), top_ids, axis=1).reshape(NB, 1)
        ys = mx.concatenate([ys, t_ys], axis=1)

        confidences = confidences[rows_in_ys]
        t_conf = mx.exp(mx.take_along_axis(
            t_top_scores.reshape(N, B * B), top_ids, axis=1)).reshape(NB, 1)
        confidences = mx.concatenate([confidences, t_conf], axis=1)

        # beam re-order: gather the K/V caches with the same row permutation
        self_kvs = [(kc[rows_in_ys], vc[rows_in_ys]) for kc, vc in self_kvs]

        is_finished = (t_ys == eos_id).astype(mx.float32)
        if float(is_finished.sum()) == NB:
            break

    # length penalty (GNMT) + nbest selection — small, do in numpy-friendly mx
    ys3 = ys.reshape(N, B, -1)
    ys_lengths = (ys3 != eos_id).sum(axis=-1)  # (N, B)
    if length_penalty > 0.0:
        pen = mx.power((5.0 + ys_lengths.astype(mx.float32)) / 6.0,
                       length_penalty)
        scores = (scores.reshape(N, B) / pen)
    nbest = int(nbest)
    nbest_scores, nbest_ids = topk_indices(scores, nbest, axis=1)

    nbest_ys = mx.take_along_axis(ys3, nbest_ids[:, :, None], axis=1)  # (N, nbest, Tmax)

    conf3 = confidences.reshape(N, B, -1)
    hyps = []
    ys_l = ys3.tolist()
    conf_l = conf3.tolist()
    nb_ids = nbest_ids.tolist()
    for n in range(N):
        entry = []
        for i in range(nbest):
            seq = ys_l[n][nb_ids[n][i]]
            L = int((mx.array(seq) != eos_id).sum())
            yseq = seq[1:L]  # drop sos, exclude eos (torch: ys[1:ys_lengths])
            # torch: mean of exp per-step scores over real tokens
            steps = conf_l[n][nb_ids[n][i]][1:L]
            confidence = sum(steps) / len(steps) if steps else 0.0
            entry.append({"yseq": yseq, "confidence": confidence})
        hyps.append(entry)
    return hyps
