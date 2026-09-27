"""FireRedASR2-LLM in pure MLX: speech-feature merge + HF-faithful beam search.

Port of fireredasr2s/fireredasr2/models/fireredasr_llm.py (merge) and of the
transformers 5.x `_beam_search` loop (GenerationMixin, do_sample=False,
single eos, early_stopping=False) with KV-cached decoding and per-step beam
cache reordering (HF `_reorder_cache`).

Batch note: the official merge supports batch > 1, but the 4-module system
feeds VAD segments one batch element at a time (asr_batch_size) and the
per-sample math is identical, so this port processes one utterance at a
time, mirroring the official arithmetic with batch_size == 1.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from .qwen2 import (qwen2_decode_step, qwen2_logprobs_sequence,
                    qwen2_prefill, qwen2_reorder_cache)


# ----------------------------------------------------------- merge (B=1) ---

def merge_input_ids_with_speech_features(
        speech_features, inputs_embeds, input_ids, attention_mask,
        speech_token_id: int, pad_token_id: int, speech_lens):
    """Port of FireRedAsrLlm._merge_input_ids_with_speech_features.

    speech_features: (1, S, D) adapter output (padded).
    inputs_embeds:   (1, U, D) text embeddings.
    input_ids:       (1, U) int array, exactly one speech token.
    attention_mask:  (1, U)
    speech_lens:     (1,) valid speech frames per utterance.
    Returns (final_embedding (1, M, D), final_attention_mask (1, M)).
    """
    ids = np.asarray(input_ids)
    am = np.asarray(attention_mask)
    embs = np.asarray(inputs_embeds)
    sf = np.asarray(speech_features)
    B, U = ids.shape
    assert B == 1, "MLX port processes one utterance at a time"

    _, speech_len, embed_dim = sf.shape
    special_mask = ids == speech_token_id                     # (1, U)
    num_special = special_mask.sum(axis=-1)                   # (1,)
    max_embed_dim = int(num_special[0] * (speech_len - 1) + U)

    batch_indices, non_speech_indices = np.where(ids != speech_token_id)
    new_token_positions = np.cumsum(special_mask * (speech_len - 1) + 1,
                                    axis=-1) - 1              # (1, U)
    nb_speech_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    left_padding = not np.sum(ids[:, -1] == pad_token_id)
    if left_padding:
        new_token_positions += nb_speech_pad[:, None]
    text_to_overwrite = new_token_positions[batch_indices,
                                            non_speech_indices]

    final_embedding = np.zeros((B, max_embed_dim, embed_dim),
                               dtype=embs.dtype)
    final_attention_mask = np.zeros((B, max_embed_dim), dtype=am.dtype)
    final_embedding[batch_indices, text_to_overwrite] = \
        embs[batch_indices, non_speech_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = \
        am[batch_indices, non_speech_indices]

    speech_to_overwrite = np.ones((B, max_embed_dim), dtype=bool)
    speech_to_overwrite[batch_indices, text_to_overwrite] = False
    speech_to_overwrite &= \
        speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_pad[:, None]
    speech_cumsum = speech_to_overwrite.astype(np.int64).cumsum(-1)
    speech_position_counter = np.where(speech_to_overwrite,
                                       speech_cumsum - 1, 0)
    sl = np.asarray(speech_lens).astype(np.int64)
    valid = speech_position_counter < sl[:, None]
    speech_to_overwrite &= valid
    assert speech_to_overwrite.sum() == int(sl.sum()), \
        "speech_lens and speech token distribution mismatch"
    bidx, sidx = np.where(speech_to_overwrite)
    sfeat_idx = speech_position_counter[speech_to_overwrite]
    final_embedding[bidx, sidx] = sf[bidx, sfeat_idx]
    final_attention_mask[speech_to_overwrite] = 1

    # mask out padding positions (none for B=1 decode prompts, kept verbatim)
    batch_indices_pad, pad_indices = np.where(ids == pad_token_id)
    if len(batch_indices_pad) > 0:
        indices_to_mask = new_token_positions[batch_indices_pad, pad_indices]
        final_embedding[batch_indices_pad, indices_to_mask] = 0
        final_attention_mask[batch_indices_pad, indices_to_mask] = 0

    return (mx.array(final_embedding), mx.array(final_attention_mask))


# ------------------------------------------------------------- beam search ---

def _topk_stable(scores: np.ndarray, k: int):
    """Descending top-k with ascending index order on ties (torch.topk CPU)."""
    order = np.argsort(-scores, kind="stable")[:k]
    return scores[order], order


def beam_search(w: dict, prompt_embeds, cfg: dict, max_new_tokens: int,
                num_beams: int, length_penalty: float,
                eos_id: int, pad_id: int,
                repetition_penalty: float = 1.0, min_length: int = 0):
    """Port of transformers 5.x GenerationMixin._beam_search (batch=1,
    do_sample=False, single eos id, early_stopping=False).

    prompt_embeds: (1, S, D) merged speech+text embeddings.
    Returns (token_ids (L,),) — the best finished hypothesis, eos excluded.
    """
    B = num_beams
    K = max(2, 2) * B               # single eos -> (1 + 1) * num_beams
    V = cfg["vocab_size"]
    maxlen = max(1, max_new_tokens)
    NEG = -1.0e9

    f32 = np.float32
    running_sequences = np.full((B, maxlen), pad_id, dtype=np.int64)
    running_beam_scores = np.full(B, NEG, dtype=f32)
    running_beam_scores[0] = 0.0
    sequences = running_sequences.copy()
    beam_scores = np.full(B, NEG, dtype=f32)
    is_sent_finished = np.zeros(B, dtype=bool)
    running_beam_indices = np.full((B, maxlen), -1, dtype=np.int64)
    beam_indices = running_beam_indices.copy()
    early_stop_unsatisfied = True

    S = prompt_embeds.shape[1]
    emb_w = w["model.embed_tokens.weight"]
    cur_len = 0

    # prefill the shared prompt once, then replicate the cache per beam
    logits, caches = qwen2_prefill(w, prompt_embeds, cfg)
    for c in caches:
        c[0] = mx.broadcast_to(c[0], (B,) + c[0].shape[1:])
        c[1] = mx.broadcast_to(c[1], (B,) + c[1].shape[1:])

    while True:
        # a. forward: first step consumes prefill logits, then cached steps
        if cur_len > 0:
            toks = mx.array(running_sequences[:, cur_len - 1],
                            dtype=mx.int32)
            logits, _ = qwen2_decode_step(
                w, emb_w[toks][:, None, :], caches, cfg)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # b. logits processors (transformers 5.x applies them to log-probs,
        #    per beam over that beam's own generated tokens)
        if repetition_penalty != 1.0 and cur_len > 0:
            lp = np.asarray(log_probs)
            for b in range(B):
                seen = np.unique(running_sequences[b, :cur_len])
                lp[b, seen] = np.where(lp[b, seen] < 0,
                                       lp[b, seen] * repetition_penalty,
                                       lp[b, seen] / repetition_penalty)
            log_probs = mx.array(lp)
        if min_length > 0 and cur_len < min_length:
            lp = np.asarray(log_probs)
            lp[:, eos_id] = -np.inf
            log_probs = mx.array(lp)

        lp = np.asarray(log_probs)                       # (B, V) float32
        acc = (lp + running_beam_scores[:, None]).reshape(-1)

        # c. top-K continuations
        topk_vals, topk_flat = _topk_stable(acc, K)
        cand_beam = topk_flat // V
        cand_tok = topk_flat % V

        topk_sequences = running_sequences[cand_beam]
        topk_sequences[:, cur_len] = cand_tok
        topk_beam_indices = running_beam_indices[cand_beam]
        topk_beam_indices[:, cur_len] = cand_beam

        # d. stopping criteria (EosTokenCriteria | MaxLengthCriteria)
        hits = (cand_tok == eos_id) | (cur_len + 1 >= maxlen)

        # e. running beams for next iteration
        run_vals = topk_vals + np.where(hits, NEG, 0.0)
        run_sel = _topk_stable(run_vals, B)[1]
        running_sequences = topk_sequences[run_sel]
        running_beam_scores = run_vals[run_sel]
        running_beam_indices = topk_beam_indices[run_sel]
        qwen2_reorder_cache(caches, cand_beam[run_sel])

        # f. update finished beams
        lp_norm = topk_vals / ((cur_len + 1) ** length_penalty)
        if not early_stop_unsatisfied:
            lp_norm = lp_norm + NEG
        top_b_finished = hits & (np.arange(K) < B)
        fin_vals = lp_norm + np.where(top_b_finished, 0.0, NEG)

        merged_sequences = np.concatenate([sequences, topk_sequences])
        merged_scores = np.concatenate([beam_scores, fin_vals])
        merged_indices = np.concatenate([beam_indices, topk_beam_indices])
        merged_finished = np.concatenate([is_sent_finished, top_b_finished])
        sel = _topk_stable(merged_scores, B)[1]
        sequences = merged_sequences[sel]
        beam_scores = merged_scores[sel]
        beam_indices = merged_indices[sel]
        is_sent_finished = merged_finished[sel]

        cur_len += 1

        # g. early-stop heuristic + loop condition
        best_hypothetical_length = cur_len        # prompt_len == 0
        best_possible = running_beam_scores[0] \
            / (best_hypothetical_length ** length_penalty)
        if is_sent_finished.any():
            worst_finished = beam_scores[is_sent_finished].min()
        else:
            worst_finished = NEG
        early_stop_unsatisfied = early_stop_unsatisfied and \
            bool(best_possible > worst_finished)
        improvement_possible = early_stop_unsatisfied
        valid_continuations = not bool(hits.all())
        if not (improvement_possible and valid_continuations):
            break

    gen_len = int(((beam_indices[0] + 1) > 0).sum())
    return sequences[0, :gen_len].tolist()


# --------------------------------------------------------------- transcribe ---

def llm_transcribe(w_enc, w_llm, cfg, feats, feat_lengths, input_ids,
                   attention_mask, beam_size, decode_max_len,
                   decode_min_len=0, repetition_penalty=1.0,
                   llm_length_penalty=0.0):
    """Full LLM ASR path for ONE utterance.

    w_enc: encoder weight dict (encoder.* keys + pe_rel)
    w_llm: LLM weight dict (HF names) + encoder_projector.*
    cfg:   dict with llm config (n_layers/n_head/n_kv_head/head_dim/rope_theta/
           rms_eps/vocab_size), speech_token_id, eos_id, pad_id, adapter ds.
    input_ids:     (1, U) mx.array with exactly one speech token.
    attention_mask:(1, U)
    Returns (token_ids (L,), confidence) — the best finished hypothesis, eos
    excluded; confidence = mean per-token probability of the sequence.
    """
    from .conformer_encoder import encoder_forward
    from .adapter import adapter_forward

    enc_out, enc_lengths, _ = encoder_forward(w_enc, feats, feat_lengths,
                                              n_layers=16)
    speech_features, speech_lens = adapter_forward(
        w_llm, enc_out, enc_lengths, ds=cfg.get("adapter_ds", 2))

    emb_w = w_llm["model.embed_tokens.weight"]
    inputs_embeds = emb_w[input_ids.astype(mx.int32)]              # (1, U, D)
    merged, merged_mask = merge_input_ids_with_speech_features(
        speech_features, inputs_embeds, np.asarray(input_ids),
        np.asarray(attention_mask), cfg["speech_token_id"],
        cfg["pad_id"], speech_lens)

    max_new = int(speech_features.shape[1]) if decode_max_len < 1 \
        else decode_max_len
    ids = beam_search(w_llm, merged, cfg, max_new, beam_size,
                      llm_length_penalty, cfg["eos_id"], cfg["pad_id"],
                      repetition_penalty=repetition_penalty,
                      min_length=decode_min_len)

    if len(ids) == 0:
        return [], 0.0
    lp = qwen2_logprobs_sequence(w_llm, merged, ids, cfg)
    confidence = float(mx.exp(lp).mean())
    return ids, confidence
