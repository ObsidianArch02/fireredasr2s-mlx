"""CTC head + NumPy Viterbi alignment for timestamps.

MLX has no torchaudio.functional.forced_align equivalent; the alignment DP
below reproduces the official ctc_align Viterbi (identical transition rules),
and alignment_to_timestamp is a verbatim port of
CTC.ctc_alignment_to_timestamp.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from .conformer_encoder import linear


def ctc_logits(w: dict, enc_out: mx.array) -> mx.array:
    """enc_out: (T, D) single utterance -> log_probs (T, odim)."""
    logits = linear(enc_out, w["ctc.ctc_lo.weight"], w["ctc.ctc_lo.bias"])
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def forced_align_np(log_probs: np.ndarray, y: np.ndarray, blank_id: int = 0):
    """Viterbi forced alignment, torch-free port of CTC.ctc_align.
    log_probs (T, V); y: ids without blank. Returns per-frame token list
    (with blanks), i.e. ys_with_blank."""
    T, _ = log_probs.shape
    ys = [blank_id]
    for tok in y:
        ys += [int(tok), blank_id]
    S = len(ys)
    neg_inf = -1e10
    log_alpha = np.full((T, S), neg_inf, dtype=np.float64)
    log_alpha[0, 0] = log_probs[0, ys[0]]
    log_alpha[0, 1] = log_probs[0, ys[1]]
    state_path = np.full((T, S), -1, dtype=np.int32)
    for t in range(1, T):
        for s in range(S):
            if (ys[s] == blank_id or s < 2 or ys[s] == ys[s - 2]):
                cands = [log_alpha[t - 1, s], log_alpha[t - 1, s - 1]]
                prev = [s, s - 1]
            else:
                cands = [log_alpha[t - 1, s], log_alpha[t - 1, s - 1],
                         log_alpha[t - 1, s - 2]]
                prev = [s, s - 1, s - 2]
            best = max(cands)
            log_alpha[t, s] = best + log_probs[t, ys[s]]
            state_path[t, s] = prev[int(np.argmax(cands))]
    state_seq = np.zeros(T, dtype=np.int32)
    cands = [log_alpha[-1, S - 1], log_alpha[-1, S - 2]]
    state_seq[-1] = [S - 1, S - 2][int(np.argmax(cands))]
    for t in range(T - 2, -1, -1):
        state_seq[t] = state_path[t + 1, state_seq[t + 1]]
    return [ys[s] for s in state_seq]


def alignment_to_timestamp(ys_with_blank, subsampling, blank_id=0):
    """Verbatim port of CTC.ctc_alignment_to_timestamp (frame_shift=10ms)."""
    frame_shift = 10  # ms, hard coded in official
    T = len(ys_with_blank)
    t = 0
    start_times, end_times, ctc_durs = [], [], []
    while t < T:
        token = ys_with_blank[t]
        t += 1
        if token != blank_id:
            start_t = t
            timestamp = frame_shift * subsampling * t / 1000.0  # s
            start_times.append(timestamp)
            if len(start_times) == len(end_times) + 2:
                end_times.append(start_times[-1])
            while t < T and token == ys_with_blank[t]:
                t += 1
            ctc_durs.append((t - start_t + 1) * frame_shift * subsampling / 1000.0)
    end_times.append((frame_shift * subsampling * T + 25) / 1000.0)
    if len(start_times) == 0:
        start_times.append(0.0)
    assert len(ctc_durs) == len(end_times) and len(start_times) == len(end_times)
    avg_dur = sum(e - s for s, e in zip(start_times, end_times)) / len(end_times)
    new_end_times = []
    for s, e, ctc_dur in zip(start_times, end_times, ctc_durs):
        if e - s > 2 * avg_dur:
            e = s + max(1.5 * avg_dur, ctc_dur)
        new_end_times.append(round(e, 3))
    return start_times, new_end_times


def token_timestamps(w: dict, enc_out: mx.array, enc_len: int, yseq,
                     blank_id: int = 0, subsampling: int = 4):
    """Return (starts, ends) in seconds for one utterance, before the official
    60ms shift (applied by the facade, mirroring _get_and_fix_timestamp)."""
    lp = ctc_logits(w, enc_out)
    lp_np = np.asarray(lp[:enc_len].astype(mx.float32))
    y = np.array([i for i in yseq if i != blank_id], dtype=np.int64)
    if len(y) == 0 or lp_np.shape[0] == 0 or len(y) > lp_np.shape[0]:
        return None
    ys_with_blank = forced_align_np(lp_np, y, blank_id)
    return alignment_to_timestamp(ys_with_blank, subsampling, blank_id)
