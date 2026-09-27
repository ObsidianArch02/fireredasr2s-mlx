# Differences from the official PyTorch CLI (not-ported items)

Compared against: official `fireredasr2s/fireredasr2s_cli.py` (the FireRedASR2S
four-module system, default AED path).

Parts verified **identical** are out of scope here: the CLI flag surface, all
`result.jsonl` fields (text / sentences / vad_segments_ms / words / dur_s / uttid /
wav_path), TextGrid, SRT, character-level timestamps, confidence, and the behavior
of the VAD/LID/Punc modules - all covered by field-level / byte-level parity tests
(see `tests/` and the "Parity against the official implementation" section).

This document lists only items that **exist upstream but are missing here, or have
different semantics**.

---

## A. Intentionally not ported

### 1. ~~`--asr_type llm`: FireRedASR2-LLM~~ (ported 2026-09-27)

FireRedASR2-LLM (Conformer encoder + Adapter + Qwen2-7B-Instruct + LoRA) is now
fully ported to MLX: `--asr_type llm --asr_model_dir pretrained_models/FireRedASR2-LLM`
runs directly (fp32, Metal GPU, zero torch). Implementation notes:

- **Qwen2 decoder**: hand-written MLX forward (eager attention, KV cache, RoPE,
  RMSNorm, SwiGLU) matching transformers 5.x numerics; the checkpoint's LoRA
  weights (r=64, alpha=16, target q/k/v/o/gate/up/down) are pre-merged into the
  base weights at load time.
- **beam search**: line-for-line translation of transformers 5.17 `_beam_search`
  (do_sample=False, single eos, length_penalty, early-stop heuristic, KV cache
  beam reordering).
- **Acceptance method**: upstream has no runnable baseline on Mac, so a golden
  baseline was self-generated on this machine (a larger-memory machine) with the official PyTorch
  code at CPU fp32 (`scripts/dump_golden_llm.py` -> `golden/golden_llm.npz`).
  MLX **CPU** fp32 is token-for-token identical (3/3 samples, ids + text,
  encoder forward max-abs-diff 2e-6; see `tests/test_llm_parity.py`).
- **Known boundary**: Metal fp32 matmul carries ~1e-2 noise vs CPU fp32; the two
  short utterances match upstream verbatim on GPU, but the `event` sample is a
  degenerate repeated sequence (upstream itself outputs "happy birthday" x30)
  whose near-tied beam scores may be perturbed into stopping early (the token
  sequences still match). Exact parity tests are therefore pinned to CPU; GPU
  behavior is locked by a smoke test.
- **Two intentional deviations** (the official 4-module CLI's LLM path crashes
  itself, so there is no "official behavior" to match):
  1. LLM results carry `confidence` (mean per-token probability of the best
     hypothesis) - upstream results lack this field and would hit a KeyError
     when assembling sentences;
  2. Assembly is forced through the no-timestamp branch (the LLM has no
     word-level timestamps anyway).

### 2. External LM shallow fusion (`--elm_dir` / `--elm_weight`, LstmLm)
- The official AED beam search can fuse the log-prob of an LSTM external LM
  (with a 3x EOS weighting).
- Upstream did not release that LM's weights with FireRedASR2S (`elm_dir`
  defaults to empty; unusable for normal users).
- MLX behavior: flag accepted, unsupported notice printed.
- If upstream ever releases the weights, porting is straightforward (LSTM +
  score fusion; MLX has the operators).

### 3. Streaming VAD and multi-label mVAD (sound-event classification)
- FireRedVAD actually ships three models: `VAD` (ported here), `Stream-VAD`
  (streaming), and `AED` (speech/singing/music multi-label). The official
  4-module CLI only uses the non-streaming `VAD`; Stream-VAD and mVAD are
  driven by the separate official `fireredvad` CLI, outside the
  `fireredasr2s_cli.py` path.
- MLX side: only the non-streaming VAD used on the CLI path is ported.
  Stream-VAD (including its streaming cache mechanism) and the mVAD head are
  not ported.

### 4. Standalone per-module CLIs (the `fireredvad` bin tools)
- Upstream also ships standalone CLIs that run only VAD / AED / Stream-VAD.
  This repo ships only the unified 4-module CLI; single-module use goes through
  the Python API (`FireRedVadMlx.detect` etc.).

### 5. TensorRT / Triton serving (official `runtime/triton_tensorrt/`)
- CUDA-only deployment stack; not applicable to MLX. Explicitly out of scope.

### 6. Training / fine-tuning
- Upstream ships LoRA fine-tuning and training scripts (LLM path, peft). This
  repo covers inference only.

---

## B. Flags accepted but with different semantics (default output unaffected)

| Flag | Official semantics | MLX semantics |
|---|---|---|
| `--asr_use_gpu` `--vad_use_gpu` `--lid_use_gpu` `--punc_use_gpu` | 1=CUDA | accepted and ignored: MLX always runs on Metal GPU (no CPU inference mode) |
| `--asr_use_half` | 1=bf16/fp16 | accepted and ignored: numeric precision is always fp32 (parity rule) |
| `--punc_with_timestamp` | defined in parser but unused in `main()` (dead flag; behavior decided by `--return_timestamp`) | wired into the Punc config and effective (superset of upstream) |
| `--sort_wav_by_dur` | defined in parser but unused in `main()` (dead flag) | implemented, sorts by audio duration (superset of upstream) |

---

## C. Precision and acceptance bounds

- The whole pipeline is **fp32, no quantization**. Quantization (8-bit/4-bit) is
  technically feasible (`mx.quantize`), but any precision loss conflicts with
  the "byte-identical to upstream" acceptance gate, so it is off by default and
  not shipped.
- Byte-level parity covers the 3 official sample wavs (longest 22s). Longer
  audio has no golden parity (confidence there relies on extrapolating the same
  decoding path without an upstream baseline to compare).

---

## Changelog

- 2026-09-27: first version. Reflects the 4-module AED system state after `3b48b98`.
- 2026-09-27: `--asr_type llm` port finished (CPU fp32 golden self-generated on a
  larger-memory machine; MLX CPU identical / GPU smoke-pinned); removed from "not ported"
  and documented here.
- 2026-09-27: fixes after a three-pass audit (official parity / idempotency /
  security): beam search early-stop semantics aligned line-for-line with
  transformers 5.17 `_check_early_stop_heuristic` (min over ALL finished slots);
  merge aligned with the official forced `speech_lens=None` semantics;
  repetition_penalty applied per beam; hand-rolled safetensors reader gained a
  header length cap and shard path allowlist; LLM facade gained upstream-style
  per-utterance exception fallback; ultra-short-audio guard; system no longer
  mutates the caller's config.
