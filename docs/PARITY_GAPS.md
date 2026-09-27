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

### 1. `--asr_type llm`: FireRedASR2-LLM (8B Encoder-Adapter-LLM)
- Upstream implementation: Conformer encoder + Adapter + Qwen2-7B-Instruct,
  CUDA-only serving (vLLM `-tp=2 --dtype=float32` / TensorRT). **The official
  build itself cannot run on Mac.**
- MLX behavior: the flag is accepted, then errors out with an explanation.
- Feasibility assessment (if ever done): the encoder part is already ported in
  this repo; the Adapter is about 30 lines; Qwen2 is a first-class mlx-lm
  architecture. The real blockers are not technical:
  1. Upstream has no runnable baseline on Mac, so a "byte-identical" acceptance
     gate is in principle impossible for this variant;
  2. 8B autoregressive decoding would push warm RTF from AED's 0.12 back to the
     ~1 range (noticeably slower);
  3. bf16 weights ~15GB + encoder 4.4GB - fits on a smaller-memory machine but with no
     headroom; quantization deviates further (with no baseline to measure the
     deviation against).
- Upstream benefit for reference: average CER on 4 Mandarin benchmarks,
  LLM 2.89% vs AED 3.05% (bigger gap on dialect sets).

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
