# Development Process Artifacts

This document explains the **procedural, one-time, or reproducible** files and
scripts that live in this repository — what each one is for, when it was run
during the port, and which ones a fresh contributor actually needs to care
about. It exists so a reader does not mistake a one-off data-dump script for
a code path that runs in production.

The split is deliberate:

- **`fireredasr2_mlx/`** — the ported library. This is the only thing that
  ships at runtime. No developer scripts live here.
- **`golden/`** — the official-PyTorch reference outputs the port is
  measured against. Treat as read-only test fixtures; do not regenerate
  casually.
- **`scripts/`** — every one-off driver used during development. Most of
  these you will never run; they are kept for audit and for re-running the
  parity workflow.
- **`tests/`** — pytest entrypoints, including the golden-parity tests.
- **`docs/`** — this file, `MLX_PORT_PLAN.md` (the original porting plan),
  `PARITY_GAPS.md` (upstream-vs-port behavior differences), `HANDOFF.md`
  (how to pick up where this left off).

## `golden/` — official PyTorch baseline outputs

| Path | Producer | What it is |
|---|---|---|
| `golden/wavs/` | manual | Three short wavs (`hello_zh.wav`, `hello_en.wav`, `event.wav`) pulled from the upstream repo's `assets/` and used as the parity corpus. |
| `golden/output_torch/` | `scripts/run_official_cli.py` | The **entire** `output/` tree produced by running the *official* PyTorch `fireredasr2s_cli` on `golden/wavs` with all-GPU flags off. Includes `result.jsonl`, `asr_tg/*.TextGrid`, `asr_srt/*.srt`. Every byte here is the golden reference — the MLX output must match it field-for-field. |
| `golden/golden.pt` | `scripts/dump_golden.py` | A torch pickle containing AED intermediate tensors: `feats`, `lengths`, `encoder_out`, decoder inputs, beam-search results. Used by `tests/test_parity.py` for **layer-level** numeric parity, not just end-to-end text. |
| `golden/golden_llm.npz` | `scripts/dump_golden_llm.py` | Same idea for the LLM variant, but stored as a torch-free `.npz` so the MLX test can read it without a torch install. |
| `golden/golden_llm.json` | `scripts/dump_golden_llm.py` | Text/ids/shapes companion to `golden_llm.npz`. |

**Why they exist**: the port's contract is "byte-identical output to the
official PyTorch build." That requires comparing intermediate tensors, not
just final text — a bug that produces the same final text through a
different path is still a bug.

## `scripts/` — one-off development drivers

These were run **during** the port, once per stage. They are not part of the
shipped CLI.

| Script | When used | What it does |
|---|---|---|
| `scripts/chunked_dl.sh` | Once, before Phase A | Generic resumable multi-range download (24 chunks × `curl -r`) used to pull `model.pth.tar` and the Qwen2 weight shards from a mirror endpoint. Takes `<url> <size> <sha256> <dest> [nchunk] [par]`. Not specific to this project; kept for provenance of how the weights arrived. |
| `scripts/dump_golden.py` | Once per Phase A | Runs the **official** PyTorch AED on `golden/wavs/` with `torch.load(weights_only=False)` and dumps intermediate tensors to `golden/golden.pt`. Needs a torch environment and the upstream checkout in `../FireRedASR2S-main` (or `FIRERED_OFFICIAL_DIR`). |
| `scripts/dump_golden_llm.py` | Once for the LLM variant | Same as `dump_golden.py` but for the Encoder-Adapter-Qwen2-7B path. Emits `golden_llm.npz` + `golden_llm.json`. Heavier — needs a machine with enough memory to hold the official LLM in fp32. |
| `scripts/run_official_cli.py` | Every parity check | Thin launcher that `sys.path`-inserts the upstream checkout and calls `cli_main()`. Exists because the workflow runner could not `cd` + set `PYTHONPATH` in one step; today it's a convenience for re-running the baseline. |
| `scripts/fireredasr2s_mlx_cli.py` | Every parity check | Mirror-image launcher for the MLX side. Used in README's "Parity against the official implementation" example. |
| `scripts/test_parity_encoder.py` | Per-layer parity debug | Manual check: loads `golden/golden.pt`, runs only the encoder in MLX, prints max-abs-diff. Superseded by `tests/test_parity.py` but kept for quick encoder-only debugging. |

## `tests/` — pytest entrypoints

| File | Purpose |
|---|---|
| `tests/conftest.py` | `sys.path` injection + `slow` marker registration. |
| `tests/test_parity.py` | AED parity: compares MLX encoder/decoder/beam outputs against `golden/golden.pt`, and end-to-end `result.jsonl` against `golden/output_torch/`. Marked `slow`. |
| `tests/test_llm_parity.py` | LLM parity: 17 checks against `golden/golden_llm.npz`. Pinned to CPU (`FIREREDASR2_MLX_DEVICE=cpu`) because Metal fp32 matmul noise flips near-tied beams. |
| `tests/acceptance_check.py` | Standalone field-by-field diff between any two `result.jsonl` files (JSON number types count as different). Used in the README parity walkthrough. |

## `docs/` — design & handoff documents

| File | Purpose |
|---|---|
| `MLX_PORT_PLAN.md` | Original porting plan — unit-by-unit breakdown of what to hand-write vs reuse, phase schedule, risk list. Historical context; the port has since evolved (LLM variant added). |
| `PARITY_GAPS.md` | Authoritative list of upstream-vs-port differences: unsupported flags (`--elm_dir`), unimplemented upstream dead flags, precision bounds, Stream-VAD/mVAD exclusions. |
| `HANDOFF.md` | "Picking up where this left off" — how to run the CLI, what is known to work, what is incomplete. |
| `DEVELOPMENT_PROCESS.md` | This file. |

## Files you should *not* see committed

These are produced locally and must stay local — they either contain machine
paths or are large binary artifacts:

- `pretrained_models/` — model weights. `.gitignore`d. Download per README.
- `.venv/`, `__pycache__/`, `.pytest_cache/` — standard Python artifacts.
- `.DS_Store` — macOS metadata.
- `output/`, `output_*/` — any run output you produce locally.

## Reproducing the baseline (if you ever need to)

1. Clone the upstream repo next to this one:
   `git clone https://github.com/FireRedTeam/FireRedASR2S.git ../FireRedASR2S-main`
2. Set up a torch venv and run `scripts/dump_golden.py` to regenerate
   `golden/golden.pt`.
3. Run `scripts/run_official_cli.py` on `golden/wavs/` to regenerate
   `golden/output_torch/`.
4. For LLM parity, run `scripts/dump_golden_llm.py` on a machine with
   enough memory to hold Qwen2-7B in fp32.
5. From here on, the golden files are inputs to `tests/` — you do not
   need to run steps 2–4 again unless upstream or the test wavs change.
