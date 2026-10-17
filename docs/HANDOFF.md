# Handoff — fireredasr2s-mlx

Date: 2026-10-16. Last commit: `fe2463b`.

## What this is

A pure-MLX port of the official PyTorch [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)
four-module ASR system (VAD -> LID -> AED ASR -> Punc) for Apple Silicon.
The CLI mirrors the official `fireredasr2s_cli.py` flag for flag and produces
byte-identical `result.jsonl`, TextGrid, and SRT on the three golden samples.
The LLM variant (`--asr_type llm`, Conformer + Adapter + Qwen2-7B-Instruct) is also ported.
Runtime has zero torch imports. torch is needed once to convert checkpoints.

## Repo layout

| Path | Contents |
|---|---|
| `fireredasr2_mlx/` | Package: `cli.py`, `system.py`, `llm.py`, `loader.py`, `models/` (conformer_encoder, transformer_decoder, ctc, vad, lid, fireredasr_llm, adapter, qwen2), `punc/` (bert_mlx, punc_mlx) |
| `docs/MLX_PORT_PLAN.md` | Original port plan: operator-by-operator notes, phase list, risk table |
| `docs/PARITY_GAPS.md` | The diff list against the official CLI: not-ported items, semantic differences, accuracy bounds |
| `golden/` | 3 sample wavs, `output_torch/` golden outputs, `golden_llm.npz` for the LLM variant |
| `tests/` | `test_parity.py` (AED golden parity), `test_llm_parity.py` (17 LLM parity items), `acceptance_check.py` |
| `scripts/` | `run_official_cli.py` (runs the upstream PyTorch CLI), `dump_golden*.py`, `fireredasr2s_mlx_cli.py` entry |
| `pretrained_models/` | VAD / LID / AED / Punc / LLM weights (local, not committed) |

## How to run and verify

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    mlx numpy soundfile textgrid sentencepiece \
    kaldi-native-fbank kaldiio safetensors tokenizers pytest

# Single-file run
.venv/bin/python -m fireredasr2_mlx.cli \
    --wav_path golden/wavs/hello_zh.wav --outdir output

# Golden parity: the slow test pins itself to CPU internally (see below)
.venv/bin/python -m pytest tests/ -m "not slow"   # fast checks
.venv/bin/python -m pytest tests/                  # includes the slow golden run
```

To compare against the official PyTorch build, clone upstream one level up
(`../FireRedASR2S-main` or set `FIRERED_OFFICIAL_DIR`), then use
`scripts/run_official_cli.py` + `tests/acceptance_check.py`.

## Known bounds, in priority order

1. **Metal fp32 matmul noise (~1e-2).** Near-tied beams can flip on GPU.
   The `event` sample decodes to a degenerate repeated sequence upstream.
   On GPU its beam may stop early. Exact byte parity is pinned to CPU through
   `FIREREDASR2_MLX_DEVICE=cpu` (the golden tests set this internally).
   GPU output is held by smoke tests only.
2. **Golden coverage is 3 samples, max 22 s.** Longer audio has no official
   baseline. Confidence on long input relies on the same decode path.
3. **External LM shallow fusion not ported.** `--elm_dir/--elm_weight` are
   accepted, warn, and are ignored. Upstream never shipped the LstmLm weights.
   The port itself is simple if weights ever appear.
4. **Stream-VAD and mVAD heads not ported.** Only the non-streaming VAD used
   by the four-module CLI is done. Streaming cache machinery is out of scope.
5. **Quantization and bf16/fp16 not delivered.** `--asr_use_half` is a no-op.
   `mx.quantize` works, but any accuracy drop breaks the byte-parity gate.
6. **No `mx.compile` on the encoder yet.** Performance tuning was not done.
   The RTF <= 0.15 target in the plan has no recorded measurement.
7. **`*_use_gpu` flags are no-ops.** MLX always runs on Metal unless the
   env var forces CPU. This is documented behavior, not a bug.

## Done, though the plan marked it "second version"

- True KV cache in the AED decoder (`transformer_decoder.py`): per-layer
  K/V appended each step, row-gathered on beam reorder.
- LLM variant: handwritten Qwen2 forward (KV cache, RoPE, RMSNorm, SwiGLU)
  aligned to transformers 5.x semantics. LoRA is pre-merged at load. Beam search
  is a line-for-line port of transformers 5.17 `_beam_search`.
- Byte-parity hardening after three audit rounds (see PARITY_GAPS change log).

## Likely next tasks

- External LM fusion, if upstream releases the LstmLm weights.
- Optional quantization (4/8-bit) behind a flag, with a non-parity quality gate.
- `mx.compile` on the Conformer block, then an RTF measurement pass.
- Long-audio golden generation on a large-memory machine to extend coverage.
- Stream-VAD / mVAD ports, if streaming or multi-label event output is needed.

## Gotchas

- `model.pth.tar` is pickle. Conversion needs `weights_only=False`.
  Only convert checkpoints from the official FireRedTeam HuggingFace repo.
- The LLM variant holds about 35 GB resident in fp32. Use a machine with sufficient memory.
  Keep `--enable_vad 1` for long audio.
- The official four-module CLI has two dead flags (`--punc_with_timestamp`,
  `--sort_wav_by_dur`). This port wires both, a documented superset.
- The mlx dependency is pinned `>=0.32,<0.40` on purpose: the parity tests
  rely on cross-device op semantics from that range (see `pyproject.toml`).

## Similar projects upstream (checked 2026-10-16)

- `mlx-community/FireRedASR2-AED-mlx` (HF): converted F32 weights only, no
  runner. The companion runner `Kn90688/mlx-fireredasr` is deleted.
- `vanch007/mlx-FireRedAudio`: sibling FireRedAudio model in MLX. Best
  engineering reference for fused attention and loader structure.
- `finch-xu/openmeow`: macOS voice server that lists FireRedASR v2 as a
  backend. Application layer, not a port.
- No public four-module FireRedASR2S MLX port exists besides this one.
