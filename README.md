# FireRedASR2S - MLX Port

A **pure MLX port** of Xiaohongshu's [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S),
the SOTA four-module Chinese-English mixed ASR system, targeting Apple Silicon (Metal GPU)
with zero torch dependency at runtime.

- The official PyTorch reference implementation is kept read-only at [FireRedASR2S-main/](https://github.com/FireRedTeam/FireRedASR2S) (unmodified).
- The CLI surface matches the official `fireredasr2s_cli.py` **flag for flag**: swap the official command line for this one and it just runs.
- Field-level parity on 3 test wavs: all `result.jsonl` fields, TextGrid, and SRT match the official PyTorch output (golden outputs in [output_torch/](./golden/output_torch)).

## Install (uv)

Requires macOS (Apple Silicon) + Python 3.12 + [uv](https://docs.astral.sh/uv/).

```bash
cd fireredasr2s-mlx
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    mlx numpy soundfile textgrid sentencepiece \
    kaldi-native-fbank kaldiio safetensors tokenizers
```

> `torch` is imported on demand only for the one-time checkpoint conversion
> (`fireredasr2_mlx.loader.convert_checkpoint`); the inference path (importing the
> `fireredasr2_mlx` package and running the CLI) never touches torch.

Model weights resolve by default to `pretrained_models/FireRedVAD/VAD`,
`pretrained_models/FireRedLID`, `pretrained_models/FireRedASR2-AED`, and
`pretrained_models/FireRedPunc` relative to the project root; see
[download_model.sh](./download_model.sh) / [download_model_hf.sh](./download_model_hf.sh).

## The four modules

| Module | Role | Official counterpart | MLX implementation |
| --- | --- | --- | --- |
| VAD | Voice activity detection, cuts out speech segments (non-streaming) | `fireredvad` | `fireredasr2_mlx/models/vad.py` |
| LID | Language identification (e.g. `zh mandarin` / `en`) | `fireredlid` | `fireredasr2_mlx/models/lid.py` |
| ASR | AED (Conformer encoder + attention/CTC decoder) with word-level timestamps, zh-en mixed | `fireredasr2` | `fireredasr2_mlx/models/` |
| Punc | BERT punctuation restoration, assembles sentence-level subtitles by timestamp | `fireredpunc` | `fireredasr2_mlx/punc/` |

Pipeline: `wav -> VAD segmentation -> per-segment ASR (+LID) -> Punc -> result.jsonl / TextGrid / SRT (/ segment wavs)`.

## Relationship to the official CLI

- **Flag-compatible**: the names, semantics, defaults, and choices of all 43 flags
  match the official `fireredasr2s_cli.py` exactly (verified by script).
- **Identical output**: for the same input, `result.jsonl`, `asr_tg/*.TextGrid`, and
  `asr_srt/*.srt` match the official PyTorch build field for field.
- **Different backend**: the official build is PyTorch (CUDA/CPU); this one is MLX
  (Metal/CPU). Therefore `--asr_use_gpu/--asr_use_half/--vad_use_gpu/--lid_use_gpu/--punc_use_gpu`
  are accepted and ignored - MLX always runs on Metal (`use_gpu=0` needs no CPU
  fallback path; numeric results are unchanged).
  - Exception: Metal fp32 matmul carries ~1e-2 noise relative to CPU, which can flip
    near-tied beams on hard samples (the event song segment); to reproduce the
    official golden byte-for-byte, set `FIREREDASR2_MLX_DEVICE=cpu` (this is how the
    golden tests run; CPU is field-identical).

## Usage examples

```bash
# 1) single wav -> output/ (result.jsonl + TextGrid + SRT)
.venv/bin/python -m fireredasr2_mlx.cli \
    --wav_path golden/wavs/hello_zh.wav --outdir output

# 2) multiple wavs + save VAD segment wavs (--sort_wav_by_dur 1 processes longest first)
.venv/bin/python -m fireredasr2_mlx.cli \
    --wav_paths golden/wavs/hello_zh.wav golden/wavs/hello_en.wav golden/wavs/event.wav \
    --sort_wav_by_dur 1 --save_segment 1 --outdir output

# 3) directory batch + scp list, disable outputs you do not need
.venv/bin/python -m fireredasr2_mlx.cli --wav_dir golden/wavs --outdir output
.venv/bin/python -m fireredasr2_mlx.cli --wav_scp wavs.scp --write_textgrid 0 --write_srt 0 --outdir output

# 4) FireRedASR2-LLM variant (Qwen2-7B decoder; download weights and run the one-time
#    conversion first, see next section)
.venv/bin/python -m fireredasr2_mlx.cli \
    --wav_path golden/wavs/hello_zh.wav --outdir output \
    --asr_type llm --asr_model_dir pretrained_models/FireRedASR2-LLM
```

## FireRedASR2-LLM variant (--asr_type llm)

Besides the default AED path, the four-module system supports the official LLM
variant (Conformer + Adapter + Qwen2-7B-Instruct, LoRA weights shipped in the
checkpoint). Setup:

```bash
# 1) download weights (~19GB, includes Qwen2-7B-Instruct bf16 shards and model.pth.tar)
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('FireRedTeam/FireRedASR2-LLM', \
    local_dir='pretrained_models/FireRedASR2-LLM')"

# 2) one-time conversion (needs a torch env; splits encoder/adapter+LoRA into safetensors)
.venv/bin/python -c "from fireredasr2_mlx.loader import convert_llm_checkpoint; \
    convert_llm_checkpoint('pretrained_models/FireRedASR2-LLM/model.pth.tar', \
    'pretrained_models/FireRedASR2-LLM')"
```

> Security note: `model.pth.tar` is a pickle file, so converting it can execute
> arbitrary code (`weights_only=False` is required). Only convert checkpoints from
> the trusted official FireRedTeam HuggingFace repo.
> Memory note: the LLM variant needs ~35GB resident in fp32 (plus ~12GB peak for
> checkpoint conversion/loading); use a machine with sufficient RAM. Keep VAD
> on for long audio (`--enable_vad 1`, segmented decoding) - decoding several
> minutes of audio in one shot would produce an oversized sequence because
> `decode_max_len=0` (the cap equals the number of speech frames), same as upstream.

Then run example 4. The LLM variant has no word-level timestamps
(`--return_timestamp` is a no-op) and results carry an extra `confidence` field
(mean per-token probability). On official benchmarks the LLM's average CER is
slightly better than AED (about 2.89% vs 3.05% on 4 Mandarin sets). Numerical
acceptance details and bounds are in [docs/PARITY_GAPS.md](docs/PARITY_GAPS.md).

## Known boundaries

- **`--asr_type llm` (FireRedASR2-LLM, Qwen2-7B decoder) is supported**: the same
  four-module pipeline as AED, with a flag surface matching upstream
  (`--beam_size`/`--decode_max_len` etc. are shared flags). LLM has no word-level
  timestamps; `--return_timestamp` is a no-op; results carry `confidence` (mean
  per-token probability). Weights need the one-time conversion after download (see
  above). Exact parity (token-for-token equal to official PyTorch CPU fp32) holds on
  CPU; Metal fp32 matmul noise of ~1e-2 may change beam near-tie outcomes on
  degenerate repeated sequences (see [docs/PARITY_GAPS.md](docs/PARITY_GAPS.md)).
- **External LM (`--elm_dir/--elm_weight`) unsupported**: the flags are accepted,
  print a notice, and are ignored; decoding continues as AED without an external LM.
- `--sort_wav_by_dur 1` orders processing by actual audio duration, longest first
  (the official 4-module CLI parses this flag but never applies it; the sort
  direction matches its standalone script `fireredasr2/utils/io.py`). It affects
  processing order only, not per-utterance results.
- `--punc_with_timestamp 0` forces punctuation through the no-timestamp branch
  (one sentence for the whole segment); the official 4-module system parses but
  never wires this flag - this port implements it.

## Differences from the official PyTorch CLI

The full list of not-ported items and semantic differences is in
[docs/PARITY_GAPS.md](docs/PARITY_GAPS.md). The purpose of each one-off
development script and golden artifact is documented in
[docs/DEVELOPMENT_PROCESS.md](docs/DEVELOPMENT_PROCESS.md). Key points: `--asr_type llm` (8B LLM
variant) and external LM fusion (`--elm_dir`) are unsupported; Stream-VAD / mVAD /
TensorRT / training code are out of scope; the four `*_use_gpu` flags and
`--asr_use_half` are accepted as no-ops (MLX always runs Metal, always fp32).

## Parity against the official implementation

This repo does not ship the official code. To run field-level parity against the
official PyTorch build, clone the upstream repo separately:

```bash
git clone https://github.com/FireRedTeam/FireRedASR2S.git ../FireRedASR2S-main
```

`scripts/run_official_cli.py` looks for the official checkout in the directory
above this repo by default; set `FIRERED_OFFICIAL_DIR` to point anywhere else.
Keep the weights layout in `pretrained_models/` (the official CLI's default model
paths match it).

```bash
.venv/bin/python scripts/run_official_cli.py \
    --wav_paths golden/wavs/hello_zh.wav golden/wavs/hello_en.wav golden/wavs/event.wav \
    --outdir /tmp/golden_torch --asr_use_gpu 0 --vad_use_gpu 0 --lid_use_gpu 0 --punc_use_gpu 0
.venv/bin/python scripts/fireredasr2s_mlx_cli.py \
    --wav_paths golden/wavs/hello_zh.wav golden/wavs/hello_en.wav golden/wavs/event.wav \
    --outdir /tmp/golden_mlx
.venv/bin/python tests/acceptance_check.py /tmp/golden_torch/result.jsonl /tmp/golden_mlx/result.jsonl
```
