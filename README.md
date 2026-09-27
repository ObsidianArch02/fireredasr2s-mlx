# FireRedASR2S - MLX Port

A **pure MLX port** of Xiaohongshu's [FireRedASR2S](./third_party/FireRedASR2S-main),
the SOTA four-module Chinese-English mixed ASR system, targeting Apple Silicon
(Metal GPU) with zero torch dependency at runtime.

- The official PyTorch reference implementation is kept read-only at
  [FireRedASR2S-main/](./third_party/FireRedASR2S-main) (unmodified).
- The CLI surface matches the official `fireredasr2s_cli.py` **flag for flag**:
  swap the official command line for this one and it just runs.
- Field-level parity on 3 test wavs: all `result.jsonl` fields, TextGrid, and SRT
  match the official PyTorch output (golden outputs in
  [output_torch/](./golden/output_torch)).

## Install (uv)

Requires macOS (Apple Silicon) + Python 3.12 + [uv](https://docs.astral.sh/uv/).

```bash
cd fireredasr2s-mlx
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    mlx numpy soundfile textgrid sentencepiece \
    kaldi-native-fbank kaldiio safetensors
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
  match `third_party/FireRedASR2S-main/fireredasr2s/fireredasr2s_cli.py` exactly
  (verified by script).
- **Identical output**: for the same input, `result.jsonl`, `asr_tg/*.TextGrid`,
  and `asr_srt/*.srt` match the official PyTorch build field for field.
- **Different backend**: the official build is PyTorch (CUDA/CPU); this one is MLX
  (Metal/CPU). Therefore `--asr_use_gpu/--asr_use_half/--vad_use_gpu/--lid_use_gpu/--punc_use_gpu`
  are accepted and ignored - MLX always runs on Metal (`use_gpu=0` needs no CPU
  fallback path; numeric results are unchanged).

## Usage examples

```bash
# 1) single wav -> output/ (result.jsonl + TextGrid + SRT)
.venv/bin/python -m fireredasr2_mlx.cli \
    --wav_path third_party/FireRedASR2S-main/assets/hello_zh.wav --outdir output

# 2) multiple wavs + save VAD segment wavs (--sort_wav_by_dur 1 processes longest first)
.venv/bin/python -m fireredasr2_mlx.cli \
    --wav_paths third_party/FireRedASR2S-main/assets/hello_zh.wav third_party/FireRedASR2S-main/assets/hello_en.wav third_party/FireRedASR2S-main/assets/event.wav \
    --sort_wav_by_dur 1 --save_segment 1 --outdir output

# 3) directory batch + scp list, disable outputs you do not need
.venv/bin/python -m fireredasr2_mlx.cli --wav_dir third_party/FireRedASR2S-main/assets --outdir output
.venv/bin/python -m fireredasr2_mlx.cli --wav_scp wavs.scp --write_textgrid 0 --write_srt 0 --outdir output
```

## Known boundaries

- **`--asr_type llm` unsupported**: FireRedASR2-LLM (Qwen2-7B decoder) is
  CUDA-only (torch + transformers) and cannot run on Mac/MLX upstream either.
  Choosing `llm` prints an explanation and exits; the default `--asr_type aed`
  is the complete four-module system implemented by this port.
- **External LM (`--elm_dir/--elm_weight`) unsupported**: the flags are accepted,
  print a notice, and are ignored; decoding continues as AED without an external LM.
- `--sort_wav_by_dur 1` orders processing by actual audio duration, longest first
  (the official 4-module CLI parses this flag but never applies it; the sort
  direction matches its standalone script `fireredasr2/utils/io.py`). It affects
  processing order only, not per-utterance results.
- `--punc_with_timestamp 0` forces punctuation through the no-timestamp branch
  (one sentence for the whole segment); the official 4-module system parses but
  never wires this flag - this port implements it.
