# FireRedASR2-AED -> MLX native port plan

Goal: port the official PyTorch FireRedASR2-AED (1.1B, SOTA on zh-en mixed) to a
pure MLX implementation running Metal GPU inference on Apple Silicon (verified on Apple Silicon), with zero PyTorch runtime dependency.

Working directory: `fireredasr2s-port/`
Reference implementation (golden baseline): `FireRedASR2S-main/` (official repo, read-only)
Weights: `pretrained_models/FireRedASR2-AED/model.pth.tar` (4.73GB fp32, sha256 known, downloading)

---

## 0. Survey of existing ports (verified 2026-09-27)

| Project | Status | Verdict |
|---|---|---|
| mlx-community/FireRedASR2-AED-mlx | Weights exist (937 F32 safetensors tensors), runner deleted (Kn90688/mlx-fireredasr 404) | **Weight conversion format can be copied verbatim** (tensor names map one-to-one to the official state_dict), runner code must be written |
| vanch007/mlx-FireRedAudio | Complete MLX port of the same-vendor audio LLM (not ASR2), RTF 0.10-0.14 on Apple Silicon | **Best engineering reference**: fused Metal SDPA, Gated Delta kernels, quantization, loader structure |
| CrispASR | ggml/C++ route, Metal | Out of scope (CTC decoding, not AED); comparison only |
| sherpa-onnx | ONNX CPU | Out of scope |

Conclusion: **there is no reusable FireRedASR2 MLX runner - it must be written by
hand. The work concentrates in the forward operators; decoding/features can reuse
large amounts of the official pure Python/NumPy code.**

## 1. Architecture inventory (checked line-by-line against official source)

### 1.1 Direct reuse (no porting, Python as-is)
- **fbank features**: `kaldi_native_fbank` (C extension, pure CPU, 80 mel / 25ms / 10ms, snip_edges) -> NumPy arrays feed MLX directly
- **CMVN**: read stats from kaldi.ark and do mean/variance normalization (pure NumPy)
- **tokenizer**: `ChineseCharEnglishSpmTokenizer` (sentencepiece + dict.txt, pure Python)
- **beam search orchestration**: `batch_beam_search` is all tensor orchestration (topk/gather/cat) - translates line-for-line to mx/numpy

### 1.2 Parts needing hand-written MLX operators (the porting surface)

**Encoder = Conv2dSubsampling + RelPositionalEncoding + 16xConformerBlock (d=1280, h=20, dk=64, k=33)**

| Unit | Contents | Pitfalls |
|---|---|---|
| U1 Conv2dSubsampling | pad to 7-frame context -> 2x[Conv2d(k3,s2,valid)+ReLU] (1->32->32ch) -> reshape -> Linear(32x19->1280); length `((T-1)//2-1)//2` | torch weights `[O,I,kH,kW]` -> mlx `[O,kH,kW,I]` layout transpose |
| U2 RelPositionalEncoding | flip(sin/cos) bidirectional pe, cat into 2*max_len-1, slice `Tmax//2-T+1 : Tmax//2+T` | pure precomputed buffer, no pitfalls |
| U3 RelPosMHSA | pre-LN on q/k/v -> projections -> `q+pos_bias_u`.k (content term) + `q+pos_bias_v`.linear_pos(p) (position term) -> `_rel_shift` -> x1/8 -> mask(-inf) -> softmax -> mask 0 again -> .v -> fc | `_rel_shift` zero-pad+view trick must be translated line-for-line; mask semantics are uint8 (1=keep), and the second mask-to-0 after softmax must be kept (padding rows are all -inf -> NaN, rescued by re-masking) |
| U4 ConformerConvolution | pre-LN -> mask to 0 -> Conv1d(1280->2560,k1,no bias) -> GLU (channel dim) -> depthwise Conv1d(2560->2560,k33,groups=2560,pad16,no bias) -> **LayerNorm(2560)** (official variable is named batch_norm but is actually LN) -> Swish -> Conv1d(2560->1280,k1) -> mask -> residual | mlx conv1d supports groups; many T<->C transpose points - standardize on (N,T,C) and transpose only around convs |
| U5 FFN macaron x2 + final LN | LN->Linear(1280->5120)->Swish->Linear->0.5/0.5 residual; LayerNorm at block end | no pitfalls |
| U6 Decoder layers x16 | pre-norm sandwich: self-attn (causal) -> cross-attn (to encoder) -> MLP(GELU); tied embedding x sqrt(1280) + absolute PE | Official cache scheme = cache whole layer output sequence, compute only the last-position query each step (full-prefix K/V recomputed per step). **v1 copies this verbatim for numeric parity; v2 switches to a true KV cache (per-layer k,v, mathematically equivalent)** |
| U7 CTC head | Linear(1280->8667)+log_softmax, used for timestamps only | forced_align (torchaudio) has no MLX counterpart -> port as NumPy Viterbi alignment (official ctc.py already has a `ctc_align` DP to copy, ~30 lines) |
| U8 batch_beam_search | see 1.1, translate | beam reordering `topB_row_number_in_ys` index arithmetic must be checked line-by-line |

**Total params ~1.1B; F32 weights 4.4GB - fits in unified memory on Apple Silicon; bf16 (2.2GB) is a later option.**

### 1.3 Weight conversion
- Source: official `model.pth.tar` (torch ckpt: `args` + `model_state_dict`, loaded with `strict=False`)
- Target: safetensors, key names copied from the mlx-community file (= official state_dict names with `encoder./decoder.` prefix normalization), F32
- Watch tied weights: `decoder.tgt_word_prj.weight == decoder.tgt_word_emb.weight` - safetensors stores one copy, aliased at load time
- `pe` is a buffer, not in state_dict - recomputed at runtime

## 2. Implementation steps

### Phase A: golden baseline (PyTorch, in progress)
- [x] uv 3.12 venv, install torch 2.14 / transformers 5.17 / kaldi_native_fbank etc. (a pip mirror)
- [ ] model.pth.tar download finished (a parallel chunked mirror download, background task)
- [ ] run `test_aed.py`: hello_zh/hello_en/event, expected output ~ official README samples ("Hello world." / "Hello speech.")
- [ ] dump `golden.pt`: feats, encoder_out (hook capture), transcript text + timestamps
- Risk: torch 2.14 too new - if `torch.load(weights_only=False)` or the import chain (transformers 5.x) breaks, downgrade to torch 2.x LTS / transformers 4.51

### Phase B: skeleton + per-unit numerical alignment (1-2 days)
Package layout (mirroring mlx-FireRedAudio):
```
fireredasr2_mlx/
  __init__.py          # FireRedAsr2Mlx facade: load -> transcribe
  loader.py            # pth.tar -> safetensors conversion + loading (tied weights)
  features.py          # import the official ASRFeatExtractor directly (numpy boundary)
  tokenizer.py         # import the official tokenizer directly
  models/
    conformer_encoder.py   # U1-U5
    transformer_decoder.py # U6
    ctc.py                 # U7
  decode.py            # U8 batch_beam_search translation
tests/
  test_parity_u1..u8   # per unit: same input vs PyTorch golden, fp32 max-abs-diff < 1e-4
  test_e2e             # 3 sample wavs, transcript text identical to PyTorch
```
Order: U1->U3->U4->U5 (encoder assembled, encoder_out identical) -> U6->U8
(decoder+beam, transcript text identical) -> U7 (timestamps). Each unit is
pytest-checked against intermediate tensors in `golden.pt` - "looks right" is not
accepted.

### Phase C: performance (1-2 days)
1. Default `mx.set_default_device(mx.gpu)`, measure RTF (PyTorch CPU baseline in parallel)
2. Cheap wins first: bf16 weights (transcript must stay identical - if it drifts, keep mixed strategy like F32 encoder + bf16 decoder), `mx.compile` around encoder blocks
3. Attention: the content/position matmuls can stay explicit; if SDPA becomes the bottleneck, fold the pos term into a variant of `mx.fast.scaled_dot_product_attention` (see mlx-FireRedAudio's fused SDPA)
4. v2 decoder switches to true KV cache (incremental K/V); remember to gather cache in sync when reordering beams inside beam search
5. Target: **RTF <= 0.15 on Apple Silicon** (mlx-FireRedAudio gets RTF 0.10-0.14 on Apple Silicon on a comparable model)

### Phase D: wrap-up
- Optional quantization (mlx affine quantize, 4/8bit, gated by transcript regression tests)
- CLI: `python -m fireredasr2_mlx audio.wav`, outputs JSONL (text/confidence/timestamp), fields aligned with upstream
- README + numerical alignment report (per-unit max-abs-diff table)

## 3. Risk list

| Risk | Mitigation |
|---|---|
| torch 2.14/transformers 5.17 incompatible with official code | requirements pin old versions for CUDA reasons; on Mac downgrade freely (transformers 4.51 / torch 2.5) |
| Conv2d/Conv1d layout transposes misaligned (silent error) | U1/U4 unit tests enforce exact-equality thresholds; reproduce once in NumPy with `np.einsum` before writing mlx |
| softmax all -inf rows -> NaN | copy the "mask again after softmax" double protection verbatim |
| tied embedding written twice in safetensors -> inconsistency | loader explicitly asserts `mx.array_equal(a,b)` on the alias |
| beam search index arithmetic wrong (garbled output order, no error) | translate the whole function verbatim + text-identical test; no "optimized rewrites" |
| RAM tight on a smaller-memory machine (F32 weights 4.4GB + PyTorch golden running simultaneously) | stagger parity runs: run PyTorch to dump golden first, exit, then run MLX |
| Metal fp32 matmul vs CPU fp32 last-bit differences | 1e-4 tolerance + text-level exact equality (decoding argmax is robust to 1e-4 perturbation) |

## 4. Acceptance criteria
1. Transcript text, confidence, and timestamps identical to the PyTorch build on 3 samples (zh/en/mixed)
2. Zero torch import (`import torch` must fail in an env with only `fireredasr2_mlx` deps)
3. RTF <= 0.15 (Apple Silicon GPU)
4. pytest all green (8 unit parity tests + 2 e2e)
