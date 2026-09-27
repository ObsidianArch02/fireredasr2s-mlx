"""Parity test-suite for the FireRedASR2S MLX port.

Three layers:

1. Zero-torch guard    — `fireredasr2_mlx` (and every submodule) must import
     cleanly with `builtins.__import__` blocking any `torch` import
     (iron rule 1: the runtime is torch-free).
2. Unit parity         — encoder / decoder exercised on RANDOM weights:
     output shapes match the official torch layout arithmetic, all values
     finite, masking / length bookkeeping matches the torch formulas.
3. Golden parity       — (`@pytest.mark.slow`) the full CLI is run on the 3
     golden wavs and result.jsonl + TextGrid + SRT must be IDENTICAL, field
     for field, to the official output in `output_torch/` (iron rule 2).

Run:  cd fireredasr2s-port && .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import builtins
import importlib
import math
import pkgutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import acceptance_check  # noqa: E402  (standalone comparator, no pytest dep)

GOLDEN_DIR = PROJECT_ROOT / "golden" / "output_torch"
# Sorted order == the order the CLI produces (sorted wav_paths), and the
# order of the records inside output_torch/result.jsonl.
GOLDEN_WAVS = [
    "golden/wavs/event.wav",
    "golden/wavs/hello_en.wav",
    "golden/wavs/hello_zh.wav",
]


# =========================================================== zero-torch ====

def _purge_modules(prefixes):
    """Pop modules (and submodules) from sys.modules; return what was removed."""
    saved = {}
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            saved[name] = sys.modules.pop(name)
    return saved


def test_zero_torch_import_guard():
    """Import fireredasr2_mlx and every submodule with torch hard-blocked."""
    saved_pkg = _purge_modules(["fireredasr2_mlx"])
    # Force a re-import of torch too, so a hidden `import torch` anywhere in
    # the package cannot silently succeed against an already-loaded copy.
    _purge_modules(["torch"])

    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] == "torch":
            blocked.append(name)
            raise ImportError(f"zero-torch guard: import of '{name}' blocked")
        return real_import(name, *args, **kwargs)

    try:
        builtins.__import__ = guarded_import
        import fireredasr2_mlx  # noqa: F401

        for mod in pkgutil.walk_packages(fireredasr2_mlx.__path__,
                                         prefix="fireredasr2_mlx."):
            importlib.import_module(mod.name)
    finally:
        builtins.__import__ = real_import
        # Restore the pre-test package modules; torch stays purged on purpose:
        # if it is missing afterwards, nothing ever imported it.
        sys.modules.update(saved_pkg)

    assert not blocked, f"fireredasr2_mlx reached for torch via: {blocked}"
    assert "torch" not in sys.modules, \
        "torch ended up in sys.modules while importing fireredasr2_mlx"


# ====================================================== unit: encoder ======
# Tiny dims mirroring the official encoder layout (conv2d subsampling on
# 80-d fbank, d_model/n_head/depth overridable).

ENC_D, ENC_HEADS, ENC_LAYERS, ENC_FFN = 16, 4, 2, 32


def _rand(rng, *shape):
    return mx.array((rng.standard_normal(shape) * 0.05).astype(np.float32))


def random_encoder_weights(seed=0):
    from fireredasr2_mlx.models.conformer_encoder import build_rel_pe

    rng = np.random.default_rng(seed)
    w = {}
    D, F = ENC_D, ENC_FFN
    # conv2d subsampling: torch [O,I,kH,kW] transposed at load time to
    # MLX [O,kH,kW,I] (see loader.convert_checkpoint) — mimic that layout.
    w["encoder.input_preprocessor.conv.0.weight"] = _rand(rng, D, 3, 3, 1)
    w["encoder.input_preprocessor.conv.0.bias"] = _rand(rng, D)
    w["encoder.input_preprocessor.conv.2.weight"] = _rand(rng, D, 3, 3, D)
    w["encoder.input_preprocessor.conv.2.bias"] = _rand(rng, D)
    # freq axis: 80 -> 39 -> 19 (k=3, s=2 twice), flatten C*W2 = D*19
    w["encoder.input_preprocessor.out.weight"] = _rand(rng, D, D * 19)
    w["encoder.input_preprocessor.out.bias"] = _rand(rng, D)
    for l in range(ENC_LAYERS):
        p = f"encoder.layer_stack.{l}"
        for ffn in ("ffn1", "ffn2"):
            w[f"{p}.{ffn}.net.0.weight"] = _rand(rng, D)   # net.0 = LayerNorm
            w[f"{p}.{ffn}.net.0.bias"] = _rand(rng, D)
            w[f"{p}.{ffn}.net.1.weight"] = _rand(rng, F, D)
            w[f"{p}.{ffn}.net.1.bias"] = _rand(rng, F)
            w[f"{p}.{ffn}.net.4.weight"] = _rand(rng, D, F)
            w[f"{p}.{ffn}.net.4.bias"] = _rand(rng, D)
        a = f"{p}.mhsa"
        for ln in ("layer_norm_q", "layer_norm_k", "layer_norm_v"):
            w[f"{a}.{ln}.weight"] = _rand(rng, D)
            w[f"{a}.{ln}.bias"] = _rand(rng, D)
        for proj in ("w_qs", "w_ks", "w_vs", "linear_pos", "fc"):
            w[f"{a}.{proj}.weight"] = _rand(rng, D, D)
        w[f"{a}.pos_bias_u"] = _rand(rng, ENC_HEADS, D // ENC_HEADS)
        w[f"{a}.pos_bias_v"] = _rand(rng, ENC_HEADS, D // ENC_HEADS)
        c = f"{p}.conv"
        w[f"{c}.pre_layer_norm.weight"] = _rand(rng, D)
        w[f"{c}.pre_layer_norm.bias"] = _rand(rng, D)
        w[f"{c}.pointwise_conv1.weight"] = _rand(rng, 2 * D, 1, D)  # O,kW,C
        w[f"{c}.depthwise_conv.weight"] = _rand(rng, D, 33, 1)  # pad=16, k=33
        w[f"{c}.batch_norm.weight"] = _rand(rng, D)
        w[f"{c}.batch_norm.bias"] = _rand(rng, D)
        w[f"{c}.pointwise_conv2.weight"] = _rand(rng, D, 1, D)
        w[f"{p}.layer_norm.weight"] = _rand(rng, D)
        w[f"{p}.layer_norm.bias"] = _rand(rng, D)
    w["pe_rel"] = build_rel_pe(D, max_len=256)
    return w


def _key_mask(lengths, T):
    """(N,1,T) 0/1 mask: 1 where position < length (padding_position_is_0)."""
    return (np.arange(T)[None, None, :]
            < np.asarray(lengths, dtype=np.int64)[:, None, None]).astype(np.uint8)


def _subsampled_mask(lengths, T):
    """Official encoder mask arithmetic: pad T->T+6, slice [:-2:2] twice."""
    src = (np.arange(T + 6)[None, None, :]
           < np.asarray(lengths, dtype=np.int64)[:, None, None]).astype(np.uint8)
    return src[:, :, :-2:2][:, :, :-2:2]


def test_encoder_random_weights_shape_finiteness():
    from fireredasr2_mlx.models.conformer_encoder import encoder_forward

    rng = np.random.default_rng(42)
    N, T = 2, 100
    feats = mx.array(rng.standard_normal((N, T, 80)).astype(np.float32))
    lengths = mx.array(np.array([T, 73], dtype=np.int32))

    enc_out, enc_lengths, enc_mask = encoder_forward(
        random_encoder_weights(), feats, lengths,
        n_layers=ENC_LAYERS, n_head=ENC_HEADS, d_model=ENC_D)

    # time axis: pad to T+6, conv k=3 s=2 twice -> ((106-3)//2+1 -3)//2+1 = 25
    T2 = ((T + 6 - 3) // 2 + 1 - 3) // 2 + 1
    assert enc_out.shape == (N, T2, ENC_D)
    assert enc_mask.shape == (N, 1, T2)

    out = np.asarray(enc_out.astype(mx.float32))
    assert np.isfinite(out).all(), "encoder produced non-finite activations"

    # length bookkeeping must match the official mask formula exactly
    m = _subsampled_mask([T, 73], T)
    assert np.array_equal(np.asarray(enc_mask.astype(mx.uint8)), m)
    assert enc_lengths.tolist() == m[:, -1, :].sum(-1).tolist()
    assert enc_lengths.tolist() == [T2, 19]  # 73 -> ceil((ceil(73/2))/2)


# ====================================================== unit: decoder ======

DEC_D, DEC_HEADS, DEC_LAYERS, DEC_FFN, VOCAB = 32, 4, 2, 64, 40
SOS, EOS, PAD = 3, 4, 2  # dict.txt ids the port shares with the official CLI


def random_decoder_weights(seed=1):
    rng = np.random.default_rng(seed)
    w = {}
    D, F = DEC_D, DEC_FFN
    for l in range(DEC_LAYERS):
        p = f"decoder.layer_stack.{l}"
        for attn in ("self_attn", "cross_attn"):
            w[f"{p}.{attn}_norm.weight"] = _rand(rng, D)
            w[f"{p}.{attn}_norm.bias"] = _rand(rng, D)
            w[f"{p}.{attn}.w_qs.weight"] = _rand(rng, D, D)
            w[f"{p}.{attn}.w_qs.bias"] = _rand(rng, D)
            w[f"{p}.{attn}.w_ks.weight"] = _rand(rng, D, D)   # no bias
            w[f"{p}.{attn}.w_vs.weight"] = _rand(rng, D, D)
            w[f"{p}.{attn}.w_vs.bias"] = _rand(rng, D)
            w[f"{p}.{attn}.fc.weight"] = _rand(rng, D, D)
            w[f"{p}.{attn}.fc.bias"] = _rand(rng, D)
        w[f"{p}.mlp_norm.weight"] = _rand(rng, D)
        w[f"{p}.mlp_norm.bias"] = _rand(rng, D)
        w[f"{p}.mlp.w_1.weight"] = _rand(rng, F, D)
        w[f"{p}.mlp.w_1.bias"] = _rand(rng, F)
        w[f"{p}.mlp.w_2.weight"] = _rand(rng, D, F)
        w[f"{p}.mlp.w_2.bias"] = _rand(rng, D)
    w["decoder.layer_norm_out.weight"] = _rand(rng, D)
    w["decoder.layer_norm_out.bias"] = _rand(rng, D)
    # tied embedding / output projection, as in the checkpoint
    w["decoder.tgt_word_emb.weight"] = _rand(rng, VOCAB, D)
    w["decoder.tgt_word_prj.weight"] = w["decoder.tgt_word_emb.weight"]
    from fireredasr2_mlx.models.conformer_encoder import (
        abs_positional_encoding)
    w["pe_abs"] = abs_positional_encoding(D, max_len=128)
    return w


def _run_beam(w, enc_out, src_mask, **kw):
    from fireredasr2_mlx.models.transformer_decoder import batch_beam_search

    return batch_beam_search(
        w, enc_out, src_mask, beam_size=3, nbest=1, decode_max_len=10,
        n_layers=DEC_LAYERS, n_head=DEC_HEADS, d_model=DEC_D,
        sos_id=SOS, eos_id=EOS, pad_id=PAD, **kw)


def test_decoder_random_weights_shape_finiteness():
    rng = np.random.default_rng(43)
    N, Ti = 2, 15
    enc_out = mx.array(rng.standard_normal((N, Ti, DEC_D)).astype(np.float32))
    src_mask = mx.array(_key_mask([Ti, 9], Ti))

    hyps = _run_beam(random_decoder_weights(), enc_out, src_mask)

    assert len(hyps) == N
    for entry in hyps:
        assert len(entry) == 1  # nbest=1
        hyp = entry[0]
        yseq = hyp["yseq"]
        assert isinstance(yseq, list)
        assert 0 <= len(yseq) <= 10  # bounded by decode_max_len
        assert all(isinstance(t, int) for t in yseq)
        assert all(0 <= t < VOCAB for t in yseq)
        assert all(t != PAD for t in yseq), "pad leaked into the hypothesis"
        assert all(t != SOS for t in yseq), "sos leaked into the hypothesis"
        assert all(t != EOS for t in yseq), "eos leaked into the hypothesis"
        conf = hyp["confidence"]
        assert math.isfinite(conf) and 0.0 <= conf <= 1.0


def test_decoder_kv_cache_deterministic_across_runs():
    """The true-KV-cache decoder must be deterministic run-to-run."""
    rng = np.random.default_rng(44)
    N, Ti = 2, 12
    enc_out = mx.array(rng.standard_normal((N, Ti, DEC_D)).astype(np.float32))
    src_mask = mx.array(_key_mask([Ti, Ti], Ti))
    w = random_decoder_weights(seed=2)

    a = _run_beam(w, enc_out, src_mask)
    b = _run_beam(w, enc_out, src_mask)
    assert a == b


# =========================================================== golden ========

@pytest.fixture(scope="module")
def mlx_cli_run(tmp_path_factory):
    """Run the full MLX CLI once on the 3 golden wavs (module-scoped).

    Pinned to CPU: the golden files are PyTorch-CPU outputs, and Metal fp32
    matmul noise (~1e-2) can flip near-tied AED beams on hard inputs (the
    event song clip), breaking byte-level parity that is exact on CPU."""
    import os
    outdir = tmp_path_factory.mktemp("output_mlx")
    cmd = [sys.executable, "-m", "fireredasr2_mlx.cli",
           "--wav_paths", *GOLDEN_WAVS, "--outdir", str(outdir)]
    env = dict(os.environ, FIREREDASR2_MLX_DEVICE="cpu")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                          text=True, env=env)
    return proc, outdir


def _require_cli_ok(mlx_cli_run):
    proc, outdir = mlx_cli_run
    assert proc.returncode == 0, (
        f"MLX CLI failed (rc={proc.returncode}):\n{proc.stderr[-3000:]}")
    return outdir


@pytest.mark.slow
def test_golden_cli_run_succeeds(mlx_cli_run):
    outdir = _require_cli_ok(mlx_cli_run)
    assert (outdir / "result.jsonl").exists()


@pytest.mark.slow
def test_golden_result_jsonl_field_parity(mlx_cli_run):
    """Iron rule 2: every field of result.jsonl must equal the official one."""
    outdir = _require_cli_ok(mlx_cli_run)
    diffs, compared = acceptance_check.compare_files(
        str(GOLDEN_DIR / "result.jsonl"), str(outdir / "result.jsonl"))
    assert compared == len(GOLDEN_WAVS), \
        f"expected {len(GOLDEN_WAVS)} records, compared {compared}"
    assert not diffs, "result.jsonl field diffs:\n" + "\n".join(diffs)


def _tree(root: Path):
    return sorted(p.relative_to(root).as_posix()
                  for p in root.rglob("*") if p.is_file())


@pytest.mark.slow
def test_golden_textgrid_and_srt_parity(mlx_cli_run):
    """Iron rule 2: TextGrid and SRT outputs must be byte-identical."""
    outdir = _require_cli_ok(mlx_cli_run)
    for sub in ("asr_tg", "asr_srt"):
        got, want = outdir / sub, GOLDEN_DIR / sub
        assert _tree(got) == _tree(want), (
            f"{sub}/ file sets differ: mlx={_tree(got)} torch={_tree(want)}")
        for rel in _tree(want):
            assert (got / rel).read_bytes() == (want / rel).read_bytes(), \
                f"{sub}/{rel} differs from the official output"
