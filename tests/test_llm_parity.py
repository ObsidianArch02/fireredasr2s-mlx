"""Golden parity tests for the FireRedASR2-LLM (Qwen2-7B) MLX port.

Layers, bottom-up:
  1. tokenizer   — decode prompt ids (no weights needed)
  2. encoder     — MLX encoder vs official encoder_out (golden npz)
  3. adapter     — MLX adapter vs official speech_features
  4. merge       — merged embeddings vs official merged embeddings
  5. qwen2 fwd   — first-step logits (argmax + top-10 ids)
  6. full decode — beam-search token ids and decoded text must be IDENTICAL

Device policy: Metal fp32 matmul carries ~1e-2 absolute noise (vs PyTorch
CPU fp32), which can flip near-tie beam-search outcomes in degenerate
repetition loops (the `event` sample). All exact-equality tests therefore run
on mx.cpu (measured max encoder diff 2e-6); a separate GPU smoke test pins
the day-to-day GPU behavior. Golden source: scripts/dump_golden_llm.py
(official PyTorch, CPU fp32, transformers 5.17). Marked slow.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_ROOT / "pretrained_models" / "FireRedASR2-LLM"
GOLDEN_NPZ = PROJECT_ROOT / "golden" / "golden_llm.npz"
GOLDEN_JSON = PROJECT_ROOT / "golden" / "golden_llm.json"

UTTIDS = ["hello_zh", "hello_en", "event"]

pytestmark = pytest.mark.slow

needs_llm = pytest.mark.skipif(
    not (LLM_DIR / "encoder.safetensors").exists()
    or not GOLDEN_NPZ.exists(),
    reason="LLM weights / golden dump missing "
           "(run convert_llm_checkpoint + scripts/dump_golden_llm.py)")

_MODEL = None


@pytest.fixture(autouse=True)
def _cpu_device():
    """Exact-parity tests run on CPU (see module docstring)."""
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(mx.gpu)


def get_model():
    global _MODEL
    if _MODEL is None:
        from fireredasr2_mlx.llm import FireRedAsr2LlmMlx
        _MODEL = FireRedAsr2LlmMlx.from_pretrained(str(LLM_DIR))
    return _MODEL


def golden():
    return np.load(GOLDEN_NPZ), json.loads(GOLDEN_JSON.read_text())


def _feats_lens(z, uttid):
    feats = mx.array(z[f"{uttid}__feats"])
    lens = mx.array(z[f"{uttid}__lengths"].astype(np.int32))
    return feats, lens


# ----------------------------------------------------------- 1. tokenizer ---

def test_llm_tokenizer_prompt():
    tok_json = LLM_DIR / "Qwen2-7B-Instruct" / "tokenizer.json"
    if not tok_json.exists():
        pytest.skip("LLM tokenizer files missing")
    from fireredasr2_mlx.llm_tokenizer import LlmTokenizerWrapperMlx

    tok = LlmTokenizerWrapperMlx.build_llm_tokenizer(
        LLM_DIR / "Qwen2-7B-Instruct")
    ids, am = tok.decode_prompt()
    # official LlmTokenizerWrapper.preprocess_texts(decode=True) for [""]
    assert ids.tolist() == [[151644, 872, 198, 151646, 14880, 46670, 61443,
                             111268, 17714, 87335, 151645, 198, 151644,
                             77091, 198]]
    assert am.tolist() == [[1] * 15]
    assert tok.pad_token_id == 151643
    assert tok.eos_token_id == 151645
    assert tok.bos_token_id == 151644
    assert tok.speech_token_id == 151646


# ---------------------------------------------------------- 2-4. encoder ---

@needs_llm
@pytest.mark.parametrize("uttid", UTTIDS)
def test_llm_encoder_parity(uttid):
    from fireredasr2_mlx.models.conformer_encoder import encoder_forward

    m = get_model()
    z, _ = golden()
    feats, lens = _feats_lens(z, uttid)
    enc_out, enc_lengths, _ = encoder_forward(m.w_enc, feats, lens,
                                              n_layers=16)
    mx.eval(enc_out)

    g_out = z[f"{uttid}__enc_out"]
    g_lens = z[f"{uttid}__enc_lengths"]
    assert enc_lengths.tolist() == g_lens.tolist()
    L = int(g_lens[0])
    diff = float(np.abs(np.asarray(enc_out)[:, :L] - g_out[:, :L]).max())
    assert diff < 1e-4, f"encoder max abs diff {diff}"


@needs_llm
@pytest.mark.parametrize("uttid", UTTIDS)
def test_llm_adapter_parity(uttid):
    from fireredasr2_mlx.models.adapter import adapter_forward
    from fireredasr2_mlx.models.conformer_encoder import encoder_forward

    m = get_model()
    z, _ = golden()
    feats, lens = _feats_lens(z, uttid)
    enc_out, enc_lengths, _ = encoder_forward(m.w_enc, feats, lens,
                                              n_layers=16)
    sf, sf_lens = adapter_forward(m.w_llm, enc_out, enc_lengths)

    g_sf = z[f"{uttid}__speech_features"]
    g_sl = z[f"{uttid}__speech_lens"]
    assert sf_lens.tolist() == g_sl.tolist()
    L = int(g_sl[0])
    diff = float(np.abs(np.asarray(sf)[:, :L] - g_sf[:, :L]).max())
    assert diff < 1e-4, f"adapter max abs diff {diff}"


@needs_llm
@pytest.mark.parametrize("uttid", UTTIDS)
def test_llm_merge_parity(uttid):
    from fireredasr2_mlx.models.adapter import adapter_forward
    from fireredasr2_mlx.models.conformer_encoder import encoder_forward
    from fireredasr2_mlx.models.fireredasr_llm import \
        merge_input_ids_with_speech_features

    m = get_model()
    z, _ = golden()
    feats, lens = _feats_lens(z, uttid)
    enc_out, enc_lengths, _ = encoder_forward(m.w_enc, feats, lens,
                                              n_layers=16)
    sf, sf_lens = adapter_forward(m.w_llm, enc_out, enc_lengths)

    input_ids, attention_mask = m.tokenizer.decode_prompt()
    embs = m.w_llm["model.embed_tokens.weight"][
        mx.array(input_ids, dtype=mx.int32)]
    merged, merged_mask = merge_input_ids_with_speech_features(
        sf, embs, input_ids, attention_mask, m.cfg["speech_token_id"],
        m.cfg["pad_id"], sf_lens)
    mx.eval(merged)

    assert merged_mask.tolist() == z[f"{uttid}__merged_mask"].tolist()
    g_merged = z[f"{uttid}__merged"]
    diff = float(np.abs(np.asarray(merged) - g_merged).max())
    assert diff < 1e-4, f"merged embedding max abs diff {diff}"


# ------------------------------------------------------------ 5. qwen2 -----

@needs_llm
@pytest.mark.parametrize("uttid", UTTIDS)
def test_llm_first_logits_parity(uttid):
    """Merged prompt through the Qwen2 stack: argmax + top-10 must match."""
    m = get_model()
    z, _ = golden()
    merged = mx.array(z[f"{uttid}__merged"])
    from fireredasr2_mlx.models.qwen2 import qwen2_prefill
    logits, _ = qwen2_prefill(m.w_llm, merged, m.cfg)
    mx.eval(logits)
    logits = np.asarray(logits)[0]

    g = z[f"{uttid}__first_logits"]
    top10 = np.argsort(-logits, kind="stable")[:10].tolist()
    g_top10 = np.argsort(-g, kind="stable")[:10].tolist()
    assert top10 == g_top10, f"top-10 mismatch: {top10} vs {g_top10}"
    assert int(np.argmax(logits)) == int(np.argmax(g))
    diff = float(np.abs(logits - g).max())
    assert diff < 1e-2, f"logits max abs diff {diff}"


# ------------------------------------------------------- 6. full decode ----

@needs_llm
@pytest.mark.parametrize("uttid", UTTIDS)
def test_llm_decode_text_parity(uttid):
    """Full LLM path incl. beam search: ids and text identical to official."""
    m = get_model()
    z, texts = golden()
    feats, lens = _feats_lens(z, uttid)
    ids, confidence = llm_transcribe_model(m, feats, lens)
    assert ids == texts[uttid]["ids"], \
        f"ids mismatch:\n{ids}\nvs\n{texts[uttid]['ids']}"
    text = m.tokenizer.batch_decode([ids])[0].lower()
    assert text == texts[uttid]["text"]
    assert 0.0 <= confidence <= 1.0


@needs_llm
def test_llm_decode_gpu_smoke():
    """GPU fp32: zh/en exact; event may stop early in its degenerate
    repetition loop (near-tie beam flips from ~1e-2 matmul noise), but the
    token set must be the official one."""
    mx.set_default_device(mx.gpu)
    m = get_model()
    z, texts = golden()
    for uttid in UTTIDS:
        feats, lens = _feats_lens(z, uttid)
        ids, _ = llm_transcribe_model(m, feats, lens)
        text = m.tokenizer.batch_decode([ids])[0].lower()
        if uttid == "event":
            assert set(ids) <= {105511, 104053, 151645}
            assert text == "生日快乐" * ((len(ids) - 1) // 2)
        else:
            assert ids == texts[uttid]["ids"]
            assert text == texts[uttid]["text"]
    mx.set_default_device(mx.cpu)


def llm_transcribe_model(m, feats, lens):
    from fireredasr2_mlx.models.fireredasr_llm import llm_transcribe
    input_ids, attention_mask = m.tokenizer.decode_prompt()
    return llm_transcribe(
        m.w_enc, m.w_llm, m.cfg, feats, lens,
        mx.array(input_ids), mx.array(attention_mask),
        beam_size=3, decode_max_len=0)
