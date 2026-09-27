"""Weight loading for the MLX port.

Two-step flow:
1. convert (run ONCE, in the PyTorch venv): torch checkpoint -> safetensors
   with official state_dict names; conv weights transposed to MLX layout.
2. load (runtime, torch-free): safetensors -> flat dict of mx.arrays, plus
   recomputed positional-encoding buffers.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

from .models.conformer_encoder import abs_positional_encoding, build_rel_pe


def convert_checkpoint(pth_path: str, out_dir: str):
    """ONE-TIME conversion, must run inside the PyTorch venv (imports torch).
    Writes <out_dir>/model.safetensors + <out_dir>/args.json.

    SECURITY: uses torch.load(weights_only=False) — only convert checkpoints
    from trusted, integrity-verified sources; untrusted .pth.tar files can
    execute arbitrary code on load."""
    import json
    import torch  # heavyweight import, conversion-only
    from safetensors.torch import save_file

    package = torch.load(pth_path, map_location="cpu", weights_only=False)
    sd = package["model_state_dict"]
    out = {}
    for k, v in sd.items():
        t = v.detach().cpu().float()
        if t.ndim == 4:  # Conv2d [O,I,kH,kW] -> [O,kH,kW,I]
            t = t.permute(0, 2, 3, 1).contiguous()
        elif t.ndim == 3:  # Conv1d [O,I,kW] -> [O,kW,I]
            t = t.permute(0, 2, 1).contiguous()
        out[k] = t
    out_path = Path(out_dir) / "model.safetensors"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_path))
    args = package.get("args")
    args_dict = vars(args) if hasattr(args, "__dict__") else args
    args_dict = {k: (v if isinstance(v, (int, float, str, bool, list, type(None)))
                     else str(v)) for k, v in (args_dict or {}).items()}
    (Path(out_dir) / "args.json").write_text(json.dumps(args_dict))
    return str(out_path), args_dict


def load_safetensors_plain(model_dir: str):
    """Load model.safetensors + args.json (for VAD/LID-style modules)."""
    model_dir = Path(model_dir)
    import json
    st = load_file(str(model_dir / "model.safetensors"))
    w = {k: mx.array(v) for k, v in st.items()}
    if "encoder.positional_encoding.pe" in w:
        # checkpoint PE buffers are stored transposed (1, D, T)
        w["pe_rel"] = mx.transpose(w["encoder.positional_encoding.pe"], (0, 2, 1))
        dec_pe = next((k for k in ("decoder.positional_encoding.pe",
                                   "lid_decoder.positional_encoding.pe")
                       if k in w), None)
        if dec_pe is not None:
            w["pe_abs"] = mx.transpose(w[dec_pe], (0, 2, 1))
    args = json.loads((model_dir / "args.json").read_text())
    return w, args


def load_weights(model_dir: str, n_layers: int = 16, d_model: int = 1280,
                 pe_maxlen: int = 5000) -> dict:
    """Runtime loader (torch-free). Returns flat weight dict + PE buffers."""
    st = load_file(str(Path(model_dir) / "model.safetensors"))
    w = {k: mx.array(v) for k, v in st.items()}
    # PE buffers ship in the checkpoint but stored transposed (1, D, T);
    # runtime code wants (1, T, D). Recompute only if absent.
    if "encoder.positional_encoding.pe" in w:
        w["pe_rel"] = mx.transpose(w["encoder.positional_encoding.pe"], (0, 2, 1))
        w["pe_abs"] = mx.transpose(w["decoder.positional_encoding.pe"], (0, 2, 1))
    else:
        w["pe_rel"] = build_rel_pe(d_model, pe_maxlen)
        w["pe_abs"] = abs_positional_encoding(d_model, pe_maxlen)
    # tied output projection: safetensors stores both names; verify they match
    a, b = w.get("decoder.tgt_word_prj.weight"), w.get("decoder.tgt_word_emb.weight")
    if a is not None and b is not None and not mx.array_equal(a, b):
        raise ValueError("tied weights diverged in checkpoint")
    return w


# ----------------------------------------------------------- LLM variant ---

# LoRA targets trained into the released checkpoint (official LoraConfig:
# r=64, lora_alpha=16 -> scaling 0.25; target q/k/v/o/gate/up/down).
_LORA_TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                 "mlp.down_proj")
LORA_SCALING = 16.0 / 64.0


def convert_llm_checkpoint(pth_path: str, out_dir: str):
    """ONE-TIME conversion for FireRedASR2-LLM, run in the PyTorch venv.

    SECURITY: uses torch.load(weights_only=False) — only convert checkpoints
    from the official FireRedTeam HuggingFace repo (or otherwise trusted,
    integrity-verified sources); untrusted .pth.tar files can execute
    arbitrary code on load.

    Splits model.pth.tar into:
      encoder.safetensors  — encoder.* (conv layout fixed, fp32)
      adapter.safetensors  — encoder_projector.* + LoRA weights (fp32)
                             (LoRA keys keep the HF layout minus the
                             "llm.base_model.model." PeftModel prefix)
      args.json            — training args (encoder_downsample_rate etc.)
    """
    import json
    import torch  # heavyweight import, conversion-only
    from safetensors.torch import save_file

    package = torch.load(pth_path, map_location="cpu", weights_only=False)
    sd = package["model_state_dict"]
    enc, adapter = {}, {}
    for k, v in sd.items():
        t = v.detach().cpu().float()
        if k.startswith("encoder."):
            if t.ndim == 4:  # Conv2d [O,I,kH,kW] -> [O,kH,kW,I]
                t = t.permute(0, 2, 3, 1).contiguous()
            elif t.ndim == 3:  # Conv1d [O,I,kW] -> [O,kW,I]
                t = t.permute(0, 2, 1).contiguous()
            enc[k] = t
        elif k.startswith("encoder_projector."):
            adapter[k] = t
        elif k.startswith("llm.base_model.model."):
            adapter[k[len("llm.base_model.model."):]] = t
        else:
            raise ValueError(f"unexpected key in LLM checkpoint: {k}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(enc, str(out / "encoder.safetensors"))
    save_file(adapter, str(out / "adapter.safetensors"))
    args = package.get("args")
    args_dict = vars(args) if hasattr(args, "__dict__") else (args or {})
    args_dict = {k: (v if isinstance(v, (int, float, str, bool, list, type(None)))
                     else str(v)) for k, v in args_dict.items()}
    (out / "args.json").write_text(json.dumps(args_dict))
    return str(out), args_dict


def _load_safetensors_shard(path):
    """Minimal torch-free safetensors reader (BF16/F32 -> fp32 mx arrays).

    safetensors 0.8's `framework="mlx"` get_tensor() chokes on BF16, so the
    shard is parsed directly: the format is an 8-byte header length, a JSON
    header with per-tensor dtype/shape/offsets, then the raw buffer. BF16 is
    widened to fp32 by a 16-bit left shift (exact).
    """
    import json
    import struct

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(8)
        if not (0 < header_len <= min(file_size - 8, 100_000_000)):
            raise ValueError(f"bad safetensors header length {header_len} "
                             f"in {path}")
        header = json.loads(f.read(header_len))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        shape = info["shape"]
        dtype = info["dtype"]
        if dtype not in ("BF16", "F32"):
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        if not (0 <= start < end <= len(data)):
            raise ValueError(f"bad offsets for {name}: ({start}, {end}) "
                             f"vs buffer {len(data)}")
        n = end - start
        itemsize = 2 if dtype == "BF16" else 4
        expected = int(np.prod(shape, dtype=np.int64)) * itemsize \
            if shape else itemsize
        if n != expected:
            raise ValueError(f"size mismatch for {name}: "
                             f"{n} bytes vs expected {expected}")
        if dtype == "BF16":
            u16 = np.frombuffer(data, dtype=np.uint16, count=n // 2,
                                offset=start)
            arr = (u16.astype(np.uint32) << 16).view(np.float32) \
                .reshape(shape)
        else:
            arr = np.frombuffer(data, dtype=np.float32, count=n // 4,
                                offset=start).reshape(shape)
        out[name] = mx.array(arr)
    return out


def load_llm_weights(model_dir: str):
    """Runtime loader (torch-free) for the LLM variant.

    Returns (w_enc, w_llm, cfg):
      w_enc — encoder weights (encoder.* keys + pe_rel) for encoder_forward
      w_llm — Qwen2 weights (LoRA pre-merged, fp32) + encoder_projector.*
      cfg   — llm config dict (n_layers/n_head/n_kv_head/head_dim/rope_theta/
              rms_eps/vocab_size/adapter_ds)
    """
    import json
    from safetensors.numpy import load_file

    model_dir = Path(model_dir)
    args = json.loads((model_dir / "args.json").read_text())

    st = load_file(str(model_dir / "encoder.safetensors"))
    w_enc = {k: mx.array(v) for k, v in st.items()}
    if "encoder.positional_encoding.pe" in w_enc:
        w_enc["pe_rel"] = mx.transpose(w_enc["encoder.positional_encoding.pe"],
                                       (0, 2, 1))

    qwen_dir = model_dir / "Qwen2-7B-Instruct"
    cfg_json = json.loads((qwen_dir / "config.json").read_text())
    w_llm = {}
    index = json.loads((qwen_dir / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        # shard names come from a JSON file; constrain to plain basenames
        if Path(shard).name != shard:
            raise ValueError(f"unsafe shard name in index: {shard!r}")
        w_llm.update(_load_safetensors_shard(str(qwen_dir / shard)))

    adapter = load_file(str(model_dir / "adapter.safetensors"))
    n_layers = cfg_json["num_hidden_layers"]
    for l in range(n_layers):
        for proj in _LORA_TARGETS:
            a = adapter[f"model.layers.{l}.{proj}.lora_A.default.weight"]
            b = adapter[f"model.layers.{l}.{proj}.lora_B.default.weight"]
            key = f"model.layers.{l}.{proj}.weight"
            w_llm[key] = w_llm[key] \
                + LORA_SCALING * (mx.array(b) @ mx.array(a))
    for k, v in adapter.items():
        if k.startswith("encoder_projector."):
            w_llm[k] = mx.array(v)

    cfg = {
        "n_layers": n_layers,
        "n_head": cfg_json["num_attention_heads"],
        "n_kv_head": cfg_json["num_key_value_heads"],
        "head_dim": cfg_json["hidden_size"] // cfg_json["num_attention_heads"],
        "rope_theta": cfg_json["rope_theta"],
        "rms_eps": cfg_json["rms_norm_eps"],
        "vocab_size": cfg_json["vocab_size"],
        "adapter_ds": int(args.get("encoder_downsample_rate", 2)),
    }
    return w_enc, w_llm, cfg
