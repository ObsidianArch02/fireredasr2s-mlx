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
from safetensors.numpy import load_file

from .models.conformer_encoder import abs_positional_encoding, build_rel_pe


def convert_checkpoint(pth_path: str, out_dir: str):
    """ONE-TIME conversion, must run inside the PyTorch venv (imports torch).
    Writes <out_dir>/model.safetensors + <out_dir>/args.json."""
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
