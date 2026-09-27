#!/usr/bin/env python3
"""Launcher for the MLX CLI (fireredasr2_mlx.cli).

Exists for script-layer acceptance runs: the workflow runner cannot `cd` and
cannot set PYTHONPATH, so this script does the path setup itself, then hands
over to `fireredasr2_mlx.cli.cli_main()` with sys.argv passed through.

Usage (from anywhere):
    python fireredasr2s_mlx_cli.py --wav_paths ... --outdir ...
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (script lives in scripts/)

sys.path.insert(0, ROOT)
# Default model dirs (--asr_model_dir etc.) are relative to the project root.
os.chdir(ROOT)

from fireredasr2_mlx.cli import cli_main  # noqa: E402

if __name__ == "__main__":
    cli_main()
