#!/usr/bin/env python3
"""Launcher for the OFFICIAL PyTorch CLI (fireredasr2s_cli).

Exists for script-layer acceptance runs: the workflow runner cannot `cd` and
cannot set PYTHONPATH, so this script does the path setup itself, then hands
over to the official `cli_main()` with sys.argv passed through unchanged.

Usage (from anywhere):
    python run_official_cli.py --wav_paths ... --outdir ...
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (script lives in scripts/)
# The official implementation is not shipped with this repo: point
# FIRERED_OFFICIAL_DIR at any local checkout of FireRedTeam/FireRedASR2S
# (defaults to the directory above this repo).
OFFICIAL_REPO = os.environ.get(
    "FIRERED_OFFICIAL_DIR",
    os.path.join(ROOT, "..", "FireRedASR2S-main"))

# `import fireredasr2s.*` resolves against the read-only official checkout.
sys.path.insert(0, OFFICIAL_REPO)
# Default model dirs in the official CLI are relative to the project root.
os.chdir(ROOT)

from fireredasr2s.fireredasr2s_cli import cli_main  # noqa: E402

if __name__ == "__main__":
    cli_main()
