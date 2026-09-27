"""Pytest fixtures for the FireRedASR2S MLX port test suite.

Inserts the project root (fireredasr2s-port/) into sys.path so that
`import fireredasr2_mlx` works no matter which cwd pytest is launched from.
Also registers the `slow` marker so the golden-parity tests can be filtered
with `-m "not slow"` without emitting PytestUnknownMarkWarning.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: end-to-end golden parity (runs the full four-module system on "
        "all 3 wavs; minutes, not seconds)")
