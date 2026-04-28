"""Gate E: runs overfit-sanity smoke once against sandbox.h5. Heavy, opt-in via SANDBOX_RUN_GATE_E=1."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.gate_e


_REPO = Path(__file__).resolve().parents[2]


def test_gate_e_overfit_runs():
    h5 = _REPO / "sandbox" / "data" / "sandbox.h5"
    assert h5.is_file(), f"missing {h5} (run sandbox.cli gates b)"
    rc = subprocess.call(
        [sys.executable, "-m", "sandbox.prepare_h5", "overfit-sanity", "--h5", str(h5), "--steps", "50"],
        cwd=str(_REPO),
    )
    assert rc == 0
