"""Gate F: validator tests against the most recent sandbox/runs/gate_f_attempt*. Heavy, opt-in."""
from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.gate_f


_REPO = Path(__file__).resolve().parents[2]


def _latest_attempt() -> Path | None:
    runs = _REPO / "sandbox" / "runs"
    if not runs.is_dir():
        return None
    cands = sorted(runs.glob("gate_f_attempt*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def test_gate_f_validator_passes_on_last_run():
    run_dir = _latest_attempt()
    assert run_dir is not None, "no gate_f_attempt* run directory found"
    from sandbox.validate_gate_f import validate

    ok, rep = validate(run_dir)
    assert ok, rep
