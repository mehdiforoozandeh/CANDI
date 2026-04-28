"""Gate G: validator tests against sandbox/runs/gate_g_type1_chr19 + gate_g_type2_loci. Heavy, opt-in."""
from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.gate_g


_REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("regime", ["type1_chr19", "type2_loci"])
def test_gate_g_validator_passes(regime: str) -> None:
    run_dir = _REPO / "sandbox" / "runs" / f"gate_g_{regime}"
    assert run_dir.is_dir(), f"missing {run_dir}"
    from sandbox.validate_gate_g import validate

    ok, rep = validate(run_dir)
    assert ok, rep
