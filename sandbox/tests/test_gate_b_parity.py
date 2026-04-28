"""Gate B: verifies parity.ok + sandbox.h5 were produced by prepare_h5 bake+validate-parity.

Heavy: opt-in via SANDBOX_RUN_GATE_B=1. The slurm script `sandbox/slurm/gate_b_bake_parity.sh`
is the primary entrypoint; this test only checks the artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.gate_b


_DATA = Path(__file__).resolve().parents[1] / "data"


def test_gate_b_artifacts_present():
    h5 = _DATA / "sandbox.h5"
    ok = _DATA / "parity.ok"
    assert h5.is_file(), f"missing {h5} (run sandbox.cli gates b)"
    assert ok.is_file(), f"missing {ok} (run sandbox.cli gates b)"
    payload = json.loads(ok.read_text())
    assert payload.get("status") == "ok" or "mismatches" not in payload, payload
