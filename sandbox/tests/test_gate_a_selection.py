"""Gate A — selection.json sanity (plan §14)."""
from __future__ import annotations

import json
from pathlib import Path


def test_selection_json_gate_a():
    path = Path(__file__).resolve().parents[1] / "data" / "selection.json"
    assert path.is_file(), f"missing {path}"
    payload = json.loads(path.read_text())
    bios = payload["biosamples"]
    assert len(bios) == 5
    vb_union: set[str] = set()
    for b in bios:
        assert len(b["T"]) >= 1, b["name"]
        vb_union |= set(b.get("V", [])) | set(b.get("B", []))
    assert len(vb_union) >= 1
