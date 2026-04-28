"""Pytest fixtures/markers for sandbox gates.

Markers `gate_b`, `gate_e`, `gate_f`, `gate_g` are skipped by default; enable via
env vars `SANDBOX_RUN_GATE_B=1`, `SANDBOX_RUN_GATE_E=1`, etc. (typically only on
SLURM-backed runs where heavy I/O or GPU is available).
"""
from __future__ import annotations

import os

import pytest


_MARKERS = ("gate_b", "gate_e", "gate_f", "gate_g")


def pytest_configure(config):  # type: ignore[no-untyped-def]
    for m in _MARKERS:
        config.addinivalue_line("markers", f"{m}: sandbox plan {m.upper()} gate (opt-in via env)")


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    for item in items:
        for m in _MARKERS:
            if m in item.keywords:
                env = f"SANDBOX_RUN_{m.upper()}"
                if os.environ.get(env) != "1":
                    item.add_marker(
                        pytest.mark.skip(reason=f"set {env}=1 to run {m} (requires heavy resources)")
                    )
