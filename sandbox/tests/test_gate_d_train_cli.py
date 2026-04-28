"""Gate D: --dry-run / --print-config exit before touching HDF5."""
from __future__ import annotations

from sandbox.train import main as train_main


def test_train_dry_run_and_print_config_exit_zero():
    assert train_main(["--dry-run"]) == 0
    assert train_main(["--print-config"]) == 0
