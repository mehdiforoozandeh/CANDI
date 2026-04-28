"""Validate a 1-epoch sandbox.train run against plan Gate G criteria."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


WALL_CLOCK_BUDGET_SEC = 90 * 60  # 90 min per plan


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolved_round_trip(cfg_path: Path) -> Tuple[bool, str]:
    from sandbox.config import deep_merge, load_yaml  # noqa: F401 (used implicitly)
    from sandbox.config_types import SandboxConfig, config_from_dict

    raw = _load_yaml(cfg_path)
    cfg_a = config_from_dict(SandboxConfig, raw)
    # Re-feed the dataclass-serialized form and ensure we arrive at the same thing.
    dumped = asdict(cfg_a)
    cfg_b = config_from_dict(SandboxConfig, dumped)
    if asdict(cfg_a) != asdict(cfg_b):
        return False, "resolved_config round-trip diverged"
    return True, "ok"


def validate(run_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    rep: Dict[str, Any] = {"run_dir": str(run_dir), "checks": {}}
    ok = True

    cfg_path = run_dir / "resolved_config.yaml"
    rep["checks"]["resolved_config_exists"] = {"ok": cfg_path.is_file(), "path": str(cfg_path)}
    ok = ok and cfg_path.is_file()

    elapsed_path = run_dir / "elapsed.txt"
    try:
        elapsed = float(elapsed_path.read_text().strip())
    except Exception as e:  # noqa: BLE001
        rep["checks"]["elapsed_txt"] = {"ok": False, "error": str(e)}
        return False, rep
    rep["checks"]["wall_clock"] = {
        "ok": elapsed <= WALL_CLOCK_BUDGET_SEC,
        "elapsed_seconds": elapsed,
        "budget_seconds": WALL_CLOCK_BUDGET_SEC,
    }
    ok = ok and elapsed <= WALL_CLOCK_BUDGET_SEC

    if cfg_path.is_file():
        rt_ok, rt_msg = _resolved_round_trip(cfg_path)
        rep["checks"]["resolved_config_round_trip"] = {"ok": rt_ok, "detail": rt_msg}
        ok = ok and rt_ok

    parity = Path(__file__).resolve().parent / "data" / "parity.ok"
    rep["checks"]["parity_ok_present"] = {"ok": parity.is_file(), "path": str(parity)}
    ok = ok and parity.is_file()

    metrics_path = run_dir / "metrics.jsonl"
    rep["checks"]["metrics_jsonl_nonempty"] = {
        "ok": metrics_path.is_file() and metrics_path.stat().st_size > 0,
        "path": str(metrics_path),
    }
    ok = ok and rep["checks"]["metrics_jsonl_nonempty"]["ok"]

    rep["ok"] = ok
    return ok, rep


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Gate G validator: walltime + config round-trip + parity.ok")
    p.add_argument("run_dir", type=Path)
    args = p.parse_args(argv)
    ok, rep = validate(Path(args.run_dir))
    print(json.dumps(rep, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
