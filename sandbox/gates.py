"""Run sandbox plan validation gates via pytest / slurm (do not edit the plan markdown)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TESTS = _REPO / "sandbox" / "tests"
_SLURM = _REPO / "sandbox" / "slurm"
_DATA = _REPO / "sandbox" / "data"
_RUNS = _REPO / "sandbox" / "runs"


def _run(cmd: list[str], env: dict | None = None) -> int:
    print("[gates] $", " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd, cwd=str(_REPO), env=env)


def _gate_b() -> int:
    """Submit or run bake + validate-parity. Writes sandbox/data/parity.ok."""
    script = _SLURM / "gate_b_bake_parity.sh"
    if os.environ.get("SANDBOX_USE_SBATCH", "1") == "0":
        return _run(["bash", str(script)])
    cmd = ["sbatch"]
    if os.environ.get("SANDBOX_WAIT", "1") == "1":
        cmd.append("--wait")
    cmd.append(str(script))
    return _run(cmd)


def _gate_e() -> int:
    """Submit or run overfit-sanity on sandbox.h5."""
    h5 = _DATA / "sandbox.h5"
    if not h5.is_file():
        print(f"gates e: missing {h5} (run gate_b first)", file=sys.stderr)
        return 2
    if os.environ.get("SANDBOX_USE_SBATCH", "1") == "0":
        extra = ["--relax-gate-e"] if os.environ.get("SANDBOX_RELAX_GATE_E") == "1" else []
        return _run(
            [sys.executable, "-m", "sandbox.prepare_h5", "overfit-sanity", "--h5", str(h5), *extra],
        )
    script = _SLURM / "gate_e_overfit.sh"
    cmd = ["sbatch"]
    if os.environ.get("SANDBOX_WAIT", "1") == "1":
        cmd.append("--wait")
    cmd.append(str(script))
    return _run(cmd)


def _gate_f() -> int:
    """Submit 3-epoch type2_loci training; then validate metrics.jsonl."""
    h5 = _DATA / "sandbox.h5"
    if not h5.is_file():
        print(f"gates f: missing {h5} (run gate_b first)", file=sys.stderr)
        return 2
    attempt = os.environ.get("SANDBOX_F_ATTEMPT", "1")
    run_dir = _RUNS / f"gate_f_attempt{attempt}"
    env = {**os.environ, "SANDBOX_F_ATTEMPT": attempt, "SANDBOX_F_RUN_DIR": str(run_dir)}
    if os.environ.get("SANDBOX_USE_SBATCH", "1") == "0":
        rc = _run(["bash", str(_SLURM / "gate_f_train.sh")], env=env)
        return rc
    cmd = ["sbatch"]
    if os.environ.get("SANDBOX_WAIT", "1") == "1":
        cmd.append("--wait")
    cmd.append(str(_SLURM / "gate_f_train.sh"))
    rc = subprocess.call(cmd, cwd=str(_REPO), env=env)
    if rc != 0:
        return rc
    return _run([sys.executable, "-m", "sandbox.validate_gate_f", str(run_dir)])


def _gate_g() -> int:
    """Submit two 1-epoch runs (type1_chr19, type2_loci); validate each."""
    h5 = _DATA / "sandbox.h5"
    if not h5.is_file():
        print(f"gates g: missing {h5} (run gate_b first)", file=sys.stderr)
        return 2
    overall = 0
    for regime in ("type1_chr19", "type2_loci"):
        run_dir = _RUNS / f"gate_g_{regime}"
        env = {
            **os.environ,
            "SANDBOX_G_REGIME": regime,
            "SANDBOX_G_RUN_DIR": str(run_dir),
        }
        if os.environ.get("SANDBOX_USE_SBATCH", "1") == "0":
            rc = _run(["bash", str(_SLURM / "gate_g_train.sh")], env=env)
        else:
            cmd = ["sbatch"]
            if os.environ.get("SANDBOX_WAIT", "1") == "1":
                cmd.append("--wait")
            cmd.append(str(_SLURM / "gate_g_train.sh"))
            rc = subprocess.call(cmd, cwd=str(_REPO), env=env)
        if rc != 0:
            overall = rc
            continue
        rc_v = _run([sys.executable, "-m", "sandbox.validate_gate_g", str(run_dir)])
        if rc_v != 0:
            overall = rc_v
    return overall


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run sandbox plan validation gates.")
    p.add_argument(
        "which",
        nargs="?",
        default="all",
        choices=("all", "a", "b", "c", "d", "e", "f", "g"),
        help="Gate id or 'all' for the full sandbox/tests suite.",
    )
    args = p.parse_args(argv)
    base = [sys.executable, "-m", "pytest"]
    if args.which == "all":
        return subprocess.call(base + [str(_TESTS)], cwd=str(_REPO))
    if args.which == "a":
        return subprocess.call(base + [str(_TESTS / "test_gate_a_selection.py")], cwd=str(_REPO))
    if args.which == "c":
        return subprocess.call(base + [str(_TESTS / "test_sandbox_model.py")], cwd=str(_REPO))
    if args.which == "d":
        return subprocess.call(base + [str(_TESTS), "-k", "gate_d"], cwd=str(_REPO))
    if args.which == "b":
        return _gate_b()
    if args.which == "e":
        return _gate_e()
    if args.which == "f":
        return _gate_f()
    if args.which == "g":
        return _gate_g()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
