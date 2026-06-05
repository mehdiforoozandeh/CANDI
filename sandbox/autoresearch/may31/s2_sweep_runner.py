#!/usr/bin/env python3
"""Session 2 autoresearch sweep — edits train.py only, commits, runs agent_step."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAIN = REPO / "sandbox/autoresearch/may31/train.py"
RESULTS = REPO / "sandbox/autoresearch/may31/results.tsv"
LOG = REPO / "sandbox/autoresearch/may31/s2_sweep.log"

BEST_COMMIT = "07c9016a"
BEST_PRIMARY = 0.750292
BEST_IMP_R2 = -0.249708
BASELINE_PEARSON = 0.372441
PEARSON_FLOOR = max(BASELINE_PEARSON - 0.05, 0.38)
DCR_LO, DCR_HI = 3.25, 4.75
DEN_KEEP_MIN = 0.28

# exp23 seed defaults (session 2 baseline)
SEED_CFG = {
    "lambda_mse_imp": 0.0,
    "lambda_mse_obs": 0.2,
    "calib_loss": "raw",
    "imp_weight": 0.5,
    "obs_weight": 3.5,
    "count_weight": 2.0,
    "depth_center": 23.0,
    "dsf_sampling": "off",
    "signal_transform": "log1p",
    "use_vb_meta_on_masked": False,
}


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True, check=check)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def git_reset_best() -> None:
    run(["git", "checkout", "autoresearch/may31"])
    run(["git", "reset", "--hard", BEST_COMMIT])


def set_train_cfg(overrides: dict) -> None:
    cfg = {**SEED_CFG, **overrides}
    text = TRAIN.read_text()
    repl = [
        (r"lambda_mse_imp: float = [\d.]+", f"lambda_mse_imp: float = {cfg['lambda_mse_imp']}"),
        (r"lambda_mse_obs: float = [\d.]+", f"lambda_mse_obs: float = {cfg['lambda_mse_obs']}"),
        (r'calib_loss: str = "[^"]+"', f'calib_loss: str = "{cfg["calib_loss"]}"'),
        (r"imp_weight: float = [\d.]+", f"imp_weight: float = {cfg['imp_weight']}"),
        (r"obs_weight: float = [\d.]+", f"obs_weight: float = {cfg['obs_weight']}"),
        (r"count_weight: float = [\d.]+", f"count_weight: float = {cfg['count_weight']}"),
        (r"depth_center: float = [\d.]+", f"depth_center: float = {cfg['depth_center']}"),
        (r'dsf_sampling: str = "[^"]+"', f'dsf_sampling: str = "{cfg["dsf_sampling"]}"'),
        (r'signal_transform: str = "[^"]+"', f'signal_transform: str = "{cfg["signal_transform"]}"'),
        (r"use_vb_meta_on_masked: bool = \w+", f"use_vb_meta_on_masked: bool = {cfg['use_vb_meta_on_masked']}"),
    ]
    for pat, rep in repl:
        text, n = re.subn(pat, rep, text, count=1)
        if n != 1:
            raise RuntimeError(f"failed to patch {pat}")
    TRAIN.write_text(text)


def parse_last_row() -> dict:
    lines = RESULTS.read_text().strip().splitlines()
    if len(lines) < 2:
        return {}
    hdr = lines[0].split("\t")
    vals = lines[-1].split("\t")
    return dict(zip(hdr, vals))


def guard_ok(row: dict) -> bool:
    if row.get("status") != "ok" or row.get("vram_ok") != "true":
        return False
    dcr = float(row["dcr"])
    if not (DCR_LO <= dcr <= DCR_HI):
        return False
    if row.get("metric_phase") == "imp":
        den = float(row["den_r2"])
        pearson = float(row["imp_pearson"])
        if den < DEN_KEEP_MIN:
            return False
        if pearson < PEARSON_FLOOR:
            return False
    return True


def should_keep(row: dict, best_primary: float, best_imp: float) -> tuple[bool, str]:
    if not guard_ok(row):
        return False, "guard_fail"
    primary = float(row["primary_score"])
    imp_r2 = float(row["imp_r2"])
    if primary > best_primary + 1e-4:
        return True, "primary_up"
    if abs(primary - best_primary) <= 1e-4 and imp_r2 > best_imp + 1e-4:
        return True, "tie_imp"
    if imp_r2 >= best_imp + 0.05:
        return True, "imp_jump"
    return False, "discard"


def commit_train(desc: str) -> None:
    run(["git", "add", "sandbox/autoresearch/may31/train.py"])
    run(["git", "commit", "-m", f"autoresearch/may31 s2: {desc}"])


def one_exp(desc: str, overrides: dict, best_primary: float, best_imp: float) -> tuple[float, float, str]:
    global BEST_COMMIT
    git_reset_best()
    set_train_cfg(overrides)
    commit_train(desc)
    proc = run(
        [sys.executable, "-m", "sandbox.autoresearch.may31.agent_step", "--description", desc],
        check=False,
    )
    row = parse_last_row()
    commit = row.get("commit", "?")
    primary = float(row.get("primary_score", -999))
    imp_r2 = float(row.get("imp_r2", -999))
    den_r2 = float(row.get("den_r2", float("nan")))
    phase = row.get("metric_phase", "?")
    keep, reason = should_keep(row, best_primary, best_imp)
    log(
        f"EXP {desc} | commit={commit} primary={primary:.6f} phase={phase} "
        f"imp_r2={imp_r2:.4f} den_r2={den_r2:.4f} -> {reason}"
    )
    if keep:
        BEST_COMMIT = commit
        log(f"KEEP {commit} new best primary={primary:.6f} imp_r2={imp_r2:.4f}")
        return primary, imp_r2, commit
    run(["git", "reset", "--hard", BEST_COMMIT])
    return best_primary, best_imp, BEST_COMMIT


def build_queue() -> list[tuple[str, dict]]:
    q: list[tuple[str, dict]] = []
    # Step 2: lam_imp=0.2 + imp_weight sweep
    for iw in (0.75, 1.0, 1.5):
        q.append((f"s2 imp={iw} lam_imp=0.2", {"lambda_mse_imp": 0.2, "imp_weight": iw}))
    # Step 3
    q.append(("s2 calib=log2 lam_imp=0.2", {"calib_loss": "log2", "lambda_mse_imp": 0.2}))
    # Step 4
    q.append(("s2 vb_meta lam_imp=0.2", {"use_vb_meta_on_masked": True, "lambda_mse_imp": 0.2}))
    # Step 5
    for st in ("none", "arcsinh"):
        q.append((f"s2 signal_transform={st} lam_imp=0.2", {"signal_transform": st, "lambda_mse_imp": 0.2}))
    # Follow-ups: small lam_imp + vb / calib combos
    q.append(("s2 lam_imp=0.05", {"lambda_mse_imp": 0.05}))
    q.append(("s2 lam_imp=0.05 vb_meta", {"lambda_mse_imp": 0.05, "use_vb_meta_on_masked": True}))
    q.append(("s2 lam_imp=0.05 calib=log2", {"lambda_mse_imp": 0.05, "calib_loss": "log2"}))
    q.append(("s2 lam_imp=0.1 imp=0.75", {"lambda_mse_imp": 0.1, "imp_weight": 0.75}))
    q.append(("s2 lam_imp=0.1 imp=1.0", {"lambda_mse_imp": 0.1, "imp_weight": 1.0}))
    q.append(("s2 lam_imp=0.1 vb_meta", {"lambda_mse_imp": 0.1, "use_vb_meta_on_masked": True}))
    q.append(("s2 lam_imp=0.1 calib=log2", {"lambda_mse_imp": 0.1, "calib_loss": "log2"}))
    q.append(("s2 vb_meta only", {"use_vb_meta_on_masked": True}))
    q.append(("s2 imp=0.75 no mse_imp", {"imp_weight": 0.75}))
    q.append(("s2 imp=1.0 no mse_imp", {"imp_weight": 1.0}))
    q.append(("s2 imp=1.5 no mse_imp", {"imp_weight": 1.5}))
    q.append(("s2 lam_imp=0.02", {"lambda_mse_imp": 0.02}))
    q.append(("s2 lam_imp=0.02 vb_meta", {"lambda_mse_imp": 0.02, "use_vb_meta_on_masked": True}))
    return q


def main() -> int:
    global BEST_COMMIT, BEST_PRIMARY, BEST_IMP_R2
    LOG.write_text(f"=== s2 sweep start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    best_p, best_imp = BEST_PRIMARY, BEST_IMP_R2
    queue = build_queue()
    cycle = 0
    while True:
        cycle += 1
        log(f"--- cycle {cycle} ---")
        improved = False
        for desc, overrides in queue:
            try:
                new_p, new_imp, _ = one_exp(desc, overrides, best_p, best_imp)
                if new_p > best_p + 1e-4 or new_imp > best_imp + 0.05:
                    improved = True
                best_p, best_imp = new_p, new_imp
                BEST_PRIMARY, BEST_IMP_R2 = best_p, best_imp
            except Exception as exc:
                log(f"ERROR {desc}: {exc}")
                git_reset_best()
        if not improved:
            log("cycle had no improvement; continuing anyway")
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
