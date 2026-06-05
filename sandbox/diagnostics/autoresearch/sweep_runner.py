#!/usr/bin/env python3
"""Autoresearch A1/A2/A3 sweep — patches train.py TrainConfig only."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "sandbox/diagnostics/autoresearch/train.py"
RESULTS = ROOT / "sandbox/diagnostics/autoresearch/results.tsv"
LOG = ROOT / "sandbox/diagnostics/autoresearch/sweep.log"

BASE = {
    "use_depth_offset": True,
    "depth_center": 27.0,
    "depth_scale_mode": "pow2",
    "depth_linexp_alpha": 0.693147,
    "n_mode": "softplus",
    "mu_eps": 1e-6,
    "optimizer": "adamax",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "sgd_momentum": 0.0,
    "clip_norm": 0.5,
    "obs_weight": 0.5,
    "imp_weight": 8.0,
    "count_weight": 1.0,
}


def log(msg: str) -> None:
    line = msg.rstrip() + "\n"
    with LOG.open("a") as f:
        f.write(line)
    print(msg, flush=True)


def best_score() -> float:
    if not RESULTS.exists():
        return 9.999
    best = 9.999
    for line in RESULTS.read_text().splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 5 and p[4] == "keep":
            best = min(best, float(p[1]))
    return best


def apply_config(cfg: dict) -> None:
    merged = {**BASE, **cfg}
    text = TRAIN.read_text()
    for key, val in merged.items():
        if isinstance(val, bool):
            rep = f"{key}: bool = {val}"
            pat = rf"^\s*{re.escape(key)}: bool = .*$"
        elif isinstance(val, str):
            rep = f'{key}: str = "{val}"'
            pat = rf'^\s*{re.escape(key)}: str = .*$'
        elif isinstance(val, int):
            rep = f"{key}: int = {val}"
            pat = rf"^\s*{re.escape(key)}: int = .*$"
        else:
            rep = f"{key}: float = {val}"
            pat = rf"^\s*{re.escape(key)}: float = .*$"
        new_text, n = re.subn(pat, f"    {rep}", text, count=1, flags=re.M)
        if n == 0:
            raise KeyError(f"field not found: {key}")
        text = new_text
    TRAIN.write_text(text)


def run_one(desc: str, cfg: dict) -> None:
    apply_config(cfg)
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)}
    subprocess.run(["git", "add", "sandbox/diagnostics/autoresearch/train.py"], cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, "-m", "sandbox.diagnostics.autoresearch.scope", "--staged"],
        cwd=ROOT, check=True, env=env,
    )
    subprocess.run(["git", "commit", "-m", f"autoresearch: {desc}"], cwd=ROOT, check=True)
    proc = subprocess.run(
        [sys.executable, "-m", "sandbox.diagnostics.autoresearch.train"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    log_text = proc.stdout + proc.stderr
    (ROOT / "sandbox/diagnostics/autoresearch/run.log").write_text(log_text)
    d: dict[str, str] = {}
    inside = False
    for line in log_text.splitlines():
        if line.strip() == "---":
            inside = not inside
            continue
        if inside and ":" in line:
            k, _, v = line.partition(":")
            d[k.strip()] = v.strip()
    score = float(d.get("composite_score", "9.999"))
    peak = float(d.get("peak_vram_mb", "0") or "0")
    pok = d.get("peak_vram_ok", "false").lower() == "true"
    stat = d.get("status", "crash")
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    b = best_score()
    keep = score < b and pok and stat == "ok"
    status = "keep" if keep else ("crash" if stat != "ok" else "discard")
    with RESULTS.open("a") as f:
        f.write(f"{commit}\t{score:.6f}\t{peak/1024:.1f}\t{str(pok).lower()}\t{status}\t{desc}\n")
    log(f"EXP {desc} -> score={score:.6f} best={b:.6f} {status}")
    if keep:
        log(f"NEW_BEST {score:.6f} {desc}")
        BASE.update(cfg)
    else:
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=ROOT, check=True)
    apply_config(BASE)


def main() -> None:
    log(f"=== sweep start best={best_score():.6f} ===")
    experiments: list[tuple[str, dict]] = []
    for cn in [0.35, 0.45, 0.55, 0.65]:
        experiments.append((f"A1 clip_norm={cn}", {"clip_norm": cn}))
    for iw in [7.0, 7.5, 8.5, 9.0]:
        experiments.append((f"A1 imp_weight={iw}", {"imp_weight": iw}))
    for ow in [0.45, 0.55, 0.6]:
        experiments.append((f"A1 obs_weight={ow}", {"obs_weight": ow}))
    for lr in [5e-4, 7.5e-4, 1.5e-3]:
        experiments.append((f"A1 lr={lr}", {"lr": lr}))
    for dc in [26.5, 27.5]:
        experiments.append((f"A1 depth_center={dc}", {"depth_center": dc}))
    for wd in [1e-4, 1e-3]:
        experiments.append((f"A1 weight_decay={wd}", {"weight_decay": wd}))
    experiments.append(("A1 beta1=0.95", {"beta1": 0.95}))
    experiments.append(("A1 beta2=0.99", {"beta2": 0.99}))
    for cw in [0.5, 1.5]:
        experiments.append((f"A1 count_weight={cw}", {"count_weight": cw}))
    experiments.append(("A2 depth_scale_mode=linexp", {"depth_scale_mode": "linexp"}))
    for alpha in [0.5, 1.0]:
        experiments.append((f"A2 linexp alpha={alpha}", {"depth_scale_mode": "linexp", "depth_linexp_alpha": alpha}))
    experiments.append(("A2 n_mode=exp", {"n_mode": "exp"}))
    for eps in [1e-5, 1e-7]:
        experiments.append((f"A2 mu_eps={eps}", {"mu_eps": eps}))
    experiments.append(("A2 use_depth_offset=False", {"use_depth_offset": False}))
    for wd in [1e-4, 1e-3]:
        experiments.append((f"A3 adamw wd={wd}", {"optimizer": "adamw", "weight_decay": wd}))
    experiments.append(("A3 adam optimizer", {"optimizer": "adam"}))
    experiments.append(("A3 clip0.5 lr5e-4", {"clip_norm": 0.5, "lr": 5e-4}))
    experiments.append(("A3 clip0.35 lr1e-3", {"clip_norm": 0.35, "lr": 1e-3}))
    experiments.append(("A3 imp7.5 obs0.55", {"imp_weight": 7.5, "obs_weight": 0.55}))

    cycle = 0
    while True:
        cycle += 1
        log(f"--- cycle {cycle} ---")
        for desc, cfg in experiments:
            try:
                run_one(desc, cfg)
            except Exception as exc:
                log(f"ERROR {desc}: {exc}")
                subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=ROOT, check=False)
                apply_config(BASE)


if __name__ == "__main__":
    main()
