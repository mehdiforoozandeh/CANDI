"""Thin read-only orchestrator (DESIGN_DOC §11). One deterministic supervisor process; NEVER edits
any train.py / judge / production code, no model calls. Duties: (1) watchdog — restart a dead loop
driver tmux, scancel its orphaned job; (2) GPU-budget arbiter — observe the ≤5 in-flight cap;
(3) dashboard backend — regenerate dashboard.html; (4) plateau detector — flag a stalled loop TO the
human (writes a flag, never acts); (5) synthesis — on global STOP, write SYNTHESIS.md.

  python orchestrator.py            # supervise loop
  python orchestrator.py --once     # one pass (watchdog+dashboard+plateau), no sleep
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

MENU = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MENU / "_harness"))
import journal        # noqa: E402
import agent_step     # noqa: E402
import dashboard      # noqa: E402 (same dir)

POLL_S = 30
GPU_CAP = 5
PLATEAU_K = 12
LOOPS = [t for t, *_ in dashboard.LOOPS]
STOP = MENU / "STOP"
VENV = "/project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate"


def _tmux_alive(sess: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", sess], capture_output=True).returncode == 0


def _restart_driver(tag: str) -> None:
    subprocess.run(["scancel", "--name", f"menu-{tag}"], check=False)        # reap orphaned job first
    cmd = (f"cd {MENU}; source {VENV}; module load samtools 2>/dev/null; "
           f"export PYTHONPATH=/project/6014832/mforooz/EpiDenoise; "
           f"python -u _harness/run_driver.py {tag} > {tag}/run_driver.out 2>&1")
    subprocess.run(["tmux", "new-session", "-d", "-s", f"menu-{tag}", cmd], check=False)


def _inflight() -> int:
    r = subprocess.run(["squeue", "-h", "-u", "mforooz", "-n",
                        ",".join(f"menu-{t}" for t in LOOPS), "-o", "%T"], capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.strip()])


def _plateau(tag: str) -> None:
    rows = journal.read_results(MENU / tag)
    recent = [r for r in rows if r.get("status") in ("keep", "reset", "crash", "gate")][-PLATEAU_K:]
    flag = MENU / tag / "PLATEAU.flag"
    if len(recent) >= PLATEAU_K and not any(r["status"] == "keep" for r in recent):
        flag.write_text(f"no keep in last {PLATEAU_K} iters @ {time.strftime('%H:%M:%S')}\n")
    elif flag.exists():
        flag.unlink()


def _synthesis() -> None:
    lines = ["# Menu-AR synthesis\n", f"_written {time.strftime('%Y-%m-%d %H:%M')}_\n"]
    for tag in LOOPS:
        rows = journal.read_results(MENU / tag)
        if not rows:
            lines.append(f"## {tag}\n- (no iterations)\n"); continue
        best = max(rows, key=lambda r: float(r.get("era_score", "-1e9") or "-1e9"))
        keeps = sum(1 for r in rows if r["status"] == "keep")
        iters = max(int(r["iter"]) for r in rows)
        lines.append(f"## {tag}\n- iters {iters}, keeps {keeps}, best ERA_SCORE {best['era_score']} "
                     f"(iter {best['iter']}: {best['change_summary']})\n")
    (MENU / "SYNTHESIS.md").write_text("\n".join(lines))


def one_pass(restart: bool = True) -> None:
    for tag in LOOPS:
        done = (MENU / tag / "DONE").exists()       # driver exited cleanly (cap reached) → do NOT restart
        if restart and not done and (MENU / tag / "train.py").exists() \
                and not _tmux_alive(f"menu-{tag}") and not STOP.exists():
            _restart_driver(tag)
        _plateau(tag)
    n = _inflight()
    if n > GPU_CAP:
        (MENU / "ORCH.warn").write_text(f"in-flight {n} > cap {GPU_CAP} @ {time.strftime('%H:%M:%S')}\n")
    try:
        (MENU / "dashboard.html").write_text(dashboard.render())
    except Exception as e:
        print("dashboard render error:", e)


def main() -> int:
    once = "--once" in sys.argv
    while True:
        one_pass(restart=not once)
        if STOP.exists():
            for tag in LOOPS:
                subprocess.run(["scancel", "--name", f"menu-{tag}"], check=False)
            agent_step.reap()
            _synthesis()
            print("[orchestrator] STOP — jobs cancelled, synthesis written")
            return 0
        if once:
            return 0
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
