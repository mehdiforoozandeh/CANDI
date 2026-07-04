"""Per-loop driver entrypoint (one per tmux session). Wires the REAL agent (claude/cursor) + REAL
GPU scorer into the ratchet and runs until the global STOP flag appears (NEVER STOP otherwise).

  python run_driver.py <loop_tag>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import driver           # noqa: E402
from agent_step import AgentStep   # noqa: E402
from scorer import GpuScorer       # noqa: E402

MENU = HERE.parent
STOP = MENU / "STOP"
MAX_ITERS = int(os.environ.get("MENU_MAX_ITERS", "1000000000"))   # smoke test sets e.g. 3


def main(tag: str) -> int:
    loop = MENU / tag
    if not (loop / "train.py").exists():
        print(f"[run_driver] {tag} not initialised (no train.py)"); return 1
    (loop / "DONE").unlink(missing_ok=True)
    driver.run_loop(loop, GpuScorer(loop), AgentStep(), max_iters=MAX_ITERS, stop_flag=STOP)
    (loop / "DONE").write_text("driver exited cleanly (cap reached or STOP)\n")   # → watchdog won't restart
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
