"""One real ratchet iteration on a loop (integration check): real claude/cursor agent edits
train.py → scope → smoke → commit → real GPU score → keep/reset → journal. Dumps diagnostics.

  python one_iter.py <loop_tag>
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import driver        # noqa: E402
import context       # noqa: E402
from agent_step import AgentStep   # noqa: E402
from scorer import GpuScorer       # noqa: E402

MENU = HERE.parent


def main(tag: str) -> int:
    loop = MENU / tag
    print(f"[{time.strftime('%H:%M:%S')}] one_iter {tag}: assembling+validating context…", flush=True)
    ok, probs = context.validate(loop, context.assemble(loop))
    print(f"  context valid: {ok} {probs}", flush=True)
    base_train = (loop / "train.py").read_text()

    print(f"[{time.strftime('%H:%M:%S')}] running iteration (real agent + GPU)…", flush=True)
    r = driver.run_iteration(loop, GpuScorer(loop), AgentStep())
    print(f"[{time.strftime('%H:%M:%S')}] RESULT: {r}", flush=True)

    # diagnostics
    log = subprocess.run(["git", "-C", str(loop), "log", "--oneline"], capture_output=True, text=True).stdout
    print("  git log:\n   " + "\n   ".join(log.strip().splitlines()), flush=True)
    diff = subprocess.run(["git", "-C", str(loop), "show", "--stat", "HEAD"], capture_output=True, text=True).stdout
    print("  HEAD commit stat:\n   " + "\n   ".join(diff.strip().splitlines()[:12]), flush=True)
    attempt_train = subprocess.run(["git", "-C", str(loop), "show", "HEAD:train.py"],
                                   capture_output=True, text=True).stdout
    print(f"  agent edited train.py: {attempt_train != base_train} "
          f"(base {len(base_train)} chars → attempt {len(attempt_train)} chars)", flush=True)
    refl = driver.journal.read_reflections(loop)
    if refl:
        last = refl[-1]
        print(f"  reflection: hyp='{last['hypothesis'][:80]}' result='{last['result'][:80]}'", flush=True)
    rows = driver.journal.read_results(loop)
    print(f"  results rows: {len(rows)} (last status={rows[-1]['status']}, era={rows[-1]['era_score']})", flush=True)
    bk_iter, _ = driver.journal.read_backlog(loop)
    print(f"  backlog updated_at_iter: {bk_iter}", flush=True)
    ok2, probs2 = context.validate(loop, context.assemble(loop))
    print(f"  context re-validates after iter: {ok2} {probs2}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
