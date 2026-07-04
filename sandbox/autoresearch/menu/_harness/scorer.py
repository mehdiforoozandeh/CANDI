"""Real GPU scorer — submit ONE MIG-slice sbatch for the candidate, poll to completion, parse the
ERA_SCORE/ERA_ASPECTS footer. Each loop holds ≤1 in-flight job (the driver is sequential within a
loop); the orchestrator caps total concurrency across loops.

`GpuScorer(loop)(loop, train_py) -> (era_score, aspects, status)`  — matches driver's scorer hook.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import driver

HARNESS = Path(__file__).resolve().parent
SBATCH = HARNESS / "sbatch_wrap.sh"
POLL_S = 30
MAX_WAIT_S = 2 * 3600        # queue + run backstop
# Leakage/degeneracy gate (ungameable — loops cannot edit _harness/). The model is fully editable and
# can read the h5, so the main gaming vector is training on chr21 (eval) windows. Legit Q_imp on this
# task tops out ~0.5-0.55 (baseline 0.486, ep14 peak ~0.48); Q_imp above this ceiling ⇒ eval leakage.
SUSPECT_QIMP = 0.80


def _squeue_active(jid: str) -> bool:
    r = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%T"], capture_output=True, text=True)
    return bool(r.stdout.strip())


class GpuScorer:
    def __init__(self, loop: Path):
        self.loop = Path(loop)

    def __call__(self, loop: Path, train_py: Path) -> tuple[float, dict, str]:
        loop = Path(loop)
        # +1: scoring happens before the in-flight attempt is journaled, so the live iter is max+1.
        it = max((int(r["iter"]) for r in driver.journal.read_results(loop)), default=0) + 1
        out = loop / "runs" / f"iter{it:04d}.out"
        out.parent.mkdir(exist_ok=True)
        env = {"CAND": str(train_py), "OUT": str(out)}
        sub = subprocess.run(
            ["sbatch", "--parsable", f"--job-name=menu-{loop.name}", f"--output={out}",
             f"--export=ALL,CAND={train_py},OUT={out}", str(SBATCH)],
            capture_output=True, text=True)
        jid = sub.stdout.strip().split(";")[0]
        if not jid.isdigit():
            return driver.FAIL_SCORE, {}, "crash"      # submission failed

        t0 = time.time()
        time.sleep(POLL_S)
        while _squeue_active(jid):
            if time.time() - t0 > MAX_WAIT_S:
                subprocess.run(["scancel", jid], check=False)
                return driver.FAIL_SCORE, {}, "crash"  # timeout backstop
            time.sleep(POLL_S)

        text = out.read_text() if out.exists() else ""
        (loop / "run.log").write_text("\n".join(text.splitlines()[-60:]))
        score, aspects, status = driver.parse_footer(text)
        aspects["vram_mb"] = _maxvram(jid)
        # leakage gate: an implausibly high Q_imp ⇒ the candidate trained on eval (chr21) data → gate it
        qi = aspects.get("Q_imp")
        if status == "ok" and isinstance(qi, (int, float)) and qi > SUSPECT_QIMP:
            (loop / "run.log").write_text(
                f"SUSPECT: Q_imp={qi:.3f} > {SUSPECT_QIMP} → likely eval (chr21) leakage; GATED to FAIL.\n"
                + (loop / "run.log").read_text())
            return driver.FAIL_SCORE, {**aspects, "suspect": f"Q_imp={qi:.3f}"}, "crash"
        return score, aspects, status


def _maxvram(jid: str) -> str:
    r = subprocess.run(["sacct", "-nj", f"{jid}.batch", "--format=MaxVMSize", "-P"],
                       capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""
