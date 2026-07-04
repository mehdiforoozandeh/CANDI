#!/usr/bin/env python3
"""execute_fn for ERA — runs a candidate program and returns its score.

A candidate `Solution.program` is a complete, self-contained script that runs
under a fixed budget and prints a parseable footer line:

    ERA_SCORE: <float>

Each candidate is written to its own throwaway workdir under `run_dir` and run
with that dir as cwd, so concurrent candidates never collide.

Two execution modes (config `exec_mode`):

  sbatch_per_run  (DEFAULT)
      Each candidate becomes its own short SLURM job. A batch is submitted
      concurrently and harvested as jobs finish (paired with batched_search).
      Cluster specifics — account, --gres, cpus, mem, time, and env setup
      lines — are all config-driven (nothing cluster-specific is hardcoded).

  interactive     (FALLBACK / off-cluster)
      Run candidates as a local subprocess (on a held salloc allocation, or
      just locally for small problems / the regression example / selftest).

Crash / OOM / timeout / missing footer -> FAIL_SCORE so FUTS abandons the node.

Self-test (no cluster, no GPU):
    python execute.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from futs import Problem, Solution

FAIL_SCORE = -1e9
_SCORE_RE = re.compile(r"ERA_SCORE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_score(text: str) -> float:
    m = list(_SCORE_RE.finditer(text or ""))
    if not m:
        return FAIL_SCORE
    try:
        return float(m[-1].group(1))
    except ValueError:
        return FAIL_SCORE


@dataclass
class Executor:
    run_dir: str
    python: str = sys.executable        # interpreter to run each candidate with
    exec_mode: str = "sbatch_per_run"
    # --- SLURM knobs (sbatch_per_run only; all cluster-specific) ------------
    account: str = ""                   # "" -> omit #SBATCH --account
    gres: str = ""                      # "" -> omit #SBATCH --gres (CPU clusters)
    cpus: int = 2
    mem: str = "4G"
    time_limit: str = "0:30:00"         # short per-run walltime
    setup_cmds: Tuple[str, ...] = ()    # shell lines run before the program (env/module)
    poll_s: int = 20
    # --- interactive knob ---------------------------------------------------
    max_wall_s: int = 5400              # hard ceiling for the local subprocess
    _counter: int = field(default=0, repr=False)

    def _next_id(self) -> str:
        self._counter += 1
        return f"cand{self._counter:04d}"

    # -- public API ----------------------------------------------------------

    def __call__(self, problem: Problem, solution: Solution) -> float:
        return self.run_batch(problem, [solution])[0]

    def run_batch(self, problem: Problem, solutions: List[Solution]) -> List[float]:
        if self.exec_mode == "sbatch_per_run":
            return self._sbatch_batch(solutions)
        return [self._interactive(s) for s in solutions]

    # -- shared --------------------------------------------------------------

    def _write_program(self, sol: Solution) -> Tuple[str, Path]:
        rid = self._next_id()
        d = Path(self.run_dir) / rid
        d.mkdir(parents=True, exist_ok=True)
        prog = d / "program.py"
        prog.write_text(sol.program)
        return rid, prog

    # -- mode A: one SLURM job per candidate ---------------------------------

    def render_submit(self, rid: str, prog: Path) -> str:
        d = prog.parent
        out = d / "slurm-%j.out"
        lines = ["#!/bin/bash", f"#SBATCH --cpus-per-task={self.cpus}"]
        if self.account:
            lines.insert(1, f"#SBATCH --account={self.account}")
        if self.gres:
            lines.append(f"#SBATCH --gres={self.gres}")
        lines += [
            f"#SBATCH --mem={self.mem}",
            f"#SBATCH --time={self.time_limit}",
            f"#SBATCH --output={out}",
            "",
            *self.setup_cmds,
            "",
            f"cd {d}",
            f"{self.python} -u program.py",
            "",
        ]
        return "\n".join(lines)

    def _sbatch_batch(self, solutions: List[Solution]) -> List[float]:
        jobs = []  # (rid, jobid, dir)
        for sol in solutions:
            rid, prog = self._write_program(sol)
            d = prog.parent
            sub = d / "submit.sh"
            sub.write_text(self.render_submit(rid, prog))
            try:
                res = subprocess.run(
                    ["sbatch", "--parsable", str(sub)],
                    capture_output=True, text=True, check=True,
                )
                jobid = res.stdout.strip().split(";")[0]
            except (subprocess.CalledProcessError, FileNotFoundError):
                jobid = ""
            jobs.append((rid, jobid, d))

        self._wait_all([j[1] for j in jobs if j[1]])

        scores = []
        for rid, jobid, d in jobs:
            outs = sorted(d.glob("slurm-*.out"))
            text = outs[-1].read_text() if outs else ""
            scores.append(parse_score(text))
        return scores

    def _wait_all(self, jobids: List[str]) -> None:
        if not jobids:
            return
        pending = set(jobids)
        while pending:
            try:
                q = subprocess.run(
                    ["squeue", "-h", "-j", ",".join(pending), "-o", "%i"],
                    capture_output=True, text=True,
                )
                still = {x.strip() for x in q.stdout.split() if x.strip()}
            except FileNotFoundError:
                return
            pending &= still
            if pending:
                time.sleep(self.poll_s)

    # -- mode B: run as a local subprocess -----------------------------------

    def _interactive(self, sol: Solution) -> float:
        rid, prog = self._write_program(sol)
        d = prog.parent
        try:
            res = subprocess.run(
                [self.python, "-u", "program.py"],
                cwd=str(d), capture_output=True, text=True, timeout=self.max_wall_s,
            )
        except subprocess.TimeoutExpired:
            return FAIL_SCORE
        (d / "stdout.log").write_text(res.stdout + "\n--- STDERR ---\n" + res.stderr)
        return parse_score(res.stdout)


def _dry_run() -> int:
    ex = Executor(run_dir="/tmp/era_dry", gres="gpu:1",
                  setup_cmds=("module load python", "source ./venv/bin/activate"))
    rid, prog = ex._write_program(Solution("print('ERA_SCORE: -0.4438')"))
    rendered = ex.render_submit(rid, prog)
    print("=== rendered submit.sh ===")
    print(rendered)
    assert "--gres=gpu:1" in rendered
    assert "module load python" in rendered
    assert "python -u program.py" in rendered
    sample = "...work...\nERA_SCORE: -0.4438\ndone\n"
    assert abs(parse_score(sample) - (-0.4438)) < 1e-9
    assert parse_score("no footer here") == FAIL_SCORE
    assert parse_score("ERA_SCORE: 1.0\nERA_SCORE: 2.5") == 2.5  # last wins
    # gres omitted when empty
    assert "--gres" not in Executor(run_dir="/tmp/era_dry").render_submit(rid, prog)
    print("execute.py dry-run: PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.parse_args()
    raise SystemExit(_dry_run())
