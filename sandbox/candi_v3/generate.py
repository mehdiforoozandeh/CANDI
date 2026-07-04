#!/usr/bin/env python3
"""generate_fn for ERA — a pluggable, subscription-authenticated CLI generator.

Shells out to a terminal coding-CLI in headless/print mode so generation is
billed against your *subscription* (included tokens), not a metered API key.
Two backends, subscription-only (no metered API fallback):

  - claude : `claude -p` with the latest Opus  (DEFAULT)
  - cursor : `cursor-agent -p -m composer`     (FAILOVER)

Pre-flight smoke test + repair: every generated program is run through `smoke.py`
in an isolated subprocess (one tiny CPU batch through corrupt->forward->loss) BEFORE
it's accepted, so structural crashes (missing batch key, shape/broadcast, non-finite
loss) are caught here instead of wasting a ~20-min GPU job. On a smoke failure the
traceback is fed back to the model and it's asked to fix it. Claude gets up to
`claude_attempts` tries (generate + repairs); if it still can't produce a program
that passes smoke, generation fails over to Cursor for `cursor_attempts` more.

Failover: Claude (Opus) is tried first; on a usage/rate limit a cooldown window
opens during which Cursor is used directly (skipping the Claude timeout), after
which Claude is probed again. Both CLIs run with a hard subprocess timeout (7 min),
in their own process group (so the Node subtree is reaped, not orphaned against the
login-node pids cap), and with the prompt piped on stdin (no TTY) — Cursor's `-p`
can otherwise hang. The smoke subprocess is likewise process-grouped and killed on
timeout, and runs single-threaded (torch) to stay light on the login node.

The generator returns a whole program (ERA's `Solution`), extracted from the
model's fenced ```python block. The candidate IS the source code.

Self-tests (no tokens spent):
  python generate.py --selftest      # prompt + code-block extraction
  python generate.py --check         # report which CLIs are installed
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import List, Tuple

from pathlib import Path

from futs import Problem, Solution

HERE = Path(__file__).resolve().parent

# Signals that Claude's rolling session / usage limit was hit.
LIMIT_PATTERNS = [
    r"5-?hour limit",
    r"usage limit",
    r"rate limit",
    r"limit reached",
    r"too many requests",
    r"reset[s]? at",
    r"please try again later",
]
_LIMIT_RE = re.compile("|".join(LIMIT_PATTERNS), re.IGNORECASE)
_PY_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _observations(note_path: str) -> str:
    """Distilled 'Observations' section from NOTE.md, injected as NEUTRAL observations (not
    directives) to preserve population diversity (PLAN §5 safeguard)."""
    if not note_path:
        return ""
    try:
        txt = Path(note_path).read_text()
    except OSError:
        return ""
    m = re.search(r"## Observations.*?(?=\n## |\Z)", txt, re.DOTALL)
    body = (m.group(0) if m else "").strip()
    if not body or ("AUTO:" in body and len(body) < 120):   # placeholder only -> inject nothing
        return ""
    return ("\n# Observations from prior trials (NEUTRAL — correlations, NOT rules; you may "
            "ignore them and explore differently)\n" + body + "\n")


def make_prompt(problem: Problem, parent_solution: Solution, parent_score: float,
                note_path: str = "") -> str:
    return f"""{problem.description}
{_observations(note_path)}
# Current program (score = {parent_score:.6f})
```python
{parent_solution.program}
```

# Your task
Propose ONE improved version of the program above that scores higher on the
stated objective. You have full freedom to redesign the program's algorithm,
structure, models, and methods within the rules in the problem description.
Keep it a single self-contained program that prints the required score footer.
Also print one line `ERA_REFLECTION: <2-3 sentences: what you changed and why>`.

If the problem description links external repositories (see "Priors from the
literature"), you MAY use the WebFetch tool to read them for architectural ideas
BEFORE writing. This is optional: borrow only mechanisms that serve CANDI's fixed
task (DNA + signal in -> impute/denoise signal out), never copy a model wholesale,
and never let fetching stop you from emitting a complete program.

Output ONLY the complete program in a single ```python code block. No prose
before or after the block."""


def make_repair_prompt(problem: Problem, broken_program: str, error_text: str) -> str:
    """Feed the failing program + its pre-flight error back to the model for a fix."""
    return f"""{problem.description}

# Your previous attempt CRASHED a pre-flight check (one tiny CPU batch run through
# Objective.corrupt -> Model.forward -> Objective.loss -> backward), so it never even
# started training. Here is that program:
```python
{broken_program}
```

# The error it produced:
```
{error_text}
```

# Your task
Return a CORRECTED complete program that fixes this crash while keeping the intended
improvement. The clean batch passed to `Objective.corrupt` has ONLY these keys:
counts, counts_dsf2, counts_dsf4, counts_dsf8 (each [B,L,F]); meta, meta_dsf2,
meta_dsf4, meta_dsf8 (each [B,4,F]); avail [B,F]; dna [B,19200,4]; control [B,L,1];
ctrl_avail [B]. There is NO ground-truth signal/pval key — derive any y_signal target
yourself (e.g. arcsinh(counts)). Keep all tensors on the input tensors' own device.

Output ONLY the complete corrected program in a single ```python code block. No prose
before or after the block."""


def extract_program(text: str) -> str:
    """Return the last fenced python block, else the last fenced block, else
    the stripped text."""
    blocks = _PY_BLOCK_RE.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    generic = re.findall(r"```\s*\n(.*?)```", text or "", re.DOTALL)
    if generic:
        return generic[-1].strip()
    return (text or "").strip()


def _kill_tree(proc: "subprocess.Popen") -> None:
    """SIGKILL a CLI subprocess's whole process group (it was started with
    start_new_session=True, so proc.pid is the group leader), reap it, and close
    its pipes. Ensures the Node child/grandchildren can't outlive the call and
    leak threads/fds against the login-node 512-pids cap. No-op if already gone."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    for f in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if f and not f.closed:
                f.close()
        except Exception:
            pass


@dataclass
class Generator:
    backend: str = "claude"            # active backend
    claude_model: str = "opus"         # latest Opus alias
    claude_effort: str = "medium"      # opus thinking budget: low|medium|high (speed/quality)
    cursor_model: str = "composer-2.5-fast"   # latest Composer (fast); flag is --model not -m
    timeout_s: int = 600
    cooldown_s: int = 1800             # how long to stay on Cursor after a limit
    fallback: bool = True
    claude_attempts: int = 3           # Claude tries (1 generate + 2 repairs) before failing over
    cursor_attempts: int = 2           # Cursor repair tries after Claude is exhausted (0 if not fallback)
    smoke_timeout_s: int = 120         # pre-flight smoke subprocess cap (CPU); timeout -> inconclusive PASS
    note_path: str = ""                # NOTE.md to inject observations from (PLAN §5)
    _cooldown_until: float = field(default=0.0, repr=False)
    _lock: "threading.Lock" = field(default_factory=threading.Lock, repr=False)

    # -- subprocess wrappers -------------------------------------------------

    def _cmd(self, backend: str) -> List[str]:
        if backend == "claude":
            # --allowedTools WebFetch: pre-approve the WebFetch tool so the generator can read
            # the repos linked in the design menu (§3.6) headlessly, without a permission prompt
            # (login node reaches github.com / raw.githubusercontent.com / api.github.com).
            return ["claude", "-p", "--model", self.claude_model,
                    "--effort", self.claude_effort, "--allowedTools", "WebFetch",
                    "--output-format", "text"]
        if backend == "cursor":
            # --force: auto-approve tool calls (incl. web fetch) in headless print mode.
            return ["cursor-agent", "-p", "--model", self.cursor_model, "--force",
                    "--output-format", "text"]
        raise ValueError(f"unknown backend: {backend}")

    def _run(self, backend: str, prompt: str):
        """Return (stdout_text, limited: bool, ok: bool).

        The CLI runs in its OWN process group; on timeout OR completion the whole
        group is SIGKILLed (see _kill_tree) so the Node subtree never lingers
        against the login-node pids cap. A timeout returns empty output, so __call__
        retries (and eventually fails over to Cursor)."""
        if shutil.which(self._cmd(backend)[0]) is None:
            return "", False, False
        proc = subprocess.Popen(
            self._cmd(backend),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            out, err = proc.communicate(input=prompt, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        finally:
            _kill_tree(proc)
        blob = (out or "") + "\n" + (err or "")
        limited = bool(_LIMIT_RE.search(blob))
        ok = proc.returncode == 0 and bool(extract_program(out))
        return out, limited, ok

    def reap(self) -> None:
        """End-of-round safety net: SIGKILL any lingering generator/smoke subprocesses
        (`claude -p` / `cursor-agent -p` / `smoke.py`) so they don't hold threads
        against the login-node pids cap between rounds. Each _run / _smoke already reaps
        its own tree; this catches escapees. Never matches an interactive `claude`
        session (it has no `-p`)."""
        user = getpass.getuser()
        for pat in ("claude -p", "cursor-agent -p", "smoke.py"):
            subprocess.run(["pkill", "-9", "-u", user, "-f", pat], check=False)

    # -- pre-flight smoke test (CPU, isolated subprocess) --------------------

    def _smoke(self, program: str) -> Tuple[bool, str]:
        """Run `smoke.py` on the candidate in an isolated, process-grouped, single-
        threaded CPU subprocess. Returns (ok, error_text). A clean run -> (True, "").
        A definite crash -> (False, <traceback tail>). A timeout, or the smoke harness
        itself failing to set up (e.g. data unavailable), -> (True, "") inconclusive
        PASS: we only REJECT definite crashes, never block a slow-but-valid candidate."""
        tmp = tempfile.NamedTemporaryFile("w", suffix=".py", prefix="cand_smoke_",
                                          dir=str(HERE), delete=False)
        try:
            tmp.write(program)
            tmp.close()
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1"}
            proc = subprocess.Popen(
                [sys.executable, str(HERE / "smoke.py"), tmp.name],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True, cwd=str(HERE), env=env,
            )
            try:
                out, _ = proc.communicate(timeout=self.smoke_timeout_s)
            except subprocess.TimeoutExpired:
                out = ""
            finally:
                _kill_tree(proc)
            if "SMOKE_OK" in out:
                return True, ""
            if "SMOKE_FAIL" not in out:                  # timeout / harness setup failure -> don't block
                return True, ""
            tail = "\n".join(out.strip().splitlines()[-40:])
            return False, tail
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # -- generation: backend scheduling + smoke-test repair loop ------------

    def _backend_for(self, attempt: int) -> str:
        """Backend schedule. With backend='cursor', use Cursor for EVERY attempt (e.g. to spare
        the Claude subscription). Otherwise Claude for the first `claude_attempts` tries, then
        Cursor (if fallback); a usage limit opens a cooldown that diverts to Cursor early."""
        if self.backend == "cursor":
            return "cursor"
        with self._lock:
            on_cooldown = time.time() < self._cooldown_until
        if attempt < self.claude_attempts and not on_cooldown:
            return "claude"
        return "cursor"

    def _valid(self, prog: str) -> bool:
        return bool(prog.strip()) and ("run_and_score" in prog) and ("class Model" in prog)

    def __call__(self, problem: Problem, parent_solution: Solution, parent_score: float) -> Solution:
        # Generate -> pre-flight smoke -> on crash, feed the error back and retry. Claude
        # gets `claude_attempts` tries (generate + repairs), then Cursor `cursor_attempts`.
        base_prompt = make_prompt(problem, parent_solution, parent_score, note_path=self.note_path)
        n_total = self.claude_attempts + (self.cursor_attempts if self.fallback else 0)
        prog, last_err = "", ""
        for attempt in range(n_total):
            backend = self._backend_for(attempt)
            prompt = base_prompt if not last_err else make_repair_prompt(problem, prog, last_err)
            out, limited, _ = self._run(backend, prompt)
            if limited and backend == "claude":
                with self._lock:
                    self._cooldown_until = time.time() + self.cooldown_s
            cand = extract_program(out)
            if not self._valid(cand):                   # empty / not a program -> retry with feedback
                last_err = ("No complete program returned (must define `class Model`, "
                            "`class Objective`, and call `run_and_score`).")
                prog = cand or prog
                print(f"[generate] attempt {attempt+1}/{n_total} ({backend}): empty/invalid output",
                      flush=True)
                continue
            prog = cand
            ok, err = self._smoke(prog)                 # CPU pre-flight before spending a GPU job
            if ok:
                if attempt:
                    print(f"[generate] attempt {attempt+1}/{n_total} ({backend}): smoke PASS (repaired)",
                          flush=True)
                return Solution(program=prog)
            last_err = err
            print(f"[generate] attempt {attempt+1}/{n_total} ({backend}): smoke FAIL -> feeding error back",
                  flush=True)
        return Solution(program=prog)                   # exhausted; GPU job will -1e9 it if still broken


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _selftest() -> int:
    sample = (
        "Sure, here is the improved program:\n\n"
        "```python\nVALUE = 3.0\nprint('ERA_SCORE:', -(VALUE-3)**2)\n```\n\nDone."
    )
    prog = extract_program(sample)
    assert "VALUE = 3.0" in prog and "```" not in prog, prog
    # no-fence fallback
    assert extract_program("VALUE = 1") == "VALUE = 1"
    # limit detection
    assert _LIMIT_RE.search("You have reached your 5-hour limit, resets at 4pm")
    assert not _LIMIT_RE.search("training finished")
    # prompt is non-empty and embeds the parent program + score
    p = make_prompt(Problem("OBJ"), Solution("X=1"), -0.5)
    assert "OBJ" in p and "X=1" in p and "-0.500000" in p
    print("generate.py selftest: PASS")
    return 0


def _check() -> int:
    for name in ("claude", "cursor-agent"):
        path = shutil.which(name)
        print(f"  {name:14s} {'OK ' + path if path else 'NOT FOUND'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        sys.exit(_check())
    sys.exit(_selftest())
