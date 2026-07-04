"""Stateless agent step — one headless `claude -p` (failover `cursor-agent -p`) invocation that
makes ONE surgical edit to the loop's train.py and emits a parseable reflection footer.

Diff-based (unlike ERA's whole-program generation): the CLI runs with cwd = the loop dir, edits
train.py in place, and may inspect its OWN loop's git history (`git log/show/diff`) read-only.
Isolation is structural — the nested loop repo + scope fence + smoke + ratchet-revert mean a bad
edit can only waste an iteration, never escape or regress the champion.

Process-group safety (ERA lesson, see memory): each CLI runs in its own session and is SIGKILLed
as a group on timeout/completion so orphans don't blow the login-node pids cap.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

NO_EDIT = "(agent produced no edit)"
RATE_LIMITED = "(rate/usage-limited: waiting for reset)"
# Signals a rate/usage/session limit in a CLI's stdout+stderr (claude or cursor). Only consulted when
# NEITHER agent produced an edit, so it sees the CLI's error text, not candidate code → low false-pos.
_RATE_LIMIT_PAT = re.compile(
    r"usage limit|rate.?limit|session limit|limit reached|limit will reset|"
    r"too many requests|\b429\b|resource_exhausted|quota|overloaded", re.I)


def _rate_limited(text: str) -> bool:
    return bool(text and _RATE_LIMIT_PAT.search(text))

FOOTER = """
When done, end your reply with EXACTLY this block (one line each), nothing after it:
CHANGE_SUMMARY: <≤12-word description of the single change you made>
HYPOTHESIS: <the one change and the bet, one sentence>
RATIONALE: <why, citing a prior/result/menu item>
EXPECTED: <predicted effect on which scored metric>
PARKED: <one idea you did NOT pursue now (for the backlog), or 'none'>
"""

TASK = """
You are one iteration of a greedy autoresearch loop. Read the context above (your thesis, the
current-best train.py, results.tsv, your loop's git history, recent reflections, the backlog).
You MAY use Bash ONLY for read-only git inspection of THIS loop — `git log`, `git show <sha>`,
`git diff <a> <b>` — to see the exact code of any prior attempt. Then make ONE small, attributable
improvement along your thesis. You MAY edit ./train.py AND/OR the vendored model package ./candi_model/
(your program.md says WHERE your thesis lives, e.g. candi_model/encoder.py or candi_model/model.py).
Keep it minimal — ONE attributable change.

VALIDATE before finishing (REQUIRED): run `python ../_harness/smoke.py train.py` (a FAST ~10s CPU
preflight) and iterate until it prints `SMOKE_OK`. It imports your train.py + ./candi_model and runs one
tiny CPU batch through corrupt→forward→loss→backward + an eval-style forward, so it DETERMINISTICALLY
catches compile / shape / dtype / non-finite / missing-output-key / broken-grad bugs in BOTH files —
cheaply, with NO GPU. Do NOT finish on a SMOKE_FAIL: a non-OK edit is wasted — the harness re-runs the
SAME smoke as a HARD GATE and auto-reverts your edit before any GPU is spent.

HARD RULES:
- DO NOT run ./train.py, `python train.py`, `run_and_score`, or any TRAINING/SCORING. There is NO GPU
  on this node; running the full candidate WILL hang for the whole budget. (`smoke.py` is fine — it is
  the tiny CPU preflight, NOT the full run.) The harness scores your edit on a GPU after you exit.
- Keep the determinism flags + the pval/peak pairing assertion + TRAIN_BS/TRAIN_SEED; never read chr21.
- Edit ONLY ./train.py and ./candi_model/ inside THIS loop dir; never touch anything outside it
  (a kernel sandbox enforces this — any write outside your loop dir fails).
{footer}""".format(footer=FOOTER)


def _bwrap_prefix(loop: Path) -> list[str]:
    """bubblewrap sandbox: read-only root, the LOOP dir the only writable project path. Reads work
    everywhere (judge, model.py, venv, h5, node); writes outside the loop are kernel-blocked. The
    agent's own state dirs (~/.claude etc.) and /tmp are writable (not project code). Network is left
    intact (claude/cursor need the API)."""
    home = os.path.expanduser("~")
    pfx = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
           "--tmpfs", "/tmp", "--bind", str(loop), str(loop), "--chdir", str(loop)]
    for d in (".claude", ".cache", ".config", ".cursor", ".local", ".npm"):
        pfx += ["--bind-try", os.path.join(home, d), os.path.join(home, d)]
    pfx += ["--setenv", "MPLCONFIGDIR", os.path.join(home, ".cache", "matplotlib")]  # fast self-smoke
    return pfx + ["--"]


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


@dataclass
class AgentStep:
    claude_model: str = "opus"
    claude_effort: str = "medium"
    cursor_model: str = "composer-2.5-fast"
    timeout_s: int = 600

    def _cmd(self, backend: str) -> list[str]:
        if backend == "claude":
            return ["claude", "-p", "--model", self.claude_model, "--effort", self.claude_effort,
                    "--allowedTools", "Edit", "Write", "Read", "Bash"]
        return ["cursor-agent", "-p", "--model", self.cursor_model, "--force"]

    def _run(self, backend: str, prompt: str, cwd: Path) -> str:
        # KERNEL-ENFORCED isolation (bubblewrap): read-only root, the loop dir is the ONLY writable
        # project path. No tool/shell/Edit/Write the agent runs can touch anything outside its loop
        # (the _judge, repo-root model.py, sibling loops, etc.) — deterministic, not prompt-dependent.
        cmd = _bwrap_prefix(cwd) + self._cmd(backend)
        proc = subprocess.Popen(cmd, cwd=str(cwd), text=True,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
        try:
            out, err = proc.communicate(input=prompt, timeout=self.timeout_s)
            return (out or "") + "\n" + (err or "")   # both streams, so rate-limit text is visible
        except subprocess.TimeoutExpired:
            return ""
        finally:
            _kill_tree(proc)

    def __call__(self, loop: Path, bundle: dict) -> tuple[str, dict]:
        loop = Path(loop)
        prompt = bundle["prompt"] + "\n" + TASK
        out_c = self._run("claude", prompt, loop)
        if _edited(loop):
            return _parse(out_c, loop)
        # claude produced no edit → fail over to cursor (a SEPARATE provider, so it covers a claude
        # rate/usage limit). A real edit with a missing footer is still a valid attempt — keep it.
        out_x = self._run("cursor", prompt, loop)
        if _edited(loop):
            return _parse(out_x, loop)
        # NEITHER produced an edit. If a rate/usage/session limit is the cause, tell the driver to WAIT
        # for the limit to reset (don't waste a GPU job re-scoring the champion, don't burn an
        # iteration). Otherwise it's a plain skip (bounded retry).
        if _rate_limited(out_c) or _rate_limited(out_x):
            return RATE_LIMITED, {**_empty_fields(), "outcome": "rate_limited"}
        return NO_EDIT, {**_empty_fields(), "outcome": "no_edit"}


def _edited(loop: Path) -> bool:
    """True iff the agent changed the editable surface — train.py OR anything under candi_model/.
    Each iteration starts clean at the champion (driver._recover), so any working-tree change there
    is this agent's edit. Critically this DETECTS candi_model-only edits (the architecture loops),
    which a train.py-only check mis-flags as 'no edit' → wrong attribution + a redundant cursor run
    that compounds two edits into one scored iteration. __pycache__/*.pyc are gitignored, so the
    self-smoke's bytecode never false-triggers this."""
    r = subprocess.run(["git", "-C", str(loop), "status", "--porcelain", "--", "train.py", "candi_model"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def _has_footer(out: str) -> bool:
    return "CHANGE_SUMMARY:" in out


def _empty_fields() -> dict:
    return {"hypothesis": "", "rationale": "", "expected": "", "parked": ""}


def _diff_summary(loop: Path) -> str:
    r = subprocess.run(["git", "-C", str(loop), "diff", "--stat", "HEAD", "--", "train.py", "candi_model"],
                       capture_output=True, text=True)
    last = [l for l in r.stdout.splitlines() if l.strip()]
    return (last[-1].strip() if last else "edited train.py / candi_model")


def _parse(out: str, loop: Path) -> tuple[str, dict]:
    def g(key: str) -> str:
        for ln in out.splitlines():
            if ln.strip().startswith(key + ":"):
                return ln.split(":", 1)[1].strip()
        return ""
    change = g("CHANGE_SUMMARY") or f"(edit w/o footer: {_diff_summary(loop)})"
    return change, {
        "hypothesis": g("HYPOTHESIS"), "rationale": g("RATIONALE"),
        "expected": g("EXPECTED"), "parked": g("PARKED"),
    }


def reap(user: str = "mforooz") -> None:
    """End-of-round safety net: SIGKILL lingering CLI/smoke subprocesses (pids-cap protection)."""
    for pat in ("claude -p", "cursor-agent -p", "smoke.py"):
        subprocess.run(["pkill", "-9", "-u", user, "-f", pat], check=False)
