"""Enforce autoresearch edit scope: sandbox/diagnostics/ only."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Set

REPO_ROOT = Path(__file__).resolve().parents[3]
DIAGNOSTICS_ROOT = REPO_ROOT / "sandbox" / "diagnostics"
ALLOWED_PREFIX = "sandbox/diagnostics/"

# During the experiment loop the agent edits this file only.
LOOP_EDIT_FILE = "sandbox/diagnostics/autoresearch/train.py"

# Gitignored runtime artifacts (never commit).
IGNORED_UNDER_AUTORESEARCH = {
    "results.tsv",
    "run.log",
    "loop.log",
    "loop.pid",
    "baseline.json",
}


def _git_lines(*args: str) -> List[str]:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def changed_paths() -> Set[str]:
    """Paths with staged or unstaged edits vs HEAD (not all untracked repo files)."""
    paths: Set[str] = set()
    paths.update(_git_lines("diff", "--name-only"))
    paths.update(_git_lines("diff", "--cached", "--name-only"))
    return paths


def paths_outside_diagnostics(paths: Set[str]) -> List[str]:
    bad: List[str] = []
    for p in sorted(paths):
        norm = p.replace("\\", "/")
        if norm.startswith(ALLOWED_PREFIX):
            continue
        bad.append(p)
    return bad


def paths_outside_loop_edit(paths: Set[str]) -> List[str]:
    """Stricter check: only train.py may change during the experiment loop."""
    bad: List[str] = []
    for p in sorted(paths):
        norm = p.replace("\\", "/")
        if norm == LOOP_EDIT_FILE:
            continue
        if norm.startswith(ALLOWED_PREFIX):
            # Other diagnostics files (docs, runs json) must not change mid-loop
            bad.append(p)
        else:
            bad.append(p)
    return bad


def check_scope(*, loop_mode: bool = True, staged_only: bool = False) -> List[str]:
    if staged_only:
        paths = set(_git_lines("diff", "--cached", "--name-only"))
    else:
        paths = changed_paths()
    if loop_mode:
        return paths_outside_loop_edit(paths)
    return paths_outside_diagnostics(paths)


def check_commit_scope(*, loop_mode: bool = True) -> List[str]:
    """Validate staged files only (run after `git add …`). Empty staged → ok."""
    staged = set(_git_lines("diff", "--cached", "--name-only"))
    if not staged:
        return []
    if loop_mode:
        return paths_outside_loop_edit(staged)
    return paths_outside_diagnostics(staged)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Verify git changes stay in scope")
    p.add_argument(
        "--loop",
        action="store_true",
        default=True,
        help="Require only autoresearch/train.py changed (default for experiment loop)",
    )
    p.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Allow any file under sandbox/diagnostics/ (e.g. end-of-session docs)",
    )
    p.add_argument(
        "--staged",
        action="store_true",
        help="Check only staged files (use after git add train.py, before commit)",
    )
    args = p.parse_args()
    if args.staged:
        violations = check_commit_scope(loop_mode=not args.diagnostics_only)
    else:
        violations = check_scope(loop_mode=not args.diagnostics_only)
    if violations:
        print("SCOPE VIOLATION — changes outside allowed paths:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        if loop_mode:
            print(
                f"\nDuring the loop, only `{LOOP_EDIT_FILE}` may be edited.",
                file=sys.stderr,
            )
        else:
            print(
                f"\nAll edits must stay under `{ALLOWED_PREFIX}`.",
                file=sys.stderr,
            )
        return 1
    print("scope ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
