"""Enforce autoresearch edit scope: sandbox/autoresearch/may31/ only."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Set

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_PREFIX = "sandbox/autoresearch/may31/"
LOOP_EDIT_FILE = "sandbox/autoresearch/may31/train.py"


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
    paths: Set[str] = set()
    paths.update(_git_lines("diff", "--name-only"))
    paths.update(_git_lines("diff", "--cached", "--name-only"))
    return paths


def paths_outside_allowed(paths: Set[str], *, loop_mode: bool) -> List[str]:
    bad: List[str] = []
    for p in sorted(paths):
        norm = p.replace("\\", "/")
        if loop_mode:
            if norm != LOOP_EDIT_FILE:
                bad.append(p)
        elif not norm.startswith(ALLOWED_PREFIX):
            bad.append(p)
    return bad


def check_scope(*, loop_mode: bool = True, staged_only: bool = False) -> List[str]:
    if staged_only:
        paths = set(_git_lines("diff", "--cached", "--name-only"))
    else:
        paths = changed_paths()
    return paths_outside_allowed(paths, loop_mode=loop_mode)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Verify git changes stay in may31 scope")
    p.add_argument(
        "--loop",
        action="store_true",
        default=True,
        help="Require only train.py changed (default for experiment loop)",
    )
    p.add_argument(
        "--may31-only",
        action="store_true",
        help="Allow any file under sandbox/autoresearch/may31/",
    )
    p.add_argument(
        "--staged",
        action="store_true",
        help="Check only staged files (use after git add train.py)",
    )
    args = p.parse_args()
    loop_mode = not args.may31_only
    violations = check_scope(loop_mode=loop_mode, staged_only=args.staged)
    if violations:
        print("SCOPE VIOLATION — changes outside allowed paths:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        if loop_mode:
            print(f"\nDuring the loop, only `{LOOP_EDIT_FILE}` may be edited.", file=sys.stderr)
        else:
            print(f"\nAll edits must stay under `{ALLOWED_PREFIX}`.", file=sys.stderr)
        return 1
    print("scope ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
