"""Hard scope fence for E34 june3 autoresearch."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Set

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_PREFIX = "sandbox/autoresearch/june3/"

# Human-only harness files (agent must not edit).
FROZEN_REL: Set[str] = {
    "sandbox/autoresearch/june3/prepare.py",
    "sandbox/autoresearch/june3/pins.py",
    "sandbox/autoresearch/june3/ar_fixed.yaml",
    "sandbox/autoresearch/june3/pin_manifest.json",
    "sandbox/autoresearch/june3/validate_parity.py",
    "sandbox/autoresearch/june3/validate_data_frac.py",
    "sandbox/autoresearch/june3/program.md",
    "sandbox/autoresearch/june3/README.md",
    "sandbox/autoresearch/june3/AGENT_SYSTEM_PROMPT.md",
    "sandbox/autoresearch/june3/eval_bridge.py",
    "sandbox/autoresearch/june3/scope.py",
    "sandbox/autoresearch/june3/agent_step.py",
    "sandbox/autoresearch/june3/keep_rule.py",
    "sandbox/autoresearch/june3/loop.sh",
    "sandbox/autoresearch/june3/hooks/pre-commit",
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


def changed_paths(*, staged_only: bool = False) -> Set[str]:
    if staged_only:
        return set(_git_lines("diff", "--cached", "--name-only"))
    paths: Set[str] = set()
    paths.update(_git_lines("diff", "--name-only"))
    paths.update(_git_lines("diff", "--cached", "--name-only"))
    return paths


def _norm(p: str) -> str:
    return p.replace("\\", "/")


def violations_for(paths: Iterable[str]) -> List[str]:
    bad: List[str] = []
    for p in sorted(set(paths)):
        norm = _norm(p)
        if norm in FROZEN_REL:
            bad.append(f"{p} (frozen harness)")
            continue
        if not norm.startswith(ALLOWED_PREFIX):
            bad.append(f"{p} (outside {ALLOWED_PREFIX})")
    return bad


def check_scope(*, staged_only: bool = False) -> List[str]:
    return violations_for(changed_paths(staged_only=staged_only))


def assert_scope(*, staged_only: bool = False) -> None:
    bad = check_scope(staged_only=staged_only)
    if bad:
        raise ScopeViolation(bad)


class ScopeViolation(RuntimeError):
    def __init__(self, paths: List[str]) -> None:
        self.paths = list(paths)
        msg = "SCOPE VIOLATION — edits forbidden:\n  " + "\n  ".join(paths)
        msg += (
            f"\n\nAllowed: any file under `{ALLOWED_PREFIX}` except frozen harness files."
            "\nNever edit sandbox/train.py, sandbox/candi_v2/, or anything outside june3/."
        )
        super().__init__(msg)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Verify june3 autoresearch edit scope")
    p.add_argument("--staged", action="store_true", help="Only staged paths")
    p.add_argument("--worktree", action="store_true", help="All unstaged+staged (default)")
    args = p.parse_args()
    bad = check_scope(staged_only=args.staged)
    if bad:
        print("SCOPE VIOLATION:", file=sys.stderr)
        for v in bad:
            print(f"  {v}", file=sys.stderr)
        return 1
    print("scope ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
