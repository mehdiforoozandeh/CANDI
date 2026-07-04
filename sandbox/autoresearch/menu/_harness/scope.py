"""Scope fence — a loop may only ever change files inside its OWN dir.

Per-loop isolation is structural: each loop dir is its own nested git repo (`git init`), so the
agent's `git add` physically cannot stage anything outside `menu/<tag>/` (the frozen _judge/
_harness/_adapter live outside the loop repo). This fence is the belt-and-suspenders check the
driver runs BEFORE every commit: it lists the loop repo's changed + untracked paths, resolves
them to absolute, and rejects if any escapes the loop dir (e.g. a symlink or `..` traversal) or
touches a protected name.

  check_scope(loop_dir) -> (ok: bool, offenders: list[str])
  CLI:  python scope.py <loop_dir>   # exit 0 ok, 1 violation
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# files a loop is allowed to own (its mutable surface)
ALLOWED = {"train.py", "program.md", "results.tsv", "reflections.md", "backlog.md", "run.log",
           ".gitignore", "run_driver.out", "DONE"}
ALLOWED_DIRS = {"runs", "__pycache__", "candi_model"}   # candi_model: vendored EDITABLE model pkg
#                                                 __pycache__: benign bytecode from the self-smoke
_IGNORE_SUFFIX = (".pyc", ".flag", ".out")


def _changed_paths(loop_dir: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(loop_dir), "status", "--porcelain", "-uall"],
                         capture_output=True, text=True)
    paths = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        p = line[3:]                       # strip XY status code
        if " -> " in p:                    # rename: take destination
            p = p.split(" -> ", 1)[1]
        paths.append(p.strip().strip('"'))
    return paths


def check_scope(loop_dir: str | Path) -> tuple[bool, list[str]]:
    loop_dir = Path(loop_dir).resolve()
    offenders: list[str] = []
    for rel in _changed_paths(loop_dir):
        ap = (loop_dir / rel).resolve()
        # must stay inside the loop dir (catches symlinks / .. traversal)
        try:
            ap.relative_to(loop_dir)
        except ValueError:
            offenders.append(f"{rel} -> escapes loop dir ({ap})")
            continue
        if rel.endswith(_IGNORE_SUFFIX):
            continue
        top = Path(rel).parts[0]
        if top not in ALLOWED and top not in ALLOWED_DIRS:
            offenders.append(f"{rel} -> not an allowed loop artifact")
    return (len(offenders) == 0, offenders)


def main(loop_dir: str) -> int:
    ok, offenders = check_scope(loop_dir)
    if ok:
        print("SCOPE_OK")
        return 0
    print("SCOPE_FAIL:")
    for o in offenders:
        print(f"  {o}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
