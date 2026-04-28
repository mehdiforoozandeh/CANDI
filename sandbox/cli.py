"""Unified entrypoint for sandbox subcommands."""
from __future__ import annotations

import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m sandbox.cli {train|prepare-h5|gates} ...", file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "gates":
        from sandbox.gates import main as gates_main

        return int(gates_main(rest))
    if cmd == "train":
        from sandbox.train import main as train_main

        return int(train_main(rest))
    if cmd in ("prepare-h5", "prepare_h5"):
        from sandbox import prepare_h5

        return int(prepare_h5.main(rest))
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
