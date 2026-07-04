#!/usr/bin/env python3
"""ERA driver — wires Problem + generate_fn + execute_fn into FUTS, logs every
node to tree.json, and reports the top-k diverse portfolio.

Search runs through `futs_batched` (the labeled extension): batch_size>1 uses
`batched_search` (concurrent execution), batch_size==1 uses the logged
sequential `search`. The frozen `futs.py` engine is the reference judge; the
extension reuses its PUCT/rank/backprop primitives unchanged.

Usage (from inside the copied tag dir):
    python run.py --config config.yaml
    python run.py --mock            # in-process self-test (no GPU, no tokens)

`tree.json` is rewritten after every node, so a search is resumable/inspectable
mid-flight and the portfolio survives a crash.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import sys
from pathlib import Path

import yaml

import futs
import futs_batched
from execute import Executor
from generate import Generator

HERE = Path(__file__).resolve().parent


def _write_tree(path: Path, nodes: list) -> None:
    path.write_text(json.dumps(nodes, indent=2))


def _portfolio(nodes: list, k: int) -> list:
    ranked = sorted([n for n in nodes if n["score"] > -1e8], key=lambda n: -n["score"])
    seen, out = set(), []
    for n in ranked:
        key = n["program"][:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
        if len(out) >= k:
            break
    return out


def _acquire_single_instance_lock():
    """Refuse to start if another driver is already running. Two concurrent run.py would
    each `--resume` the SAME tree (identical Executor counter) and submit the SAME candidate
    batch -> duplicate GPU jobs. An exclusive flock makes that structurally impossible: the
    second driver dies loudly instead of silently double-submitting. The OS releases the lock
    automatically when the holder exits (even on SIGKILL by the login-node watchdog)."""
    lock_path = HERE / "runs" / ".driver.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        prev = ""
        try:
            prev = lock_path.read_text().strip()
        except OSError:
            pass
        raise SystemExit(f"[run.py] another ERA driver already holds {lock_path} ({prev}) -> "
                         f"refusing to start (prevents concurrent drivers double-submitting). "
                         f"Kill the other run.py first, then retry.")
    fh.write(f"pid {os.getpid()}\n")
    fh.flush()
    return fh                                    # keep alive for the process lifetime


def run_real(cfg: dict, resume: bool = False) -> None:
    _LOCK_FH = _acquire_single_instance_lock()   # noqa: F841 (held until process exit)
    mod = importlib.import_module(cfg["problem_module"])
    problem, init_sol, init_score = mod.get_problem()

    gen = Generator(
        backend=cfg.get("backend", "claude"),
        claude_model=cfg.get("claude_model", "opus"),
        claude_effort=cfg.get("claude_effort", "medium"),
        cursor_model=cfg.get("cursor_model", "composer-2.5-fast"),
        timeout_s=int(cfg.get("generate_timeout_s", 600)),
        cooldown_s=int(cfg.get("cooldown_s", 1800)),
        fallback=bool(cfg.get("fallback", True)),
        claude_attempts=int(cfg.get("claude_attempts", 3)),
        cursor_attempts=int(cfg.get("cursor_attempts", 2)),
        smoke_timeout_s=int(cfg.get("smoke_timeout_s", 120)),
        note_path=str(HERE / "NOTE.md"),          # NOTE.md observations -> generation (PLAN §5)
    )
    ex = Executor(
        run_dir=str(HERE / "runs"),
        python=cfg.get("python", sys.executable),
        exec_mode=cfg.get("exec_mode", "sbatch_per_run"),
        account=cfg.get("account", ""),
        gres=cfg.get("gres", ""),
        cpus=int(cfg.get("cpus", 2)),
        mem=cfg.get("mem", "4G"),
        time_limit=cfg.get("time_limit", "0:30:00"),
        setup_cmds=tuple(cfg.get("setup_cmds", ())),
        max_wall_s=int(cfg.get("max_wall_s", 5400)),
    )

    tree_path = HERE / "tree.json"
    nodes: list = []
    from notebook import Notebook
    nb = Notebook(HERE)                            # RESULTS.tsv + tree.mmd + NOTE.md (PLAN §5)

    def on_node(nd: futs.Node) -> None:
        run_dir = HERE / "runs"
        prog_path = ""
        for p in sorted(run_dir.glob("cand*/program.py")):
            if p.read_text() == nd.solution.program:
                prog_path = str(p.relative_to(HERE))
                break
        nodes.append({
            "index": nd.index, "parent_index": nd.parent_index,
            "score": nd.score, "num_visits": nd.num_visits,
            "program_path": prog_path, "program": nd.solution.program,
        })
        _write_tree(tree_path, nodes)
        nb.on_node(nd)

    num_it = int(cfg.get("num_iterations", 20))
    c_puct = float(cfg.get("c_puct", 1.0))
    batch = int(cfg.get("batch_size", 4))
    gen_workers = int(cfg.get("gen_workers", 0))      # 0 -> = batch_size (full parallel gen)

    # --- warm-start vs cold-start -----------------------------------------------------------
    init_nodes = None
    if resume:
        if not tree_path.exists():
            raise SystemExit("[resume] no tree.json found to continue from")
        init_nodes = futs_batched.nodes_from_records(json.loads(tree_path.read_text()))
        existing = [int(p.name[4:]) for p in (HERE / "runs").glob("cand*") if p.name[4:].isdigit()]
        ex._counter = max(existing) if existing else 0        # new runs continue past cand{max}, never overwrite
        for nd in init_nodes:                                 # rebuild tree.json + notebook from the loaded tree
            on_node(nd)
        print(f"[resume] loaded {len(init_nodes)} nodes from tree.json; continuing for {num_it} "
              f"MORE (new candidates start at cand{ex._counter + 1:04d})")
    elif init_score is None:
        print("Anchoring root: executing baseline once ...")
        init_score = ex(problem, init_sol)
        print(f"baseline score = {init_score:.6f}")

    if batch > 1:
        best_sol, best_score = futs_batched.batched_search(
            problem, init_sol, init_score, gen, ex.run_batch,
            num_iterations=num_it, batch_size=batch, c_puct=c_puct, on_node=on_node,
            gen_workers=gen_workers, initial_nodes=init_nodes,
        )
    else:
        best_sol, best_score = futs_batched.search(
            problem, init_sol, init_score, gen, ex,
            num_iterations=num_it, c_puct=c_puct, on_node=on_node, initial_nodes=init_nodes,
        )

    print(f"\nBEST score = {best_score:.6f}")
    k = int(cfg.get("portfolio_k", 5))
    print(f"\nTop-{k} portfolio:")
    for n in _portfolio(nodes, k):
        print(f"  node {n['index']:>3}  score={n['score']:.6f}  {n['program_path']}")


def run_mock() -> int:
    """End-to-end driver test with a trivial numeric problem (optimum at 3.0)."""
    problem = futs.Problem("toy: maximise -(VALUE-3)^2")
    init = futs.Solution("VALUE = 0.0")

    import random
    rng = random.Random(0)

    def gen(_p, parent, _s):
        cur = float(parent.program.split("=")[1])
        return futs.Solution(f"VALUE = {cur + rng.uniform(-2, 2):.4f}")

    def ex_batch(_p, sols):
        return [-(float(s.program.split("=")[1]) - 3.0) ** 2 for s in sols]

    nodes: list = []
    best_sol, best_score = futs_batched.batched_search(
        problem, init, ex_batch(problem, [init])[0], gen, ex_batch,
        num_iterations=60, batch_size=4, c_puct=1.0,
        on_node=lambda nd: nodes.append(nd),
    )
    val = float(best_sol.program.split("=")[1])
    assert best_score > -0.25, (best_score, val)          # converged near optimum
    assert len(nodes) == 1 + 60                            # root + 60 children
    print(f"run.py mock: PASS  (best VALUE={val:.3f}, score={best_score:.4f}, nodes={len(nodes)})")
    return 0


def run_resume_selftest() -> int:
    """In-process proof of the warm-start path (no GPU, no tokens, no SLURM): run a toy
    search, serialise it like tree.json (with deliberately WRONG num_visits), reconstruct,
    and verify (a) recomputed num_visits match the live run exactly, (b) resuming continues
    node indices and adds exactly the requested new nodes without re-emitting the root."""
    problem = futs.Problem("toy: maximise -(VALUE-3)^2")
    init = futs.Solution("VALUE = 0.0")

    import random
    rng = random.Random(0)

    def gen(_p, parent, _s):
        cur = float(parent.program.split("=")[1])
        return futs.Solution(f"VALUE = {cur + rng.uniform(-2, 2):.4f}")

    def ex_batch(_p, sols):
        return [-(float(s.program.split("=")[1]) - 3.0) ** 2 for s in sols]

    # 1) run a search, capturing LIVE node refs (so we read their FINAL num_visits)
    live: list = []
    futs_batched.batched_search(problem, init, ex_batch(problem, [init])[0], gen, ex_batch,
                                num_iterations=20, batch_size=4, c_puct=1.0,
                                on_node=lambda nd: live.append(nd))
    n0 = len(live)                                    # root + 20 expansions = 21
    assert n0 == 21, n0

    # 2) serialise like tree.json — and corrupt num_visits to prove we never trust it
    records = [{"index": nd.index, "parent_index": nd.parent_index, "score": nd.score,
                "program": nd.solution.program, "num_visits": 999} for nd in live]

    # 3) reconstruct + faithfulness check (recomputed visits == live visits)
    recon = futs_batched.nodes_from_records(records)
    assert len(recon) == n0
    for a, b in zip(sorted(live, key=lambda n: n.index), recon):
        assert a.index == b.index and a.parent_index == b.parent_index
        assert abs(a.score - b.score) < 1e-12
        assert a.num_visits == b.num_visits, (a.index, a.num_visits, b.num_visits)

    # 4) resume for 10 MORE; on_node must fire only for the 10 new nodes, indices contiguous
    seen: list = []
    _, best_score = futs_batched.batched_search(problem, init, 0.0, gen, ex_batch,
                                                num_iterations=10, batch_size=4, c_puct=1.0,
                                                on_node=lambda nd: seen.append(nd),
                                                initial_nodes=recon)
    assert len(seen) == 10, len(seen)
    assert [nd.index for nd in seen] == list(range(n0, n0 + 10)), [nd.index for nd in seen]
    print(f"run.py resume-selftest: PASS  (live={n0} nodes; recomputed num_visits match; "
          f"resumed +10 -> indices {n0}..{n0 + 9}; best={best_score:.4f})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue a finished/interrupted search from tree.json "
                         "(num_iterations is then how many ADDITIONAL nodes to add)")
    ap.add_argument("--resume-selftest", action="store_true",
                    help="in-process proof of the warm-start path (no GPU/tokens)")
    args = ap.parse_args()
    if args.resume_selftest:
        sys.exit(run_resume_selftest())
    if args.mock:
        sys.exit(run_mock())
    cfg = yaml.safe_load((HERE / args.config).read_text())
    run_real(cfg, resume=args.resume)
