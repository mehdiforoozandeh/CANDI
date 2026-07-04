#!/usr/bin/env python3
"""NON-UPSTREAM extension to the frozen `futs.py` engine.

Upstream ERA ships only the sequential `futs.search()` with no node-logging hook.
This module adds two things the driver needs WITHOUT touching the frozen judge:

  search(...,  on_node=...)         # = futs.search + a per-node callback (for tree.json)
  batched_search(..., batch_size)   # selects B parents/round (virtual visits) so B
                                    #   candidates can be EXECUTED concurrently

Both reuse `futs.{Node, compute_rank_scores, compute_pucts, backpropagate_visit}`
verbatim, so the PUCT selection / rank / backprop math is identical to upstream —
the only additions are the callback and the per-round batching. `batch_size=1`
makes `batched_search` behave like `search`.

Faithfulness note: with `batch_size=1` and `on_node=None`, `search()` here is
behaviourally identical to `futs.search()`; the batched path is the documented
extension for cluster throughput.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

from futs import (
    Problem,
    Solution,
    Node,
    compute_rank_scores,
    compute_pucts,
    backpropagate_visit,
)

OnNode = Optional[Callable[[Node], None]]


def nodes_from_records(records: List[dict]) -> List[Node]:
    """Reconstruct a FUTS population from saved `tree.json` records so a finished search
    can be CONTINUED (warm-started) without re-running any candidate.

    Each record needs `index`, `parent_index`, `score`, `program`. `num_visits` in
    tree.json is NOT trusted: the driver snapshots it at node-creation time, BEFORE later
    descendants backpropagate through the node, so it understates the final visit count.
    We recompute it exactly by replaying the frozen engine's `backpropagate_visit` (one
    visit per non-root node, walking parent links — order-independent), so the resumed
    rank/PUCT statistics are identical to an uninterrupted run.
    """
    recs = sorted(records, key=lambda r: r["index"])
    nodes: List[Node] = []
    for i, r in enumerate(recs):
        if r["index"] != i:
            raise ValueError(f"tree.json records not contiguous: expected index {i}, got {r['index']}")
        nodes.append(Node(index=r["index"], parent_index=r["parent_index"],
                          solution=Solution(r["program"]), score=float(r["score"]), num_visits=0))
    for nd in nodes:                          # replay visits exactly as the original search did
        if nd.parent_index is not None:
            backpropagate_visit(nodes, nd)
    return nodes


def search(
    problem: Problem,
    initial_solution: Solution,
    initial_score: float,
    generate_fn,
    execute_fn,
    num_iterations: int,
    c_puct: float = 1.0,
    on_node: OnNode = None,
    initial_nodes: Optional[List[Node]] = None,
) -> Tuple[Solution, float]:
    """Sequential FUTS, identical to futs.search plus an `on_node` callback.

    `initial_nodes` (from `nodes_from_records`) warm-starts the population to CONTINUE a
    finished search; `num_iterations` is then how many MORE nodes to add. When given, the
    root/initial solution+score are ignored and on_node fires only for newly added nodes.
    """
    if initial_nodes is None:
        nodes = [Node(index=0, parent_index=None,
                      solution=initial_solution, score=initial_score)]
        if on_node:
            on_node(nodes[0])
    else:
        nodes = list(initial_nodes)

    for _ in range(num_iterations):
        compute_rank_scores(nodes)
        compute_pucts(nodes, c_puct)
        best = max(nodes, key=lambda n: n.puct)

        new_solution = generate_fn(problem, best.solution, best.score)
        new_score = execute_fn(problem, new_solution)
        node = Node(index=len(nodes), parent_index=best.index,
                    solution=new_solution, score=new_score)
        nodes.append(node)
        backpropagate_visit(nodes, node)
        if on_node:
            on_node(node)

    best = max(nodes, key=lambda n: n.score)
    return best.solution, best.score


def _select_parents(nodes: List[Node], c_puct: float, k: int) -> List[Node]:
    """Pick k parents by PUCT using virtual visits, so a round spreads across
    distinct lineages instead of picking the same max k times. Visits added here
    are virtual (selection-only) and restored before real children backpropagate."""
    compute_rank_scores(nodes)
    picked: List[Node] = []
    bumped: List[Node] = []
    for _ in range(k):
        compute_pucts(nodes, c_puct)
        best = max(nodes, key=lambda n: n.puct)
        picked.append(best)
        best.num_visits += 1          # virtual visit
        bumped.append(best)
    for n in bumped:                  # undo virtual visits
        n.num_visits -= 1
    return picked


def batched_search(
    problem: Problem,
    initial_solution: Solution,
    initial_score: float,
    generate_fn,
    execute_batch_fn,
    num_iterations: int,
    batch_size: int = 4,
    c_puct: float = 1.0,
    on_node: OnNode = None,
    gen_workers: int = 0,
    initial_nodes: Optional[List[Node]] = None,
) -> Tuple[Solution, float]:
    """FUTS with B candidates generated+executed per round.

    `execute_batch_fn(problem, [Solution, ...]) -> [float, ...]` runs the batch
    concurrently (e.g. B independent SLURM jobs). Generation runs the B LLM calls
    concurrently too (up to `gen_workers`, default = batch_size): the B parents are
    selected BEFORE generation and each call is independent, so concurrent generation is
    semantically identical to the sequential version — only wall-clock changes (PUCT
    selection / rank / backprop math is unchanged). `gen_workers=1` => sequential.

    `initial_nodes` (from `nodes_from_records`) warm-starts the population to CONTINUE a
    finished search; `num_iterations` is then how many MORE nodes to add. When given, the
    root/initial solution+score are ignored and on_node fires only for newly added nodes.
    """
    if initial_nodes is None:
        nodes = [Node(index=0, parent_index=None,
                      solution=initial_solution, score=initial_score)]
        if on_node:
            on_node(nodes[0])
    else:
        nodes = list(initial_nodes)

    done = 0
    while done < num_iterations:
        b = min(batch_size, num_iterations - done)
        parents = _select_parents(nodes, c_puct, b)
        workers = max(1, gen_workers or b)
        if workers == 1 or b == 1:
            new_solutions = [generate_fn(problem, p.solution, p.score) for p in parents]
        else:
            with ThreadPoolExecutor(max_workers=min(workers, b)) as pool:
                new_solutions = list(pool.map(
                    lambda p: generate_fn(problem, p.solution, p.score), parents))
        # End-of-round cleanup (no search-semantics effect): reap any lingering
        # generator CLI subprocesses so they don't hold the login-node pids/fork
        # budget during the long GPU execution wait. Mock generate_fns lack reap().
        reap = getattr(generate_fn, "reap", None)
        if reap:
            reap()
        scores = execute_batch_fn(problem, new_solutions)
        for parent, sol, score in zip(parents, new_solutions, scores):
            node = Node(index=len(nodes), parent_index=parent.index,
                        solution=sol, score=score)
            nodes.append(node)
            backpropagate_visit(nodes, node)
            if on_node:
                on_node(node)
        done += b

    best = max(nodes, key=lambda n: n.score)
    return best.solution, best.score
