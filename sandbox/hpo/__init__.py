"""HPO graph log: persistent, append-only graph of all sandbox runs.

Each node represents a completed run (config + key results). Each edge records
parent → child derivation (i.e. which knobs changed). The graph is stored as a
single JSON file (`sandbox/hpo_graph.json` by default) so it is easy to diff,
version-control, and embed in a publication.

Public API:
- :func:`update_graph_for_run` — invoked by `sandbox.train` at the end of a run.
- :func:`load_graph` / :func:`save_graph` — primitives used by the viewer script.

See ``sandbox/hpo/axes.py`` for the curated allowlist of "consequential" config
leaves recorded on every node (so the graph stays interpretable as it grows).
"""
from sandbox.hpo.graph import (
    GRAPH_SCHEMA_VERSION,
    diff_axes,
    load_graph,
    save_graph,
    update_graph_for_run,
)

__all__ = [
    "GRAPH_SCHEMA_VERSION",
    "diff_axes",
    "load_graph",
    "save_graph",
    "update_graph_for_run",
]
