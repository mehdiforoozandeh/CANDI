"""june3 eval shim — never edit sandbox/train.py; forward only supported kwargs."""
from __future__ import annotations

import inspect
from typing import Any, Dict

from sandbox.train import run_eval_pass as _run_eval_pass


def run_eval_pass(*args: Any, **kwargs: Any) -> Dict[str, float]:
    sig = inspect.signature(_run_eval_pass)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return _run_eval_pass(*args, **filtered)
