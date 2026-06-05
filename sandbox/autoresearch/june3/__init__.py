"""june3 package init — restores old run_eval_pass key format for frozen prepare.py.

The production sandbox.train.run_eval_pass changed its return format after the
harness was frozen: keys are now prefixed with "eval_metrics/" and "eval_losses/".
The frozen prepare.py expects unprefixed keys (imp_count_r2_gw, count_imp_loss, …).

This shim runs before eval_bridge.py imports sandbox.train.run_eval_pass, so the
patched (unprefixed) version is what eval_bridge._run_eval_pass binds to.
"""
from __future__ import annotations

import sandbox.train as _sandbox_train

_orig = _sandbox_train.run_eval_pass


def _compat_run_eval_pass(*args, **kwargs):
    result = _orig(*args, **kwargs)
    out = {}
    for k, v in result.items():
        if k.startswith("eval_metrics/"):
            out[k[len("eval_metrics/"):]] = v
        elif k.startswith("eval_losses/"):
            out[k[len("eval_losses/"):]] = v
        else:
            out[k] = v
    return out


_sandbox_train.run_eval_pass = _compat_run_eval_pass
