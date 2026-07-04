"""Determinism launcher (NOT loop-editable; lives in _harness/).

sbatch_wrap.sh runs the scored candidate through THIS, so cuDNN-determinism is forced BEFORE the
candidate imports torch. A loop editing its train.py therefore cannot reintroduce the σ≈0.05
run-to-run nondeterminism we measured (it would only affect its own redundant in-file flags).

  python _det_launch.py <candidate_train.py>
"""
import os
import runpy
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before cuBLAS init (det. matmul)
import torch  # noqa: E402
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

runpy.run_path(sys.argv[1], run_name="__main__")              # run the candidate as __main__
