#!/bin/bash
# candi_kit — THE GATE. Fail-fast pre-flight, propagates the first failing tier's exit code:
#   1 pytest  ->  2 compat-q19 (the bit-exact vendoring proof)  ->  3 CPU smoke  ->  4 GPU smoke.
# Green is the precondition for submitting slurm/train.sh; red means STOP (a red tier 2 is almost
# always constructor-RNG drift in model.py — see .BUILD_PLAN RISKS).
# Adapted from sandbox/diagnostics/dual_conditioning_real/jobs/gate.sh.
#   sbatch candi_kit/slurm/gate.sh
# Logs go to ./slurm-logs/ RELATIVE TO WHERE YOU SUBMIT FROM. SLURM resolves --output against
# the submitting cwd, not the script's location, so `mkdir -p slurm-logs` first (or pass
# `sbatch --output=... --error=...` to override). Submitting from anywhere is then fine.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_kit_gate
#SBATCH --output=slurm-logs/gate_%j.out
#SBATCH --error=slurm-logs/gate_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -uo pipefail

KIT="${KIT:-/project/6014832/mforooz/EpiDenoise/candi_kit}"
# VENV: defaults to the environment you are ALREADY in ($VIRTUAL_ENV), which is what README §4.0
# leaves you in. Override with VENV=/path/to/venv. It must NOT default to someone else's venv --
# sourcing a path you do not own fails late, inside python, not at the source line.
VENV="${VENV:-${VIRTUAL_ENV:-}}"
# Historical checkpoints are NOT part of this kit and you almost certainly do not have them.
# Left empty by default: compat then verifies the parts that do not need them (parameter count,
# state_dict hash, golden forward outputs), which is what actually proves the port is faithful.
CKPT_DIR="${CKPT_DIR:-}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export WANDB_MODE=disabled
if [ -n "$VENV" ]; then
  source "$VENV/bin/activate"
else
  echo "[error] no environment: set VENV=/path/to/venv, or sbatch from an active venv" >&2; exit 1
fi

cd "$KIT"
echo "[gate] host=$(hostname)"; nvidia-smi -L || true

python -m pytest tests/ -q
rc=$?; if [ $rc -ne 0 ]; then echo "[gate] TIER 1 pytest FAILED (rc=$rc)"; exit $rc; fi

if [ -n "$CKPT_DIR" ]; then
  python -m candi_kit.compat --ckpt-dir "$CKPT_DIR"
else
  python -m candi_kit.compat
fi
rc=$?; if [ $rc -ne 0 ]; then echo "[gate] TIER 2 compat-q19 FAILED (rc=$rc)"; exit $rc; fi

# Tiers 3-4: build the shipped model and run a forward+backward, first on CPU then on the GPU.
# Synthetic tensors only, so the gate needs no baked h5 and no data root.
SMOKE='
import sys, torch
from candi_kit.model import build_real_model, forward_full, nb_nll
dev = sys.argv[1]
A, L, B = 8, 768, 2
torch.manual_seed(0)
m = build_real_model(num_assays=A, context_length=L).to(dev).train()
meta = torch.zeros(B, 4, A + 1, device=dev)
meta[:, 0, :] = 24.0
meta[:, 1, :] = torch.arange(A + 1, dtype=torch.float32, device=dev)
meta[:, 2, :] = 36.0
x = (torch.rand(B, L, A + 1, device=dev) * 50).round()
dna = torch.zeros(B, L * 25, 4, device=dev); dna[:, :, 0] = 1.0
out = forward_full(m, dict(x_data=x, x_dna=dna, x_meta=meta, y_meta=meta[:, :, :A]))
loss = nb_nll(out["p"], out["n"], (torch.rand(B, L, A, device=dev) * 50).round(),
              torch.ones(B, A, device=dev))
assert torch.isfinite(loss), "non-finite loss"
loss.backward()
assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in m.parameters()), "no grad"
print(f"[smoke:{dev}] ok loss={loss.item():.4f} d_model={m.encoder.d_model}")
'

python -c "$SMOKE" cpu
rc=$?; if [ $rc -ne 0 ]; then echo "[gate] TIER 3 CPU SMOKE FAILED (rc=$rc)"; exit $rc; fi

python -c "$SMOKE" cuda
rc=$?; if [ $rc -ne 0 ]; then echo "[gate] TIER 4 GPU SMOKE FAILED (rc=$rc)"; exit $rc; fi

echo "[gate] ALL TIERS GREEN"
exit 0
