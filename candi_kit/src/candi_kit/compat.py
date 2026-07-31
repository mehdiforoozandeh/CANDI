"""Gate A: prove the vendored model is a bit-exact COPY of the historical q19 recipe.

NEW FILE (no vendored source). It reconstructs the q19 model under `torch.manual_seed(0)` and checks it
against the recorded anchors and the four historical checkpoints:

  A1  seed 0 -> 3,103,194 params and state_dict sha1 fd0e9493ac92a15f
  A2  every historical checkpoint loads with strict=True, zero missing / unexpected keys
  A3  a fixed synthetic forward matches the stored golden to 0 ULP on CPU (both arms)

If A1-A3 fail, the construction order in `candi_kit.model` has drifted — see the FROZEN banner there.

CLI: python -m candi_kit.compat [--ckpt-dir <repo>/sandbox/diagnostics/dual_conditioning_real/results]
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path

import torch

from candi_kit.model import build_real_model, forward_full

Q19 = dict(embed_dim=32, dropout=0.1, n_transformer_layers=2, feat_per_assay=16,
           depth_center=25.1, num_assays=8, context_length=768, d_model=0, nhead=4)
Q19_NUM_PARAMS = 3_103_194
Q19_STATE_SHA1 = "fd0e9493ac92a15f"
XTRANSFORMERS_PIN = "2.11.23"

# name -> use_offset. The four recorded arms in dual_conditioning_real/results.
CKPT_ARMS = {"main_s0_perassay": True, "offoff_s0_perassay": False,
             "wd0_on_s0": True, "wd0_off_s0": False}

GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"


def build_compat_q19(use_offset: bool):
    torch.manual_seed(0)
    return build_real_model(use_offset=use_offset, **Q19)


def state_dict_sha1(model) -> str:
    """sha1 over sorted (key bytes + raw tensor bytes), truncated to 16 hex chars."""
    sd = model.state_dict()
    h = hashlib.sha1()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def synthetic_batch(B: int = 2, ctx: int = 768, num_assays: int = 8, seed: int = 0) -> dict:
    """The frozen Gate-A batch. Uses a local Generator so the global RNG is never touched."""
    g = torch.Generator().manual_seed(seed)
    readlens = torch.tensor([30.0, 36.0, 76.0, 100.0, 101.0])

    def meta(F, s):
        gm = torch.Generator().manual_seed(s)
        m = torch.zeros(B, 4, F)
        for c in range(F):
            m[:, 0, c] = 22.0 + 6.0 * torch.rand(B, generator=gm)
            m[:, 1, c] = float(min(c, num_assays))
            m[:, 2, c] = readlens[torch.randint(0, 5, (B,), generator=gm)]
            m[:, 3, c] = torch.randint(0, 2, (B,), generator=gm).float()
        return m

    x_data = (torch.rand(B, ctx, num_assays + 1, generator=g) * 50).round()
    x_dna = torch.zeros(B, ctx * 25, 4)
    x_dna.scatter_(2, torch.randint(0, 4, (B, ctx * 25), generator=g).unsqueeze(-1), 1.0)
    return dict(x_data=x_data, x_dna=x_dna,
                x_meta=meta(num_assays + 1, seed), y_meta=meta(num_assays, seed + 1))


def _forward_outputs(model) -> dict:
    # Thread count is LOAD-BEARING for a 0-ULP comparison: CPU reductions re-associate with the number
    # of intra-op threads, which moved eta by ~9e-7 between a 32-core login node and a 2-core SLURM
    # allocation. Pinning to 1 makes the golden reproducible on any machine. The goldens on disk were
    # generated from the ORIGINAL repo implementation under this same pin.
    torch.set_num_threads(1)
    model.eval()
    with torch.no_grad():
        return forward_full(model, synthetic_batch())


def _check_golden(name: str, out: dict) -> None:
    path = GOLDEN_DIR / f"{name}.pt"
    if not path.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
        print(f"  [golden] WROTE {path} (no prior golden — this run is now the anchor)")
        return
    ref = torch.load(path)
    for k in sorted(ref):
        if not torch.equal(ref[k], out[k]):
            raise AssertionError(f"golden mismatch for {name}.{k}: max|d| = "
                                 f"{(ref[k] - out[k]).abs().max().item():.3e}")
    print(f"  [golden] {name}: {len(ref)} tensors match to 0 ULP")


def verify(ckpt_paths: dict) -> None:
    """ckpt_paths: {name: (Path, use_offset)}. Runs A1 -> A3 and raises on the first failure."""
    xt = version("x-transformers")
    if xt != XTRANSFORMERS_PIN:
        raise AssertionError(f"x-transformers {xt} != pinned {XTRANSFORMERS_PIN}; the checkpoint keys "
                             f"and the per-module init order are version-dependent")
    print(f"[compat] torch {torch.__version__}, x-transformers {xt}")

    m = build_compat_q19(use_offset=True)
    n_params = sum(p.numel() for p in m.parameters())
    sha1 = state_dict_sha1(m)
    print(f"[A1] params={n_params} sha1={sha1}")
    if n_params != Q19_NUM_PARAMS or sha1 != Q19_STATE_SHA1:
        raise AssertionError(f"A1 FAILED: expected {Q19_NUM_PARAMS} params / sha1 {Q19_STATE_SHA1}. "
                             f"Construction order in candi_kit.model has drifted — read its banner.")

    print("[A3] seed-0 init forward")
    for arm in (True, False):
        _check_golden(f"init_{'on' if arm else 'off'}", _forward_outputs(build_compat_q19(arm)))

    for name, (path, use_offset) in sorted(ckpt_paths.items()):
        model = build_compat_q19(use_offset)
        missing, unexpected = model.load_state_dict(
            torch.load(path, map_location="cpu"), strict=True)
        if missing or unexpected:
            raise AssertionError(f"A2 FAILED for {name}: missing={list(missing)} "
                                 f"unexpected={list(unexpected)}")
        print(f"[A2] {name}: strict load OK (offset={'on' if use_offset else 'off'})")
        _check_golden(name, _forward_outputs(model))
    print("[compat] OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate A — q19 compat reproduction")
    ap.add_argument("--ckpt-dir", type=Path, default=None,
                    help="dual_conditioning_real/results; omit to run A1/A3 only")
    args = ap.parse_args()
    ckpts = {}
    if args.ckpt_dir is not None:
        for name, use_offset in CKPT_ARMS.items():
            p = args.ckpt_dir / f"{name}.ckpt"
            if p.exists():
                ckpts[name] = (p, use_offset)
        if not ckpts:
            raise SystemExit(f"no q19 checkpoints found under {args.ckpt_dir}")
    verify(ckpts)


if __name__ == "__main__":
    main()
