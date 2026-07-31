"""Train + eval driver for q19 — dual conditioning on REAL CANDI sandbox data (counts-only).

One invocation = one arm. Trains the golden-reference model (real 4-row metadata, per-assay `per_conv`
encoder + `RealDualCondDecoder` depth-offset NB head, COUNTS ONLY) on `T_*` chr19 with per-assay
independent DSF ON + cloze masking, then evaluates M1/M2/M3 (rebuilt on real batches, `metrics_real.py`)
on chr21 and writes `results/{tag}.json` for the report.

Reuses the sandbox harness verbatim: `SandboxH5Dataset` (data + per-assay DSF + V/B ground truth),
`make_masker`/`prepare_masked_batch` (cloze masking; control appended, never masked), and the testbed
NB primitives. The count loss is the masked NB NLL split into obs (unmasked) + imp (cloze) branches.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from sandbox import SANDBOX_ASSAYS
from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.data import SandboxH5Dataset, build_canonical_meta
from sandbox.diagnostics.dual_conditioning_real import metrics_real as MR
from sandbox.diagnostics.dual_conditioning_real.model_real import build_real_model, forward_full

H5_DEFAULT = "sandbox/data/sandbox.h5"
EIC_META = "data/eic_metadata.csv"
OUTDIR = "sandbox/diagnostics/dual_conditioning_real/results"


# ---------------------------------------------------------------------------
# loss + schedule
# ---------------------------------------------------------------------------

def _elem_nb_nll(p, n, target, eps: float = 1e-6):
    probs = (1.0 - p).clamp(eps, 1.0 - eps)
    total = n.clamp_min(eps)
    dist = torch.distributions.NegativeBinomial(total_count=total, probs=probs)
    return -dist.log_prob(target.clamp_min(0.0))       # [B, L, A]


def nb_count_loss(out, prep):
    """Masked NB NLL over available assays, split obs (unmasked) + imp (cloze) — counts only."""
    elem = _elem_nb_nll(out["p"], out["n"], prep["y_data"])
    obs, msk = prep["observed_map"], prep["masked_map"]
    lo = elem[obs].mean() if bool(obs.any()) else out["p"].sum() * 0.0
    li = elem[msk].mean() if bool(msk.any()) else out["p"].sum() * 0.0
    return lo + li, dict(obs=float(lo), imp=float(li))


def cosine_warmup(opt, total_steps, warmup_frac=0.1, min_ratio=0.1):
    warm = max(1, int(warmup_frac * total_steps))

    def fn(s):
        if s < warm:
            return s / warm
        t = (s - warm) / max(1, total_steps - warm)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def _t_biosamples(h5_path) -> list:
    import json
    import h5py
    with h5py.File(h5_path, "r") as h:
        order = json.loads(h["biosamples"].attrs["order"])
    return [b for b in order if b.startswith("T_")]


def _train_step(model, prep, opt, sched, losses):
    out = forward_full(model, prep)
    loss, terms = nb_count_loss(out, prep)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    losses.append(float(loss))
    return terms


def train(model, h5_path, device, *, regime="type1_chr19", epochs=25, steps_per_epoch=200,
          batch_size=8, lr=5e-4, weight_decay=1e-4, seed=0, dsf_sampling="uniform",
          log_every=0, full_coverage=False) -> list:
    masker = make_masker(p_full_assay=1.0)          # whole-assay cloze -> imputation supervision
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    losses, step = [], 0

    if full_coverage:
        # DETERMINISTIC full coverage: every epoch iterates ALL chr19 windows for ALL T_ biosamples
        # (no random-biosample sampling). One per-biosample dataset each, sharing ONE RAM buffer, pulled
        # round-robin so biosamples interleave within the epoch.
        bios = _t_biosamples(h5_path)
        shared = Path(h5_path).read_bytes()
        datasets = []
        for b in bios:
            ds = SandboxH5Dataset(h5_path, regime, train=True, batch_size=batch_size,
                                  biosample_prefix=b, dsf_sampling=dsf_sampling, seed=seed,
                                  shuffle=True, h5_cache_ram=False)
            ds._ram_buf = shared                     # share the single 1.6 GB buffer (no duplication)
            datasets.append(ds)
        per_epoch = sum(d.estimate_steps_per_epoch() for d in datasets)
        sched = cosine_warmup(opt, max(1, epochs * per_epoch))
        print(f"[train] full-coverage: {len(bios)} T_ biosamples x all chr19 windows "
              f"= ~{per_epoch} batches/epoch x {epochs} epochs", flush=True)
        for ep in range(epochs):
            model.train()
            iters = []
            for ds in datasets:
                ds.seed = seed + ep                  # per-epoch reshuffle (per biosample)
                iters.append(iter(ds))
            active = list(range(len(iters)))
            while active:
                for i in list(active):
                    try:
                        batch = next(iters[i])
                    except StopIteration:
                        active.remove(i)
                        continue
                    prep = prepare_masked_batch(batch, masker, device, apply_mask=True)
                    if prep is None:
                        continue
                    terms = _train_step(model, prep, opt, sched, losses)
                    step += 1
                    if log_every and step % log_every == 0:
                        print(f"  ep{ep} step{step} nll={losses[-1]:.3f} (obs={terms['obs']:.3f} "
                              f"imp={terms['imp']:.3f}) lr={sched.get_last_lr()[0]:.2e}", flush=True)
        return losses

    # sampled path: random T_ biosample per batch, steps_per_epoch batches/epoch
    ds = SandboxH5Dataset(h5_path, regime, train=True, batch_size=batch_size, biosample_prefix="T_",
                          dsf_sampling=dsf_sampling, seed=seed, shuffle=True)
    sched = cosine_warmup(opt, epochs * steps_per_epoch)
    for ep in range(epochs):
        model.train()
        ds.seed = seed + ep                          # per-epoch variety (RAM buffer persists)
        it = iter(ds)
        for _ in range(steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(ds)
                batch = next(it)
            prep = prepare_masked_batch(batch, masker, device, apply_mask=True)
            if prep is None:
                continue
            terms = _train_step(model, prep, opt, sched, losses)
            step += 1
            if log_every and step % log_every == 0:
                print(f"  ep{ep} step{step} nll={losses[-1]:.3f} (obs={terms['obs']:.3f} "
                      f"imp={terms['imp']:.3f}) lr={sched.get_last_lr()[0]:.2e}", flush=True)
    return losses


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, h5_path, device, *, regime="type1_chr19", batch_size=4, max_batches=None,
             fg_frac=0.02, n_boot=1000, use_canonical=False, seed=0,
             eval_budget=200_000, m3_regions=8) -> dict:
    canonical_meta = None
    if Path(EIC_META).exists():
        canonical_meta = build_canonical_meta(EIC_META, list(SANDBOX_ASSAYS))
    units = MR.build_eval_units(model, h5_path, device, regime=regime, batch_size=batch_size,
                                max_batches=max_batches, canonical_meta=canonical_meta,
                                use_canonical=use_canonical, seed=seed)
    return dict(
        n_units=len(units),
        M1=MR.eval_M1(model, units, device, budget=eval_budget, seed=seed),
        M2=MR.eval_M2(model, units, device, fg_frac=fg_frac, n_boot=n_boot, seed=seed),
        M3=MR.eval_M3(model, h5_path, device, n_regions=m3_regions, seed=seed),
    )


# ---------------------------------------------------------------------------
# json + main
# ---------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, dict):
        return {(f"{k[0]}|{k[1]}|{k[2]}" if isinstance(k, tuple) else str(k)): _jsonable(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def train_and_eval(*, h5_path=H5_DEFAULT, regime="type1_chr19", epochs=25, steps_per_epoch=200,
                   batch_size=8, lr=5e-4, weight_decay=1e-4, use_offset=True, dsf_sampling="uniform",
                   device="cpu", seed=0, eval_batch_size=4, eval_max_batches=None, fg_frac=0.02,
                   n_boot=1000, use_canonical=False, log_every=0, full_coverage=False,
                   eval_budget=200_000, m3_regions=8, ckpt_path=None) -> dict:
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    model = build_real_model(use_offset=use_offset).to(device)
    losses = train(model, h5_path, device, regime=regime, epochs=epochs, steps_per_epoch=steps_per_epoch,
                   batch_size=batch_size, lr=lr, weight_decay=weight_decay, seed=seed,
                   dsf_sampling=dsf_sampling, log_every=log_every, full_coverage=full_coverage)
    model.eval()
    if ckpt_path:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
    ev = evaluate(model, h5_path, device, regime=regime, batch_size=eval_batch_size,
                  max_batches=eval_max_batches, fg_frac=fg_frac, n_boot=n_boot,
                  use_canonical=use_canonical, seed=seed, eval_budget=eval_budget, m3_regions=m3_regions)
    return dict(config=dict(regime=regime, epochs=epochs, steps_per_epoch=steps_per_epoch,
                            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                            use_offset=use_offset, dsf_sampling=dsf_sampling, seed=seed,
                            fg_frac=fg_frac, n_boot=n_boot, use_canonical=use_canonical,
                            full_coverage=full_coverage, eval_max_batches=eval_max_batches,
                            eval_budget=eval_budget, m3_regions=m3_regions),
                train_losses=losses, **ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=H5_DEFAULT)
    ap.add_argument("--regime", default="type1_chr19", choices=["type1_chr19", "type2_loci"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--offset", default="on", choices=["on", "off"])
    ap.add_argument("--dsf-sampling", default="uniform", choices=["uniform", "off", "x_eq_y", "upsample_only"])
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--eval-max-batches", type=int, default=0)
    ap.add_argument("--fg-frac", type=float, default=0.02)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--use-canonical", action="store_true", help="canonical EIC medians for missing meta (control)")
    ap.add_argument("--full-coverage", action="store_true",
                    help="deterministic full coverage: every epoch = all chr19 windows x all T_ biosamples")
    ap.add_argument("--eval-budget", type=int, default=200_000,
                    help="max eval points for M1 corr/CRPS (set very high for no subsampling)")
    ap.add_argument("--m3-regions", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = a.tag or f"real_off-{a.offset}_dsf-{a.dsf_sampling}_ep{a.epochs}_seed{a.seed}"
    print(f"[run] tag={tag} device={device} full_coverage={a.full_coverage} "
          f"eval_max_batches={a.eval_max_batches or 'ALL'}", flush=True)
    t0 = time.time()
    res = train_and_eval(
        h5_path=a.h5, regime=a.regime, epochs=a.epochs, steps_per_epoch=a.steps_per_epoch,
        batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay, use_offset=(a.offset == "on"),
        dsf_sampling=a.dsf_sampling, device=device, seed=a.seed, eval_batch_size=a.eval_batch_size,
        eval_max_batches=(a.eval_max_batches or None), fg_frac=a.fg_frac, n_boot=a.n_boot,
        use_canonical=a.use_canonical, log_every=200, full_coverage=a.full_coverage,
        eval_budget=a.eval_budget, m3_regions=a.m3_regions, ckpt_path=f"{OUTDIR}/{tag}.ckpt")
    res["config"]["tag"] = tag
    res["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(f"{OUTDIR}/{tag}.json", "w") as f:
        json.dump(_jsonable(res), f, indent=2)
    m1, m2 = res["M1"], res["M2"]
    print(f"[{tag}] imp_spear={m1['imp'].get('spearman', float('nan')):.3f} "
          f"den_spear={m1['den'].get('spearman', float('nan')):.3f} "
          f"eff_rank={m1['encoder_eff_rank']:.2f} health={m1['health_gate_den_ge_imp']} | "
          f"M2 depth_min@true={m2['depth']['frac_min_at_true']:.2f} "
          f"eta_slope={m2['depth']['median_eta_slope']:.3f} "
          f"runtype_dir={m2['run_type']['frac_direction']:.2f} | "
          f"M3 ratio={res['M3']['ratio']:.3f} wall={res['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
