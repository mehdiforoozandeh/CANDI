"""Train + eval driver for the CANDI dual-conditioning recipe on real data (counts-only).

One invocation = one arm. Trains the golden-reference model (real 4-row metadata, per-assay `per_conv`
encoder + depth-offset NB head, COUNTS ONLY) on the h5's train chromosomes with per-assay independent
DSF ON + cloze masking, then evaluates M1/M2/M3 on the eval chromosomes and writes
`{out_dir}/{tag}.json` + `{out_dir}/{tag}.ckpt`.

Provenance: COPY+EDIT of
EpiDenoise/sandbox/diagnostics/dual_conditioning_real/run_real.py:1-286.

FROZEN: the optimizer/schedule/clip/loss/step order is load-bearing. Adam (coupled L2, NOT AdamW,
positional lr), grad-clip 1.0, cosine warmup_frac 0.1 / min_ratio 0.1, loss = mean NB NLL over unmasked
+ mean over cloze (unweighted sum), and the cudnn.deterministic / cudnn.benchmark pair with NO
`torch.use_deterministic_algorithms` (it crashes the candi_v2 encoder). Scale (num_assays,
context_length, assay order, DSF ladder, chromosomes) comes from the h5 attrs and is NEVER a flag.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from candi_kit.batch import make_masker, prepare_masked_batch
from candi_kit.eval import evaluate
from candi_kit.dataset import CandiKitH5Dataset, h5_depth_center
from candi_kit.model import build_real_model, forward_full


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


def train(model, h5_path, device, *, regime="type1", epochs=25, steps_per_epoch=200,
          batch_size=8, lr=5e-4, weight_decay=0.0, seed=0, dsf_sampling="uniform",
          log_every=0, full_coverage=False, p_full_assay=1.0, mask_fraction=0.2) -> list:
    if p_full_assay == 1.0 and mask_fraction != 0.2:
        print("[train] WARNING: --mask-fraction is INERT under --p-full-assay 1.0 "
              "(DataMasker._mask_full_assay never reads it)", flush=True)
    masker = make_masker(p_full_assay=p_full_assay, mask_fraction=mask_fraction,
                         p_full_loci=0.0, p_chunks=0.0)   # whole-assay cloze -> imputation supervision
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    losses, step = [], 0

    if full_coverage:
        # DETERMINISTIC full coverage: every epoch iterates ALL train windows for ALL T_ biosamples
        # (no random-biosample sampling). One per-biosample dataset each, sharing ONE RAM buffer, pulled
        # round-robin so biosamples interleave within the epoch.
        bios = _t_biosamples(h5_path)
        shared = Path(h5_path).read_bytes()
        datasets = []
        for b in bios:
            ds = CandiKitH5Dataset(h5_path, regime, train=True, batch_size=batch_size,
                                   biosample_prefix=b, dsf_sampling=dsf_sampling, seed=seed,
                                   shuffle=True, h5_cache_ram=False)
            ds._ram_buf = shared                     # share the single 1.6 GB buffer (no duplication)
            datasets.append(ds)
        per_epoch = sum(d.estimate_steps_per_epoch() for d in datasets)
        sched = cosine_warmup(opt, max(1, epochs * per_epoch))
        print(f"[train] full-coverage: {len(bios)} T_ biosamples x all train windows "
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
    ds = CandiKitH5Dataset(h5_path, regime, train=True, batch_size=batch_size, biosample_prefix="T_",
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
# `evaluate` is imported from candi_kit.eval, which owns it (plan edit 11 wires _dsf_counterfactual
# into that one top-level entry point). It derives the assay order from build_eval_units and the eval
# region from the h5's own `eval_chroms` attr, so neither is threaded through here.


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


def _pkg_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _g(d, *keys, default=float("nan")):
    for k in keys:
        if k in d:
            return d[k]
    return default


def train_and_eval(*, h5_path, out_dir, regime="type1", epochs=25, steps_per_epoch=200, batch_size=8,
                   lr=5e-4, weight_decay=0.0, use_offset=True, dsf_sampling="uniform", device="cpu",
                   seed=0, embed_dim=32, dropout=0.1, n_transformer_layers=2, feat_per_assay=16,
                   depth_center=None, d_model=0, nhead=4, p_full_assay=1.0, mask_fraction=0.2,
                   eval_batch_size=4, eval_max_batches=None, fg_frac=0.02, n_boot=1000,
                   eval_budget=200_000, m3_regions=8, include_deprecated=False, log_every=0,
                   full_coverage=False, compat_q19=False, ckpt_path=None) -> dict:
    # Scale is read from the h5, never from a flag. `h5_cache_ram=False` so this probe does not
    # duplicate the shared RAM buffer that the full-coverage loop allocates.
    ds = CandiKitH5Dataset(h5_path, regime, train=True, batch_size=batch_size,
                           dsf_sampling=dsf_sampling, seed=seed, h5_cache_ram=False)
    if compat_q19 and (ds.num_assays != 8 or ds.context_bins != 768):
        raise ValueError(f"--compat-q19 requires an 8-assay / 768-bin h5; got num_assays="
                         f"{ds.num_assays}, context_bins={ds.context_bins}")
    if depth_center is None:
        depth_center = h5_depth_center(h5_path)
        print(f"[train] depth_center derived from h5 (median T_ meta_dsf1[0]) = {depth_center:.4f} "
              "— override with --depth-center", flush=True)

    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    model = build_real_model(embed_dim=embed_dim, dropout=dropout,
                             n_transformer_layers=n_transformer_layers, feat_per_assay=feat_per_assay,
                             depth_center=depth_center, use_offset=use_offset,
                             num_assays=ds.num_assays, context_length=ds.context_bins,
                             d_model=d_model, nhead=nhead).to(device)
    losses = train(model, h5_path, device, regime=regime, epochs=epochs, steps_per_epoch=steps_per_epoch,
                   batch_size=batch_size, lr=lr, weight_decay=weight_decay, seed=seed,
                   dsf_sampling=dsf_sampling, log_every=log_every, full_coverage=full_coverage,
                   p_full_assay=p_full_assay, mask_fraction=mask_fraction)
    model.eval()
    if ckpt_path:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
    ev = evaluate(model, h5_path, device, regime=regime,
                  batch_size=eval_batch_size, max_batches=eval_max_batches, fg_frac=fg_frac,
                  n_boot=n_boot, seed=seed, eval_budget=eval_budget, m3_regions=m3_regions,
                  include_deprecated=include_deprecated)
    return dict(config=dict(regime=regime, epochs=epochs, steps_per_epoch=steps_per_epoch,
                            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                            use_offset=use_offset, dsf_sampling=dsf_sampling, seed=seed,
                            fg_frac=fg_frac, n_boot=n_boot, full_coverage=full_coverage,
                            eval_max_batches=eval_max_batches, eval_budget=eval_budget,
                            m3_regions=m3_regions, p_full_assay=p_full_assay,
                            mask_fraction=mask_fraction, embed_dim=embed_dim, dropout=dropout,
                            n_transformer_layers=n_transformer_layers, feat_per_assay=feat_per_assay,
                            include_deprecated=include_deprecated, compat_q19=compat_q19,
                            out_dir=str(out_dir),
                            assays=list(ds.assays), num_assays=ds.num_assays,
                            context_bins=ds.context_bins, resolution=ds.resolution,
                            dsf_list=list(ds.dsf_list), train_chroms=list(ds.train_chroms),
                            eval_chroms=list(ds.eval_chroms), d_model=int(model.encoder.d_model),
                            nhead=nhead, depth_center=float(depth_center),
                            kit_version=_pkg_version("candi_kit"), torch_version=torch.__version__,
                            x_transformers_version=_pkg_version("x-transformers")),
                train_losses=losses, **ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--regime", default="type1", choices=["type1", "type2_loci"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--offset", "--arm", dest="offset", default="on",
                    choices=["on", "off", "offset_on", "offset_off"])
    ap.add_argument("--dsf-sampling", default="uniform", choices=["uniform", "off", "x_eq_y", "upsample_only"])
    ap.add_argument("--p-full-assay", type=float, default=1.0)
    ap.add_argument("--mask-fraction", type=float, default=0.2, help="INERT under --p-full-assay 1.0")
    ap.add_argument("--depth-center", type=float, default=None,
                    help="default: median of finite meta_dsf1[0] over T_ biosamples (printed)")
    ap.add_argument("--d-model", type=int, default=0,
                    help="0 = auto = (num_assays+1)*expansion**n_cnn_layers; SET IT when num_assays != 8")
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--n-transformer-layers", type=int, default=2)
    ap.add_argument("--feat-per-assay", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--eval-max-batches", type=int, default=0)
    ap.add_argument("--fg-frac", type=float, default=0.02)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--full-coverage", action="store_true",
                    help="deterministic full coverage: every epoch = all train windows x all T_ biosamples")
    ap.add_argument("--eval-budget", type=int, default=200_000,
                    help="max eval points for M1 corr/CRPS (set very high for no subsampling)")
    ap.add_argument("--m3-regions", type=int, default=8)
    ap.add_argument("--include-deprecated", action="store_true",
                    help="also emit the deprecated metric keys, each with its verdict string attached")
    ap.add_argument("--compat-q19", action="store_true",
                    help="pin the q19 architecture knobs; requires an 8-assay / 768-bin h5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    if a.compat_q19:
        a.embed_dim, a.dropout, a.n_transformer_layers, a.feat_per_assay = 32, 0.1, 2, 16
        a.depth_center, a.d_model, a.nhead = 25.1, 0, 4
        print("[run] --compat-q19: embed_dim=32 dropout=0.1 n_transformer_layers=2 feat_per_assay=16 "
              "depth_center=25.1 d_model=0 nhead=4", flush=True)
    use_offset = a.offset in ("on", "offset_on")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = a.tag or f"real_off-{a.offset}_dsf-{a.dsf_sampling}_ep{a.epochs}_seed{a.seed}"
    print(f"[run] tag={tag} device={device} full_coverage={a.full_coverage} "
          f"eval_max_batches={a.eval_max_batches or 'ALL'}", flush=True)
    t0 = time.time()
    res = train_and_eval(
        h5_path=a.h5, out_dir=out_dir, regime=a.regime, epochs=a.epochs,
        steps_per_epoch=a.steps_per_epoch, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, use_offset=use_offset, dsf_sampling=a.dsf_sampling, device=device,
        seed=a.seed, embed_dim=a.embed_dim, dropout=a.dropout,
        n_transformer_layers=a.n_transformer_layers, feat_per_assay=a.feat_per_assay,
        depth_center=a.depth_center, d_model=a.d_model, nhead=a.nhead, p_full_assay=a.p_full_assay,
        mask_fraction=a.mask_fraction, eval_batch_size=a.eval_batch_size,
        eval_max_batches=(a.eval_max_batches or None), fg_frac=a.fg_frac, n_boot=a.n_boot,
        eval_budget=a.eval_budget, m3_regions=a.m3_regions, include_deprecated=a.include_deprecated,
        log_every=200, full_coverage=a.full_coverage, compat_q19=a.compat_q19,
        ckpt_path=str(out_dir / f"{tag}.ckpt"))
    res["config"]["tag"] = tag
    res["wall_s"] = round(time.time() - t0, 1)
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(_jsonable(res), f, indent=2)
    m1, m2 = res["M1"], res["M2"]
    print(f"[{tag}] imp_spear={_g(m1['imp'], 'spearman_raw', 'spearman'):.3f} "
          f"den_spear={_g(m1['den'], 'spearman_raw', 'spearman'):.3f} "
          f"eff_rank={_g(m1, 'encoder_eff_rank_perpos', 'encoder_eff_rank'):.2f} | "
          f"M2 total_slope={_g(m2['depth'], 'median_total_slope'):.3f} | "
          f"M3 ratio={res['M3']['ratio']:.3f} wall={res['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
