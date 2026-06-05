"""JEPA Stage 2 decoder trainer.

Loads a JEPA encoder+predictor checkpoint and trains CANDI-style decoder heads
on top of the predictor output.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from _utils import reverse_complement_dna, reverse_signal
from candi_loss import CANDI_LOSS
from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.config import deep_merge, dump_yaml, load_yaml
from sandbox.config_types import AdamHParams, AdamWHParams, AdamaxHParams, config_from_dict
from sandbox.data import SandboxH5Dataset, build_canonical_meta
from sandbox.eval import (
    prompt_sensitivity_depth_count_ratio,
    prompt_sensitivity_readlen_mse,
    prompt_sensitivity_runtype_mse,
)
from sandbox.hpo import update_graph_for_run
from sandbox.jepa_config import JEPADecoderConfig
from sandbox.jepa_decoder import JEPADecoderModel
from sandbox.losses import SandboxCompositeLoss
from sandbox.train import _split_eval_families, run_eval_pass
from sandbox import SANDBOX_ASSAYS


_JEPA_DEFAULT_YAML = Path(__file__).resolve().parent / "configs" / "jepa_default.yaml"
_DECODER_DEFAULT_YAML = Path(__file__).resolve().parent / "configs" / "decoder_training.yaml"
_BRANCH_RAW_TO_LOG = {
    "count_obs_raw": "count_obs",
    "count_imp_raw": "count_imp",
    "pval_obs_raw": "pval_obs",
    "pval_imp_raw": "pval_imp",
    "peak_obs_raw": "peak_obs",
    "peak_imp_raw": "peak_imp",
}


def _dotted_overrides_from_argv(argv: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tok in argv:
        if "=" not in tok or tok.startswith("-"):
            continue
        k, v = tok.split("=", 1)
        parts = k.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        for conv in (int, float):
            try:
                v = conv(v)  # type: ignore[assignment]
                break
            except (ValueError, TypeError):
                pass
        if v == "true":
            v = True  # type: ignore[assignment]
        elif v == "false":
            v = False  # type: ignore[assignment]
        node[parts[-1]] = v
    return out


def _dist_type_for_model(s: str) -> str:
    return "studentst" if s == "student_t" else s


def _active_base_lr(cfg: JEPADecoderConfig) -> float:
    on = cfg.training.optimizer.name
    if on == "adam":
        return float(cfg.training.optimizer.adam.lr)
    if on == "adamw":
        return float(cfg.training.optimizer.adamw.lr)
    if on == "adamax":
        return float(cfg.training.optimizer.adamax.lr)
    if on == "sgd":
        return float(cfg.training.optimizer.sgd.lr)
    raise ValueError(on)


def _warn_inactive_optimizer_overrides(cfg: JEPADecoderConfig) -> None:
    active = cfg.training.optimizer.name
    defaults = {
        "adam": AdamHParams().lr,
        "adamw": AdamWHParams().lr,
        "adamax": AdamaxHParams().lr,
    }
    for name, default_lr in defaults.items():
        if name == active:
            continue
        actual_lr = getattr(cfg.training.optimizer, name).lr
        if actual_lr != default_lr:
            print(
                f"[WARN] training.optimizer.{name}.lr={actual_lr} set while active optimizer is {active}",
                file=sys.stderr,
            )


def _maybe_augment_reverse_complement(batch: Dict[str, Any], prob: float) -> Dict[str, Any]:
    if prob <= 0.0 or float(torch.rand(1).item()) >= float(prob):
        return batch
    out = dict(batch)
    out["x_dna"] = reverse_complement_dna(out["x_dna"].clone())
    for k in ("x_data", "y_data", "y_pval", "y_peaks", "control_data"):
        if k in out and isinstance(out[k], torch.Tensor):
            out[k] = reverse_signal(out[k].clone())
    return out


def _make_criterion(cfg: JEPADecoderConfig, *, eval_mode: bool = False) -> SandboxCompositeLoss:
    lw = cfg.training.loss_weights
    count_w = float(lw.count_weight)
    pval_w = float(lw.pval_weight)
    peak_w = float(lw.peak_weight)
    if not eval_mode:
        if cfg.decoder.heads == "count_only":
            pval_w = 0.0
            peak_w = 0.0
        elif cfg.decoder.heads == "pval_only":
            count_w = 0.0
            peak_w = 0.0
        elif cfg.decoder.heads == "peak_only":
            count_w = 0.0
            pval_w = 0.0
    cand = CANDI_LOSS(
        dist_type=_dist_type_for_model(str(cfg.decoder.signal_dist)),
        count_weight=count_w,
        pval_weight=pval_w,
        peak_weight=peak_w,
        obs_weight=1.0 if str(cfg.decoder.loss_mode) == "unified" and not eval_mode else float(lw.obs_weight),
        imp_weight=0.0 if str(cfg.decoder.loss_mode) == "unified" and not eval_mode else float(lw.imp_weight),
    )
    return SandboxCompositeLoss(cand)


def build_optimizer(model: JEPADecoderModel, cfg: JEPADecoderConfig) -> torch.optim.Optimizer:
    params = list(model.trainable_parameters())
    if not params:
        raise ValueError("no trainable parameters")
    name = cfg.training.optimizer.name
    if name == "adam":
        h = cfg.training.optimizer.adam
        return torch.optim.Adam(params, lr=h.lr, betas=(h.beta1, h.beta2), eps=h.eps, weight_decay=h.weight_decay, amsgrad=h.amsgrad)
    if name == "adamw":
        h = cfg.training.optimizer.adamw
        return torch.optim.AdamW(params, lr=h.lr, betas=(h.beta1, h.beta2), eps=h.eps, weight_decay=h.weight_decay, amsgrad=h.amsgrad)
    if name == "adamax":
        h = cfg.training.optimizer.adamax
        return torch.optim.Adamax(params, lr=h.lr, betas=(h.beta1, h.beta2), eps=h.eps, weight_decay=h.weight_decay)
    if name == "sgd":
        h = cfg.training.optimizer.sgd
        return torch.optim.SGD(params, lr=h.lr, momentum=h.momentum, dampening=h.dampening, weight_decay=h.weight_decay, nesterov=h.nesterov)
    raise ValueError(f"unknown optimizer {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: JEPADecoderConfig, *, total_steps: int):
    sch = cfg.training.schedule
    base_lr = _active_base_lr(cfg)
    eta_min = base_lr * float(sch.min_lr_ratio)
    if sch.name == "constant" or total_steps <= 0:
        return None
    if sch.name == "linear":
        return LinearLR(optimizer, start_factor=1.0, end_factor=float(sch.min_lr_ratio), total_iters=total_steps)
    if sch.name == "cosine":
        wf = float(sch.warmup_frac)
        if wf <= 0.0 or total_steps <= 1:
            return CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=eta_min)
        warmup_steps = min(max(1, int(round(total_steps * wf))), max(1, total_steps - 1))
        cosine_steps = max(1, total_steps - warmup_steps)
        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min),
            ],
            milestones=[warmup_steps],
        )
    return None


def _global_grad_norm_for_loss(model: torch.nn.Module, loss: torch.Tensor) -> float:
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return float("nan")
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = None
    for g in grads:
        if g is None:
            continue
        cur = g.detach().float().pow(2).sum()
        sq = cur if sq is None else sq + cur
    return float(torch.sqrt(sq).item()) if sq is not None else float("nan")


def _max_abs_grad_value(model: torch.nn.Module) -> float:
    out = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None and p.grad.numel() > 0:
            out = max(out, float(p.grad.detach().abs().max().item()))
    return out


def _prepare_loss_maps(prep: Dict[str, torch.Tensor], cfg: JEPADecoderConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if str(cfg.decoder.loss_mode) != "unified":
        return (
            prep["observed_map"],
            prep["masked_map"],
            prep["signal_observed_map"],
            prep["signal_masked_map"],
        )
    all_map = prep["observed_map"] | prep["masked_map"]
    all_signal = prep["signal_observed_map"] | prep["signal_masked_map"]
    zeros = torch.zeros_like(all_map, dtype=torch.bool)
    zeros_signal = torch.zeros_like(all_signal, dtype=torch.bool)
    return all_map, zeros, all_signal, zeros_signal


def _build_training_step_log(
    *,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    pre_clip_grad_norm: float,
    clipped: int,
    clip_cap: float,
    clipped_running: Optional[float],
    clipped_window: Optional[float],
    loss_stats: Dict[str, float],
    loss_term_grad_norms: Dict[str, float],
) -> Dict[str, float]:
    logd: Dict[str, float] = {
        "training_stats/step": float(global_step),
        "training_stats/lr": float(optimizer.param_groups[0]["lr"]),
        "training_stats/total_loss": float(loss_stats["loss_total_weighted"]),
        "training_stats/grad_pre_clip_norm": float(pre_clip_grad_norm),
        "training_stats/grad_clipped": float(clipped),
    }
    if clip_cap > 0.0:
        logd["training_stats/grad_clip_cap"] = float(clip_cap)
    if clipped_running is not None:
        logd["training_stats/grad_clipped_frac_running"] = float(clipped_running)
    if clipped_window is not None:
        logd["training_stats/grad_clipped_frac_window"] = float(clipped_window)
    for raw_key, log_key in _BRANCH_RAW_TO_LOG.items():
        stats_key = f"loss_branch_{raw_key}"
        if stats_key in loss_stats and math.isfinite(float(loss_stats[stats_key])):
            logd[f"training_losses/{log_key}"] = float(loss_stats[stats_key])
        grad_key = f"grad_norm_{raw_key}"
        if grad_key in loss_term_grad_norms:
            logd[f"training_grad_norms/{log_key}"] = float(loss_term_grad_norms[grad_key])
    return logd


def train_one_epoch(
    model: JEPADecoderModel,
    dataset: SandboxH5Dataset,
    device: torch.device,
    masker,
    criterion: SandboxCompositeLoss,
    optimizer: torch.optim.Optimizer,
    cfg: JEPADecoderConfig,
    *,
    scheduler=None,
    use_amp: bool = False,
    global_step: int = 0,
    wandb_run=None,
    max_batches: Optional[int] = None,
    training_stats_jsonl_every_n_steps: int = 0,
    on_training_step_snapshot=None,
    epoch_index: int = 0,
) -> int:
    model.train()
    amp_on = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    batches = 0
    clip_cap = float(cfg.training.grad.clip_norm)
    clip_mode = str(cfg.training.grad.clip_mode)
    clip_win = max(1, int(cfg.training.grad.clip_log_window))
    clipped_hist: deque[int] = deque(maxlen=clip_win)
    all_clipped: list[int] = []
    rc_p = float(cfg.training.augment_reverse_complement_prob)

    for batch in dataset:
        batch = _maybe_augment_reverse_complement(dict(batch), rc_p)
        prep = prepare_masked_batch(batch, masker, device)
        if prep is None:
            continue
        obs_map, masked_map, sig_obs_map, sig_masked_map = _prepare_loss_maps(prep, cfg)
        with torch.amp.autocast("cuda", enabled=amp_on):
            p, n, mu, scale, df, peak = model(
                prep["x_data"],
                prep["x_dna"],
                prep["x_meta"],
                prep["y_meta"],
                query_mask=prep["query_mask"],
                query_mask_signal=prep["query_mask_signal"],
            )
            loss, stats, loss_terms = criterion.forward_with_terms(
                p,
                n,
                mu,
                scale,
                df,
                peak,
                prep["y_data"],
                prep["y_pval"],
                prep["y_peaks"],
                obs_map,
                masked_map,
                sig_obs_map,
                sig_masked_map,
                global_step=global_step,
                fallback_imp_to_observed_when_no_masked=False,
            )

        loss_term_grad_norms: Dict[str, float] = {}
        for raw_key in _BRANCH_RAW_TO_LOG:
            term = loss_terms.get(raw_key)
            if term is None or not torch.isfinite(term.detach()):
                loss_term_grad_norms[f"grad_norm_{raw_key}"] = float("nan")
            else:
                loss_term_grad_norms[f"grad_norm_{raw_key}"] = _global_grad_norm_for_loss(model, term)

        optimizer.zero_grad(set_to_none=True)
        if amp_on:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        params = list(model.trainable_parameters())
        if clip_mode == "norm":
            if clip_cap > 0.0:
                pre = float(torch.nn.utils.clip_grad_norm_(params, clip_cap).item())
                clipped = int(pre > clip_cap + 1e-12)
            else:
                pre = float(torch.nn.utils.clip_grad_norm_(params, float("inf")).item())
                clipped = 0
        else:
            pre = float(torch.nn.utils.clip_grad_norm_(params, float("inf")).item())
            if clip_cap > 0.0:
                max_abs_pre = _max_abs_grad_value(model)
                torch.nn.utils.clip_grad_value_(params, clip_cap)
                clipped = int(max_abs_pre > clip_cap + 1e-12)
            else:
                clipped = 0
        clipped_hist.append(clipped)
        all_clipped.append(clipped)

        if amp_on:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (
            int(cfg.eval.meta_sensitivity_probe_every_n_steps) > 0
            and wandb_run is not None
            and global_step % int(cfg.eval.meta_sensitivity_probe_every_n_steps) == 0
        ):
            model.eval()
            with torch.no_grad():
                _kw = dict(
                    query_mask=prep["query_mask"],
                    query_mask_signal=prep["query_mask_signal"],
                )
                out_norm = model(prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"], **_kw)
                y_meta_null = torch.full_like(prep["y_meta"], -1.0)
                out_null = model(prep["x_data"], prep["x_dna"], prep["x_meta"], y_meta_null, **_kw)
            model.train()
            head_names = ["p", "n", "mu", "scale", "df", "peak"]
            sens_logs: Dict[str, Any] = {}
            for hi, hname in enumerate(head_names):
                o_n = out_norm[hi]
                o_z = out_null[hi]
                if o_n is None or o_z is None:
                    continue
                mse_val = float(((o_n.float() - o_z.float()) ** 2).mean().item())
                sens_logs[f"training_metadata_probes/meta_sensitivity_{hname}_mse"] = mse_val
            if sens_logs:
                sens_logs["training_metadata_probes/meta_sensitivity_mean_mse"] = float(sum(sens_logs.values()) / len(sens_logs))
                wandb_run.log(sens_logs, step=global_step)

        global_step += 1
        batches += 1
        clipped_running = float(sum(all_clipped)) / float(len(all_clipped)) if all_clipped else None
        clipped_window = float(sum(clipped_hist)) / float(len(clipped_hist)) if clipped_hist else None
        logd = _build_training_step_log(
            global_step=global_step,
            optimizer=optimizer,
            pre_clip_grad_norm=pre,
            clipped=clipped,
            clip_cap=clip_cap,
            clipped_running=clipped_running,
            clipped_window=clipped_window,
            loss_stats=stats,
            loss_term_grad_norms=loss_term_grad_norms,
        )
        if wandb_run is not None:
            wandb_run.log(logd, step=global_step)
        if (
            on_training_step_snapshot is not None
            and training_stats_jsonl_every_n_steps > 0
            and (global_step % training_stats_jsonl_every_n_steps == 0)
        ):
            on_training_step_snapshot(epoch_index, global_step, logd)
        if max_batches is not None and batches >= max_batches:
            break
    return global_step


def _maybe_wandb(cfg: JEPADecoderConfig):
    if cfg.wandb.mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; logging disabled", file=sys.stderr)
        return None
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        mode=cfg.wandb.mode,
        tags=list(cfg.wandb.tags),
        config=asdict(cfg),
    )
    return wandb


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Sandbox CANDI JEPA decoder trainer (E28 Stage 2)")
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--h5", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override decoder.checkpoint_path")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("dotted_overrides", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg_dict: Dict[str, Any] = asdict(JEPADecoderConfig())
    for base in (_JEPA_DEFAULT_YAML, _DECODER_DEFAULT_YAML):
        if base.exists():
            cfg_dict = deep_merge(cfg_dict, load_yaml(base))
    for cpath in args.config:
        cfg_dict = deep_merge(cfg_dict, load_yaml(Path(cpath)))
    short: Dict[str, Any] = {}
    if args.h5 is not None:
        short.setdefault("data", {})["h5_path"] = str(args.h5)
    if args.checkpoint is not None:
        short.setdefault("decoder", {})["checkpoint_path"] = str(args.checkpoint)
    if args.epochs is not None:
        short.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        short.setdefault("training", {})["batch_size"] = int(args.batch_size)
    if args.device is not None:
        short.setdefault("training", {})["device"] = str(args.device)
    if args.seed is not None:
        short.setdefault("training", {})["seed"] = int(args.seed)
    if args.save_checkpoint:
        short.setdefault("training", {})["save_checkpoint"] = True
    if short:
        cfg_dict = deep_merge(cfg_dict, short)
    if args.no_wandb:
        cfg_dict.setdefault("wandb", {})["mode"] = "disabled"
    elif args.wandb:
        cfg_dict.setdefault("wandb", {})["mode"] = "online"
    if args.run_name is not None:
        cfg_dict.setdefault("wandb", {})["run_name"] = str(args.run_name)
    dot_ov = _dotted_overrides_from_argv(args.dotted_overrides)
    if dot_ov:
        cfg_dict = deep_merge(cfg_dict, dot_ov)

    cfg = config_from_dict(JEPADecoderConfig, cfg_dict)
    _warn_inactive_optimizer_overrides(cfg)
    if args.print_config or args.dry_run:
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return 0

    h5p = Path(cfg.data.h5_path)
    if not h5p.exists():
        print(f"error: HDF5 not found: {h5p}", file=sys.stderr)
        return 2
    if not cfg.decoder.checkpoint_path:
        print("error: decoder.checkpoint_path or --checkpoint is required", file=sys.stderr)
        return 2

    run_dir = Path(cfg.training.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, run_dir / "resolved_config.yaml")  # type: ignore[arg-type]
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")

    def _append_metrics(record: Dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")

    t_start = time.time()
    created_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))
    random.seed(int(cfg.training.seed))
    torch.manual_seed(int(cfg.training.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.training.seed))
    if bool(cfg.training.strict_determinism):
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(cfg.training.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[jdec] device={device}", file=sys.stderr)

    ds_train = SandboxH5Dataset(
        h5p,
        str(cfg.data.regime),
        train=True,
        batch_size=int(cfg.training.batch_size),
        biosample_prefix="T_",
        dsf_list=list(cfg.training.dsf.dsf_list),
        dsf_sampling=str(cfg.training.dsf.sampling),
        seed=int(cfg.training.seed),
        shuffle=True,
        h5_cache_ram=bool(cfg.data.h5_cache_ram),
        ram_cache_max_bytes=int(cfg.data.ram_cache_max_bytes),
    )
    masker = make_masker(
        p_full_loci=float(cfg.training.masking.p_full_loci),
        p_full_assay=float(cfg.training.masking.p_full_assay),
        p_chunks=float(cfg.training.masking.p_chunks),
        mask_fraction=float(cfg.training.masking.mask_fraction),
        chunk_size=int(cfg.training.masking.chunk_size),
        min_available_frac=float(cfg.training.masking.min_available_frac),
        preserve_assay_id=bool(cfg.training.masking.preserve_assay_id),
    )
    canonical_meta: Optional[torch.Tensor] = None
    if Path(str(cfg.eval.eic_metadata_path)).exists():
        canonical_meta = build_canonical_meta(str(cfg.eval.eic_metadata_path), SANDBOX_ASSAYS)

    model = JEPADecoderModel.from_checkpoint(cfg, device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[jdec] trainable params: {n_train:,} / {n_total:,}", file=sys.stderr)

    train_crit = _make_criterion(cfg, eval_mode=False).to(device)
    eval_crit = _make_criterion(cfg, eval_mode=True).to(device)
    steps_per_epoch = int(cfg.training.steps_per_epoch) if int(cfg.training.steps_per_epoch) > 0 else ds_train.estimate_steps_per_epoch()
    total_steps = max(1, int(cfg.training.epochs)) * steps_per_epoch
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_steps=total_steps)
    wb = _maybe_wandb(cfg)

    def _emit_training_step_snapshot(ep_idx: int, gstep: int, logd: Dict[str, float]) -> None:
        rec = {"kind": "training_step", "epoch": int(ep_idx), "global_step": int(gstep)}
        ts: Dict[str, float] = {}
        tl: Dict[str, float] = {}
        tg: Dict[str, float] = {}
        for k, v in logd.items():
            if k.startswith("training_stats/"):
                ts[k] = float(v)
            elif k.startswith("training_losses/"):
                tl[k] = float(v)
            elif k.startswith("training_grad_norms/"):
                tg[k] = float(v)
        if ts:
            rec["training_stats"] = ts
        if tl:
            rec["training_losses"] = tl
        if tg:
            rec["training_grad_norms"] = tg
        _append_metrics(rec)

    best_eval_total: Optional[float] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    early_stop_strikes = 0
    global_step = 0
    for ep in range(int(cfg.training.epochs)):
        ep_t0 = time.time()
        global_step = train_one_epoch(
            model,
            ds_train,
            device,
            masker,
            train_crit,
            opt,
            cfg,
            scheduler=sched,
            use_amp=bool(cfg.training.amp),
            global_step=global_step,
            wandb_run=wb,
            max_batches=cfg.training.max_train_batches,
            training_stats_jsonl_every_n_steps=int(cfg.training.training_stats_jsonl_every_n_steps),
            on_training_step_snapshot=_emit_training_step_snapshot,
            epoch_index=ep,
        )
        if wb is not None:
            wb.log({"epoch": ep}, step=global_step)
        epoch_record: Dict[str, Any] = {
            "kind": "epoch",
            "epoch": ep,
            "global_step": int(global_step),
            "epoch_seconds": float(time.time() - ep_t0),
        }
        if cfg.training.eval_each_epoch and (ep + 1) % int(cfg.eval.eval_every_n_epochs) == 0:
            metrics = run_eval_pass(
                model,
                eval_crit,
                h5p,
                str(cfg.data.regime),
                device,
                masker,
                batch_size=int(cfg.training.batch_size),
                seed=int(cfg.training.seed) + ep,
                max_batches=int(cfg.training.eval_max_batches),
                imp_prefixes=tuple(cfg.eval.eval_imp_prefixes),
                h5_cache_ram=bool(cfg.data.h5_cache_ram),
                ram_cache_max_bytes=int(cfg.data.ram_cache_max_bytes),
                canonical_meta=canonical_meta,
                use_canonical_missing_meta=bool(cfg.eval.use_canonical_missing_meta),
                include_median_metrics=True,
            )
            print(json.dumps({"epoch": ep, "eval_prefixed": metrics}, indent=2), file=sys.stderr)
            eval_metrics_dict, eval_metrics_median_dict, eval_losses_dict = _split_eval_families(metrics)
            if wb is not None:
                wb.log(metrics, step=global_step)
            epoch_record["eval_metrics"] = eval_metrics_dict
            if eval_metrics_median_dict:
                epoch_record["eval_metrics_median"] = eval_metrics_median_dict
            epoch_record["eval_losses"] = eval_losses_dict

            if int(cfg.eval.meta_sensitivity_probe_every_n_steps) > 0:
                ds_ev = SandboxH5Dataset(
                    h5p,
                    str(cfg.data.regime),
                    train=False,
                    batch_size=min(2, int(cfg.training.batch_size)),
                    dsf_sampling="off",
                    seed=int(cfg.training.seed),
                    shuffle=False,
                    eval_include_vb_ground_truth=True,
                    imp_prefixes=tuple(cfg.eval.eval_imp_prefixes),
                    h5_cache_ram=bool(cfg.data.h5_cache_ram),
                    ram_cache_max_bytes=int(cfg.data.ram_cache_max_bytes),
                )
                batch0 = next(iter(ds_ev), None)
                if batch0 is not None:
                    eval_masker_probe = make_masker(p_full_loci=0.0, p_full_assay=0.0, p_chunks=0.0, mask_fraction=0.0, chunk_size=int(cfg.training.masking.chunk_size))
                    prep0 = prepare_masked_batch(batch0, eval_masker_probe, device, apply_mask=False)
                    if prep0 is not None:
                        pr = {
                            "training_metadata_probes/depth_count_ratio": prompt_sensitivity_depth_count_ratio(model, prep0, prep0["y_meta"], device),
                            "training_metadata_probes/runtype_mse": prompt_sensitivity_runtype_mse(model, prep0, prep0["y_meta"], device),
                            "training_metadata_probes/readlen_mse": prompt_sensitivity_readlen_mse(model, prep0, prep0["y_meta"], device),
                        }
                        if wb is not None:
                            wb.log(pr, step=global_step)
                        epoch_record["training_metadata_probes"] = pr

        if "eval_losses" in epoch_record:
            cur_total = epoch_record["eval_losses"].get("eval_losses/total_loss")
            if isinstance(cur_total, (int, float)) and math.isfinite(cur_total):
                if best_eval_total is None or float(cur_total) < best_eval_total:
                    best_eval_total = float(cur_total)
                    early_stop_strikes = 0
                    if bool(cfg.training.save_best_checkpoint):
                        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                else:
                    early_stop_strikes += 1
                if bool(cfg.training.early_stop_enabled) and early_stop_strikes >= int(cfg.training.early_stop_patience):
                    epoch_record["early_stop_triggered"] = True
        _append_metrics(epoch_record)
        if epoch_record.get("early_stop_triggered"):
            break

    if bool(cfg.training.save_checkpoint):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "global_step": int(global_step),
                "best_eval_total_loss": best_eval_total,
                "config": asdict(cfg),
            },
            run_dir / "jepa_decoder_checkpoint_last.pt",
        )
    if bool(cfg.training.save_best_checkpoint) and best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "global_step": int(global_step),
                "best_eval_total_loss": best_eval_total,
                "config": asdict(cfg),
            },
            run_dir / "jepa_decoder_checkpoint_best.pt",
        )

    elapsed = float(time.time() - t_start)
    (run_dir / "elapsed.txt").write_text(f"{elapsed:.3f}\n")
    if wb is not None:
        wb.log({"run/elapsed_seconds": elapsed}, step=global_step)
        wb.finish()
    if not bool(getattr(cfg.hpo, "disable", False)):
        try:
            graph_path = Path(cfg.hpo.graph_path)
            if not graph_path.is_absolute():
                graph_path = Path.cwd() / graph_path
            update_graph_for_run(
                run_id=run_dir.name,
                run_dir=run_dir,
                resolved_cfg_dict=asdict(cfg),
                parent_run_ids=list(cfg.hpo.parent),
                experiment_label=str(cfg.hpo.experiment_label),
                notes=str(cfg.hpo.notes),
                elapsed_seconds=elapsed,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                wandb_run_name=cfg.wandb.run_name,
                graph_path=graph_path,
                created_at=created_at_iso,
            )
        except Exception as exc:
            print(f"[jdec] HPO graph update failed: {exc}", file=sys.stderr)
    print(f"Finished {cfg.training.epochs} epoch(s), global_step={global_step}, elapsed={elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
