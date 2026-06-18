#!/usr/bin/env python3
"""Real (non-diagnostic) training entrypoint for CANDI v2.

Emits ``resolved_config.yaml`` + ``metrics.jsonl`` in the run dir so runs are
log-observability compatible. Reuses the production sandbox data pipeline
(``SandboxH5Dataset``), masker, ``build_v2_loss``, and core training/eval
machinery from ``sandbox.train`` (``train_one_epoch``, ``run_eval_pass``,
``_maybe_wandb``, metadata sensitivity probes).

This is the entrypoint for the E30 A/B (baseline vs NB depth-offset count head):
the two runs differ only by ``decoder.count_head`` in their config overlays.

Does NOT import anything from ``sandbox/diagnostics/autoresearch/``.

Config merge order:
    CANDIv2Config dataclass defaults
      -> sandbox/configs/candi_v2_default.yaml (always)
      -> each --config overlay (in order)
      -> --set dotted overrides

Usage:
    python -m sandbox.train_candi_v2 \\
        --config sandbox/configs/e30_v2_common.yaml \\
        --config sandbox/configs/e30_v2_baseline.yaml \\
        --run-dir sandbox/runs/e30_v2_baseline \\
        --run-name e30_v2_baseline
        # wandb.mode=online is the candi_v2_default; use --no-wandb to disable.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

from sandbox import SANDBOX_ASSAYS
from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.candi_v2.config import CANDIv2Config, validate_v2_config
from sandbox.candi_v2.loss import build_v2_loss
from sandbox.candi_v2.model import CANDIv2
from sandbox.config import deep_merge, load_yaml
from sandbox.config_types import config_from_dict
from sandbox.data import SandboxH5Dataset, build_canonical_meta
from sandbox.eval import (
    prompt_sensitivity_depth_count_ratio,
    prompt_sensitivity_readlen_mse,
    prompt_sensitivity_runtype_mse,
)
from sandbox.hpo import update_graph_for_run
from sandbox.train import (
    _maybe_wandb,
    _split_eval_families,
    run_eval_pass,
    train_one_epoch,
)

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

DEFAULT_V2_YAML = Path(__file__).resolve().parent / "configs" / "candi_v2_default.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value


def _apply_set(tree: Dict[str, Any], dotted: str) -> None:
    if "=" not in dotted:
        raise ValueError(f"--set expects key=value, got {dotted!r}")
    key, raw = dotted.split("=", 1)
    cur = tree
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = _coerce(raw)


def load_v2_config(
    config_paths: List[Path],
    set_overrides: List[str],
) -> CANDIv2Config:
    merged: Dict[str, Any] = dataclasses.asdict(CANDIv2Config())
    if DEFAULT_V2_YAML.exists():
        merged = deep_merge(merged, load_yaml(DEFAULT_V2_YAML))
    for p in config_paths:
        merged = deep_merge(merged, load_yaml(Path(p)))
    overrides: Dict[str, Any] = {}
    for s in set_overrides:
        _apply_set(overrides, s)
    if overrides:
        merged = deep_merge(merged, overrides)
    merged["encoder"]["num_assays"] = len(SANDBOX_ASSAYS)
    # Keep data.signal_transform aligned with encoder (encoder is canonical at runtime).
    enc_st = merged.get("encoder", {}).get("signal_transform")
    data_st = merged.get("data", {}).get("signal_transform")
    if enc_st is not None:
        merged.setdefault("data", {})["signal_transform"] = enc_st
    elif data_st is not None:
        merged.setdefault("encoder", {})["signal_transform"] = data_st
    return config_from_dict(CANDIv2Config, merged)


def dump_resolved_config(cfg: CANDIv2Config, path: Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML required to dump resolved config.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, tcfg) -> torch.optim.Optimizer:
    name = tcfg.optimizer.name
    if name == "adam":
        h = tcfg.optimizer.adam
        return torch.optim.Adam(model.parameters(), lr=h.lr, betas=(h.beta1, h.beta2),
                                eps=h.eps, weight_decay=h.weight_decay, amsgrad=h.amsgrad)
    if name == "adamw":
        h = tcfg.optimizer.adamw
        return torch.optim.AdamW(model.parameters(), lr=h.lr, betas=(h.beta1, h.beta2),
                                 eps=h.eps, weight_decay=h.weight_decay, amsgrad=h.amsgrad)
    if name == "adamax":
        h = tcfg.optimizer.adamax
        return torch.optim.Adamax(model.parameters(), lr=h.lr, betas=(h.beta1, h.beta2),
                                  eps=h.eps, weight_decay=h.weight_decay)
    if name == "sgd":
        h = tcfg.optimizer.sgd
        return torch.optim.SGD(model.parameters(), lr=h.lr, momentum=h.momentum,
                               dampening=h.dampening, weight_decay=h.weight_decay, nesterov=h.nesterov)
    raise ValueError(f"unknown optimizer {name}")


def _active_base_lr(tcfg) -> float:
    return float(getattr(tcfg.optimizer, tcfg.optimizer.name).lr)


def build_scheduler(optimizer, tcfg, *, total_steps: int):
    sch = tcfg.schedule
    base_lr = _active_base_lr(tcfg)
    eta_min = base_lr * float(sch.min_lr_ratio)
    if sch.name == "constant" or total_steps <= 0:
        return None
    if sch.name == "linear":
        return LinearLR(optimizer, start_factor=1.0, end_factor=float(sch.min_lr_ratio),
                        total_iters=total_steps)
    if sch.name == "cosine":
        wf = float(sch.warmup_frac)
        if wf <= 0.0 or total_steps <= 1:
            return CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=eta_min)
        warmup = min(max(1, int(round(total_steps * wf))), max(1, total_steps - 1))
        cosine = max(1, total_steps - warmup)
        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup),
                CosineAnnealingLR(optimizer, T_max=cosine, eta_min=eta_min),
            ],
            milestones=[warmup],
        )
    return None


# ---------------------------------------------------------------------------
# Model wrapper (6-tuple forward for train.py compatibility)
# ---------------------------------------------------------------------------

class _V2TupleWrapper(nn.Module):
    """6-tuple forward so ``train_one_epoch`` / ``run_eval_pass`` can call v2."""

    def __init__(self, model: CANDIv2) -> None:
        super().__init__()
        self.v2 = model

    def forward(self, x_data, x_dna, x_meta, y_meta, **kwargs):
        return self.v2.forward_tuple(x_data, x_dna, x_meta, y_meta, **kwargs)


def _save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    best_eval_total_loss: Optional[float],
) -> None:
    """Checkpoint payload for resume and best-epoch selection."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": int(global_step),
            "epoch": int(epoch),
            "best_eval_total_loss": best_eval_total_loss,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(cfg: CANDIv2Config, args) -> int:
    validate_v2_config(cfg)

    h5_path = Path(args.h5 or cfg.data.h5_path)
    if not h5_path.exists():
        print(f"error: HDF5 not found: {h5_path}", file=sys.stderr)
        return 2

    run_dir = Path(args.run_dir or cfg.training.run_dir)
    cfg.data.h5_path = str(h5_path)
    cfg.training.run_dir = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_resolved_config(cfg, run_dir / "resolved_config.yaml")
    metrics_path = run_dir / "metrics.jsonl"
    resume_path = getattr(args, "resume", None)
    if not resume_path:
        metrics_path.write_text("")

    def _append_metrics(record: Dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")

    t_start = time.time()
    created_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))

    dev = args.device or cfg.training.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(dev).lower() == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(str(dev))
    seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    wb = _maybe_wandb(cfg)

    model = CANDIv2(cfg).to(device)
    train_model = _V2TupleWrapper(model).to(device)
    loss_fn = build_v2_loss(cfg).to(device)
    optimizer = build_optimizer(model, cfg.training)

    _eic_path = str(cfg.eval.eic_metadata_path)
    canonical_meta: Optional[torch.Tensor] = None
    if Path(_eic_path).exists():
        canonical_meta = build_canonical_meta(_eic_path, SANDBOX_ASSAYS)
        print(
            f"[train_candi_v2] Loaded canonical metadata for {len(SANDBOX_ASSAYS)} assays "
            f"from {_eic_path}",
            file=sys.stderr,
        )
    else:
        print(
            f"[train_candi_v2] Warning: eic_metadata_path={_eic_path!r} not found; "
            "scenario-2 imputation eval disabled.",
            file=sys.stderr,
        )

    ds_train = SandboxH5Dataset(
        h5_path, cfg.data.regime, train=True,
        batch_size=int(cfg.training.batch_size), biosample_prefix="T_",
        dsf_list=tuple(cfg.training.dsf.dsf_list),
        dsf_sampling=str(cfg.training.dsf.sampling),
        seed=seed, shuffle=True,
        eval_include_vb_ground_truth=False,
        imp_prefixes=tuple(cfg.eval.eval_imp_prefixes),
        h5_cache_ram=bool(cfg.data.h5_cache_ram),
        ram_cache_max_bytes=int(cfg.data.ram_cache_max_bytes),
    )
    masker = make_masker(
        p_full_assay=float(cfg.training.masking.p_full_assay),
        p_full_loci=float(cfg.training.masking.p_full_loci),
        p_chunks=float(cfg.training.masking.p_chunks),
        mask_fraction=float(cfg.training.masking.mask_fraction),
        chunk_size=int(cfg.training.masking.chunk_size),
    )

    cfg_steps = int(cfg.training.steps_per_epoch)
    steps_per_epoch = cfg_steps if cfg_steps > 0 else ds_train.estimate_steps_per_epoch()
    total_steps = max(1, int(cfg.training.epochs)) * steps_per_epoch
    scheduler = build_scheduler(optimizer, cfg.training, total_steps=total_steps)

    print(
        f"[train_candi_v2] device={device} count_head={cfg.decoder.count_head} "
        f"depth_center={cfg.decoder.depth_center} heads={cfg.decoder.heads} "
        f"regime={cfg.data.regime} epochs={cfg.training.epochs} steps/epoch~{steps_per_epoch} "
        f"eval_every={cfg.eval.eval_every_n_epochs} wandb={cfg.wandb.mode} "
        f"params={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )

    def _emit_training_step_snapshot(ep_idx: int, gstep: int, logd: Dict[str, float]) -> None:
        rec: Dict[str, Any] = {
            "kind": "training_step",
            "epoch": int(ep_idx),
            "global_step": int(gstep),
        }
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

    global_step = 0
    start_epoch = 0
    best_eval_total: Optional[float] = None
    early_stop_strikes = 0
    early_stop_triggered = False
    last_best_save_epoch = -10**9

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_eval_total = ckpt.get("best_eval_total_loss")
        print(
            f"[train_candi_v2] Resumed from {resume_path} epoch={start_epoch} "
            f"global_step={global_step} best_total={best_eval_total}",
            file=sys.stderr,
        )

    last_ep = max(start_epoch - 1, 0)

    for ep in range(start_epoch, int(cfg.training.epochs)):
        last_ep = ep
        ep_t0 = time.time()
        global_step = train_one_epoch(
            train_model,
            ds_train,
            device,
            masker,
            loss_fn,
            optimizer,
            cfg,
            scheduler=scheduler,
            use_amp=bool(cfg.training.amp),
            global_step=global_step,
            wandb_run=wb,
            max_batches=cfg.training.max_train_batches,
            meta_sensitivity_probe_every_n_steps=int(cfg.eval.meta_sensitivity_probe_every_n_steps),
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
                train_model,
                loss_fn,
                h5_path,
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
            )
            print(json.dumps({"epoch": ep, "eval_prefixed": metrics}, indent=2), file=sys.stderr)
            eval_metrics_dict, eval_losses_dict = _split_eval_families(metrics)
            if wb is not None:
                wb.log(metrics, step=global_step)
            epoch_record["eval_metrics"] = eval_metrics_dict
            epoch_record["eval_losses"] = eval_losses_dict

            ds_ev = SandboxH5Dataset(
                h5_path,
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
                probe_masker = make_masker(
                    p_full_loci=0.0,
                    p_full_assay=0.0,
                    p_chunks=0.0,
                    mask_fraction=0.0,
                    chunk_size=int(cfg.training.masking.chunk_size),
                )
                prep0 = prepare_masked_batch(batch0, probe_masker, device, apply_mask=False)
                if prep0 is not None:
                    pr = {
                        "training_metadata_probes/depth_count_ratio": prompt_sensitivity_depth_count_ratio(
                            train_model,
                            prep0,
                            prep0["y_meta"],
                            device,
                            depth_lo=float(cfg.eval.probe_depth_lo),
                            depth_hi=float(cfg.eval.probe_depth_hi),
                        ),
                        "training_metadata_probes/runtype_mse": prompt_sensitivity_runtype_mse(
                            train_model, prep0, prep0["y_meta"], device,
                        ),
                        "training_metadata_probes/readlen_mse": prompt_sensitivity_readlen_mse(
                            train_model,
                            prep0,
                            prep0["y_meta"],
                            device,
                            readlen_a=float(cfg.eval.probe_read_length_lo),
                            readlen_b=float(cfg.eval.probe_read_length_hi),
                        ),
                    }
                    if wb is not None:
                        wb.log(pr, step=global_step)
                    print(json.dumps({"epoch": ep, "training_metadata_probes": pr}, indent=2), file=sys.stderr)
                    epoch_record["training_metadata_probes"] = pr

            cur_total = eval_losses_dict.get("eval_losses/total_loss")
            if isinstance(cur_total, (int, float)) and math.isfinite(cur_total):
                if best_eval_total is None or float(cur_total) < best_eval_total:
                    best_eval_total = float(cur_total)
                    early_stop_strikes = 0
                    cooldown = int(cfg.training.best_checkpoint_cooldown_epochs)
                    if cfg.training.save_best_checkpoint and (
                        last_best_save_epoch < 0
                        or (ep - last_best_save_epoch) >= cooldown
                    ):
                        _save_training_checkpoint(
                            run_dir / "checkpoint_best.pt",
                            model=model,
                            optimizer=optimizer,
                            global_step=global_step,
                            epoch=ep,
                            best_eval_total_loss=best_eval_total,
                        )
                        last_best_save_epoch = ep
                else:
                    early_stop_strikes += 1
                if (
                    bool(cfg.training.early_stop_enabled)
                    and early_stop_strikes >= int(cfg.training.early_stop_patience)
                ):
                    early_stop_triggered = True
                    epoch_record["early_stop_triggered"] = True
                    epoch_record["early_stop_best_total_loss"] = best_eval_total
                    epoch_record["early_stop_strikes"] = early_stop_strikes

        _append_metrics(epoch_record)
        if early_stop_triggered:
            print(
                f"[train_candi_v2] early stopping at epoch {ep}: "
                f"strikes={early_stop_strikes} >= patience={int(cfg.training.early_stop_patience)}, "
                f"best_total_loss={best_eval_total:.4f}",
                file=sys.stderr,
            )
            break

    if cfg.training.save_checkpoint:
        ckpt_path = run_dir / "checkpoint_last.pt"
        _save_training_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            epoch=last_ep,
            best_eval_total_loss=best_eval_total,
        )
        print(f"[train_candi_v2] Saved checkpoint to {ckpt_path}", file=sys.stderr)

    elapsed = float(time.time() - t_start)
    (run_dir / "elapsed.txt").write_text(f"{elapsed:.3f}\n")
    if wb is not None:
        wb.log({"run/elapsed_seconds": elapsed}, step=global_step)
        wb.finish()

    if not bool(getattr(cfg.hpo, "disable", False)):
        try:
            run_id = run_dir.name
            graph_path = Path(cfg.hpo.graph_path)
            if not graph_path.is_absolute():
                graph_path = Path.cwd() / graph_path
            update_graph_for_run(
                run_id=run_id,
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
            print(f"[train_candi_v2] HPO graph updated: {graph_path} (run_id={run_id})", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[train_candi_v2] HPO graph update failed: {e}", file=sys.stderr)

    print(
        f"[train_candi_v2] done in {elapsed:.1f}s global_step={global_step} -> {metrics_path}",
        file=sys.stderr,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="CANDI v2 trainer (E30 A/B)")
    p.add_argument("--config", action="append", default=[], help="YAML overlay (repeatable).")
    p.add_argument("--set", action="append", default=[], help="Dotted override key=value (repeatable).")
    p.add_argument("--print-config", action="store_true", help="Print resolved CANDIv2Config and exit.")
    p.add_argument("--dry-run", action="store_true", help="Validate config and exit (no train).")
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--h5", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None, help="W&B run name (wandb.run_name).")
    p.add_argument("--wandb", action="store_true", help="Force wandb online mode.")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    p.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save final checkpoint to <run_dir>/checkpoint_last.pt (default: off).",
    )
    p.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping on rising eval_losses/total_loss (default: off).",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Number of consecutive eval points where total_loss > best before stopping.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (.pt) with model/optimizer/global_step/epoch state.",
    )
    args = p.parse_args(argv)

    set_overrides = list(args.set)
    if args.epochs is not None:
        set_overrides.append(f"training.epochs={args.epochs}")
    if args.batch_size is not None:
        set_overrides.append(f"training.batch_size={args.batch_size}")
    if args.device is not None:
        set_overrides.append(f"training.device={args.device}")
    if args.seed is not None:
        set_overrides.append(f"training.seed={args.seed}")
    if args.lr is not None or args.weight_decay is not None:
        # Optimizer name resolved after first merge pass; apply via dotted keys in second pass.
        pass
    if args.no_wandb:
        set_overrides.append("wandb.mode=disabled")
    elif args.wandb:
        set_overrides.append("wandb.mode=online")
    if args.run_name is not None:
        set_overrides.append(f"wandb.run_name={args.run_name}")
    if args.save_checkpoint:
        set_overrides.append("training.save_checkpoint=true")
    if args.early_stop:
        set_overrides.append("training.early_stop_enabled=true")
    if args.early_stop_patience is not None:
        set_overrides.append(f"training.early_stop_patience={args.early_stop_patience}")
    if args.run_dir is not None:
        set_overrides.append(f"training.run_dir={args.run_dir}")
    if args.h5 is not None:
        set_overrides.append(f"data.h5_path={args.h5}")

    cfg = load_v2_config([Path(c) for c in args.config], set_overrides)

    if args.lr is not None or args.weight_decay is not None:
        on = cfg.training.optimizer.name
        if args.lr is not None:
            setattr(getattr(cfg.training.optimizer, on), "lr", float(args.lr))
        if args.weight_decay is not None:
            setattr(getattr(cfg.training.optimizer, on), "weight_decay", float(args.weight_decay))

    if args.print_config or args.dry_run:
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return 0

    return train(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
