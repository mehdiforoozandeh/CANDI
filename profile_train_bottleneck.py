#!/usr/bin/env python3

import argparse
import contextlib
import io
import json
import multiprocessing
import os
import random
import statistics
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import _utils
from train import (
    CANDI_TRAINER,
    create_argument_parser,
    create_model_from_args,
    resolve_dataset_class,
    setup_device,
    validate_arguments,
)


class StageProfiler:
    def __init__(self, device):
        self.device = device
        self.current = None

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def timed(self, name, func):
        def wrapped(*args, **kwargs):
            self._sync()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self._sync()
                elapsed = time.perf_counter() - start
                if self.current is not None:
                    self.current[name] += elapsed

        return wrapped


def summarise(values):
    if not values:
        return {"n": 0, "mean_s": None, "median_s": None, "min_s": None, "max_s": None}
    return {
        "n": len(values),
        "mean_s": statistics.mean(values),
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Profile train.py bottlenecks without modifying core modules.")
    parser.add_argument("--batches", type=int, default=30, help="Measured batches.")
    parser.add_argument("--warmup-batches", type=int, default=5, help="Warmup batches before measuring.")
    parser.add_argument("--num-workers-override", type=int, default=None, help="Override DataLoader num_workers.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--loci-gen", type=str, default="full_chr")
    parser.add_argument("--dsf-list", type=str, default="1")
    parser.add_argument("--name-suffix", type=str, default="profile")
    parser.add_argument("--dist-type", type=str, default="gaussian")
    parser.add_argument("--signal-transform", type=str, default="log1p")
    parser.add_argument("--decoder-type", type=str, default="fixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-backend", type=str, default="npz", choices=["npz", "h5", "zarr"])
    parser.add_argument("--prepared-data-path", type=str, default=None)
    parser.add_argument("--wandb-log-every", type=int, default=50)
    parser.add_argument("--supertrack-train-monitor-every", type=int, default=20)
    parser.add_argument("--mask-stem", action="store_true")
    parser.add_argument("--xl-dna", action="store_true")
    parser.add_argument("--enable-supertrack-train-monitor", action="store_true")
    return parser.parse_args()


def build_train_namespace(cli):
    parser = create_argument_parser()
    train_args = [
        "--eic",
        "--batch-size",
        str(cli.batch_size),
        "--grad-accum-steps",
        str(cli.grad_accum_steps),
        "--epochs",
        str(cli.epochs),
        "--loci-gen",
        cli.loci_gen,
        "--dsf-list",
        cli.dsf_list,
        "--name-suffix",
        cli.name_suffix,
        "--dist-type",
        cli.dist_type,
        "--signal-transform",
        cli.signal_transform,
        "--decoder-type",
        cli.decoder_type,
        "--seed",
        str(cli.seed),
        "--data-path",
        cli.data_path,
        "--data-backend",
        cli.data_backend,
        "--disable-validation",
        "--no-save",
    ]
    if cli.prepared_data_path:
        train_args.extend(["--prepared-data-path", cli.prepared_data_path])
    if cli.mask_stem:
        train_args.append("--mask-stem")
    if cli.xl_dna:
        train_args.append("--xl-dna")
    if cli.enable_supertrack_train_monitor:
        train_args.extend(
            [
                "--enable-supertrack-train-monitor",
                "--supertrack-train-monitor-every",
                str(cli.supertrack_train_monitor_every),
            ]
        )
    train_args.extend(["--wandb-log-every", str(cli.wandb_log_every)])
    args = parser.parse_args(train_args)
    return validate_arguments(args)


def build_paths_and_params(args):
    dataset_type = "eic" if args.eic else "merged"
    base_path = args.data_path
    if not base_path.endswith("/"):
        base_path += "/"
    data_path = base_path + ("DATA_CANDI_EIC/" if args.eic else "DATA_CANDI_MERGED/")

    prepared_data_path = None
    if args.data_backend in ("zarr", "h5"):
        if args.prepared_data_path is None:
            raise ValueError("prepared data path must be set for h5/zarr profiling")
        prepared_data_path = args.prepared_data_path
        if not str(prepared_data_path).endswith("/"):
            prepared_data_path = str(prepared_data_path) + "/"
    active_data_path = prepared_data_path if args.data_backend in ("zarr", "h5") else data_path

    dataset_params = {
        "base_path": active_data_path,
        "dataset_type": dataset_type,
        "m": args.num_loci,
        "context_length": args.context_length * 25,
        "split": "train",
        "loci_gen_strategy": args.loci_gen,
        "ccre_fraction": args.ccre_fraction,
        "dsf_list": args.dsf_list,
        "DNA": True,
        "must_have_chr_access": args.must_have_chr_access,
        "bios_min_exp_avail_threshold": args.min_avail,
        "shuffle_bios": True,
        "balanced_bios_order": args.balanced_bios_order,
        "fill_prompt_mode": args.fill_prompt_mode,
        "signal_transform": args.signal_transform,
        "enable_per_assay_dsf_sampling": args.enable_per_assay_dsf_sampling,
        "per_assay_dsf_sampling_mode": args.per_assay_dsf_sampling_mode,
        "seed": args.seed,
        "data_backend": args.data_backend,
    }

    training_params = {
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "inner_epochs": args.inner_epochs,
        "enable_validation": not args.disable_validation,
        "val_freq": args.val_freq,
        "enable_supertrack_train_monitor": args.enable_supertrack_train_monitor,
        "supertrack_train_monitor_every": args.supertrack_train_monitor_every,
        "supertrack_train_monitor_max_batch": args.supertrack_train_monitor_max_batch,
        "wandb_log_every": args.wandb_log_every,
        "use_mixed_precision": args.mixed_precision,
        "specific_ema_alpha": args.specific_ema_alpha,
        "progress_dir": args.progress_dir,
        "debug": args.debug,
        "DNA": True,
        "no_save": args.no_save,
        "count_weight": args.count_weight,
        "pval_weight": args.pval_weight,
        "peak_weight": args.peak_weight,
        "obs_weight": args.obs_weight,
        "imp_weight": args.imp_weight,
        "enable_assay_ema_balance": args.enable_assay_ema_balance,
        "enable_hier_reduction": args.enable_hier_reduction,
        "assay_ema_decay": args.assay_ema_decay,
        "assay_ema_eps": args.assay_ema_eps,
        "assay_ema_warmup_steps": args.assay_ema_warmup_steps,
        "assay_ema_weight_min": args.assay_ema_weight_min,
        "assay_ema_weight_max": args.assay_ema_weight_max,
        "enable_fg_bg_balance": args.enable_fg_bg_balance,
        "fg_weight": args.fg_weight,
        "fg_min_fraction": args.fg_min_fraction,
        "enable_uncertainty_weighting": args.enable_uncertainty_weighting,
        "uncertainty_warmup_steps": args.uncertainty_warmup_steps,
        "uncertainty_init_logvar": args.uncertainty_init_logvar,
        "enable_count_rstable_objective": args.enable_count_rstable_objective,
        "count_rstable_eps": args.count_rstable_eps,
        "count_rstable_ema_decay": args.count_rstable_ema_decay,
        "count_rstable_warmup_steps": args.count_rstable_warmup_steps,
        "count_rstable_denom_min": args.count_rstable_denom_min,
        "count_rstable_r_max": args.count_rstable_r_max,
        "count_rstable_dispersion_min": args.count_rstable_dispersion_min,
        "count_rstable_dispersion_max": args.count_rstable_dispersion_max,
        "p_full_loci": args.p_full_loci,
        "p_full_assay": args.p_full_assay,
        "p_chunks": args.p_chunks,
        "mask_fraction": args.mask_fraction,
        "chunk_size": args.chunk_size,
        "reverse_complement_prob": args.reverse_complement_prob,
        "clip_mode": args.clip_mode,
        "clip_value": args.clip_value,
        "dist_type": args.dist_type,
        "signal_transform": args.signal_transform,
        "enable_latent_kl": args.enable_latent_kl,
        "latent_kl_weight": args.latent_kl_weight,
        "latent_kl_warmup_steps": args.latent_kl_warmup_steps,
        "latent_std_min": args.latent_std_min,
        "latent_std_max": args.latent_std_max,
        "latent_reparam_mode": args.latent_reparam_mode,
        "latent_sample_train_only": args.latent_sample_train_only,
        "latent_deterministic_warmup_steps": args.latent_deterministic_warmup_steps,
    }
    return dataset_params, training_params


def install_wrappers(trainer, profiler):
    original_data_masker_apply_mask = _utils.DataMasker.apply_mask
    _utils.DataMasker.apply_mask = profiler.timed("mask_apply_s", original_data_masker_apply_mask)

    trainer.model.forward = types.MethodType(profiler.timed("model_forward_s", trainer.model.forward.__func__), trainer.model)
    trainer.criterion.forward = types.MethodType(
        profiler.timed("criterion_forward_s", trainer.criterion.forward.__func__),
        trainer.criterion,
    )
    trainer._compute_metrics = types.MethodType(
        profiler.timed("compute_metrics_s", trainer._compute_metrics.__func__),
        trainer,
    )
    trainer._monitor_supertrack_on_batch = types.MethodType(
        profiler.timed("supertrack_monitor_s", trainer._monitor_supertrack_on_batch.__func__),
        trainer,
    )

    original_backward = torch.Tensor.backward

    def timed_backward(self, *args, **kwargs):
        return profiler.timed("backward_s", original_backward)(self, *args, **kwargs)

    torch.Tensor.backward = timed_backward

    def restore():
        _utils.DataMasker.apply_mask = original_data_masker_apply_mask
        torch.Tensor.backward = original_backward

    return restore


def main():
    cli = parse_args()
    args = build_train_namespace(cli)

    os.environ["WANDB_MODE"] = "offline"
    os.environ["PYTHONUNBUFFERED"] = "1"

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = setup_device(args)
    dataset_params, training_params = build_paths_and_params(args)

    with contextlib.redirect_stdout(io.StringIO()):
        dataset_cls = resolve_dataset_class(dataset_params)
        temp_dataset = dataset_cls(**dataset_params)
        signal_dim = len(temp_dataset.aliases["experiment_aliases"])
        num_assays = temp_dataset.num_assays
        num_runtypes = 2
        dataset_params["signal_dim"] = signal_dim
        dataset_params["num_assays"] = num_assays
        dataset_params["num_runtypes"] = num_runtypes

        model = create_model_from_args(args, signal_dim, num_assays, num_runtypes)
        trainer = CANDI_TRAINER(
            model=model,
            dataset_params=dataset_params,
            training_params=training_params,
            device=device,
            rank=args.rank if args.ddp else None,
            world_size=args.world_size if args.ddp else None,
        )
        trainer.is_main_process = False
        trainer._setup()
        estimated_batches_per_epoch = trainer._estimate_batches_per_epoch()
        if trainer.scheduler is None:
            trainer._setup_cosine_scheduler(estimated_batches_per_epoch)

    cpu_count = multiprocessing.cpu_count()
    if trainer.is_ddp:
        num_workers = min(cpu_count // trainer.world_size, 4)
    else:
        num_workers = min(cpu_count, 4)
    if cli.num_workers_override is not None:
        num_workers = cli.num_workers_override

    dataloader = torch.utils.data.DataLoader(
        trainer.dataset,
        batch_size=training_params["batch_size"],
        num_workers=num_workers,
        pin_memory=(trainer.device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    profiler = StageProfiler(trainer.device)
    restore_wrappers = install_wrappers(trainer, profiler)
    trainer.optimizer.zero_grad(set_to_none=True)

    iterator = iter(dataloader)
    records = []
    total_batches = cli.warmup_batches + cli.batches
    micro_batches_since_step = 0

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for batch_idx in range(total_batches):
                wait_start = time.perf_counter()
                batch = next(iterator)
                wait_elapsed = time.perf_counter() - wait_start

                trainer.current_epoch = 0
                trainer.current_batch_idx = batch_idx
                trainer.current_global_step = batch_idx

                h2d_start = time.perf_counter()
                batch = trainer._move_batch_to_device(batch)
                if trainer.device.type == "cuda":
                    torch.cuda.synchronize(trainer.device)
                h2d_elapsed = time.perf_counter() - h2d_start

                per_batch = defaultdict(float)
                profiler.current = per_batch

                process_start = time.perf_counter()
                result = trainer._process_batch(batch, loss_scale=(1.0 / trainer.grad_accum_steps))
                if trainer.device.type == "cuda":
                    torch.cuda.synchronize(trainer.device)
                process_elapsed = time.perf_counter() - process_start
                per_batch["process_total_s"] = process_elapsed

                should_step = ((micro_batches_since_step + 1) % trainer.grad_accum_steps == 0)
                optimizer_elapsed = 0.0
                scheduler_elapsed = 0.0
                if result is not None:
                    micro_batches_since_step += 1
                    if should_step:
                        opt_start = time.perf_counter()
                        trainer._apply_optimizer_step()
                        if trainer.device.type == "cuda":
                            torch.cuda.synchronize(trainer.device)
                        optimizer_elapsed = time.perf_counter() - opt_start
                        micro_batches_since_step = 0

                        if trainer.scheduler is not None:
                            sch_start = time.perf_counter()
                            trainer.scheduler.step()
                            scheduler_elapsed = time.perf_counter() - sch_start

                per_batch["dataloader_wait_s"] = wait_elapsed
                per_batch["host_to_device_s"] = h2d_elapsed
                per_batch["optimizer_step_s"] = optimizer_elapsed
                per_batch["scheduler_step_s"] = scheduler_elapsed
                per_batch["loop_total_s"] = (
                    wait_elapsed + h2d_elapsed + process_elapsed + optimizer_elapsed + scheduler_elapsed
                )

                wrapped_sum = sum(
                    per_batch.get(name, 0.0)
                    for name in (
                        "mask_apply_s",
                        "model_forward_s",
                        "criterion_forward_s",
                        "backward_s",
                        "compute_metrics_s",
                        "supertrack_monitor_s",
                    )
                )
                per_batch["other_process_s"] = max(0.0, process_elapsed - wrapped_sum)

                if batch_idx >= cli.warmup_batches:
                    records.append(dict(per_batch))
    finally:
        restore_wrappers()

    stages = [
        "dataloader_wait_s",
        "host_to_device_s",
        "process_total_s",
        "mask_apply_s",
        "model_forward_s",
        "criterion_forward_s",
        "backward_s",
        "compute_metrics_s",
        "supertrack_monitor_s",
        "other_process_s",
        "optimizer_step_s",
        "scheduler_step_s",
        "loop_total_s",
    ]
    summary = {stage: summarise([r.get(stage, 0.0) for r in records]) for stage in stages}

    total_mean = summary["loop_total_s"]["mean_s"] or 0.0
    fractions = {}
    if total_mean > 0:
        for stage in stages:
            if stage == "loop_total_s":
                continue
            mean_stage = summary[stage]["mean_s"] or 0.0
            fractions[stage] = mean_stage / total_mean

    payload = {
        "backend": args.data_backend,
        "prepared_data_path": args.prepared_data_path,
        "enable_supertrack_train_monitor": bool(args.enable_supertrack_train_monitor),
        "num_workers": num_workers,
        "warmup_batches": cli.warmup_batches,
        "measured_batches": cli.batches,
        "device": str(device),
        "summary": summary,
        "fractions_of_loop_total": fractions,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
