#!/usr/bin/env python3

import argparse
import json
import os
import re
import statistics
import subprocess


def build_cmd(args):
    cmd = [
        "python",
        "train.py",
        "--eic",
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--epochs",
        str(args.epochs),
        "--loci-gen",
        args.loci_gen,
        "--dsf-list",
        args.dsf_list,
        "--name-suffix",
        args.name_suffix,
        "--dist-type",
        args.dist_type,
        "--signal-transform",
        args.signal_transform,
        "--decoder-type",
        args.decoder_type,
        "--seed",
        str(args.seed),
        "--data-path",
        args.data_path,
        "--disable-validation",
        "--no-save",
    ]
    if args.mask_stem:
        cmd.append("--mask-stem")
    if args.xl_dna:
        cmd.append("--xl-dna")
    if args.enable_supertrack_train_monitor:
        cmd.extend(
            [
                "--enable-supertrack-train-monitor",
                "--supertrack-train-monitor-every",
                str(args.supertrack_train_monitor_every),
            ]
        )
    cmd.extend(["--wandb-log-every", str(args.wandb_log_every)])
    if args.data_backend:
        cmd.extend(["--data-backend", args.data_backend])
    if args.prepared_data_path:
        cmd.extend(["--prepared-data-path", args.prepared_data_path])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run train.py briefly and summarize printed batch times.")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--loci-gen", type=str, default="full_chr")
    parser.add_argument("--dsf-list", type=str, default="1")
    parser.add_argument("--name-suffix", type=str, default="bench")
    parser.add_argument("--dist-type", type=str, default="gaussian")
    parser.add_argument("--signal-transform", type=str, default="log1p")
    parser.add_argument("--decoder-type", type=str, default="fixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-backend", type=str, default=None)
    parser.add_argument("--prepared-data-path", type=str, default=None)
    parser.add_argument("--wandb-log-every", type=int, default=50)
    parser.add_argument("--supertrack-train-monitor-every", type=int, default=20)
    parser.add_argument("--mask-stem", action="store_true")
    parser.add_argument("--xl-dna", action="store_true")
    parser.add_argument("--enable-supertrack-train-monitor", action="store_true")
    args = parser.parse_args()

    env = dict(os.environ)
    env["WANDB_MODE"] = "offline"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = build_cmd(args)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_seconds,
            env=env,
        )
        output = proc.stdout
        status = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        status = 124

    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="ignore")

    times = [float(x) for x in re.findall(r"Batch time: ([0-9.]+)s", output)]
    payload = {
        "status": status,
        "num_batch_times": len(times),
        "batch_times_s": times[:20],
        "mean_s": statistics.mean(times) if times else None,
        "median_s": statistics.median(times) if times else None,
        "min_s": min(times) if times else None,
        "max_s": max(times) if times else None,
        "command": cmd,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
