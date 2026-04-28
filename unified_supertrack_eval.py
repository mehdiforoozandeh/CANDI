#!/usr/bin/env python3
"""
Unified supertrack evaluation/visualization runner.

Runs the two supertrack-specific pipelines:
1) eval_scripts/smoking_gun_supertrack.py
2) viz_supertrack.py (metadata sweep visualizations)

Outputs are saved under:
  <model_dir>/supertrack_evals/
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_SWEEP_VALUES = {
    "depth": "10000000,30000000,50000000,100000000",
    "read_length": "36,50,75,100",
    "run_type": "single-ended,paired-ended",
}


def _read_metadata(repo_root: Path, dataset: str) -> pd.DataFrame:
    metadata_path = repo_root / "data" / f"{dataset}_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    return pd.read_csv(metadata_path)


def _choose_default_bios_and_assay(df: pd.DataFrame, dataset: str) -> Tuple[str, str]:
    bios_col = "biosample_name"
    assay_col = "assay_name"
    if bios_col not in df.columns or assay_col not in df.columns:
        raise ValueError("Metadata CSV must contain biosample_name and assay_name columns.")

    bios_values = sorted(df[bios_col].dropna().astype(str).unique().tolist())
    if not bios_values:
        raise ValueError("No biosample_name entries found in metadata.")

    if dataset == "eic":
        preferred = [b for b in bios_values if b.startswith("B_") or b.startswith("V_")]
        if not preferred:
            preferred = [b for b in bios_values if b.startswith("T_")]
        bios = sorted(preferred)[0] if preferred else bios_values[0]
    else:
        bios = bios_values[0]

    assay_values = (
        df[df[bios_col].astype(str) == bios][assay_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if not assay_values:
        assay_values = sorted(df[assay_col].dropna().astype(str).unique().tolist())
    if not assay_values:
        raise ValueError("No assay_name entries found in metadata.")
    assay = sorted(assay_values)[0]
    return bios, assay


def _resolve_eic_pair(bios_name: str) -> Tuple[str, str]:
    """Return paired (T_bios, B_or_V_bios) names for EIC."""
    if bios_name.startswith("B_") or bios_name.startswith("V_"):
        return bios_name.replace("B_", "T_").replace("V_", "T_"), bios_name
    if bios_name.startswith("T_"):
        return bios_name, bios_name.replace("T_", "B_")
    return bios_name, bios_name


def _resolve_noisy_partner_bios(df: pd.DataFrame, t_bios: str) -> str:
    """Prefer B_* partner, fallback to V_* if B_* not present."""
    bios_col = "biosample_name"
    bios_values = set(df[bios_col].dropna().astype(str).unique().tolist())
    b_candidate = t_bios.replace("T_", "B_")
    v_candidate = t_bios.replace("T_", "V_")
    if b_candidate in bios_values:
        return b_candidate
    if v_candidate in bios_values:
        return v_candidate
    return b_candidate


def _build_eic_assay_task_specs(df: pd.DataFrame, bios_name: str) -> List[Tuple[str, str, str]]:
    """
    Build (assay, task, bios_name) list from EIC pairing:
    - assays in T_* => denoise using T_* bios
    - assays in B_/V_* => impute using B_/V_* bios
    """
    bios_col = "biosample_name"
    assay_col = "assay_name"
    if bios_col not in df.columns or assay_col not in df.columns:
        raise ValueError("Metadata CSV must contain biosample_name and assay_name columns.")

    t_bios, b_bios = _resolve_eic_pair(bios_name)
    if t_bios.startswith("T_") and (b_bios.startswith("B_") or b_bios.startswith("V_")):
        if bios_name.startswith("T_"):
            b_bios = _resolve_noisy_partner_bios(df, t_bios)
    t_assays = set(
        df[df[bios_col].astype(str) == t_bios][assay_col].dropna().astype(str).unique().tolist()
    )
    b_assays = set(
        df[df[bios_col].astype(str) == b_bios][assay_col].dropna().astype(str).unique().tolist()
    )

    specs: List[Tuple[str, str, str]] = []
    for assay in sorted(t_assays):
        specs.append((assay, "denoise", t_bios))
    for assay in sorted(b_assays):
        specs.append((assay, "impute", b_bios))
    return specs


def _run_cmd(
    cmd: List[str], description: str, log_path: Path, env_overrides: Dict[str, str] | None = None
) -> int:
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Log: {log_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# {description}\n")
        logf.write("Command:\n")
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if env_overrides:
            env.update(env_overrides)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
        rc = int(proc.returncode)
        logf.write(f"\n[exit_code] {rc}\n")
    print(f"\nFinished: {description} (exit_code={rc})")
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all supertrack-specific evals/viz and save to model_dir/supertrack_evals."
    )
    parser.add_argument("--model-dir", required=True, help="Trained model directory.")
    parser.add_argument("--data-path", required=True, help="Dataset root path.")
    parser.add_argument("--dataset", required=True, choices=["eic", "merged"], help="Dataset type.")
    parser.add_argument("--bios-name", default=None, help="Optional biosample; auto-selected if omitted.")
    parser.add_argument("--assay-name", default=None, help="Optional assay; auto-selected if omitted.")
    parser.add_argument("--task", default="impute", choices=["impute", "denoise"], help="Task for sweep viz.")
    parser.add_argument(
        "--sweep-params",
        default="depth,read_length,run_type",
        help="Comma-separated sweep params for viz_supertrack.",
    )
    parser.add_argument("--prompt-spec", default=None, help="Optional prompt JSON path.")
    parser.add_argument("--dsf", type=int, default=1, help="Downsampling factor for viz_supertrack.")
    parser.add_argument(
        "--tracks-only",
        action="store_true",
        help="Run only supertrack signal-track visualization passes (skip smoking-gun and metric plots).",
    )
    parser.add_argument(
        "--pred-batch-size",
        type=int,
        default=16,
        help="Fixed prediction batch size for child runs (avoids repeated auto batch-size search).",
    )
    parser.add_argument("--smoking-dsf-list", default="1,2,4", help="DSF list for smoking-gun eval.")
    parser.add_argument("--x-fixed-dsf", type=int, default=1, help="Fixed x_dsf for smoking-gun y_metadata test.")
    parser.add_argument("--y-fixed-dsf", type=int, default=1, help="Fixed y_dsf for smoking-gun x_metadata test.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    repo_root = Path(__file__).resolve().parent
    out_root = model_dir / "supertrack_evals"
    logs_dir = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.prompt_spec is not None:
        prompt_spec = Path(args.prompt_spec).resolve()
    else:
        prompt_spec = repo_root / "prompts" / ("eic_mode.json" if args.dataset == "eic" else "merged_mode.json")

    if not prompt_spec.exists():
        raise FileNotFoundError(f"Prompt spec not found: {prompt_spec}")

    metadata_df = _read_metadata(repo_root, args.dataset)
    bios_name, assay_name = _choose_default_bios_and_assay(metadata_df, args.dataset)
    if args.bios_name is not None:
        bios_name = args.bios_name
    if args.assay_name is not None:
        assay_name = args.assay_name

    eval_scripts_dir = repo_root / "eval_scripts"
    smoke_script = eval_scripts_dir / "smoking_gun_supertrack.py"
    sweep_script = repo_root / "viz_supertrack.py"
    if not smoke_script.exists():
        raise FileNotFoundError(f"Missing script: {smoke_script}")
    if not sweep_script.exists():
        raise FileNotFoundError(f"Missing script: {sweep_script}")

    assay_task_specs: List[Tuple[str, str, str]]
    if args.dataset == "eic" and args.assay_name is None:
        assay_task_specs = _build_eic_assay_task_specs(metadata_df, bios_name)
        if len(assay_task_specs) == 0:
            raise ValueError(f"No EIC assays found for biosample pairing derived from {bios_name}.")
        print(f"Auto-selected {len(assay_task_specs)} assay/task/bios entries from EIC metadata:")
        for assay_i, task_i, bios_i in assay_task_specs:
            print(f"  - {assay_i} ({task_i}) on {bios_i}")
    else:
        assay_task_specs = [(assay_name, args.task, bios_name)]

    manifest: Dict[str, object] = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "model_dir": str(model_dir),
        "data_path": str(Path(args.data_path).resolve()),
        "dataset": args.dataset,
        "bios_name": bios_name,
        "assay_name": assay_name if args.assay_name is not None else None,
        "task": args.task if args.assay_name is not None else "auto_by_eic_availability",
        "assay_task_specs": [{"assay": a, "task": t, "bios_name": b} for a, t, b in assay_task_specs],
        "pred_batch_size": args.pred_batch_size,
        "prompt_spec": str(prompt_spec),
        "output_root": str(out_root),
        "runs": [],
    }
    env_overrides = {"CANDI_PRED_BATCH_SIZE": str(max(1, int(args.pred_batch_size)))}

    if not args.tracks_only:
        smoke_cmd = [
            sys.executable,
            str(smoke_script),
            "--model-dir",
            str(model_dir),
            "--data-path",
            str(Path(args.data_path).resolve()),
            "--dataset",
            args.dataset,
            "--bios-name",
            bios_name,
            "--dsf-list",
            args.smoking_dsf_list,
            "--x-fixed-dsf",
            str(args.x_fixed_dsf),
            "--y-fixed-dsf",
            str(args.y_fixed_dsf),
            "--output-dir",
            str(out_root / "smoking_gun"),
        ]
        smoke_log = logs_dir / "smoking_gun.log"
        smoke_rc = _run_cmd(smoke_cmd, "smoking_gun_supertrack", smoke_log, env_overrides=env_overrides)
        manifest["runs"].append(
            {
                "name": "smoking_gun_supertrack",
                "command": smoke_cmd,
                "exit_code": smoke_rc,
                "log": str(smoke_log),
            }
        )

    sweep_params = [x.strip() for x in args.sweep_params.split(",") if x.strip()]
    valid_params = set(DEFAULT_SWEEP_VALUES.keys())
    for p in sweep_params:
        if p not in valid_params:
            raise ValueError(f"Unsupported sweep param '{p}'. Must be one of: {sorted(valid_params)}")

    for assay_i, task_i, bios_i in assay_task_specs:
        for p in sweep_params:
            print(f"\nPreparing sweep step for: {assay_i} ({task_i}) on {bios_i} / {p}")
            sweep_cmd = [
                sys.executable,
                str(sweep_script),
                "--model-dir",
                str(model_dir),
                "--data-path",
                str(Path(args.data_path).resolve()),
                "--bios-name",
                bios_i,
                "--assay-name",
                assay_i,
                "--task",
                task_i,
                "--dataset",
                args.dataset,
                "--sweep-param",
                p,
                "--sweep-values",
                DEFAULT_SWEEP_VALUES[p],
                "--prompt-spec",
                str(prompt_spec),
                "--dsf",
                str(args.dsf),
                "--output-dir",
                str(out_root / "sweeps"),
            ]
            if args.tracks_only:
                sweep_cmd.extend(["--tracks-only", "--visual-loci-only"])
            safe_assay = assay_i.replace("/", "_").replace(" ", "_")
            safe_bios = bios_i.replace("/", "_").replace(" ", "_")
            sweep_log = logs_dir / f"viz_supertrack_{safe_bios}_{safe_assay}_{task_i}_{p}.log"
            sweep_rc = _run_cmd(
                sweep_cmd,
                f"viz_supertrack ({bios_i} | {assay_i} | {task_i} | {p})",
                sweep_log,
                env_overrides=env_overrides,
            )
            manifest["runs"].append(
                {
                    "name": f"viz_supertrack_{safe_bios}_{safe_assay}_{task_i}_{p}",
                    "command": sweep_cmd,
                    "exit_code": sweep_rc,
                    "log": str(sweep_log),
                }
            )

    run_exit_codes = [int(r["exit_code"]) for r in manifest["runs"]]
    manifest["finished_at"] = datetime.utcnow().isoformat() + "Z"
    manifest["all_success"] = all(code == 0 for code in run_exit_codes)

    manifest_path = out_root / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Supertrack eval outputs: {out_root}")
    print(f"Run manifest: {manifest_path}")
    print(f"All successful: {manifest['all_success']}")

    if not manifest["all_success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
