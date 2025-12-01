#!/usr/bin/env python3
"""
Run All Evaluation Scripts

This script orchestrates running all evaluation scripts in the correct order
for a given model directory and dataset type.

Usage:
    python eval_scripts/run_all.py --model-dir models/my_model/ --dataset merged
    python eval_scripts/run_all.py --model-dir models/my_model/ --dataset eic
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Hardcoded data paths
DATA_PATHS = {
    'merged': '/project/6014832/mforooz/DATA_CANDI_MERGED',
    'eic': '/project/6014832/mforooz/DATA_CANDI_EIC'
}


def run_command(cmd, description):
    """
    Run a shell command and handle errors.
    
    Args:
        cmd: List of command arguments
        description: Description of what the command does
    """
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ Completed: {description}")
        return True


def main():
    """Main function to orchestrate all evaluation scripts."""
    parser = argparse.ArgumentParser(
        description="Run all evaluation scripts in order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python eval_scripts/run_all.py --model-dir models/my_model/ --dataset merged
    python eval_scripts/run_all.py --model-dir models/my_model/ --dataset eic
        """
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Path to model directory'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['merged', 'eic'],
        help='Dataset type: merged or eic'
    )
    
    args = parser.parse_args()
    
    # Validate model directory
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory does not exist: {model_dir}")
        sys.exit(1)
    
    # Get data path based on dataset
    data_path = DATA_PATHS[args.dataset]
    
    # Get script directories
    eval_scripts_dir = Path(__file__).parent  # eval_scripts/
    project_root = eval_scripts_dir.parent     # project root
    
    # Track if any step failed
    failed_steps = []
    
    # Step 1: viz_training_progress.py
    cmd = [
        sys.executable,
        str(eval_scripts_dir / "viz_training_progress.py"),
        "--model-dir", str(model_dir)
    ]
    if not run_command(cmd, "Training progress visualization"):
        failed_steps.append("viz_training_progress.py")
    
    # Step 2: pred.py (all biosamples)
    cmd = [
        sys.executable,
        str(project_root / "pred.py"),
        "--model-dir", str(model_dir),
        "--data-path", data_path,
        "--dataset", args.dataset,
        "--all-biosamples"
    ]
    if not run_command(cmd, "Generate predictions for all biosamples"):
        failed_steps.append("pred.py")
    
    # Step 3: compute_metrics.py
    cmd = [
        sys.executable,
        str(eval_scripts_dir / "compute_metrics.py"),
        "--model-dir", str(model_dir),
        "--dataset", args.dataset
    ]
    if not run_command(cmd, "Compute metrics from predictions"):
        failed_steps.append("compute_metrics.py")
    
    # Step 4: viz_pred_perf.py
    cmd = [
        sys.executable,
        str(eval_scripts_dir / "viz_pred_perf.py"),
        "--model-dir", str(model_dir)
    ]
    if not run_command(cmd, "Visualize prediction performance"):
        failed_steps.append("viz_pred_perf.py")
    
    # Step 5: viz_scatter_density.py
    cmd = [
        sys.executable,
        str(eval_scripts_dir / "viz_scatter_density.py"),
        "--model-dir", str(model_dir)
    ]
    if not run_command(cmd, "Scatter density plots"):
        failed_steps.append("viz_scatter_density.py")
    
    # Step 6: viz_conf.py
    cmd = [
        sys.executable,
        str(eval_scripts_dir / "viz_conf.py"),
        "--model-dir", str(model_dir),
        "--dataset", args.dataset
    ]
    if not run_command(cmd, "Confidence calibration visualization"):
        failed_steps.append("viz_conf.py")
    
    # Special handling: viz_rnaseq.py (always uses merged dataset)
    print(f"\n{'='*80}")
    print("Running dataset-specific scripts...")
    print(f"{'='*80}")
    
    merged_data_path = DATA_PATHS['merged']
    cmd = [
        sys.executable,
        str(eval_scripts_dir / "viz_rnaseq.py"),
        "--model-dir", str(model_dir),
        "--data-path", merged_data_path
    ]
    if not run_command(cmd, "RNA-seq evaluation (always uses merged dataset)"):
        failed_steps.append("viz_rnaseq.py")
    
    # Special handling: viz_eic_bench.py (always uses eic dataset)
    # If user specified merged dataset, we need to compute metrics for eic first
    if args.dataset == 'merged':
        print(f"\n{'='*80}")
        print("Computing EIC metrics for viz_eic_bench.py...")
        print(f"{'='*80}")
        
        eic_metrics_path = model_dir / "preds" / "metrics_eic.csv"
        cmd = [
            sys.executable,
            str(eval_scripts_dir / "compute_metrics.py"),
            "--model-dir", str(model_dir),
            "--dataset", "eic",
            "--output", str(eic_metrics_path)
        ]
        if not run_command(cmd, "Compute EIC metrics for benchmark comparison"):
            print("Warning: Failed to compute EIC metrics. Skipping viz_eic_bench.py")
            failed_steps.append("compute_metrics.py (EIC)")
        else:
            # Run viz_eic_bench.py with the EIC metrics
            cmd = [
                sys.executable,
                str(eval_scripts_dir / "viz_eic_bench.py"),
                "--candi-metrics", str(eic_metrics_path),
                "--output-dir", str(model_dir / "viz")
            ]
            if not run_command(cmd, "EIC benchmark visualization"):
                failed_steps.append("viz_eic_bench.py")
    else:
        # User specified eic dataset, so metrics.csv should already be for eic
        metrics_path = model_dir / "preds" / "metrics.csv"
        cmd = [
            sys.executable,
            str(eval_scripts_dir / "viz_eic_bench.py"),
            "--candi-metrics", str(metrics_path),
            "--output-dir", str(model_dir / "viz")
        ]
        if not run_command(cmd, "EIC benchmark visualization"):
            failed_steps.append("viz_eic_bench.py")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if failed_steps:
        print(f"⚠️  Some steps failed:")
        for step in failed_steps:
            print(f"   - {step}")
        sys.exit(1)
    else:
        print("✅ All evaluation scripts completed successfully!")
        print(f"   Results saved to: {model_dir}/viz/")
        print(f"   Metrics saved to: {model_dir}/preds/metrics.csv")


if __name__ == "__main__":
    main()

