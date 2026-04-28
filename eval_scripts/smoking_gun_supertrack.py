#!/usr/bin/env python3
"""
Smoking-gun metadata sensitivity evaluation.

Runs two matched-DSF tests on one biosample over whole chr21:
1) X-metadata test: vary x_dsf, keep y_dsf fixed
2) Y-metadata test: vary y_dsf and matched y-prompt (prompt_dsf=y_dsf), keep x_dsf fixed

Outputs:
- model_dir/supertrack_SmokingGun/<bios_name>/smoking_gun_losses.csv
- model_dir/supertrack_SmokingGun/<bios_name>/smoking_gun_summary.csv
- model_dir/supertrack_SmokingGun/<bios_name>/smoking_gun_kde.png
- model_dir/supertrack_SmokingGun/<bios_name>/run_config.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt

# Make repo-root imports work when running as:
# python eval_scripts/smoking_gun_supertrack.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pred import CANDIPredictor
from _utils import METRICS, negative_binomial_loss, students_t_nll_loss, gamma_nll_loss
from model import LaplaceNLLLoss


def parse_dsf_list(s: str) -> List[int]:
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v < 1:
            raise ValueError("DSF values must be positive integers.")
        vals.append(v)
    if not vals:
        raise ValueError("Empty dsf list.")
    return vals


def get_full_chr21_locus(handler) -> List:
    chr21_len = int(handler.chr_sizes.get("chr21", 46709983))
    return ["chr21", 0, chr21_len]


def resolve_eic_x_y_bios(bios_name: str) -> Tuple[str, str]:
    """
    Resolve EIC paired biosamples:
      - B_* or V_* is target side (Y)
      - T_* is input side (X)
    """
    if bios_name.startswith("B_"):
        return bios_name.replace("B_", "T_"), bios_name
    if bios_name.startswith("V_"):
        return bios_name.replace("V_", "T_"), bios_name
    if bios_name.startswith("T_"):
        # If user passes T_ directly, infer paired B_ as target (test convention).
        return bios_name, bios_name.replace("T_", "B_")
    return bios_name, bios_name


def load_case_tensors(
    predictor: CANDIPredictor,
    dataset: str,
    bios_name: str,
    locus: List,
    x_dsf: int,
    y_dsf: int,
    prompt_dsf: int,
) -> Tuple[torch.Tensor, ...]:
    """
    Load tensors with explicit x/y DSF and matched y prompt.
    Returns:
      X_in, Y_gt, P_gt, Peak_gt, seq, mX_in, mY_prompt, avX_in, avY_count
    """
    h = predictor.data_handler
    if dataset == "eic":
        x_bios, y_bios = resolve_eic_x_y_bios(bios_name)
    else:
        x_bios, y_bios = bios_name, bios_name

    # X side (counts + metadata at x_dsf)
    temp_x, temp_mx = h.load_bios_Counts(x_bios, locus, DSF=x_dsf)
    X, mX, avX = h.make_bios_tensor_Counts(temp_x, temp_mx)
    del temp_x, temp_mx

    # Y side GT (counts + metadata at y_dsf)
    temp_y, temp_my = h.load_bios_Counts(y_bios, locus, DSF=y_dsf)
    Y, mY, avY = h.make_bios_tensor_Counts(temp_y, temp_my)
    del temp_y, temp_my

    # Y prompt metadata from prompt_dsf (matched in this experiment)
    temp_pmy_data, temp_pmy_meta = h.load_bios_Counts(y_bios, locus, DSF=prompt_dsf)
    _, mY_prompt, avY_prompt = h.make_bios_tensor_Counts(temp_pmy_data, temp_pmy_meta)
    del temp_pmy_data, temp_pmy_meta
    # Overwrite depth row where prompt metadata exists.
    prompt_mask = avY_prompt == 1
    mY[0, prompt_mask] = mY_prompt[0, prompt_mask]
    mY_prompt = mY

    # Signal/peak GT are DSF=1 by project constraint
    if dataset == "eic":
        temp_py = h.load_bios_BW(y_bios, locus)
        temp_px = h.load_bios_BW(x_bios, locus)
        temp_p = {**temp_py, **temp_px}
        P, avP = h.make_bios_tensor_BW(temp_p)
        del temp_py, temp_px, temp_p

        temp_peak_y = h.load_bios_Peaks(y_bios, locus)
        temp_peak_x = h.load_bios_Peaks(x_bios, locus)
        temp_peak = {**temp_peak_y, **temp_peak_x}
        Peak, avPeak = h.make_bios_tensor_Peaks(temp_peak)
        del temp_peak_y, temp_peak_x, temp_peak
    else:
        temp_p = h.load_bios_BW(y_bios, locus)
        P, avP = h.make_bios_tensor_BW(temp_p)
        del temp_p

        temp_peak = h.load_bios_Peaks(y_bios, locus)
        Peak, avPeak = h.make_bios_tensor_Peaks(temp_peak)
        del temp_peak

    # Control on input side uses x_dsf
    try:
        temp_control_data, temp_control_metadata = h.load_bios_Control(x_bios, locus, DSF=x_dsf)
        if temp_control_data and "chipseq-control" in temp_control_data:
            control_data, control_meta, control_avail = h.make_bios_tensor_Control(temp_control_data, temp_control_metadata)
        else:
            temp_control_data, temp_control_metadata = h.load_bios_Control(y_bios, locus, DSF=x_dsf)
            if temp_control_data and "chipseq-control" in temp_control_data:
                control_data, control_meta, control_avail = h.make_bios_tensor_Control(temp_control_data, temp_control_metadata)
            else:
                raise RuntimeError("Control missing in both X and Y biosamples.")
        del temp_control_data, temp_control_metadata
    except Exception:
        L = X.shape[0]
        control_data = torch.full((L, 1), -1.0)
        control_meta = torch.full((4, 1), -1.0)
        control_avail = torch.zeros(1)

    X = torch.cat([X, control_data], dim=1)
    mX = torch.cat([mX, control_meta], dim=1)
    avX = torch.cat([avX, control_avail], dim=0)

    # DNA
    seq = h._dna_to_onehot(h._get_DNA_sequence(locus[0], locus[1], locus[2]))

    # Truncate and reshape to context windows (same pattern as pred.py)
    num_rows = (X.shape[0] // predictor.context_length) * predictor.context_length
    X, Y, P, Peak = X[:num_rows, :], Y[:num_rows, :], P[:num_rows, :], Peak[:num_rows, :]
    seq = seq[: num_rows * h.resolution, :]

    X = X.view(-1, predictor.context_length, X.shape[-1])
    Y = Y.view(-1, predictor.context_length, Y.shape[-1])
    P = P.view(-1, predictor.context_length, P.shape[-1])
    Peak = Peak.view(-1, predictor.context_length, Peak.shape[-1])
    seq = seq.view(-1, predictor.context_length * h.resolution, seq.shape[-1])

    mX = mX.expand(X.shape[0], -1, -1)
    mY_prompt = mY_prompt.expand(Y.shape[0], -1, -1)
    avX = avX.expand(X.shape[0], -1)
    avY = avY.expand(Y.shape[0], -1)
    avP = avP.expand(P.shape[0], -1)
    avPeak = avPeak.expand(Peak.shape[0], -1)

    return X, Y, P, Peak, seq, mX, mY_prompt, avX, avY


def compute_signal_nll(dist_type: str, mu: torch.Tensor, var: torch.Tensor, df: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if dist_type in ["mse"]:
        return (mu - target) ** 2
    if dist_type in ["mae"]:
        return torch.abs(mu - target)
    if dist_type in ["laplace", "laplace_const"]:
        lap = LaplaceNLLLoss(reduction="none")
        return lap(mu, target, var)
    if dist_type == "studentst":
        return students_t_nll_loss(target, mu, var, df, reduction="none")
    if dist_type == "gamma":
        return gamma_nll_loss(target, mu, var, reduction="none")
    # gaussian / gaussian_const
    g = torch.nn.GaussianNLLLoss(reduction="none", full=True)
    v = torch.clamp(var, min=1e-6)
    return g(mu, target, v)


def compute_per_assay_losses(
    predictor: CANDIPredictor,
    metrics_obj: METRICS,
    outputs: Tuple[torch.Tensor, ...],
    X: torch.Tensor,
    Y: torch.Tensor,
    P: torch.Tensor,
    Peak: torch.Tensor,
    avX: torch.Tensor,
    avY_count: torch.Tensor,
    test_type: str,
    dsf_condition: int,
    assay_names: List[str],
) -> Tuple[List[Dict], List[Dict]]:
    p_pred, n_pred, mu_pred, var_pred, df_pred, peak_pred = outputs

    loss_records = []
    pearson_records = []
    B, L, F = Y.shape
    avX0 = avX[0, :F].detach().cpu().numpy()
    avY0 = avY_count[0, :F].detach().cpu().numpy()

    bce = torch.nn.BCELoss(reduction="none")

    def safe_metric(fn, y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
        try:
            if y_true_arr.size < 2 or y_pred_arr.size < 2:
                return np.nan
            return float(fn(y_true_arr, y_pred_arr))
        except Exception:
            return np.nan

    for a in range(F):
        has_x = avX0[a] > 0
        has_y = avY0[a] > 0
        if not has_x and not has_y:
            continue

        task_type = "denoised" if has_x else "imputed"
        assay_name = assay_names[a]

        # For denoised assays (present in X), use X counts as count GT.
        # For imputed assays, use Y counts as count GT.
        y = X[:, :, a] if has_x else Y[:, :, a]
        p = p_pred[:, :, a]
        n = n_pred[:, :, a]
        mu = mu_pred[:, :, a]
        var = var_pred[:, :, a]
        peak_hat = peak_pred[:, :, a]
        p_gt = P[:, :, a]
        peak_gt = Peak[:, :, a]
        df = df_pred[:, :, a] if df_pred is not None else None

        # Masks
        m_count = (y >= 0) & torch.isfinite(y) & torch.isfinite(p) & torch.isfinite(n)
        m_signal = (p_gt > -100) & torch.isfinite(p_gt) & torch.isfinite(mu) & torch.isfinite(var)
        m_peak = (peak_gt >= 0) & torch.isfinite(peak_gt) & torch.isfinite(peak_hat)

        nb_nll = np.nan
        signal_nll = np.nan
        peak_bce = np.nan

        if m_count.any():
            nb_elem = negative_binomial_loss(y[m_count], n[m_count], p[m_count])
            nb_nll = float(nb_elem.mean().item())
            p_safe = torch.clamp(p[m_count], min=1e-6, max=1.0 - 1e-6)
            nb_mean_pred = (n[m_count] * (1.0 - p_safe) / p_safe).detach().cpu().numpy().reshape(-1)
            nb_mean_true = y[m_count].detach().cpu().numpy().reshape(-1)
            pearson_records.append({
                "test_type": test_type,
                "task_type": task_type,
                "metric_family": "pearson_gw",
                "metric_target": "nb_mean",
                "dsf_condition": dsf_condition,
                "assay_name": assay_name,
                "metric_value": safe_metric(metrics_obj.pearson, nb_mean_true, nb_mean_pred),
            })
            pearson_records.append({
                "test_type": test_type,
                "task_type": task_type,
                "metric_family": "pearson_1obs",
                "metric_target": "nb_mean",
                "dsf_condition": dsf_condition,
                "assay_name": assay_name,
                "metric_value": safe_metric(metrics_obj.pearson1_obs, nb_mean_true, nb_mean_pred),
            })
        if m_signal.any():
            sig_elem = compute_signal_nll(predictor.dist_type, mu[m_signal], var[m_signal], df[m_signal] if df is not None else None, p_gt[m_signal])
            signal_nll = float(sig_elem.mean().item())
            signal_pred = mu[m_signal].detach().cpu().numpy().reshape(-1)
            signal_true = p_gt[m_signal].detach().cpu().numpy().reshape(-1)
            pearson_records.append({
                "test_type": test_type,
                "task_type": task_type,
                "metric_family": "pearson_gw",
                "metric_target": "signal_mean",
                "dsf_condition": dsf_condition,
                "assay_name": assay_name,
                "metric_value": safe_metric(metrics_obj.pearson, signal_true, signal_pred),
            })
            pearson_records.append({
                "test_type": test_type,
                "task_type": task_type,
                "metric_family": "pearson_1obs",
                "metric_target": "signal_mean",
                "dsf_condition": dsf_condition,
                "assay_name": assay_name,
                "metric_value": safe_metric(metrics_obj.pearson1_obs, signal_true, signal_pred),
            })
        if m_peak.any():
            peak_elem = bce(torch.clamp(peak_hat[m_peak].float(), 1e-6, 1 - 1e-6), peak_gt[m_peak].float())
            peak_bce = float(peak_elem.mean().item())

        loss_records.extend([
            {
                "test_type": test_type,
                "task_type": task_type,
                "loss_type": "nb_nll",
                "dsf_condition": dsf_condition,
                "assay_name": assay_name,
                "loss_value": nb_nll,
            },
            {
                "test_type": test_type,
                "task_type": task_type,
                "loss_type": "signal_nll",
                "dsf_condition": dsf_condition,
                "assay_name": assay_name,
                "loss_value": signal_nll,
            },
            {
                "test_type": test_type,
                "task_type": task_type,
                "loss_type": "peak_bce",
                "dsf_condition": dsf_condition,
                "assay_name": assay_name,
                "loss_value": peak_bce,
            },
        ])

    return loss_records, pearson_records


def plot_kde_panel(df: pd.DataFrame, output_path: Path, dsf_list: List[int]) -> None:
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(4, 3, figsize=(22, 18), sharex=False, sharey=False)
    row_specs = [
        ("x_metadata", "imputed"),
        ("x_metadata", "denoised"),
        ("y_metadata", "imputed"),
        ("y_metadata", "denoised"),
    ]
    col_specs = ["nb_nll", "signal_nll", "peak_bce"]
    palette = sns.color_palette("tab10", n_colors=max(3, len(dsf_list)))
    hue_order = [int(x) for x in sorted(set(df["dsf_condition"].dropna().astype(int).tolist()))]

    for r, (test_type, task_type) in enumerate(row_specs):
        for c, loss_type in enumerate(col_specs):
            ax = axes[r, c]
            sub = df[
                (df["test_type"] == test_type) &
                (df["task_type"] == task_type) &
                (df["loss_type"] == loss_type)
            ].copy()
            sub = sub[np.isfinite(sub["loss_value"].to_numpy())]
            ax.set_title(f"{test_type} | {task_type} | {loss_type}")
            if len(sub) < 2:
                ax.text(0.5, 0.5, "insufficient samples", ha="center", va="center", transform=ax.transAxes)
                continue

            per_dsf_n = []
            for d in hue_order:
                dsub = sub[sub["dsf_condition"].astype(int) == d]
                per_dsf_n.append(len(dsub))
            can_kde = len(sub) >= 15 and all(n >= 5 for n in per_dsf_n if n > 0)

            plotted = False
            if can_kde:
                for i, d in enumerate(hue_order):
                    dsub = sub[sub["dsf_condition"].astype(int) == d]
                    if len(dsub) < 2:
                        continue
                    sns.kdeplot(
                        data=dsub,
                        x="loss_value",
                        ax=ax,
                        label=f"DSF{d}",
                        color=palette[i % len(palette)],
                        fill=False,
                        linewidth=2.0,
                        common_norm=False,
                    )
                    plotted = True
            else:
                # Fallback for sparse groups: histogram for n>=2, single bar for n==1.
                sub_min = float(sub["loss_value"].min())
                sub_max = float(sub["loss_value"].max())
                bar_w = max((sub_max - sub_min) / 50.0, 1e-3)
                for i, d in enumerate(hue_order):
                    dsub = sub[sub["dsf_condition"].astype(int) == d]
                    n = len(dsub)
                    if n >= 2:
                        bins = max(3, min(10, int(np.sqrt(n)) + 1))
                        sns.histplot(
                            data=dsub,
                            x="loss_value",
                            bins=bins,
                            ax=ax,
                            color=palette[i % len(palette)],
                            stat="count",
                            element="step",
                            fill=False,
                            label=f"DSF{d} (hist)",
                        )
                        plotted = True
                    elif n == 1:
                        v = float(dsub["loss_value"].iloc[0])
                        ax.bar(v, 1.0, width=bar_w, color=palette[i % len(palette)], alpha=0.7, label=f"DSF{d} (single)")
                        plotted = True

            if plotted:
                ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "insufficient samples per DSF", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_panel(metric_df: pd.DataFrame, metric_family: str, output_path: Path, dsf_list: List[int]) -> None:
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(18, 18), sharex=False, sharey=False)
    row_specs = [
        ("x_metadata", "imputed"),
        ("x_metadata", "denoised"),
        ("y_metadata", "imputed"),
        ("y_metadata", "denoised"),
    ]
    col_specs = ["nb_mean", "signal_mean"]
    palette = sns.color_palette("tab10", n_colors=max(3, len(dsf_list)))
    hue_order = [int(x) for x in sorted(set(metric_df["dsf_condition"].dropna().astype(int).tolist()))]

    for r, (test_type, task_type) in enumerate(row_specs):
        for c, metric_target in enumerate(col_specs):
            ax = axes[r, c]
            sub = metric_df[
                (metric_df["metric_family"] == metric_family) &
                (metric_df["test_type"] == test_type) &
                (metric_df["task_type"] == task_type) &
                (metric_df["metric_target"] == metric_target)
            ].copy()
            sub = sub[np.isfinite(sub["metric_value"].to_numpy())]
            ax.set_title(f"{metric_family} | {test_type} | {task_type} | {metric_target}")
            if len(sub) < 1:
                ax.text(0.5, 0.5, "insufficient samples", ha="center", va="center", transform=ax.transAxes)
                continue

            per_dsf_n = []
            for d in hue_order:
                dsub = sub[sub["dsf_condition"].astype(int) == d]
                per_dsf_n.append(len(dsub))
            can_kde = len(sub) >= 15 and all(n >= 5 for n in per_dsf_n if n > 0)

            plotted = False
            if can_kde:
                for i, d in enumerate(hue_order):
                    dsub = sub[sub["dsf_condition"].astype(int) == d]
                    if len(dsub) < 2:
                        continue
                    sns.kdeplot(
                        data=dsub,
                        x="metric_value",
                        ax=ax,
                        label=f"DSF{d}",
                        color=palette[i % len(palette)],
                        fill=False,
                        linewidth=2.0,
                        common_norm=False,
                    )
                    plotted = True
            else:
                sub_min = float(sub["metric_value"].min())
                sub_max = float(sub["metric_value"].max())
                bar_w = max((sub_max - sub_min) / 50.0, 1e-3)
                for i, d in enumerate(hue_order):
                    dsub = sub[sub["dsf_condition"].astype(int) == d]
                    n = len(dsub)
                    if n >= 2:
                        bins = max(3, min(10, int(np.sqrt(n)) + 1))
                        sns.histplot(
                            data=dsub,
                            x="metric_value",
                            bins=bins,
                            ax=ax,
                            color=palette[i % len(palette)],
                            stat="count",
                            element="step",
                            fill=False,
                            label=f"DSF{d} (hist)",
                        )
                        plotted = True
                    elif n == 1:
                        v = float(dsub["metric_value"].iloc[0])
                        ax.bar(v, 1.0, width=bar_w, color=palette[i % len(palette)], alpha=0.7, label=f"DSF{d} (single)")
                        plotted = True

            ax.set_xlim(-1.05, 1.05)
            ax.set_ylabel("density/count")
            ax.set_xlabel("Pearson")
            if plotted:
                ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "insufficient samples per DSF", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_smoking_gun(
    model_dir: str,
    data_path: str,
    dataset: str,
    bios_name: str,
    dsf_list: List[int],
    x_fixed_dsf: int,
    y_fixed_dsf: int,
    locus: List = None,
    output_dir: str = None,
):
    t0 = time.time()
    print("[smoking-gun] initializing predictor...", flush=True)
    predictor = CANDIPredictor(model_dir=model_dir, DNA=True)
    print("[smoking-gun] setting up data handler...", flush=True)
    predictor.setup_data_handler(data_path=data_path, dataset_type=dataset, split="test")

    if locus is None:
        locus = get_full_chr21_locus(predictor.data_handler)

    if dataset == "eic":
        x_bios, y_bios = resolve_eic_x_y_bios(bios_name)
        print(f"[EIC pairing] X biosample: {x_bios} | Y biosample: {y_bios}")
    else:
        x_bios, y_bios = bios_name, bios_name

    if output_dir is None:
        out_dir = Path(model_dir) / "supertrack_SmokingGun" / bios_name
    else:
        out_dir = Path(output_dir) / bios_name
    out_dir.mkdir(parents=True, exist_ok=True)

    assay_names = list(predictor.data_handler.aliases["experiment_aliases"].keys())
    metrics_obj = METRICS(chrom="chr21", bin_size=25)
    all_records: List[Dict] = []
    pearson_records: List[Dict] = []
    skip_counts = {"x_metadata": {}, "y_metadata": {}}

    # X-metadata test: vary x_dsf, fixed y target/prompt at y_fixed_dsf
    print(f"[smoking-gun] starting x_metadata sweep over DSFs={dsf_list}", flush=True)
    for x_dsf in dsf_list:
        tx = time.time()
        print(f"[smoking-gun] x_metadata DSF={x_dsf}: loading tensors...", flush=True)
        X, Y, P, Peak, seq, mX, mY_prompt, avX, avY = load_case_tensors(
            predictor, dataset, bios_name, locus, x_dsf=x_dsf, y_dsf=y_fixed_dsf, prompt_dsf=y_fixed_dsf
        )
        skip_counts["x_metadata"][str(x_dsf)] = int((avY[0, :Y.shape[-1]] <= 0).sum().item())
        print(f"[smoking-gun] x_metadata DSF={x_dsf}: running prediction...", flush=True)
        outputs = predictor.predict(X, mX, mY_prompt, avX, seq, imp_target=[])
        recs, pears = compute_per_assay_losses(
            predictor, metrics_obj, outputs, X, Y, P, Peak, avX, avY,
            test_type="x_metadata", dsf_condition=x_dsf, assay_names=assay_names
        )
        all_records.extend(recs)
        pearson_records.extend(pears)
        print(f"[smoking-gun] x_metadata DSF={x_dsf}: done in {time.time() - tx:.1f}s", flush=True)

    # Y-metadata test (matched): fixed x at x_fixed_dsf, vary y_dsf and prompt_dsf together
    print(f"[smoking-gun] starting y_metadata sweep over DSFs={dsf_list}", flush=True)
    for d in dsf_list:
        ty = time.time()
        print(f"[smoking-gun] y_metadata DSF={d}: loading tensors...", flush=True)
        assert d == int(d), "Prompt/GT DSF mismatch in y_metadata test."
        X, Y, P, Peak, seq, mX, mY_prompt, avX, avY = load_case_tensors(
            predictor, dataset, bios_name, locus, x_dsf=x_fixed_dsf, y_dsf=d, prompt_dsf=d
        )
        skip_counts["y_metadata"][str(d)] = int((avY[0, :Y.shape[-1]] <= 0).sum().item())
        print(f"[smoking-gun] y_metadata DSF={d}: running prediction...", flush=True)
        outputs = predictor.predict(X, mX, mY_prompt, avX, seq, imp_target=[])
        recs, pears = compute_per_assay_losses(
            predictor, metrics_obj, outputs, X, Y, P, Peak, avX, avY,
            test_type="y_metadata", dsf_condition=d, assay_names=assay_names
        )
        all_records.extend(recs)
        pearson_records.extend(pears)
        print(f"[smoking-gun] y_metadata DSF={d}: done in {time.time() - ty:.1f}s", flush=True)

    df = pd.DataFrame(all_records)
    losses_csv = out_dir / "smoking_gun_losses.csv"
    df.to_csv(losses_csv, index=False)

    summary = (
        df[np.isfinite(df["loss_value"].to_numpy())]
        .groupby(["test_type", "task_type", "loss_type", "dsf_condition"])
        .agg(
            n=("loss_value", "size"),
            median=("loss_value", "median"),
            q25=("loss_value", lambda s: np.quantile(s, 0.25)),
            q75=("loss_value", lambda s: np.quantile(s, 0.75)),
        )
        .reset_index()
    )
    summary_csv = out_dir / "smoking_gun_summary.csv"
    summary.to_csv(summary_csv, index=False)

    fig_path = out_dir / "smoking_gun_kde.png"
    plot_kde_panel(df, fig_path, dsf_list=dsf_list)

    pearson_df = pd.DataFrame(pearson_records)
    pearson_csv = out_dir / "smoking_gun_pearson_metrics.csv"
    pearson_df.to_csv(pearson_csv, index=False)
    pearson_gw_fig = out_dir / "smoking_gun_pearson_gw.png"
    pearson_1obs_fig = out_dir / "smoking_gun_pearson_1obs.png"
    plot_metric_panel(pearson_df, "pearson_gw", pearson_gw_fig, dsf_list=dsf_list)
    plot_metric_panel(pearson_df, "pearson_1obs", pearson_1obs_fig, dsf_list=dsf_list)

    run_cfg = {
        "model_dir": str(model_dir),
        "data_path": str(data_path),
        "dataset": dataset,
        "bios_name": bios_name,
        "resolved_x_bios_name": x_bios,
        "resolved_y_bios_name": y_bios,
        "locus": locus,
        "dsf_list": dsf_list,
        "x_fixed_dsf": x_fixed_dsf,
        "y_fixed_dsf": y_fixed_dsf,
        "y_metadata_matched": True,
        "skipped_assays_per_condition": skip_counts,
        "artifacts": {
            "losses_csv": str(losses_csv),
            "summary_csv": str(summary_csv),
            "pearson_csv": str(pearson_csv),
            "figure_png": str(fig_path),
            "pearson_gw_figure_png": str(pearson_gw_fig),
            "pearson_1obs_figure_png": str(pearson_1obs_fig),
        },
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"Saved smoking-gun artifacts to: {out_dir}")
    print(f"- {losses_csv.name}")
    print(f"- {summary_csv.name}")
    print(f"- {fig_path.name}")
    print(f"- {pearson_csv.name}")
    print(f"- {pearson_gw_fig.name}")
    print(f"- {pearson_1obs_fig.name}")
    print(f"[smoking-gun] total elapsed: {time.time() - t0:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Smoking-gun metadata sensitivity test on whole chr21.")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory.")
    parser.add_argument("--data-path", required=True, help="Path to dataset root.")
    parser.add_argument("--dataset", required=True, choices=["eic", "merged"], help="Dataset type.")
    parser.add_argument("--bios-name", required=True, help="Specific biosample to evaluate.")
    parser.add_argument("--dsf-list", type=str, default="1,2,4", help="Comma-separated DSFs, e.g. 1,2,4")
    parser.add_argument("--x-fixed-dsf", type=int, default=1, help="Fixed x_dsf for y_metadata test.")
    parser.add_argument("--y-fixed-dsf", type=int, default=1, help="Fixed y_dsf for x_metadata test.")
    parser.add_argument("--locus", type=str, nargs=3, default=None, help='Optional locus: "chr21 0 46709983"')
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output root (default: model_dir/supertrack_SmokingGun)")
    args = parser.parse_args()

    dsf_list = parse_dsf_list(args.dsf_list)
    locus = None
    if args.locus is not None:
        locus = [args.locus[0], int(args.locus[1]), int(args.locus[2])]

    run_smoking_gun(
        model_dir=args.model_dir,
        data_path=args.data_path,
        dataset=args.dataset,
        bios_name=args.bios_name,
        dsf_list=dsf_list,
        x_fixed_dsf=args.x_fixed_dsf,
        y_fixed_dsf=args.y_fixed_dsf,
        locus=locus,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
