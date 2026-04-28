#!/usr/bin/env python3
"""
Lightweight, self-contained supertrack visualization for EIC.

Overview
--------
This script provides a lightweight and auditable alternative to the larger
supertrack orchestration flow. It runs direct model inference and generates
track figures for metadata sweep analysis.

What it does
------------
1) Loads a trained CANDI model checkpoint.
2) Resolves paired EIC biosamples:
   - T_* biosample is used as input-side source for denoise rows.
   - B_*/V_* biosample is used as GT source for impute rows.
3) Loads a default prompt spec from prompts/eic_mode.json.
4) For each sweep field/value, applies a global prompt mutation to y_metadata.
5) Runs one full forward pass per (locus, sweep value) and slices assay outputs.
6) Produces one figure per sweep field:
   - columns: loci
   - rows: GT-available assays (denoise + impute)
   - sections per row group: count / pval(signal) / peak.

Decoder compatibility
---------------------
- Works with both fixed and query-based decoders.
- Inference intentionally omits query masks so model follows default full-F path.

Quick start
-----------
Run all three fields:

  python lightweight_supertrack_viz.py \
    --model-dir models/<your_model_dir> \
    --data-path /project/6014832/mforooz/DATA_CANDI_EIC \
    --dataset eic \
    --bios-name B_BE2C \
    --pred-batch-size 8 \
    --output-dir models/<your_model_dir>/supertrack_evals/lightweight_tracks

Run a single field:

  python lightweight_supertrack_viz.py \
    --model-dir models/<your_model_dir> \
    --data-path /project/6014832/mforooz/DATA_CANDI_EIC \
    --dataset eic \
    --bios-name B_BE2C \
    --fields run_type

Expected outputs
----------------
- lightweight_tracks_depth.png / .svg
- lightweight_tracks_read_length.png / .svg
- lightweight_tracks_run_type.png / .svg
- lightweight_supertrack_manifest.json
- lightweight_supertrack_warnings.log (only if warnings exist)

Notes
-----
- This script is EIC-only by design (`--dataset eic`).
- Prompt row semantics are preserved as:
  [depth_log2, assay_idx, read_length, run_type].
- If any paired biosample/control/locus issue occurs, the script fails fast
  or records explicit warnings in the manifest/log.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from _utils import NegativeBinomial
from pred import CANDIPredictor


DEFAULT_SWEEPS = {
    "depth": [1.0e7, 3.0e7, 5.0e7, 1.0e8],
    "read_length": [36.0, 50.0, 75.0, 100.0],
    "run_type": ["single-ended", "paired-ended"],
}

NAMED_LOCI = {
    "example_genes": [
        ("GART", "chr21", 33481539, 33588914),
        ("APP", "chr21", 25800151, 26235914),
        ("SOD1", "chr21", 31589009, 31745788),
        ("B3GALT5", "chr21", 39526359, 39802081),
        ("ITSN1", "chr21", 33577551, 33919338),
    ]
}


@dataclass
class LocusBundle:
    name: str
    chrom: str
    start_bp: int
    end_bp: int
    x_bp: np.ndarray
    X: torch.Tensor
    mX: torch.Tensor
    avX: torch.Tensor
    seq: torch.Tensor | None
    base_mY_2d: torch.Tensor
    gt_count_T: np.ndarray
    gt_pval_T: np.ndarray
    gt_peak_T: np.ndarray
    gt_count_B: np.ndarray
    gt_pval_B: np.ndarray
    gt_peak_B: np.ndarray
    avY_T: np.ndarray
    avY_B: np.ndarray


def _resolve_eic_pair(
    bios_name: str, navigation_keys: List[str]
) -> Tuple[str, str]:
    nav_set = set(navigation_keys)
    if bios_name.startswith("T_"):
        t_bios = bios_name
        b_candidate = bios_name.replace("T_", "B_")
        v_candidate = bios_name.replace("T_", "V_")
        if b_candidate in nav_set:
            return t_bios, b_candidate
        if v_candidate in nav_set:
            return t_bios, v_candidate
        raise ValueError(f"No B_/V_ pair found for {bios_name}.")

    if bios_name.startswith("B_") or bios_name.startswith("V_"):
        t_bios = bios_name.replace("B_", "T_").replace("V_", "T_")
        if t_bios not in nav_set:
            raise ValueError(f"No T_* pair found for {bios_name}.")
        return t_bios, bios_name

    raise ValueError(f"Unexpected EIC biosample name format: {bios_name}")


def _run_type_to_id(value: str) -> int:
    return 1 if "pair" in str(value).lower() else 0


def _flatten_blf_to_nf(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(-1, t.shape[-1]).detach().cpu().numpy()


def _load_prompt_spec(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _enforce_assay_ids(mY_2d: torch.Tensor) -> torch.Tensor:
    num_assays = mY_2d.shape[-1]
    mY_2d[1, :] = torch.arange(num_assays, dtype=mY_2d.dtype)
    return mY_2d


def _load_locus_bundle(
    predictor: CANDIPredictor,
    t_bios: str,
    b_bios: str,
    locus_name: str,
    chrom: str,
    start_bp: int,
    end_bp: int,
    dsf: int,
    baseline_spec: Dict,
    warnings: List[str],
) -> LocusBundle:
    dh = predictor.data_handler
    assert dh is not None
    locus = [chrom, int(start_bp), int(end_bp)]

    # Input comes from T_* with requested DSF
    temp_x, temp_mx = dh.load_bios_Counts(t_bios, locus, dsf)
    X, mX, avX = dh.make_bios_tensor_Counts(temp_x, temp_mx)
    del temp_x, temp_mx

    # T_* GT at DSF=1 (denoise rows)
    temp_y_t, temp_my_t = dh.load_bios_Counts(t_bios, locus, 1)
    Y_T, mY_T, avY_T = dh.make_bios_tensor_Counts(temp_y_t, temp_my_t)
    del temp_y_t, temp_my_t

    # B_/V_* GT at DSF=1 (impute rows)
    temp_y_b, temp_my_b = dh.load_bios_Counts(b_bios, locus, 1)
    Y_B, mY_B, avY_B = dh.make_bios_tensor_Counts(temp_y_b, temp_my_b)
    del temp_y_b, temp_my_b

    # Signal/peak GT
    temp_p_t = dh.load_bios_BW(t_bios, locus)
    P_T, _ = dh.make_bios_tensor_BW(temp_p_t)
    del temp_p_t
    temp_p_b = dh.load_bios_BW(b_bios, locus)
    P_B, _ = dh.make_bios_tensor_BW(temp_p_b)
    del temp_p_b

    temp_peak_t = dh.load_bios_Peaks(t_bios, locus)
    Peak_T, _ = dh.make_bios_tensor_Peaks(temp_peak_t)
    del temp_peak_t
    temp_peak_b = dh.load_bios_Peaks(b_bios, locus)
    Peak_B, _ = dh.make_bios_tensor_Peaks(temp_peak_b)
    del temp_peak_b

    # Control from T_ first, fallback to B_/V_
    try:
        ctrl_data, ctrl_meta = dh.load_bios_Control(t_bios, locus, dsf)
        control_data, control_mx, control_av = dh.make_bios_tensor_Control(ctrl_data, ctrl_meta)
        if float(control_av.item()) != 1.0:
            raise ValueError("No control found in T_*")
    except Exception:
        ctrl_data, ctrl_meta = dh.load_bios_Control(b_bios, locus, dsf)
        control_data, control_mx, control_av = dh.make_bios_tensor_Control(ctrl_data, ctrl_meta)
        if float(control_av.item()) != 1.0:
            msg = f"[{locus_name}] control track missing in both {t_bios} and {b_bios}; using sentinel control."
            warnings.append(msg)
            L = X.shape[0]
            control_data = torch.full((L, 1), -1.0)
            control_mx = torch.full((4, 1), -1.0)
            control_av = torch.zeros(1)

    X = torch.cat([X, control_data], dim=1)
    mX = torch.cat([mX, control_mx], dim=1)
    avX = torch.cat([avX, control_av], dim=0)

    # DNA sequence
    seq = None
    if predictor.DNA:
        seq = dh._dna_to_onehot(dh._get_DNA_sequence(chrom, start_bp, end_bp))

    # Truncate all tracks to aligned context windows
    candidate_rows = [
        X.shape[0],
        Y_T.shape[0],
        Y_B.shape[0],
        P_T.shape[0],
        P_B.shape[0],
        Peak_T.shape[0],
        Peak_B.shape[0],
    ]
    num_rows = min(candidate_rows)
    num_rows = (num_rows // predictor.context_length) * predictor.context_length
    if num_rows <= 0:
        raise ValueError(f"[{locus_name}] no rows after context alignment.")

    X = X[:num_rows, :]
    Y_T = Y_T[:num_rows, :]
    Y_B = Y_B[:num_rows, :]
    P_T = P_T[:num_rows, :]
    P_B = P_B[:num_rows, :]
    Peak_T = Peak_T[:num_rows, :]
    Peak_B = Peak_B[:num_rows, :]
    if seq is not None:
        seq = seq[: num_rows * predictor.data_handler.resolution, :]

    X = X.view(-1, predictor.context_length, X.shape[-1])
    Y_T = Y_T.view(-1, predictor.context_length, Y_T.shape[-1])
    Y_B = Y_B.view(-1, predictor.context_length, Y_B.shape[-1])
    P_T = P_T.view(-1, predictor.context_length, P_T.shape[-1])
    P_B = P_B.view(-1, predictor.context_length, P_B.shape[-1])
    Peak_T = Peak_T.view(-1, predictor.context_length, Peak_T.shape[-1])
    Peak_B = Peak_B.view(-1, predictor.context_length, Peak_B.shape[-1])
    if seq is not None:
        seq = seq.view(-1, predictor.context_length * predictor.data_handler.resolution, seq.shape[-1])

    mX = mX.expand(X.shape[0], -1, -1)
    avX = avX.expand(X.shape[0], -1)

    # Base prompt metadata from default prompt file, fallback to unified observed metadata.
    mY_base = mY_B.clone()
    missing = mY_base == -1
    mY_base[missing] = mY_T[missing]
    mY_base = dh.fill_in_prompt_manual(mY_base, baseline_spec, overwrite=True)
    mY_base = _enforce_assay_ids(mY_base)

    # Convert GT signals to plotting space.
    gt_count_t = _flatten_blf_to_nf(Y_T)
    gt_count_b = _flatten_blf_to_nf(Y_B)
    gt_pval_t = predictor.inverse_transform(_flatten_blf_to_nf(P_T))
    gt_pval_b = predictor.inverse_transform(_flatten_blf_to_nf(P_B))
    gt_peak_t = _flatten_blf_to_nf(Peak_T)
    gt_peak_b = _flatten_blf_to_nf(Peak_B)

    n_bins = gt_count_t.shape[0]
    x_bp = start_bp + np.arange(n_bins, dtype=np.int64) * predictor.data_handler.resolution

    return LocusBundle(
        name=locus_name,
        chrom=chrom,
        start_bp=start_bp,
        end_bp=end_bp,
        x_bp=x_bp,
        X=X,
        mX=mX,
        avX=avX,
        seq=seq,
        base_mY_2d=mY_base,
        gt_count_T=gt_count_t,
        gt_pval_T=gt_pval_t,
        gt_peak_T=gt_peak_t,
        gt_count_B=gt_count_b,
        gt_pval_B=gt_pval_b,
        gt_peak_B=gt_peak_b,
        avY_T=avY_T.detach().cpu().numpy(),
        avY_B=avY_B.detach().cpu().numpy(),
    )


def _build_row_specs(bundle: LocusBundle, expnames: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for idx, assay in enumerate(expnames):
        if idx < len(bundle.avY_T) and int(bundle.avY_T[idx]) == 1:
            rows.append({"task": "denoise", "gt_source": "T", "idx": idx, "label": f"D:{assay}"})
    for idx, assay in enumerate(expnames):
        if idx < len(bundle.avY_B) and int(bundle.avY_B[idx]) == 1:
            rows.append({"task": "impute", "gt_source": "B", "idx": idx, "label": f"I:{assay}"})
    return rows


def _apply_global_sweep(mY_2d: torch.Tensor, field: str, value) -> torch.Tensor:
    out = mY_2d.clone()
    if field == "depth":
        out[0, :] = float(np.log2(float(value)))
    elif field == "read_length":
        out[2, :] = float(value)
    elif field == "run_type":
        out[3, :] = float(_run_type_to_id(str(value)))
    else:
        raise ValueError(f"Unsupported sweep field: {field}")
    return out


def _run_forward_all_assays(
    predictor: CANDIPredictor, bundle: LocusBundle, mY_2d: torch.Tensor
) -> Dict[str, np.ndarray]:
    B = bundle.X.shape[0]
    mY = mY_2d.unsqueeze(0).repeat(B, 1, 1)
    n, p, mu, _, _, peak = predictor.predict(
        bundle.X, bundle.mX, mY, bundle.avX, bundle.seq, imp_target=[]
    )
    n_nf = n.contiguous().view(-1, n.shape[-1])
    p_nf = p.contiguous().view(-1, p.shape[-1])
    mu_nf = mu.contiguous().view(-1, mu.shape[-1])
    peak_nf = peak.contiguous().view(-1, peak.shape[-1])

    count_pred = NegativeBinomial(p_nf, n_nf).mean().detach().cpu().numpy()
    pval_pred = predictor.inverse_transform(mu_nf).detach().cpu().numpy()
    peak_pred = peak_nf.detach().cpu().numpy()

    return {"pred_count": count_pred, "pred_pval": pval_pred, "pred_peak": peak_pred}


def _plot_field(
    field: str,
    sweep_values: List,
    cache: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    bundles: Dict[str, LocusBundle],
    row_specs: List[Dict],
    output_dir: Path,
) -> Dict[str, str]:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.55,
            "grid.color": "#B8BCC4",
            "grid.linewidth": 0.35,
            "font.size": 7,
        }
    )

    def _fmt_sweep(v) -> str:
        if isinstance(v, float):
            if abs(v) >= 1e5:
                return f"{v:.0e}"
            if float(v).is_integer():
                return f"{int(v)}"
            return f"{v:.2f}"
        return str(v)

    loci = list(bundles.keys())
    n_loci = len(loci)
    n_rows = len(row_specs)
    total_rows = n_rows * 3  # count, pval, peak sections

    fig, axes = plt.subplots(
        total_rows,
        n_loci,
        figsize=(4.9 * n_loci, 1.08 * total_rows + 3.0),
        squeeze=False,
        sharex="col",
    )

    # High-contrast sweep palette (GT remains black).
    base_colors = [
        "#1F77B4",  # blue
        "#D62728",  # red
        "#2CA02C",  # green
        "#FF7F0E",  # orange
        "#9467BD",  # purple
        "#17BECF",  # cyan
    ]
    colors = [base_colors[i % len(base_colors)] for i in range(len(sweep_values))]
    gt_color = "#111111"
    sig_order = [("count", "Count"), ("pval", "Signal"), ("peak", "Peak")]

    for sec_idx, (sig_key, sig_label) in enumerate(sig_order):
        sec_start = sec_idx * n_rows
        # Strong separator line at section boundaries.
        for col in range(n_loci):
            if sec_idx > 0:
                axes[sec_start][col].spines["top"].set_linewidth(1.5)
                axes[sec_start][col].spines["top"].set_color("#222222")
            # Section tint for readability.
            for r in range(sec_start, sec_start + n_rows):
                axes[r][col].set_facecolor("#FCFCFD" if sec_idx % 2 == 0 else "#FAFBFF")

        for ridx, row in enumerate(row_specs):
            plot_row = sec_start + ridx
            assay_idx = row["idx"]

            for col, locus_name in enumerate(loci):
                ax = axes[plot_row][col]
                bundle = bundles[locus_name]

                if row["gt_source"] == "T":
                    gt = bundle.gt_count_T[:, assay_idx] if sig_key == "count" else (
                        bundle.gt_pval_T[:, assay_idx] if sig_key == "pval" else bundle.gt_peak_T[:, assay_idx]
                    )
                else:
                    gt = bundle.gt_count_B[:, assay_idx] if sig_key == "count" else (
                        bundle.gt_pval_B[:, assay_idx] if sig_key == "pval" else bundle.gt_peak_B[:, assay_idx]
                    )
                x_bp = bundle.x_bp
                # Plot GT first (reference), then sweep predictions.
                ax.plot(x_bp, gt, color=gt_color, linewidth=0.62, alpha=0.78, zorder=3)

                pred_key = f"pred_{sig_key}"
                series_stack = [gt]
                for sv, color in zip(sweep_values, colors):
                    pred = cache[str(sv)][locus_name][pred_key][:, assay_idx]
                    series_stack.append(pred)
                    ax.plot(x_bp, pred, color=color, linewidth=0.8, alpha=0.52, zorder=2)

                # Robust y-limits so rare spikes don't flatten everything.
                merged = np.concatenate([np.asarray(s).reshape(-1) for s in series_stack])
                finite = merged[np.isfinite(merged)]
                if finite.size > 8:
                    lo = np.percentile(finite, 0.5)
                    hi = np.percentile(finite, 99.5)
                    if hi > lo:
                        pad = 0.08 * (hi - lo + 1e-9)
                        ax.set_ylim(lo - pad, hi + pad)

                if col == 0:
                    ax.set_ylabel(f"{row['label']} [{sig_label}]", fontsize=6.6, color="#1E1E1E")
                else:
                    ax.set_yticklabels([])

                if plot_row == 0:
                    ax.set_title(locus_name, fontsize=10, pad=6.0, color="#111111")

                if plot_row < total_rows - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Genomic position (bp)", fontsize=7.2, color="#1E1E1E")

                ax.grid(True, axis="y", alpha=0.28)
                ax.tick_params(axis="both", labelsize=5.8, length=1.8, width=0.5, color="#444444")
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(True)
                ax.spines["left"].set_color("#2E2E2E")
                ax.spines["bottom"].set_color("#2E2E2E")

    legend_lines = [plt.Line2D([0], [0], color=gt_color, lw=1.2, label="GT")]
    legend_lines += [
        plt.Line2D([0], [0], color=c, lw=1.2, label=f"{field}={_fmt_sweep(sv)}")
        for sv, c in zip(sweep_values, colors)
    ]
    legend = fig.legend(
        handles=legend_lines,
        loc="upper center",
        ncol=min(6, len(legend_lines)),
        fontsize=8,
        frameon=True,
        handlelength=2.0,
        columnspacing=1.2,
        borderpad=0.4,
        bbox_to_anchor=(0.5, 0.985),
    )
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_edgecolor("#B8BCC4")
    legend.get_frame().set_alpha(0.95)

    fig.suptitle(
        f"Lightweight Supertrack Tracks - {field}",
        fontsize=13,
        y=0.998,
        color="#121212",
    )
    fig.tight_layout(rect=[0.02, 0.02, 0.998, 0.955], h_pad=0.23, w_pad=0.25)

    out_png = output_dir / f"lightweight_tracks_{field}.png"
    out_svg = output_dir / f"lightweight_tracks_{field}.svg"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg, dpi=220)
    plt.close(fig)
    return {"png": str(out_png), "svg": str(out_svg)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight self-contained supertrack visualization for EIC.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", default="eic", choices=["eic"])
    parser.add_argument("--bios-name", required=True, help="Any paired biosample name, e.g., B_BE2C or T_BE2C")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--loci-set-name", default="example_genes", choices=list(NAMED_LOCI.keys()))
    parser.add_argument("--prompt-spec", default=None, help="Default: prompts/eic_mode.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dsf", type=int, default=1, help="Input DSF for T_* source.")
    parser.add_argument("--pred-batch-size", type=int, default=16)
    parser.add_argument("--fields", default="depth,read_length,run_type")
    args = parser.parse_args()

    os.environ["CANDI_PRED_BATCH_SIZE"] = str(max(1, int(args.pred_batch_size)))

    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_dir) / "supertrack_evals" / "lightweight_tracks"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_spec_path = Path(args.prompt_spec) if args.prompt_spec else (repo_root / "prompts" / "eic_mode.json")
    if not prompt_spec_path.exists():
        raise FileNotFoundError(f"Prompt spec not found: {prompt_spec_path}")
    baseline_spec = _load_prompt_spec(prompt_spec_path)

    predictor = CANDIPredictor(args.model_dir, device=args.device, DNA=True)
    predictor.setup_data_handler(
        data_path=args.data_path,
        dataset_type="eic",
        context_length=predictor.context_length,
        resolution=predictor.data_handler.resolution if predictor.data_handler is not None else 25,
        split="test",
    )
    dh = predictor.data_handler
    assert dh is not None

    t_bios, b_bios = _resolve_eic_pair(args.bios_name, list(dh.navigation.keys()))
    if args.loci_set_name not in NAMED_LOCI:
        raise ValueError(f"Unknown loci set: {args.loci_set_name}")
    loci = NAMED_LOCI[args.loci_set_name]
    warnings: List[str] = []

    bundles: Dict[str, LocusBundle] = {}
    for name, chrom, start_bp, end_bp in loci:
        bundle = _load_locus_bundle(
            predictor=predictor,
            t_bios=t_bios,
            b_bios=b_bios,
            locus_name=name,
            chrom=chrom,
            start_bp=start_bp,
            end_bp=end_bp,
            dsf=args.dsf,
            baseline_spec=baseline_spec,
            warnings=warnings,
        )
        # Shape guardrails
        if bundle.base_mY_2d.ndim != 2 or bundle.base_mY_2d.shape[0] != 4:
            raise ValueError(f"[{name}] base_mY has invalid shape: {tuple(bundle.base_mY_2d.shape)}")
        if bundle.X.ndim != 3 or bundle.mX.ndim != 3 or bundle.avX.ndim != 2:
            raise ValueError(f"[{name}] input tensors have invalid shapes.")
        bundles[name] = bundle

    first_bundle = next(iter(bundles.values()))
    expnames = list(predictor.data_handler.aliases["experiment_aliases"].keys())
    row_specs = _build_row_specs(first_bundle, expnames)
    if len(row_specs) == 0:
        raise ValueError("No GT-available assays found for denoise/impute rows.")

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    for f in fields:
        if f not in DEFAULT_SWEEPS:
            raise ValueError(f"Unsupported sweep field '{f}'. Allowed: {sorted(DEFAULT_SWEEPS.keys())}")

    outputs_by_field: Dict[str, Dict[str, str]] = {}
    for field in fields:
        print(f"\n=== Sweep field: {field} ===")
        sweep_values = DEFAULT_SWEEPS[field]
        field_cache: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        for sv in sweep_values:
            print(f"  -> value: {sv}")
            field_cache[str(sv)] = {}
            for locus_name, bundle in bundles.items():
                mY_cur = _apply_global_sweep(bundle.base_mY_2d, field, sv)
                preds = _run_forward_all_assays(predictor, bundle, mY_cur)
                field_cache[str(sv)][locus_name] = preds

        outputs_by_field[field] = _plot_field(
            field=field,
            sweep_values=sweep_values,
            cache=field_cache,
            bundles=bundles,
            row_specs=row_specs,
            output_dir=output_dir,
        )
        print(f"  Saved: {outputs_by_field[field]['png']}")

    manifest = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "dataset": "eic",
        "bios_input": args.bios_name,
        "t_bios": t_bios,
        "b_or_v_bios": b_bios,
        "decoder_type": predictor.config.get("decoder_type", "fixed") if predictor.config else "unknown",
        "prompt_spec": str(prompt_spec_path.resolve()),
        "loci_set_name": args.loci_set_name,
        "loci": [{"name": b.name, "chrom": b.chrom, "start_bp": b.start_bp, "end_bp": b.end_bp} for b in bundles.values()],
        "num_rows_assays": len(row_specs),
        "row_labels": [r["label"] for r in row_specs],
        "fields": fields,
        "sweep_values": {k: DEFAULT_SWEEPS[k] for k in fields},
        "pred_batch_size": max(1, int(args.pred_batch_size)),
        "outputs_by_field": outputs_by_field,
        "warnings": warnings,
    }
    manifest_path = output_dir / "lightweight_supertrack_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if warnings:
        warn_path = output_dir / "lightweight_supertrack_warnings.log"
        with warn_path.open("w", encoding="utf-8") as f:
            for w in warnings:
                f.write(w + "\n")
        print(f"Warnings: {len(warnings)} (saved to {warn_path})")

    print(f"\nDone. Outputs: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
