"""Chr21 encoder embedding UMAP diagnostic for JEPA training.

After training, runs the CANDI encoder in eval mode on all chr21 windows
(target view: DSF=1, unmasked metadata) for every T_* biosample.  Each
per-position encoder token ``z[b, l2, :]`` is used directly — no mean-pooling.
A joint 2D UMAP is fit **per biosample** on all ``N_windows × L2`` tokens.

Figure layout (one combined PNG):
  rows  = one per T_* biosample
  cols  = [color by region_type | color by genomic position (chr21 start)]

Saved to ``{run_dir}/chr21umap/chr21_umap_step{N}.png`` and uploaded to W&B.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from sandbox import SANDBOX_ASSAYS
from sandbox.data import SandboxH5Dataset
from sandbox.jepa import CANDIJepa
from sandbox.jepa_config import JEPAConfig


# ─── dimensionality reduction ─────────────────────────────────────────────────

def _get_reducer(random_state: int = 42):
    """Return a UMAP reducer, falling back to t-SNE if umap-learn is missing."""
    try:
        from umap import UMAP  # type: ignore
        return UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="euclidean", random_state=random_state)
    except ImportError:
        warnings.warn(
            "[chr21_umap] umap-learn not installed; falling back to sklearn t-SNE.",
            stacklevel=3,
        )
        from sklearn.manifold import TSNE  # type: ignore
        return TSNE(n_components=2, perplexity=30,
                    random_state=random_state, n_iter=1000)


# ─── region-type labels ────────────────────────────────────────────────────────

# Assay indices (matching SANDBOX_ASSAYS order):
#   ATAC-seq(0), DNase-seq(1), H3K4me3(2), H3K4me1(3), H3K27ac(4), H3K27me3(5), H3K36me3(6), H3K9me3(7)
_ACTIVITY_ASSAY_INDICES:   Tuple[int, ...] = (0, 1, 2, 3, 4, 6)   # accessibility/transcription/enhancer/promoter
_REPRESSION_ASSAY_INDICES: Tuple[int, ...] = (5, 7)                # H3K27me3, H3K9me3


# ─── per-biosample collection result ─────────────────────────────────────────

class BiosampleTokens(NamedTuple):
    embeddings:    np.ndarray   # [N_tokens, F2]   float32
    activity_sig:  np.ndarray   # [N_tokens]        float32  mean arcsinh of activity assays
    repression_sig: np.ndarray  # [N_tokens]        float32  mean arcsinh of repression assays
    genomic_pos:   np.ndarray   # [N_tokens]        int      chr21 start bp, broadcast to L2


# ─── embedding collection ──────────────────────────────────────────────────────

@torch.no_grad()
def collect_chr21_embeddings(
    model: CANDIJepa,
    h5_path: Path,
    cfg: JEPAConfig,
    device: torch.device,
    *,
    max_tokens_per_bios: int = 60_000,
) -> Dict[str, BiosampleTokens]:
    """Collect per-position encoder tokens for every T_* biosample on chr21.

    Uses the *target view*: DSF=1 signal + control, fully unmasked metadata.
    Each window contributes ``L2`` tokens (no spatial pooling).

    ``max_tokens_per_bios`` caps total tokens via uniform subsampling to keep
    UMAP tractable when chr21 × L2 is large.
    """
    model.eval()

    ds_eval = SandboxH5Dataset(
        h5_path,
        cfg.data.regime,
        train=False,
        batch_size=min(16, int(cfg.training.batch_size)),
        biosample_prefix="T_",
        dsf_list=(1,),
        dsf_sampling="off",       # always DSF=1
        seed=0,
        shuffle=False,
        eval_include_vb_ground_truth=False,
        imp_prefixes=(),
        h5_cache_ram=False,
    )

    # Access window coordinates for genomic-position coloring.
    eval_indices: List[int] = ds_eval._eval_indices   # sorted window indices
    windows = ds_eval._windows                         # (chrom, start, end, rt)

    bios_embs: Dict[str, List[np.ndarray]] = {}
    bios_act:  Dict[str, List[np.ndarray]] = {}
    bios_rep:  Dict[str, List[np.ndarray]] = {}
    bios_pos:  Dict[str, List[np.ndarray]] = {}

    act_idx = list(_ACTIVITY_ASSAY_INDICES)
    rep_idx = list(_REPRESSION_ASSAY_INDICES)
    window_cursor = 0  # tracks which eval_indices we're at

    for raw_batch in ds_eval:
        bios_name: str = str(raw_batch["biosample_name"])
        x_data  = raw_batch["y_data"].to(device)        # [B, L, F] DSF=1 arcsinh counts
        x_meta  = raw_batch["x_meta"].to(device)        # [B, 4, F] unmasked
        x_dna   = raw_batch["x_dna"].to(device)
        ctrl_d  = raw_batch["control_data"].to(device)  # [B, L, 1]
        ctrl_m  = raw_batch["control_meta"].to(device)  # [B, 4, 1]

        B = x_data.shape[0]

        # Genomic start positions for this batch (from window index lookup)
        batch_wi = eval_indices[window_cursor: window_cursor + B]
        starts = np.array([windows[wi][1] for wi in batch_wi], dtype=np.int64)
        window_cursor += B

        # Mean signal per window, excluding unavailable assays (sentinel = -1).
        def _mean_signal(data: torch.Tensor, indices: list) -> np.ndarray:
            d = data[:, :, indices].cpu().float()       # [B, L, n]
            mask = (d >= 0.0)
            s = (d * mask).sum(dim=(1, 2))
            c = mask.sum(dim=(1, 2)).clamp(min=1)
            return (s / c).numpy()                      # [B]

        act_mean = _mean_signal(x_data, act_idx)
        rep_mean = _mean_signal(x_data, rep_idx)

        # Build target-view inputs (same as prepare_jepa_batch target branch)
        x_tgt  = torch.cat([x_data, ctrl_d], dim=2)   # [B, L, F+1]
        meta_t = torch.cat([x_meta, ctrl_m], dim=2)   # [B, 4, F+1]

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"),
                                dtype=torch.bfloat16):
            z_raw = model.candi.encode(x_tgt, x_dna, meta_t)  # [B, L2, F2]

        B_, L2, F2 = z_raw.shape
        tokens = z_raw.float().cpu().numpy().reshape(B_ * L2, F2)  # [B*L2, F2]

        # Broadcast per-window scalars to all L2 position tokens of each window.
        act_exp = np.repeat(act_mean, L2)    # [B*L2]
        rep_exp = np.repeat(rep_mean, L2)    # [B*L2]
        pos_exp = np.repeat(starts,   L2)    # [B*L2]

        bios_embs.setdefault(bios_name, []).append(tokens)
        bios_act.setdefault(bios_name,  []).append(act_exp)
        bios_rep.setdefault(bios_name,  []).append(rep_exp)
        bios_pos.setdefault(bios_name,  []).append(pos_exp)

    result: Dict[str, BiosampleTokens] = {}
    for bios in bios_embs:
        emb = np.concatenate(bios_embs[bios], axis=0)
        act = np.concatenate(bios_act[bios],  axis=0)
        rep = np.concatenate(bios_rep[bios],  axis=0)
        pos = np.concatenate(bios_pos[bios],  axis=0)

        # Uniform subsample if over budget
        N = len(emb)
        if N > max_tokens_per_bios:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(N, size=max_tokens_per_bios, replace=False)
            idx.sort()
            emb, act, rep, pos = emb[idx], act[idx], rep[idx], pos[idx]

        result[bios] = BiosampleTokens(emb, act, rep, pos)

    return result


def _umap_available() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except ImportError:
        return False


def _fit_pca_coords(emb: np.ndarray) -> np.ndarray:
    """Return PC1 × PC2 projection.

    Uses centered (but NOT scaled) embeddings so that high-variance dimensions
    — the biologically active ones — dominate the top principal components.
    Scaling to unit variance before PCA would randomise the component ordering
    and destroy this property.
    """
    from sklearn.decomposition import PCA  # type: ignore
    centered = emb - emb.mean(axis=0, keepdims=True)
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(centered)


# ─── shared figure builder ────────────────────────────────────────────────────

def _make_embedding_figure(
    tokens_by_bios: Dict[str, "BiosampleTokens"],
    coords_by_bios: Dict[str, np.ndarray],
    bios_names: List[str],
    method: str,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Build the 3-column multipanel figure (activity | repression | genomic pos).

    ``method`` is displayed in the title and axis labels (e.g. ``"UMAP"`` or ``"PCA"``).
    Independent per-biosample colour ranges are used for activity and repression so
    every row's full dynamic range is exploited.
    """
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(bios_names)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows), squeeze=False)

    # Genomic position range is shared (all biosamples cover the same chr21 coords).
    all_pos = np.concatenate([tokens_by_bios[b].genomic_pos for b in bios_names])
    pos_min_bp, pos_max_bp = float(all_pos.min()), float(all_pos.max())

    _SCATTER_KW = dict(s=2, alpha=0.5, linewidths=0)
    dim1_lbl = "PC1" if method == "PCA" else "dim 1"
    dim2_lbl = "PC2" if method == "PCA" else "dim 2"

    for row_i, bios in enumerate(bios_names):
        toks   = tokens_by_bios[bios]
        coords = coords_by_bios[bios]

        def _add_colorbar(fig, ax, sc, label: str) -> None:
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(label, fontsize=5)
            cb.ax.tick_params(labelsize=5)

        act_vmin = float(np.percentile(toks.activity_sig, 2))
        act_vmax = float(np.percentile(toks.activity_sig, 98))
        rep_vmin = float(np.percentile(toks.repression_sig, 2))
        rep_vmax = float(np.percentile(toks.repression_sig, 98))

        # col 0: activity signal
        ax0 = axes[row_i][0]
        sc0 = ax0.scatter(coords[:, 0], coords[:, 1], c=toks.activity_sig,
                          cmap="YlOrRd", vmin=act_vmin, vmax=act_vmax, **_SCATTER_KW)
        ax0.set_title(f"{bios}\nactivity (ATAC/DNase/H3K4me3/H3K4me1/H3K27ac/H3K36me3)",
                      fontsize=7)
        ax0.set_xlabel(dim1_lbl, fontsize=6); ax0.set_ylabel(dim2_lbl, fontsize=6)
        ax0.tick_params(labelsize=5)
        _add_colorbar(fig, ax0, sc0, "mean arcsinh (act assays per window)")

        # col 1: repression signal
        ax1 = axes[row_i][1]
        sc1 = ax1.scatter(coords[:, 0], coords[:, 1], c=toks.repression_sig,
                          cmap="PuBu", vmin=rep_vmin, vmax=rep_vmax, **_SCATTER_KW)
        ax1.set_title(f"{bios}\nrepression (H3K27me3 / H3K9me3)", fontsize=7)
        ax1.set_xlabel(dim1_lbl, fontsize=6); ax1.set_ylabel(dim2_lbl, fontsize=6)
        ax1.tick_params(labelsize=5)
        _add_colorbar(fig, ax1, sc1, "mean arcsinh (rep assays per window)")

        # col 2: genomic position in Mb
        pos_mb = toks.genomic_pos / 1e6
        ax2 = axes[row_i][2]
        sc2 = ax2.scatter(coords[:, 0], coords[:, 1], c=pos_mb,
                          cmap="plasma",
                          vmin=pos_min_bp / 1e6, vmax=pos_max_bp / 1e6,
                          **_SCATTER_KW)
        ax2.set_title(f"{bios}\nchr21 genomic position", fontsize=7)
        ax2.set_xlabel(dim1_lbl, fontsize=6); ax2.set_ylabel(dim2_lbl, fontsize=6)
        ax2.tick_params(labelsize=5)
        _add_colorbar(fig, ax2, sc2, "chr21 position (Mb)")

    fig.suptitle(
        f"Chr21 encoder embeddings — {method}  "
        f"|  col 0: activity  |  col 1: repression  |  col 2: genomic position",
        fontsize=9, y=1.002,
    )
    fig.tight_layout()
    return fig


# ─── legacy alias (keeps old call sites working) ─────────────────────────────

def make_chr21_umap_figure(
    tokens_by_bios: Dict[str, "BiosampleTokens"],
    coords_by_bios: Dict[str, np.ndarray],
    bios_names: List[str],
) -> "plt.Figure":  # type: ignore[name-defined]
    return _make_embedding_figure(tokens_by_bios, coords_by_bios, bios_names, "UMAP")


# ─── top-level entry point ────────────────────────────────────────────────────

def run_chr21_umap(
    model: CANDIJepa,
    h5_path: Path,
    cfg: JEPAConfig,
    device: torch.device,
    run_dir: Path,
    global_step: int,
    wandb_run=None,
    *,
    max_tokens_per_bios: int = 60_000,
) -> None:
    """Compute chr21 encoder embeddings and save UMAP + PCA figures; upload to W&B."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[chr21_umap] matplotlib not available — skipping.", flush=True)
        return

    print("[chr21_umap] collecting chr21 encoder tokens...", flush=True)
    try:
        tokens_by_bios = collect_chr21_embeddings(
            model, h5_path, cfg, device,
            max_tokens_per_bios=max_tokens_per_bios,
        )
    except Exception as e:
        print(f"[chr21_umap] embedding collection failed: {e}", flush=True)
        return

    if not tokens_by_bios:
        print("[chr21_umap] no embeddings collected — skipping.", flush=True)
        return

    bios_names = sorted(tokens_by_bios.keys())

    # ── fit UMAP per biosample (z-scored input) ───────────────────────────
    umap_coords: Dict[str, np.ndarray] = {}
    for bios in bios_names:
        emb = tokens_by_bios[bios].embeddings
        mean_e = emb.mean(axis=0, keepdims=True)
        std_e  = emb.std(axis=0, keepdims=True).clip(1e-6)
        emb_norm = (emb - mean_e) / std_e
        n_tok = emb_norm.shape[0]
        print(f"[chr21_umap] {bios}: fitting UMAP on {n_tok} × {emb_norm.shape[1]}...",
              flush=True)
        try:
            umap_coords[bios] = _get_reducer().fit_transform(emb_norm)
        except Exception as e:
            print(f"[chr21_umap] UMAP failed for {bios}: {e}", flush=True)
            return

    # ── fit PCA per biosample (centered, not scaled) ──────────────────────
    pca_coords: Dict[str, np.ndarray] = {}
    for bios in bios_names:
        emb = tokens_by_bios[bios].embeddings
        print(f"[chr21_umap] {bios}: fitting PCA...", flush=True)
        try:
            pca_coords[bios] = _fit_pca_coords(emb)
        except Exception as e:
            print(f"[chr21_umap] PCA failed for {bios}: {e}", flush=True)
            return

    # ── save figures ──────────────────────────────────────────────────────
    umap_dir = run_dir / "chr21umap"
    umap_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    wb_images: Dict[str, str] = {}

    for method, coords_map, suffix in [
        ("UMAP", umap_coords, "umap"),
        ("PCA",  pca_coords,  "pca"),
    ]:
        try:
            fig = _make_embedding_figure(tokens_by_bios, coords_map, bios_names, method)
            fig_path = umap_dir / f"chr21_{suffix}_step{global_step}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[chr21_umap] {method} figure saved → {fig_path}", flush=True)
            wb_images[f"lejepa/chr21_{suffix}"] = str(fig_path)
        except Exception as e:
            print(f"[chr21_umap] {method} figure save failed: {e}", flush=True)

    # ── upload both to W&B ────────────────────────────────────────────────
    if wandb_run is not None and wb_images:
        try:
            import wandb  # type: ignore
            wandb_run.log(
                {k: wandb.Image(v) for k, v in wb_images.items()},
                step=global_step,
            )
            print(f"[chr21_umap] uploaded {list(wb_images.keys())} to W&B.", flush=True)
        except Exception as e:
            print(f"[chr21_umap] W&B upload failed: {e}", flush=True)
