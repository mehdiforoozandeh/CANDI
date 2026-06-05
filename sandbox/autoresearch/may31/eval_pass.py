"""Fixed eval suite for E32 autoresearch — vb_natural, canonical, cloze-T, den, DCR."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from sandbox import SANDBOX_ASSAYS
from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.candi_v2.model import CANDIv2
from sandbox.data import build_canonical_meta
from sandbox.diagnostics.meta_probes import run_probe_battery
from sandbox.diagnostics.real_overfit import _V2TupleWrapper
from sandbox.eval import eval_batch_metrics, prompt_sensitivity_depth_count_ratio

log = logging.getLogger(__name__)


def audit_tv_pairing(batch: Dict[str, torch.Tensor], t_bios: str, imp_bios: str) -> None:
    """Assert T/V/B same-base pairing; log imp assay counts."""
    if not t_bios.startswith("T_"):
        raise ValueError(f"eval t_bios must start with T_: got {t_bios!r}")
    if not imp_bios.startswith(("V_", "B_")):
        raise ValueError(f"eval imp_bios must start with V_/B_: got {imp_bios!r}")
    if t_bios[2:] != imp_bios[2:]:
        raise ValueError(
            f"T/V/B base mismatch: {t_bios!r} vs {imp_bios!r} "
            f"(bases {t_bios[2:]!r} vs {imp_bios[2:]!r})"
        )
    bn = batch.get("biosample_name", t_bios)
    imp_bn = batch.get("imp_biosample_name", imp_bios)
    if isinstance(bn, str) and bn != t_bios:
        raise ValueError(f"batch biosample_name {bn!r} != manifest t_bios {t_bios!r}")
    if isinstance(imp_bn, str) and imp_bn != imp_bios:
        raise ValueError(f"batch imp_biosample_name {imp_bn!r} != manifest imp_bios {imp_bios!r}")

    y_avail = batch.get("y_avail")
    y_data_imp = batch.get("y_data_imp")
    if isinstance(y_avail, torch.Tensor) and isinstance(y_data_imp, torch.Tensor):
        missing = (y_avail == 0)
        valid_imp = (y_data_imp != -1).any(dim=1)
        n_missing = int(missing.sum().item())
        n_imp_pos = int((missing & valid_imp).sum().item())
        log.info(
            "T/V/B audit %s -> %s: missing_in_T=%d imp_eval_positions=%d",
            t_bios, imp_bios, n_missing, n_imp_pos,
        )


def build_y_meta_vb_natural(
    prep: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    canonical_meta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V/B natural metadata for y_avail==0 slots; canonical fallback when invalid."""
    y_meta_fwd = prep["y_meta"].clone()
    query_mask_fwd = prep["query_mask"].clone()
    y_avail = batch["y_avail"].to(device)
    missing = (y_avail == 0).unsqueeze(1).expand_as(y_meta_fwd)

    y_meta_imp = batch.get("y_meta_imp")
    if isinstance(y_meta_imp, torch.Tensor):
        ymi = y_meta_imp.to(device)
        valid_vb = (ymi[:, 0:1, :] != -1.0).expand_as(y_meta_fwd)
        use_vb = missing & valid_vb
        y_meta_fwd[use_vb] = ymi[use_vb]

    can = canonical_meta.to(device)
    can_exp = can.unsqueeze(0).expand_as(y_meta_fwd)
    still_missing = missing & (y_meta_fwd[:, 0:1, :] == -1.0).expand_as(y_meta_fwd)
    y_meta_fwd[still_missing] = can_exp[still_missing]
    query_mask_fwd = query_mask_fwd | missing
    return y_meta_fwd, query_mask_fwd


def build_y_meta_canonical(
    prep: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    canonical_meta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Production-style canonical metadata for y_avail==0 (A1 diagnostic)."""
    y_meta_fwd = prep["y_meta"].clone()
    query_mask_fwd = prep["query_mask"].clone()
    y_avail = batch["y_avail"].to(device)
    missing = (y_avail == 0).unsqueeze(1).expand_as(y_meta_fwd)
    can = canonical_meta.to(device)
    can_exp = can.unsqueeze(0).expand_as(y_meta_fwd)
    y_meta_fwd[missing] = can_exp[missing]
    query_mask_fwd = query_mask_fwd | missing
    return y_meta_fwd, query_mask_fwd


def _imp_eval_maps(
    batch: Dict[str, torch.Tensor],
    prep: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    y_avail = batch.get("y_avail")
    y_pval_imp = batch.get("y_pval_imp")
    y_dsf = batch.get("y_dsf")
    if not isinstance(y_avail, torch.Tensor) or not isinstance(y_pval_imp, torch.Tensor):
        return None, None
    y_av_d = y_avail.to(device)
    y_pi_d = y_pval_imp.to(device)
    t_unavail = (y_av_d <= 0).unsqueeze(1).expand_as(prep["masked_map"])
    vb_valid = (y_pi_d != -1)
    imp_eval_map = t_unavail & vb_valid
    if isinstance(y_dsf, torch.Tensor):
        dsf1 = (y_dsf.to(device) == 1).unsqueeze(1).expand_as(prep["masked_map"])
        imp_eval_signal_map = imp_eval_map & dsf1
    else:
        imp_eval_signal_map = imp_eval_map
    return imp_eval_map, imp_eval_signal_map


def _forward_batch(
    model: CANDIv2,
    prep: Dict[str, torch.Tensor],
    y_meta_fwd: torch.Tensor,
    query_mask_fwd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        p, n, mu, _var, _df, peak = model.forward_tuple(
            prep["x_data"],
            prep["x_dna"],
            prep["x_meta"],
            y_meta_fwd,
            query_mask=query_mask_fwd,
            query_mask_signal=prep["query_mask_signal"],
        )
    return p, n, mu, peak


def _batch_imp_metrics(
    model: CANDIv2,
    batch: Dict[str, torch.Tensor],
    prep: Dict[str, torch.Tensor],
    device: torch.device,
    y_meta_fwd: torch.Tensor,
    query_mask_fwd: torch.Tensor,
) -> Dict[str, float]:
    p, n, mu, peak = _forward_batch(model, prep, y_meta_fwd, query_mask_fwd)
    imp_eval_map, imp_eval_signal_map = _imp_eval_maps(batch, prep, device)
    ydi = batch.get("y_data_imp")
    ypi = batch.get("y_pval_imp")
    ypk = batch.get("y_peaks_imp")
    rt = batch.get("region_type")
    return eval_batch_metrics(
        p, n, mu, peak,
        prep["y_data"], prep["y_pval"], prep["y_peaks"],
        prep["observed_map"], prep["masked_map"],
        prep["signal_observed_map"], prep["signal_masked_map"],
        regime="type1_chr19",
        region_type=rt.to(device) if isinstance(rt, torch.Tensor) else rt,
        imp_eval_map=imp_eval_map,
        imp_eval_signal_map=imp_eval_signal_map,
        y_data_imp=ydi.to(device) if isinstance(ydi, torch.Tensor) else ydi,
        y_pval_imp=ypi.to(device) if isinstance(ypi, torch.Tensor) else ypi,
        y_peaks_imp=ypk.to(device) if isinstance(ypk, torch.Tensor) else ypk,
    )


def _mean_finite(values: List[float]) -> float:
    good = [v for v in values if math.isfinite(v)]
    if not good:
        return float("nan")
    return sum(good) / len(good)


def _cloze_imp_r2(
    model: CANDIv2,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Optional[float]:
    """Cloze-on-T imp count R² (train-aligned task). Returns None if no masked positions."""
    cloze_masker = make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0)
    prep_cloze = prepare_masked_batch(batch, cloze_masker, device, apply_mask=True)
    if prep_cloze is None or not prep_cloze["masked_map"].any():
        return None
    p, n, mu, peak = _forward_batch(
        model, prep_cloze, prep_cloze["y_meta"], prep_cloze["query_mask"],
    )
    imp_map = prep_cloze["masked_map"]
    m_cloze = eval_batch_metrics(
        p, n, mu, peak,
        prep_cloze["y_data"], prep_cloze["y_pval"], prep_cloze["y_peaks"],
        prep_cloze["observed_map"], prep_cloze["masked_map"],
        prep_cloze["signal_observed_map"], prep_cloze["signal_masked_map"],
        imp_eval_map=imp_map,
        y_data_imp=prep_cloze["y_data"],
        y_pval_imp=prep_cloze["y_pval"],
        y_peaks_imp=prep_cloze["y_peaks"],
    )
    r2 = m_cloze.get("imp_count_r2_gw", float("nan"))
    return r2 if math.isfinite(r2) else None


def evaluate_suite(
    model: CANDIv2,
    device: torch.device,
    eval_batches: List[Dict[str, torch.Tensor]],
    eval_entries: List[Dict[str, Any]],
    canonical_meta: torch.Tensor,
    *,
    eval_cloze_t_index: int = 0,
    cloze_train_batch: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Run pinned 8-batch chr21 eval: vb_natural primary + diagnostics."""
    eval_masker = make_masker(
        p_full_loci=0.0, p_full_assay=0.0, p_chunks=0.0, mask_fraction=0.0,
    )
    imp_agg: Dict[str, List[float]] = {}
    den_agg: Dict[str, List[float]] = {}
    canonical_r2: List[float] = []
    prep0: Optional[Dict[str, torch.Tensor]] = None

    for batch, entry in zip(eval_batches, eval_entries):
        t_bios = str(entry["t_bios"])
        imp_bios = str(entry["imp_bios"])
        audit_tv_pairing(batch, t_bios, imp_bios)

        prep = prepare_masked_batch(batch, eval_masker, device, apply_mask=False)
        if prep is None:
            continue

        y_meta_vb, qm_vb = build_y_meta_vb_natural(prep, batch, device, canonical_meta)
        m = _batch_imp_metrics(model, batch, prep, device, y_meta_vb, qm_vb)
        for k in ("imp_count_r2_gw", "imp_count_pearson_gw", "imp_count_spearman_gw",
                  "den_count_r2_gw", "den_count_pearson_gw"):
            if k in m and math.isfinite(m[k]):
                (imp_agg if k.startswith("imp_") else den_agg).setdefault(k, []).append(m[k])

        y_meta_can, qm_can = build_y_meta_canonical(prep, batch, device, canonical_meta)
        m_can = _batch_imp_metrics(model, batch, prep, device, y_meta_can, qm_can)
        if "imp_count_r2_gw" in m_can and math.isfinite(m_can["imp_count_r2_gw"]):
            canonical_r2.append(m_can["imp_count_r2_gw"])

        if prep0 is None:
            prep0 = prep

    metrics: Dict[str, float] = {
        "imp_count_r2_gw": _mean_finite(imp_agg.get("imp_count_r2_gw", [])),
        "imp_count_pearson_gw": _mean_finite(imp_agg.get("imp_count_pearson_gw", [])),
        "imp_count_spearman_gw": _mean_finite(imp_agg.get("imp_count_spearman_gw", [])),
        "den_count_r2_gw": _mean_finite(den_agg.get("den_count_r2_gw", [])),
        "imp_count_r2_gw_canonical": _mean_finite(canonical_r2),
    }

    # Cloze-T diagnostic: eval batch first, then train pin fallback (train-aligned).
    cloze_r2: Optional[float] = None
    if 0 <= eval_cloze_t_index < len(eval_batches):
        cloze_r2 = _cloze_imp_r2(model, eval_batches[eval_cloze_t_index], device)
    if cloze_r2 is None and cloze_train_batch is not None:
        cloze_r2 = _cloze_imp_r2(model, cloze_train_batch, device)
        if cloze_r2 is not None:
            log.info("cloze-T: used train pin batch (eval batch had no cloze mask)")
    metrics["imp_count_r2_gw_cloze_T"] = (
        cloze_r2 if cloze_r2 is not None else float("nan")
    )

    # DCR on eval batch 0 (real T y_meta)
    if prep0 is not None:
        probe_model = _V2TupleWrapper(model).to(device)
        metrics["depth_count_ratio"] = prompt_sensitivity_depth_count_ratio(
            probe_model, prep0, prep0["y_meta"], device,
        )
    else:
        metrics["depth_count_ratio"] = float("nan")

    # Optional masked-bin DCR on cloze prep (train batch preferred)
    metrics["dcr_masked_bins"] = float("nan")
    cloze_for_dcr = cloze_train_batch
    if cloze_for_dcr is None and 0 <= eval_cloze_t_index < len(eval_batches):
        cloze_for_dcr = eval_batches[eval_cloze_t_index]
    if cloze_for_dcr is not None:
        cloze_masker = make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0)
        prep_cloze = prepare_masked_batch(cloze_for_dcr, cloze_masker, device, apply_mask=True)
        if prep_cloze is not None:
            probe = _V2TupleWrapper(model).to(device)
            probes = run_probe_battery(model, probe, prep_cloze)
            metrics["dcr_masked_bins"] = probes.get("y_depth_dcr_on_masked_bins", float("nan"))

    return metrics
