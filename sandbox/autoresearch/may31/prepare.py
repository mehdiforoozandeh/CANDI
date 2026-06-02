#!/usr/bin/env python3
"""Fixed E32 autoresearch harness — DO NOT MODIFY during agent loop."""
from __future__ import annotations

import gc
import io
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from sandbox import SANDBOX_ASSAYS
from sandbox.autoresearch.may31 import eval_pass, train as agent_train
from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.candi_v2.loss import build_v2_loss
from sandbox.candi_v2.model import CANDIv2
from sandbox.data import _sample_xy_dsf, build_canonical_meta
from sandbox.diagnostics.real_data import build_real_v2_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --- Fixed constants ---
MAX_STEPS = 5000
BATCH_SIZE = 4
SEED = 42
MAX_PEAK_VRAM_MB = 9500.0
TRAIN_P_FULL_ASSAY = 1.0
TRAIN_P_FULL_LOCI = 0.0
TRAIN_P_CHUNKS = 0.0
# Session 2: was 0.35 (session 1); 0.28 after +4 VB train pins (36 batches, den ~0.29–0.33).
DEN_GATE = 0.28
DEN_KEEP_MIN = 0.28  # imp-phase keep guard (program.md)
DCR_LO = 3.25
DCR_HI = 4.75
IMP_PHASE_BIAS = 1.0

ARTIFACT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = ARTIFACT_DIR / "pin_manifest.json"
BASELINE_JSON = ARTIFACT_DIR / "baseline.json"
H5_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "sandbox.h5"
EIC_METADATA = Path(__file__).resolve().parents[3] / "data" / "eic_metadata.csv"

T_BIOS = ["T_DND-41", "T_RWPE2", "T_heart_left_ventricle", "T_H1-hESC", "T_H9"]
EVAL_PAIRS = [
    ("T_DND-41", "V_DND-41"),
    ("T_DND-41", "B_DND-41"),
    ("T_RWPE2", "B_RWPE2"),
    ("T_heart_left_ventricle", "V_heart_left_ventricle"),
    ("T_H1-hESC", "V_H1-hESC"),
    ("T_DND-41", "B_DND-41"),
    ("T_RWPE2", "B_RWPE2"),
    ("T_heart_left_ventricle", "V_heart_left_ventricle"),
]


def _decode_ch(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    return str(x)


def spread_pick(idxs: List[int], n: int, *, offset: int = 0) -> List[int]:
    if not idxs:
        raise ValueError("empty index pool")
    s = sorted(idxs)
    q = max(1, len(s) // 4)
    quartiles = [s[i * q : (i + 1) * q if i < 3 else len(s)] for i in range(4)]
    out: List[int] = []
    qi, pi = offset % 4, 0
    while len(out) < n:
        pool = quartiles[qi]
        if pool:
            out.append(pool[pi % len(pool)])
            pi += 1
        qi = (qi + 1) % 4
        if pi > len(s) * 2:
            break
    return out[:n]


def build_pin_manifest(h5_path: Path) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        chroms = [_decode_ch(x) for x in h5["windows/chrom"][:]]
    windows = list(range(len(chroms)))
    chr19_train_wi = [i for i in windows if chroms[i] == "chr19"]
    chr21_eval_wi = [i for i in windows if chroms[i] == "chr21"]

    train: List[Dict[str, Any]] = []
    for bi, t in enumerate(T_BIOS):
        for k in range(6):
            train.append({
                "t_bios": t,
                "window_indices": spread_pick(chr19_train_wi, BATCH_SIZE, offset=bi * 6 + k),
            })
    train.append({
        "t_bios": "T_DND-41",
        "window_indices": spread_pick(chr19_train_wi, BATCH_SIZE, offset=30),
    })
    train.append({
        "t_bios": "T_RWPE2",
        "window_indices": spread_pick(chr19_train_wi, BATCH_SIZE, offset=31),
    })

    eval_imp: List[Dict[str, Any]] = []
    for i, (t, imp) in enumerate(EVAL_PAIRS):
        eval_imp.append({
            "t_bios": t,
            "imp_bios": imp,
            "window_indices": spread_pick(chr21_eval_wi, BATCH_SIZE, offset=i),
        })

    return {
        "seed": SEED,
        "h5_path": "sandbox/data/sandbox.h5",
        "train": train,
        "eval_imp": eval_imp,
        "eval_cloze_t_index": 0,
    }


def load_or_create_manifest(h5_path: Path) -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    manifest = build_pin_manifest(h5_path)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote pin manifest to %s", MANIFEST_PATH)
    return manifest


class H5PinCache:
    """RAM-cached H5 reader for pinned batch assembly."""

    def __init__(self, h5_path: Path) -> None:
        self.h5_path = Path(h5_path)
        with open(self.h5_path, "rb") as rf:
            self._buf = rf.read()
        with h5py.File(io.BytesIO(self._buf), "r") as h5:
            self.bios_order: List[str] = json.loads(h5["biosamples"].attrs["order"])
            chroms = [_decode_ch(x) for x in h5["windows/chrom"][:]]
            rtypes = np.array(h5["windows/region_type"][:])
            starts = np.array(h5["windows/start"][:])
            ends = np.array(h5["windows/end"][:])
        self.windows = [
            (chroms[i], int(starts[i]), int(ends[i]), int(rtypes[i]))
            for i in range(len(chroms))
        ]
        self.signal_dim = len(SANDBOX_ASSAYS)
        self.dsf_list = (1, 2, 4, 8)

    def _open(self) -> h5py.File:
        return h5py.File(io.BytesIO(self._buf), "r")

    def load_pinned_batch(
        self,
        entry: Dict[str, Any],
        *,
        dsf_sampling: str = "off",
        train: bool = True,
        rng_seed: int = SEED,
    ) -> Dict[str, torch.Tensor]:
        t_bios = str(entry["t_bios"])
        imp_bios = entry.get("imp_bios")
        window_indices = list(entry["window_indices"])
        B = len(window_indices)
        rng = random.Random(rng_seed)

        with self._open() as h5:
            gname = t_bios.replace("/", "_")
            g = h5["biosamples"][gname]
            L = g["pval"].shape[1]
            F = self.signal_dim
            Lbp = g["dna"].shape[1]

            if train:
                xs_x, xs_y = _sample_xy_dsf(rng, self.dsf_list, dsf_sampling, F)
            else:
                xs_x, xs_y = [1] * F, [1] * F

            x_dsf = torch.tensor(xs_x, dtype=torch.int64).unsqueeze(0).expand(B, -1).clone()
            y_dsf = torch.tensor(xs_y, dtype=torch.int64).unsqueeze(0).expand(B, -1).clone()

            x_data = torch.full((B, L, F), -1.0)
            y_data = torch.full((B, L, F), -1.0)
            x_meta = torch.full((B, 4, F), -1.0)
            y_meta = torch.full((B, 4, F), -1.0)
            x_avail = torch.zeros(B, F, dtype=torch.float32)
            y_avail = torch.zeros(B, F, dtype=torch.float32)
            y_pval = torch.zeros(B, L, F)
            y_peaks = torch.zeros(B, L, F)
            x_dna = torch.zeros(B, Lbp, 4)
            control_data = torch.zeros(B, L, 1)
            control_meta = torch.zeros(B, 4, 1)
            control_avail = torch.zeros(B, 1)
            region_type = torch.zeros(B, dtype=torch.uint8)

            for j, wi in enumerate(window_indices):
                region_type[j] = int(self.windows[wi][3])
                y_pval[j] = torch.tensor(np.array(g["pval"][wi]), dtype=torch.float32)
                y_peaks[j] = torch.tensor(np.array(g["peaks"][wi]), dtype=torch.float32)
                x_dna[j] = torch.tensor(np.array(g["dna"][wi]), dtype=torch.float32)
                control_data[j] = torch.tensor(np.array(g["control"][wi]), dtype=torch.float32)
                control_meta[j] = torch.tensor(np.array(g["control_meta"][wi]), dtype=torch.float32)
                control_avail[j, 0] = 1.0 if bool((control_data[j] != 0).any().item()) else 0.0
                for fi in range(F):
                    xd = int(x_dsf[j, fi])
                    yd = int(y_dsf[j, fi])
                    if xd > 0:
                        xm = torch.tensor(np.array(g[f"meta_dsf{xd}"][:, fi]), dtype=torch.float32)
                        x_meta[j, :, fi] = xm
                        if float(xm[0]) != -1.0:
                            x_data[j, :, fi] = torch.tensor(
                                np.array(g[f"counts_dsf{xd}"][wi, :, fi]), dtype=torch.float32,
                            )
                            x_avail[j, fi] = 1.0
                    if yd > 0:
                        ym = torch.tensor(np.array(g[f"meta_dsf{yd}"][:, fi]), dtype=torch.float32)
                        y_meta[j, :, fi] = ym
                        if float(ym[0]) != -1.0:
                            y_data[j, :, fi] = torch.tensor(
                                np.array(g[f"counts_dsf{yd}"][wi, :, fi]), dtype=torch.float32,
                            )
                            y_avail[j, fi] = 1.0

            out: Dict[str, Any] = {
                "x_data": x_data,
                "x_meta": x_meta,
                "x_avail": x_avail,
                "x_dna": x_dna,
                "y_data": y_data,
                "y_meta": y_meta,
                "y_avail": y_avail,
                "y_pval": y_pval,
                "y_peaks": y_peaks,
                "control_data": control_data,
                "control_meta": control_meta,
                "control_avail": control_avail,
                "x_dsf": x_dsf,
                "y_dsf": y_dsf,
                "control_x_dsf": torch.ones(B, dtype=torch.int64),
                "biosample_name": t_bios,
                "region_type": region_type,
            }

            if imp_bios is not None:
                imp_gname = str(imp_bios).replace("/", "_")
                if imp_gname in h5["biosamples"]:
                    g_imp = h5["biosamples"][imp_gname]
                    y_data_imp = torch.full((B, L, F), -1.0)
                    y_pval_imp = torch.zeros(B, L, F)
                    y_peaks_imp = torch.zeros(B, L, F)
                    for j, wi in enumerate(window_indices):
                        for fi in range(F):
                            yd = int(y_dsf[j, fi])
                            if yd > 0:
                                ym_imp = torch.tensor(
                                    np.array(g_imp[f"meta_dsf{yd}"][:, fi]), dtype=torch.float32,
                                )
                                if float(ym_imp[0]) != -1.0:
                                    y_data_imp[j, :, fi] = torch.tensor(
                                        np.array(g_imp[f"counts_dsf{yd}"][wi, :, fi]),
                                        dtype=torch.float32,
                                    )
                        y_pval_imp[j] = torch.tensor(np.array(g_imp["pval"][wi]), dtype=torch.float32)
                        y_peaks_imp[j] = torch.tensor(np.array(g_imp["peaks"][wi]), dtype=torch.float32)
                    out["y_data_imp"] = y_data_imp
                    out["y_pval_imp"] = y_pval_imp
                    out["y_peaks_imp"] = y_peaks_imp
                    out["imp_biosample_name"] = str(imp_bios)
                    vb_meta_np = np.array(g_imp["meta_dsf1"])
                    vb_meta = torch.tensor(vb_meta_np, dtype=torch.float32)
                    out["y_meta_imp"] = vb_meta.unsqueeze(0).expand(B, -1, -1).clone()

        return out


SESSION2_VB_TRAIN_PAIRS = [
    ("T_DND-41", "V_DND-41"),
    ("T_RWPE2", "B_RWPE2"),
    ("T_heart_left_ventricle", "V_heart_left_ventricle"),
    ("T_H1-hESC", "V_H1-hESC"),
]


def _ensure_session2_vb_train_pins(manifest: Dict[str, Any], h5_path: Path) -> Dict[str, Any]:
    """Append 4 chr19 train entries with V/B meta loaded (A2 lite batches). Idempotent."""
    if manifest.get("session2_vb_train_added"):
        return manifest
    with h5py.File(h5_path, "r") as h5:
        chroms = [_decode_ch(x) for x in h5["windows/chrom"][:]]
    chr19_wi = [i for i, c in enumerate(chroms) if c == "chr19"]
    train: List[Dict[str, Any]] = list(manifest.get("train", []))
    offset_base = len(train)
    for i, (t, imp) in enumerate(SESSION2_VB_TRAIN_PAIRS):
        train.append({
            "t_bios": t,
            "imp_bios": imp,
            "window_indices": spread_pick(chr19_wi, BATCH_SIZE, offset=offset_base + i),
        })
    manifest = dict(manifest)
    manifest["train"] = train
    manifest["session2_vb_train_added"] = True
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    log.info("Session 2: appended %d VB-meta train pins (total train=%d)", len(SESSION2_VB_TRAIN_PAIRS), len(train))
    return manifest


def compute_primary(imp_r2: float, den_r2: float, dcr: float) -> Tuple[float, str]:
    in_imp = (
        math.isfinite(den_r2) and den_r2 >= DEN_GATE
        and math.isfinite(dcr) and DCR_LO <= dcr <= DCR_HI
    )
    if in_imp:
        return imp_r2 + IMP_PHASE_BIAS, "imp"
    return den_r2 - DEN_GATE, "den"


def load_baseline() -> Dict[str, Any]:
    if not BASELINE_JSON.exists():
        return {}
    try:
        return json.loads(BASELINE_JSON.read_text())
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def save_baseline(result: Dict[str, Any]) -> None:
    payload = {
        "peak_vram_mb": result["peak_vram_mb"],
        "training_seconds": result["training_seconds"],
        "timeout_seconds": result["training_seconds"] * 2,
        "primary_score": result["primary_score"],
        "metric_phase": result["metric_phase"],
        "imp_count_r2_gw": result["imp_count_r2_gw"],
        "den_count_r2_gw": result["den_count_r2_gw"],
        "depth_count_ratio": result["depth_count_ratio"],
        "imp_count_pearson_gw": result["imp_count_pearson_gw"],
    }
    BASELINE_JSON.write_text(json.dumps(payload, indent=2))


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def reset_vram_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def build_shell_model(device: torch.device, tc: agent_train.TrainConfig) -> Tuple[CANDIv2, Any]:
    cfg = build_real_v2_config(heads="count_only", lr=tc.lr, clip_norm=tc.clip_norm)
    cfg.decoder.count_head = "depth_offset"
    cfg.decoder.depth_center = float(tc.depth_center)
    cfg.encoder.signal_transform = tc.signal_transform
    cfg.training.loss_weights.obs_weight = tc.obs_weight
    cfg.training.loss_weights.imp_weight = tc.imp_weight
    cfg.training.loss_weights.count_weight = tc.count_weight
    torch.manual_seed(SEED)
    model = CANDIv2(cfg).to(device)
    loss_fn = build_v2_loss(cfg)
    return model, loss_fn


def train_loop(
    model: CANDIv2,
    loss_fn,
    train_batches: List[Dict[str, torch.Tensor]],
    masker,
    device: torch.device,
    tc: agent_train.TrainConfig,
    *,
    timeout_s: Optional[float],
) -> Tuple[float, float, float]:
    opt = agent_train.build_optimizer(model, tc)
    rng = random.Random(SEED)
    last_obs = float("nan")
    last_imp = float("nan")
    t_deadline = time.time() + timeout_s if timeout_s else None

    for step in range(1, MAX_STEPS + 1):
        if t_deadline is not None and time.time() > t_deadline:
            break
        batch = train_batches[(step - 1) % len(train_batches)]
        prep = prepare_masked_batch(batch, masker, device)
        if prep is None or not prep["masked_map"].any():
            continue
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, stats = agent_train.train_step(
            model, batch, prep, loss_fn, tc,
            global_step=step, rng=rng,
        )
        last_obs = stats.get("count_obs_loss", last_obs)
        last_imp = stats.get("count_imp_loss", last_imp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip_norm)
        opt.step()

    return last_obs, last_imp, float(step)


def print_summary(result: Dict[str, Any]) -> None:
    print("---")
    keys = (
        "primary_score", "metric_phase",
        "imp_count_r2_gw", "den_count_r2_gw", "depth_count_ratio",
        "imp_count_pearson_gw", "imp_count_spearman_gw",
        "        imp_count_r2_gw_cloze_T", "imp_count_r2_gw_canonical", "imp_r2_task_gap",
        "dcr_masked_bins", "count_imp_loss", "count_obs_loss",
        "training_seconds", "peak_vram_mb", "peak_vram_ok", "num_steps", "status", "error",
    )
    for key in keys:
        val = result.get(key)
        if isinstance(val, bool):
            print(f"{key + ':':28s}{val}")
        elif isinstance(val, float):
            print(f"{key + ':':28s}{val:.6f}")
        else:
            print(f"{key + ':':28s}{val}")
    print("---")


def run_experiment(device: Optional[torch.device] = None) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.92)

    tc = agent_train.get_config()
    baseline = load_baseline()
    timeout_s = baseline.get("timeout_seconds")
    timeout_s = float(timeout_s) if timeout_s is not None else None

    h5_path = H5_DEFAULT
    manifest = load_or_create_manifest(h5_path)
    manifest = _ensure_session2_vb_train_pins(manifest, h5_path)
    cache = H5PinCache(h5_path)
    canonical_meta = build_canonical_meta(str(EIC_METADATA), SANDBOX_ASSAYS)

    train_batches = [
        cache.load_pinned_batch(
            e, dsf_sampling=tc.dsf_sampling, train=True, rng_seed=SEED + i,
        )
        for i, e in enumerate(manifest["train"])
    ]
    eval_batches = [
        cache.load_pinned_batch(e, dsf_sampling="off", train=False, rng_seed=SEED + 1000 + i)
        for i, e in enumerate(manifest["eval_imp"])
    ]

    reset_vram_stats()
    t0 = time.time()
    status = "ok"
    model = None

    try:
        model, loss_fn = build_shell_model(device, tc)
        train_masker = make_masker(
            p_full_assay=TRAIN_P_FULL_ASSAY,
            p_full_loci=TRAIN_P_FULL_LOCI,
            p_chunks=TRAIN_P_CHUNKS,
        )
        reset_vram_stats()
        count_obs, count_imp, n_steps = train_loop(
            model, loss_fn, train_batches, train_masker, device, tc,
            timeout_s=timeout_s,
        )
        elapsed = time.time() - t0
        if timeout_s is not None and elapsed > timeout_s:
            status = "crash"

        eval_metrics = eval_pass.evaluate_suite(
            model, device, eval_batches, manifest["eval_imp"], canonical_meta,
            eval_cloze_t_index=int(manifest.get("eval_cloze_t_index", 0)),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        peak = peak_vram_mb()
        peak_ok = peak <= MAX_PEAK_VRAM_MB

        imp_r2 = eval_metrics.get("imp_count_r2_gw", float("nan"))
        den_r2 = eval_metrics.get("den_count_r2_gw", float("nan"))
        dcr = eval_metrics.get("depth_count_ratio", float("nan"))
        primary, phase = compute_primary(imp_r2, den_r2, dcr)
        cloze_r2 = eval_metrics.get("imp_count_r2_gw_cloze_T", float("nan"))
        task_gap = (
            cloze_r2 - imp_r2
            if math.isfinite(cloze_r2) and math.isfinite(imp_r2)
            else float("nan")
        )

        if not peak_ok:
            status = "crash"

        result: Dict[str, Any] = {
            "primary_score": primary,
            "metric_phase": phase,
            "imp_r2_task_gap": task_gap,
            "training_seconds": elapsed,
            "peak_vram_mb": peak,
            "peak_vram_ok": peak_ok,
            "num_steps": int(n_steps),
            "count_obs_loss": count_obs,
            "count_imp_loss": count_imp,
            "status": status,
            **eval_metrics,
        }

        if not BASELINE_JSON.exists() and status == "ok" and peak_ok:
            save_baseline(result)

    except torch.cuda.OutOfMemoryError:
        status = "crash"
        result = {
            "primary_score": -999.0,
            "metric_phase": "den",
            "training_seconds": time.time() - t0,
            "peak_vram_mb": peak_vram_mb(),
            "peak_vram_ok": False,
            "num_steps": MAX_STEPS,
            "status": "crash",
            "error": "OOM",
        }
    except Exception as exc:
        log.exception("run_experiment failed")
        result = {
            "primary_score": -999.0,
            "metric_phase": "den",
            "training_seconds": time.time() - t0,
            "peak_vram_mb": peak_vram_mb(),
            "peak_vram_ok": False,
            "num_steps": MAX_STEPS,
            "status": "crash",
            "error": str(exc),
        }
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print_summary(result)
    return result


def run_from_train() -> int:
    run_experiment()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_from_train())
