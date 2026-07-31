"""h48 — REAL-DATA (R) leg of the instrument validation gate. CPU only, no GPU, no SLURM.

The (L) logical and (C) code-correctness legs live in `tests/test_metrics_real.py` and run under pytest.
This script is the third leg: every fixed instrument is run on the FOUR existing checkpoints (and/or
`sandbox/data/sandbox.h5` directly) and the result is checked against a ground truth computed a
*different* way, plus the numeric predictions pre-registered in the h48 node.

    source candi_venv/bin/activate && export PYTHONNOUSERSITE=1
    python sandbox/diagnostics/dual_conditioning_real/h48_real_checks.py --fix s1 [--max-batches 60]

Eval units are expensive to rebuild (one encoder pass per batch, per checkpoint), so they are cached
under the scratch dir keyed by (tag, max_batches).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = "/project/6014832/mforooz/EpiDenoise"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np                                                              # noqa: E402
import torch                                                                    # noqa: E402

import sandbox.diagnostics.dual_conditioning_real.metrics_real as MR            # noqa: E402
from sandbox.diagnostics.dual_conditioning_real.metrics_real import EvalUnit    # noqa: E402
from sandbox.diagnostics.dual_conditioning_real.model_real import build_real_model  # noqa: E402

H5 = f"{REPO}/sandbox/data/sandbox.h5"
RES = f"{REPO}/sandbox/diagnostics/dual_conditioning_real/results"
CKPTS = {"main_s0_perassay": True, "offoff_s0_perassay": False,   # tag -> use_offset
         "wd0_on_s0": True, "wd0_off_s0": False}
CACHE = Path(os.environ.get("H48_CACHE", "/scratch/mforooz/tmp/h48_units"))
CPU = torch.device("cpu")

_UNIT_FIELDS = ("z", "y_meta_true", "den_map", "y_data", "imp_map", "y_data_imp", "y_meta_imp", "y_avail")


def load_model(tag: str):
    m = build_real_model(use_offset=CKPTS[tag])
    m.load_state_dict(torch.load(f"{RES}/{tag}.ckpt", map_location="cpu"))
    m.eval()
    return m


def get_units(tag: str, max_batches, model=None):
    """Build (or load) the chr21 eval units for `tag`. Units are encoder-dependent => cached per tag."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{tag}_{max_batches or 'all'}.pt"
    if f.exists():
        blob = torch.load(f, map_location="cpu", weights_only=False)
        return [EvalUnit(**d) for d in blob]
    t = time.time()
    units = MR.build_eval_units(model or load_model(tag), H5, CPU, max_batches=max_batches)
    print(f"  [{tag}] built {len(units)} units in {time.time() - t:.0f}s", flush=True)
    torch.save([{**{k: getattr(u, k) for k in _UNIT_FIELDS},
                 "biosample": u.biosample, "imp_biosample": u.imp_biosample} for u in units], f)
    return units


# ---------------------------------------------------------------------------
# S1 — total told-depth slope + cross-target metadata ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _byhand_assay_swap_eta(model, u, a, new_assay_id, n_pos=5):
    """Independent by-hand decode: change ONE scalar in the prompt, read eta at a few raw positions.
    Deliberately bypasses _gather/_metadata_ablation entirely."""
    ym = u.y_meta_true.clone()
    ym[:, 1, a] = float(new_assay_id)
    e0 = model.decoder(u.z, u.y_meta_true)["eta"][0, :n_pos, a].double().numpy()
    e1 = model.decoder(u.z, ym)["eta"][0, :n_pos, a].double().numpy()
    return e0, e1


@torch.no_grad()
def _anchor_probe(model, units, stride=1):
    """Replicate h47's V2 anchor protocol (whole-`assay_id`-row permutation, max|d eta| over ALL slots)
    and decompose it into the REAL->REAL identity part and the MISSING/CLOZE-sentinel part.

    `stride` SUBSAMPLES EVENLY -- never a prefix. These are max-statistics over a genomically ordered
    unit list whose leading units are ~8x sparser than the panel (audit S24), so `units[:n]` both
    understates the maxima and misattributes the coverage. The real->real mask also requires the rolled
    value to actually CHANGE, so a no-op roll cannot be scored as "no response".
    """
    allm, realm, sentm = [], [], []
    for u in units[::stride]:
        r1 = u.y_meta_true[:, 1, :]
        e0 = model.decoder(u.z, u.y_meta_true)["eta"]
        for sh in range(1, 8):
            rolled = torch.roll(r1, shifts=sh, dims=1)
            ym = u.y_meta_true.clone()
            ym[:, 1, :] = rolled
            d = (model.decoder(u.z, ym)["eta"] - e0).abs()
            allm.append(d.max().item())
            real = ((r1 >= 0) & (rolled >= 0) & (rolled != r1)).unsqueeze(1).expand_as(d)
            if real.any():
                realm.append(d[real].max().item())
            if (~real).any():
                sentm.append(d[~real].max().item())
    return (max(allm) if allm else float("nan"), max(realm) if realm else float("nan"),
            max(sentm) if sentm else float("nan"))


@torch.no_grad()
def _assay_id_sweep(model, units, stride=5):
    """Third, independent identity probe: at each REAL slot hold z and every other row fixed and sweep
    `assay_id` over all 8 values; report the largest eta spread. No permutation, no sentinel.
    `stride` subsamples EVENLY (see `_anchor_probe`) -- never a prefix."""
    spreads = []
    for u in units[::stride]:
        for slot in range(8):
            if float(u.y_meta_true[0, 1, slot]) < 0:
                continue
            etas = []
            for aid in range(8):
                ym = u.y_meta_true.clone()
                ym[:, 1, slot] = float(aid)
                etas.append(model.decoder(u.z, ym)["eta"][:, :, slot])
            E = torch.stack(etas)
            spreads.append((E.max(0).values - E.min(0).values).max().item())
    return max(spreads) if spreads else float("nan")


def check_s1(max_batches):
    print("\n=== S1 (R): total told-depth slope + real-z cross-target metadata ablation ===")
    print("PREDICT: main_s0 total slope = 1.000 (offset ON, embedder annihilated);")
    print("         assay-ablation: main ~ 0 (dead embedder) vs wd0_on > 0 (h47 anchor max|dEta| 0.833)\n")
    rows = []
    for tag in CKPTS:
        m = load_model(tag)
        units = get_units(tag, max_batches, m)
        sw = MR._depth_sweep(m, units, CPU, fg_frac=0.02, n_boot=200, seed=0)
        abl = {r: MR._metadata_ablation(m, units, r, fg_frac=0.02, seed=0) for r in (0, 1, 2, 3)}
        # independent cross-check of the assay ablation on the first held-out target
        u = next(u for u in units if u.imp_map is not None and u.imp_target_assays())
        a = u.imp_target_assays()[0]
        e0, e1 = _byhand_assay_swap_eta(m, u, a, (a + 3) % 8)
        anc_all, anc_real, anc_sent = _anchor_probe(m, units)
        rows.append(dict(tag=tag, n_targets=sw["n_targets"],
                         total=sw["median_total_slope"], eta=sw["median_eta_slope"],
                         assay_deta=abl[1]["mean_abs_d_eta"], assay_deta_max=abl[1]["max_abs_d_eta"],
                         assay_dcrps=abl[1]["mean_d_crps"],
                         depth_deta=abl[0]["mean_abs_d_eta"], rt_deta=abl[3]["mean_abs_d_eta"],
                         byhand=float(np.max(np.abs(e1 - e0))),
                         anc_all=anc_all, anc_real=anc_real, anc_sent=anc_sent,
                         sweep=_assay_id_sweep(m, units)))
    print(f"{'tag':<20}{'n_t':>4}{'total_slope':>13}{'eta_slope':>11}"
          f"{'assay|dEta|':>13}{'assay dCRPS':>13}{'depth|dEta|':>13}{'rt|dEta|':>12}{'byhand':>12}")
    for r in rows:
        print(f"{r['tag']:<20}{r['n_targets']:>4}{r['total']:>13.4f}{r['eta']:>11.4f}"
              f"{r['assay_deta']:>13.4e}{r['assay_dcrps']:>13.4f}{r['depth_deta']:>13.4e}"
              f"{r['rt_deta']:>12.4e}{r['byhand']:>12.4e}")
    print("\n  assay-IDENTITY steering, max|d eta| -- four protocols, three of them independent of the "
          "instrument:")
    print(f"  {'tag':<20}{'h47 anchor':>12}{'  |  REAL->REAL':>16}{'sentinel':>12}"
          f"{'  ||  8-id sweep':>17}{'x-target abl':>14}")
    for r in rows:
        print(f"  {r['tag']:<20}{r['anc_all']:>12.4f}{r['anc_real']:>16.4f}{r['anc_sent']:>12.4f}"
              f"{r['sweep']:>17.4f}{r['assay_deta_max']:>14.4f}")
    d = {r["tag"]: r for r in rows}
    ok = [
        ("main_s0 total slope == 1.000 (+-0.001)", abs(d["main_s0_perassay"]["total"] - 1.0) < 1e-3),
        ("main_s0 assay-ablation |dEta| ~ 0 (dead embedder)", d["main_s0_perassay"]["assay_deta"] < 1e-6),
        ("wd0_on assay-ablation |dEta| > main (revived, however weakly)",
         d["wd0_on_s0"]["assay_deta"] > 100 * max(d["main_s0_perassay"]["assay_deta"], 1e-12)),
        # the anchor protocol must be REPRODUCED before it can be corrected
        ("h47 anchor protocol reproduced on wd0_on (max|dEta| >= 0.833)",
         d["wd0_on_s0"]["anc_all"] >= 0.833),
        ("h47 anchor is sentinel-driven (sentinel part >= 100x the REAL->REAL part)",
         d["wd0_on_s0"]["anc_sent"] > 100 * d["wd0_on_s0"]["anc_real"]),
        # the three sentinel-free identity probes must agree with each other to within an order of magnitude
        ("3 independent identity probes agree on wd0_on (REAL->REAL ~ sweep ~ x-target)",
         max(d["wd0_on_s0"][k] for k in ("anc_real", "sweep", "assay_deta_max"))
         < 10 * min(d["wd0_on_s0"][k] for k in ("anc_real", "sweep", "assay_deta_max"))),
    ]
    for tag in CKPTS:                      # by-hand decode must agree with the instrument's verdict
        instr_dead = d[tag]["assay_deta"] < 1e-6
        ok.append((f"{tag}: by-hand decode agrees with instrument (dead={instr_dead})",
                   (d[tag]["byhand"] < 1e-6) == instr_dead))
    for name, passed in ok:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return all(p for _, p in ok)


# ---------------------------------------------------------------------------
# S3 — rank-based foreground on the REAL held-out targets
# ---------------------------------------------------------------------------

def check_s3(max_batches):
    print("\n=== S3 (R): realised foreground fraction on the real held-out targets ===")
    print("PREDICT: realised ~ 2% (requested), not the 23.9% the quantile selector gave;")
    print("         no target at 100%; selected positions are genuinely the highest counts\n")
    # the mask depends only on the TARGET, so one checkpoint's units suffice (identical GT for all four)
    units = get_units("main_s0_perassay", max_batches)
    old, new, purity, empt, full, per_key = [], [], [], 0, 0, {}
    for u in units:
        if u.imp_map is None:
            continue
        for a in u.imp_target_assays():
            m = u.imp_map[:, :, a]
            tgt = u.y_data_imp[:, :, a][m].double().cpu().numpy()
            if tgt.size == 0:
                continue
            old.append(float((tgt >= np.quantile(tgt, 0.98)).mean()))
            fg = MR._foreground_mask(tgt, 0.02)
            new.append(float(fg.mean()))
            empt += int(fg.sum() == 0)
            full += int(fg.all())
            purity.append(bool(fg.sum() == 0 or (~fg).sum() == 0
                               or tgt[fg].min() >= tgt[~fg].max()))
            key = (u.biosample, u.imp_biosample, MR.SANDBOX_ASSAYS[a])
            per_key.setdefault(key, []).append((old[-1], new[-1]))
    old, new = np.array(old), np.array(new)
    print(f"  {len(old)} target-records over {len(per_key)} distinct (biosample, imp_biosample, assay) "
          f"targets\n")
    print(f"  {'target':<44}{'OLD realised':>14}{'NEW realised':>14}")
    for key, v in sorted(per_key.items()):
        v = np.array(v)
        print(f"  {'/'.join(key):<44}{v[:, 0].mean():>14.4f}{v[:, 1].mean():>14.4f}")
    print(f"\n  OLD: mean {old.mean():.4f}  max {old.max():.4f}  frac-of-records-at-100% "
          f"{float((old >= 0.999).mean()):.4f}")
    print(f"  NEW: mean {new.mean():.4f}  max {new.max():.4f}  frac-of-records-at-100% "
          f"{float((new >= 0.999).mean()):.4f}")
    ok = [("OLD selector reproduces the audit pathology (mean >> 2%, some records at 100%)",
           old.mean() > 0.10 and (old >= 0.999).any()),
          ("NEW realised fraction ~ requested 2% on every target (<= 0.025)", new.max() <= 0.025),
          ("no target selected at 100%", full == 0),
          ("no target selected empty", empt == 0),
          (f"selected positions are the highest counts on all {len(purity)} records", all(purity))]
    for name, passed in ok:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return all(p for _, p in ok)


# ---------------------------------------------------------------------------
# S4 — target-clustered CIs, re-aggregated from the STORED result JSONs (no re-decode)
# ---------------------------------------------------------------------------

def check_s4(max_batches):
    import json
    print("\n=== S4 (R): target-clustered CIs re-aggregated from the stored M2 records ===")
    print("PREDICT: offoff_s0 run_type [+0.0655,+0.0739] -> ~[-0.022,+0.181], crossing 0 (audit S4)\n")
    print(f"  {'tag':<20}{'cov':<12}{'position-level CI':>28}{'n':>9}   {'target-clustered CI':>26}"
          f"{'n_cl':>6}{'signp':>8}  dir?")
    ok, got = [], {}
    for tag in CKPTS:
        with open(f"{RES}/{tag}.json") as fh:
            d = json.load(fh)
        for cov in ("run_type", "read_length"):
            blk = d["M2"][cov]
            old, pt = blk["overall"], blk["per_target"]
            new = MR._cluster_bootstrap_ci(pt, n_boot=2000, seed=0)
            got[(tag, cov)] = (old, new)
            print(f"  {tag:<20}{cov:<12}[{old['lo']:+.4f},{old['hi']:+.4f}]{old['n']:>16,}   "
                  f"[{new['lo']:+.4f},{new['hi']:+.4f}]{new['n_clusters']:>10}{new['sign_test_p']:>8.3f}"
                  f"  {'YES' if new['supports_direction'] else 'no'}")
    o, n = got[("offoff_s0_perassay", "run_type")]
    ok.append(("stored offoff run_type position CI == the audit's [+0.0655,+0.0739]",
               abs(o["lo"] - 0.06546) < 1e-4 and abs(o["hi"] - 0.07391) < 1e-4))
    ok.append((f"clustered CI reproduces the audit's [-0.022,+0.181] (got [{n['lo']:+.4f},{n['hi']:+.4f}])",
               abs(n["lo"] - (-0.0215)) < 0.02 and abs(n["hi"] - 0.1810) < 0.03))
    ok.append(("...and it now CROSSES ZERO (h42's headline does not survive clustering)", n["lo"] < 0.0))
    ok.append(("n_clusters == 12 real targets everywhere",
               all(v[1]["n_clusters"] == 12 for v in got.values())))
    # (a bit-exactly blind arm gives a zero-width CI under BOTH schemes -- require strict widening only
    #  where the position-level CI had width to begin with)
    ok.append(("every non-degenerate position-level CI was narrower than its clustered counterpart",
               all((v[1]["hi"] - v[1]["lo"]) >= (v[0]["hi"] - v[0]["lo"]) for v in got.values())
               and all((v[1]["hi"] - v[1]["lo"]) > (v[0]["hi"] - v[0]["lo"])
                       for v in got.values() if (v[0]["hi"] - v[0]["lo"]) > 1e-9)))
    ok.append(("no clustered CI still supports a direction the audit says is unsupported "
               "(offoff/wd0_off run_type)",
               not got[("offoff_s0_perassay", "run_type")][1]["supports_direction"]
               and not got[("wd0_off_s0", "run_type")][1]["supports_direction"]))
    widen = [(v[1]["hi"] - v[1]["lo"]) / (v[0]["hi"] - v[0]["lo"])
             for v in got.values() if (v[0]["hi"] - v[0]["lo"]) > 1e-9]
    print(f"\n  CI widening from clustering: median {np.median(widen):.1f}x  (audit expected ~24x)")
    for name, passed in ok:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return all(p for _, p in ok)


# ---------------------------------------------------------------------------
# S5/S18 — honest marginal + oracle per-assay scale decomposition
# ---------------------------------------------------------------------------

def check_s5(max_batches, budget=100_000):
    print("\n=== S5/S18 (R): honest per-assay marginal + oracle scale decomposition ===")
    print("PREDICT: marginal non-degenerate on all 8 assays (legacy is a point mass at 0 on ~4/8);")
    print("         decomposition additive; ON/OFF macro-CRPS ordering COMPRESSES under oracle scale;")
    print("         beats-marginal falls 8/8 -> ~5/8\n")
    out = {}
    for tag in CKPTS:
        m = load_model(tag)
        units = get_units(tag, max_batches, m)
        out[tag] = MR.eval_M1(m, units, CPU, budget=budget, seed=0)
    print(f"  {'tag':<20}{'macroCRPS':>11}{'oracle-scaled':>15}{'scale_err':>11}{'+n-oracle':>11}"
          f"{'beats-marg':>12}{'(oracle)':>10}{'legacy-marg':>13}")
    for tag, r in out.items():
        pa = r["imp_per_assay"]
        on = float(np.mean([v["crps_oracle_scaled_and_n"] for v in pa.values()]))
        legacy = int(sum(v["crps"] <= v["marg_crps_legacy_median"] for v in pa.values()))
        print(f"  {tag:<20}{r['imp_macro_crps']:>11.4f}{r['imp_macro_crps_oracle_scaled']:>15.4f}"
              f"{r['imp_macro_scale_error']:>11.4f}{on:>11.4f}"
              f"{r['imp_beats_marginal_n']:>10}/8{r['imp_beats_marginal_oracle_scaled_n']:>8}/8"
              f"{legacy:>11}/8")
    print(f"\n  per-assay oracle scale c* (log2 units; |c*| large => a fixable level error):")
    for tag, r in out.items():
        cs = {a: v["c_star"] for a, v in r["imp_per_assay"].items()}
        print(f"  {tag:<20}" + "  ".join(f"{a[:9]}:{c:+.2f}" for a, c in sorted(cs.items())))
    print(f"\n  degenerate LEGACY marginals (mu ~ 0 => 'predict nothing'), per checkpoint:")
    for tag, r in out.items():
        deg = [a for a, v in r["imp_per_assay"].items() if v["marg_mu_legacy_median"] < 1e-3]
        new_min = min(v["marg_mu"] for v in r["imp_per_assay"].values())
        print(f"  {tag:<20}legacy degenerate {len(deg)}/8 {sorted(deg)}; new min marg_mu {new_min:.4f}")

    spread = lambda key: (max(r[key] for r in out.values()) - min(r[key] for r in out.values()))
    raw, orc = spread("imp_macro_crps"), spread("imp_macro_crps_oracle_scaled")
    print(f"\n  macro-CRPS spread across the 4 arms: raw {raw:.4f} -> oracle-scaled {orc:.4f} "
          f"({100 * (1 - orc / raw):.0f}% compression)")
    ok = [
        ("honest marginal is non-degenerate (mu > 0) on all 8 assays, all 4 ckpts",
         all(v["marg_mu"] > 0 for r in out.values() for v in r["imp_per_assay"].values())),
        ("legacy marginal WAS degenerate on >= 3/8 assays (reproduces the audit's 4/8)",
         all(sum(v["marg_mu_legacy_median"] < 1e-3 for v in r["imp_per_assay"].values()) >= 3
             for r in out.values())),
        ("honest marginal is never worse than the legacy one (it is the CRPS-optimal constant)",
         all(v["marg_crps"] <= v["marg_crps_legacy_median"] + 1e-9
             for r in out.values() for v in r["imp_per_assay"].values())),
        ("decomposition additive: crps == crps_oracle_scaled + scale_error",
         all(abs(v["crps"] - v["crps_oracle_scaled"] - v["scale_error"]) < 1e-9
             for r in out.values() for v in r["imp_per_assay"].values())),
        ("oracle scaling never hurts (crps_oracle_scaled <= crps)",
         all(v["crps_oracle_scaled"] <= v["crps"] + 1e-9
             for r in out.values() for v in r["imp_per_assay"].values())),
        ("the 4-arm macro-CRPS ordering COMPRESSES under oracle scale (>= 25%)", orc < 0.75 * raw),
        ("beats-marginal falls below 8/8 on at least one arm (audit: ON 8/8 -> 5/8)",
         any(r["imp_beats_marginal_n"] < 8 for r in out.values())),
    ]
    for name, passed in ok:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return all(p for _, p in ok)


# ---------------------------------------------------------------------------
# S6 — RECOVER the h5 assay column order from the data (h5 <-> eic_metadata.csv full-triple join)
# ---------------------------------------------------------------------------

def check_s6(max_batches):
    import csv
    import json
    import h5py
    print("\n=== S6 (R): recover the h5 assay column order by joining meta_dsf1 to eic_metadata.csv ===")
    print("PREDICT: 38/38 (biosample, column) triples match H5_ASSAY_ORDER, 5/38 match SANDBOX_ASSAYS\n")
    rows = list(csv.DictReader(open(f"{REPO}/data/eic_metadata.csv")))
    meta = {}
    for r in rows:
        meta[(r["biosample_name"], r["assay_name"])] = (
            float(np.log2(float(r["depth"]))), float(r["read_length"]),
            1.0 if r["run_type"] == "paired-ended" else 0.0)

    def _matches(triple, name, bio):
        ref = meta.get((bio, name))
        if ref is None:
            return False
        return (abs(triple[0] - ref[0]) < 0.02 and triple[1] == ref[1] and triple[2] == ref[2])

    observed, recovered, n_amb = [], [], 0
    with h5py.File(f"{REPO}/sandbox/data/sandbox.h5", "r") as h:
        order = json.loads(h["biosamples"].attrs["order"])
        for bio in order:
            md = np.array(h["biosamples"][bio]["meta_dsf1"])            # [4, 8]
            for col in range(md.shape[1]):
                if float(md[0, col]) == -1.0:
                    continue
                triple = (float(md[0, col]), float(md[2, col]), float(md[3, col]))
                observed.append((bio, col, triple))
                hits = [a for a in MR.SANDBOX_ASSAYS if _matches(triple, a, bio)]
                n_amb += int(len(hits) > 1)
                recovered.append((col, hits[0] if len(hits) == 1 else None))
    n = len(observed)
    hit_new = sum(_matches(t, MR.H5_ASSAY_ORDER[c], b) for b, c, t in observed)
    hit_old = sum(_matches(t, MR.SANDBOX_ASSAYS[c], b) for b, c, t in observed)
    print(f"  {n} (biosample, column) records with a real metadata triple")
    print(f"    H5_ASSAY_ORDER  : {hit_new}/{n} match")
    print(f"    SANDBOX_ASSAYS  : {hit_old}/{n} match")
    print(f"    ambiguous joins (a triple matching >1 assay for its biosample): {n_amb}")
    # independently RECOVER the order: each column must resolve to exactly one assay, consistently
    by_col = {}
    for col, name in recovered:
        by_col.setdefault(col, set()).add(name)
    rec_order = [next(iter(by_col[c])) if len(by_col.get(c, set())) == 1 else None
                 for c in range(8)]
    print(f"\n  recovered from data : {rec_order}")
    print(f"  H5_ASSAY_ORDER      : {MR.H5_ASSAY_ORDER}")
    print(f"  SANDBOX_ASSAYS      : {list(MR.SANDBOX_ASSAYS)}")
    print(f"\n  relabel of the two headline per-assay claims:")
    for col, claim in ((0, "'ATAC-seq contributes 56% of the CRPS gap'"),
                       (4, "'H3K27ac collapses (0.625 -> 0.010)'")):
        print(f"    col {col}: {claim}  ->  really {MR.H5_ASSAY_ORDER[col]} "
              f"(was labelled {MR.SANDBOX_ASSAYS[col]})")
    ok = [(f"all {n} triples match H5_ASSAY_ORDER", hit_new == n),
          (f"SANDBOX_ASSAYS matches only a handful ({hit_old}/{n}, audit says 5/38)", hit_old < n // 2),
          ("the order recovered from the data IS H5_ASSAY_ORDER", rec_order == MR.H5_ASSAY_ORDER),
          ("no ambiguous joins (each triple identifies exactly one assay)", n_amb == 0),
          ("relabel is a bijection", sorted(MR.H5_ASSAY_ORDER) == sorted(MR.SANDBOX_ASSAYS))]
    for name, passed in ok:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return all(p for _, p in ok)


# ---------------------------------------------------------------------------
# S14/S23 — DSF counterfactual + z-controlled recoverability
# ---------------------------------------------------------------------------

def check_s14(max_batches):
    import json
    import h5py
    print("\n=== S14/S23 (R): DSF counterfactual + z-controlled condition recoverability ===")
    print("GATE: real counts_dsf{k} really are ~1/k downsamples of counts_dsf1")
    print("PREDICT: assay recoverable for wd0_on, chance for main; run_type chance for ALL (bound B1)\n")

    # --- gate: is counts_dsf{k} a 1/k downsample? -------------------------------------------------
    ratios = {}
    with h5py.File(f"{REPO}/sandbox/data/sandbox.h5", "r") as h:
        order = json.loads(h["biosamples"].attrs["order"])
        for k in (2, 4, 8):
            rs = []
            for b in order:
                g = h["biosamples"][b]
                md = np.array(g["meta_dsf1"])
                for a in range(8):
                    if float(md[0, a]) == -1.0:
                        continue
                    c1 = np.array(g["counts_dsf1"][:40, :, a]).astype(float)
                    ck = np.array(g[f"counts_dsf{k}"][:40, :, a]).astype(float)
                    if c1.sum() > 0:
                        rs.append(ck.sum() / c1.sum())
            ratios[k] = (float(np.mean(rs)), float(np.min(rs)), float(np.max(rs)), len(rs))
    print("  counts_dsf{k} / counts_dsf1 total-count ratio (expect ~1/k):")
    for k, (mn, lo, hi, n) in ratios.items():
        print(f"    k={k}: mean {mn:.4f} (expect {1/k:.4f})  range [{lo:.4f},{hi:.4f}]  n={n}")

    # --- the two instruments on the 4 checkpoints -------------------------------------------------
    rows = []
    for tag in CKPTS:
        m = load_model(tag)
        units = get_units(tag, max_batches, m)
        dsf = MR._dsf_counterfactual(m, H5, CPU, n_region_draws=4, seed=0)
        p_assay = MR._recoverability_probe(m, units, 1, list(range(8)), seed=0)
        p_assay_naive = MR._recoverability_probe(m, units, 1, list(range(8)),
                                                 z_controlled=False, seed=0)
        p_rt = MR._recoverability_probe(m, units, 3, [0, 1], seed=0)
        p_depth = MR._recoverability_probe(
            m, units, 0, [22.0, 24.0, 26.0, 28.0], seed=0)
        rows.append(dict(tag=tag, dsf=dsf, assay=p_assay, assay_naive=p_assay_naive,
                         rt=p_rt, depth=p_depth))
    print("\n  S14 depth counterfactual (each told depth vs its OWN counts_dsf{k} ground truth):")
    print(f"  {'tag':<20}{'n_targets':>10}{'frac_min_at_true':>18}{'frac beats told=1':>20}")
    for r in rows:
        d = r["dsf"]
        print(f"  {r['tag']:<20}{d['n_targets']:>10}{d['frac_min_at_true']:>18.4f}"
              f"{d['frac_beats_told1']:>20.4f}")
    print("  (chance for frac_min_at_true is 0.25 with 4 told levels)")
    print("\n  S23 recoverability from the decoder OUTPUT (leave-one-target-out accuracy),")
    print("  with the probe's feature energy = the MAGNITUDE of the response it is classifying:")
    print(f"  {'tag':<20}{'assay (z-ctrl)':>16}{'energy':>10}{'assay (NAIVE)':>15}"
          f"{'run_type':>11}{'energy':>10}{'depth':>9}{'energy':>10}")
    for r in rows:
        print(f"  {r['tag']:<20}{r['assay']['accuracy']:>16.4f}{r['assay']['feature_energy']:>10.2e}"
              f"{r['assay_naive']['accuracy']:>15.4f}"
              f"{r['rt']['accuracy']:>11.4f}{r['rt']['feature_energy']:>10.2e}"
              f"{r['depth']['accuracy']:>9.4f}{r['depth']['feature_energy']:>10.2e}")
    print("  chance: assay 0.125 / run_type 0.500 / depth 0.250")
    d = {r["tag"]: r for r in rows}
    ok = [
        # --- the h48 gate for S14/S23: the dsf ground truth is real (the stub legs live in tests/) ---
        ("GATE: counts_dsf{k} is a ~1/k downsample of counts_dsf1 (within 10%)",
         all(abs(ratios[k][0] - 1.0 / k) < 0.1 / k for k in (2, 4, 8))),
        ("GATE: the DSF counterfactual resolved all 12 held-out targets on all 4 arms",
         all(r["dsf"]["n_targets"] == 12 for r in rows)),
        # --- instrument sanity on real checkpoints ---
        ("main_s0 (bit-exactly dead embedder) reads CHANCE on both z-controlled probes",
         d["main_s0_perassay"]["assay"]["accuracy"] <= 0.125 + 1e-9
         and d["main_s0_perassay"]["rt"]["accuracy"] <= 0.5 + 1e-9),
        # (centring identical rows leaves float-rounding residue, not an exact 0, on large features)
        ("...with nothing to classify: its probe feature energy is at float-rounding level",
         d["main_s0_perassay"]["assay"]["feature_energy"] < 1e-12
         and d["main_s0_perassay"]["rt"]["feature_energy"] < 1e-12),
        ("the probe is not simply dead: it recovers assay well above chance on >= 1 live arm",
         any(r["assay"]["accuracy"] > 0.125 + 0.10 for r in rows)),
        ("the NAIVE (non-z-controlled) probe is NOT the one carrying the signal",
         all(r["assay_naive"]["accuracy"] < 0.5 for r in rows)),
    ]
    print()
    for name, passed in ok:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("""
  READING THESE NUMBERS (results, not gate criteria):
  * S14: NO arm passes the real depth counterfactual -- `frac_min_at_true` sits at chance (0.25) and
    told=k beats told=1 on 3-17% of (target, level) pairs. The CRPS matrix shows why: CRPS rises
    monotonically as the told depth falls, at EVERY ground-truth level, because the models already
    under-predict mu. So the old sweep's `dir_mean_delta > 0` really was guaranteed by construction --
    against a genuine counterfactual ground truth the depth response is not beneficial.
  * S23 is a pathway ALIVE/DEAD detector, not a magnitude measure and NOT a test of bound B1. With `z`
    held fixed and features target-centred there is no noise, so ANY deterministic response classifies
    perfectly however tiny it is -- note `wd0_on` recovers run_type at 1.000 off a feature energy of
    ~1e-3. B1 is a property of the training DATA (H(run_type|assay,read_length)=0 bits); no output probe
    can test it. Read every "recoverable" together with the S1 effect sizes.""")
    return all(p for _, p in ok)


FIXES = {"s1": check_s1, "s3": check_s3, "s4": check_s4, "s5": check_s5, "s6": check_s6,
         "s14": check_s14}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", required=True, choices=sorted(FIXES) + ["all"])
    # 0 = ALL 608 chr21 eval units. Do NOT truncate for anything quantitative: `shuffle=False` makes a
    # prefix a spatially-contiguous block, and the first units are ~8x sparser than the panel (audit
    # S24) -- at 60 batches every held-out target is essentially all-zero and macro CRPS reads ~0.
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    names = sorted(FIXES) if a.fix == "all" else [a.fix]
    results = {n: FIXES[n](a.max_batches or None) for n in names}
    print("\n" + "=" * 60)
    for n, ok in results.items():
        print(f"{n.upper()}: {'GREEN' if ok else 'RED'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
