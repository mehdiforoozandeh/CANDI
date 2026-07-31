"""h48 — CPU re-score of the four existing checkpoints with the CORRECTED instruments.

Runs only AFTER the validation gate is green (all six fixes verified three ways: `pytest tests/` for the
(L)/(C) legs, `h48_real_checks.py --fix all` for the (R) legs). No retraining, no GPU, no SLURM -- the
four checkpoints are the only input.

    source candi_venv/bin/activate && export PYTHONNOUSERSITE=1
    python sandbox/diagnostics/dual_conditioning_real/h48_rescore.py

Writes `results/{tag}_h48rescored.json` per checkpoint plus `results/H48_SCORECARD.md`.

M3 is NOT re-run: it is encoder-only and none of the six fixes touch it, so its stored values are
carried forward unchanged rather than burning CPU to reproduce them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = "/project/6014832/mforooz/EpiDenoise"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np                                                              # noqa: E402
import torch                                                                    # noqa: E402

import sandbox.diagnostics.dual_conditioning_real.metrics_real as MR            # noqa: E402
from sandbox.diagnostics.dual_conditioning_real.h48_real_checks import (        # noqa: E402
    CKPTS, CPU, H5, RES, get_units, load_model,
)

# the recorded (broken-instrument) anchors these re-scores replace
OLD = {"main_s0_perassay": dict(macro_crps=1.495, imp_sp=0.533, ece=0.062, beats=8),
       "offoff_s0_perassay": dict(macro_crps=1.902, imp_sp=0.401, ece=0.097, beats=3),
       "wd0_on_s0": dict(macro_crps=1.341, imp_sp=0.637, ece=0.053, beats=8),
       "wd0_off_s0": dict(macro_crps=2.056, imp_sp=0.380, ece=0.078, beats=1)}


def rescore(tag, *, budget, fg_frac, n_boot, seed, max_batches):
    m = load_model(tag)
    units = get_units(tag, max_batches, m)
    t0 = time.time()
    M1 = MR.eval_M1(m, units, CPU, budget=budget, seed=seed)
    print(f"    M1 {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    M2 = MR.eval_M2(m, units, CPU, fg_frac=fg_frac, n_boot=n_boot, seed=seed)
    print(f"    M2 {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    extra = dict(
        dsf_counterfactual=MR._dsf_counterfactual(m, H5, CPU, n_region_draws=4, fg_frac=fg_frac,
                                                  seed=seed),
        recoverability=dict(
            assay=MR._recoverability_probe(m, units, 1, list(range(8)), fg_frac=fg_frac, seed=seed),
            assay_not_z_controlled=MR._recoverability_probe(m, units, 1, list(range(8)),
                                                            z_controlled=False, fg_frac=fg_frac,
                                                            seed=seed),
            run_type=MR._recoverability_probe(m, units, 3, [0, 1], fg_frac=fg_frac, seed=seed),
            depth=MR._recoverability_probe(m, units, 0, [22.0, 24.0, 26.0, 28.0], fg_frac=fg_frac,
                                           seed=seed)))
    print(f"    S14/S23 {time.time() - t0:.0f}s", flush=True)
    with open(f"{RES}/{tag}.json") as fh:
        M3 = json.load(fh).get("M3")
    return dict(tag=tag, n_units=len(units), M1=M1, M2=M2, M3_carried_forward=M3, **extra)


def _fmt(v, nd=4):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def scorecard(res):
    L = ["# h48 — proposed re-scored scorecard (corrected instruments)", "",
         "CPU re-score of the four existing q19/h47 checkpoints with the six h48 instrument fixes "
         "(S1, S3, S4, S5/S18, S6, S14/S23). No retraining, no GPU. M1 and M2 use full chr21 eval "
         f"coverage ({res[list(res)[0]]['n_units']} units, 1215 target-records over 12 held-out "
         "targets); the S14 columns use 4 region-draws x 4 windows and the S23 columns a 120-record "
         "even stride — **the header coverage does NOT apply to the last table.**", "",
         "**This is a PROPOSED scorecard. The verdict on h48 is the PI's call — the node is not closed.**",
         "Post-verification (6 adversarial skeptics + a critic); read `../H48_REPORT.md` §2-§3 for the "
         "claims these numbers do and do not support.", "",
         "## M1 — magnitude, decomposed (S5/S18)", "",
         "`crps_oracle_scaled` is the CAPABILITY term (CRPS after granting each assay its oracle "
         "multiplicative scale `c*`); `scale_error` is the FIXABLE per-assay calibration term. "
         "Macro CRPS mixes the two, which is what made the ON/OFF difference look like a Pareto.", "",
         "> **The first two columns are a REGRESSION CHECK — they are expected to be identical.** Both "
         "are the same computation on the same points; differences of ~1e-4 are CPU-vs-GPU float and "
         "confirm the re-score recomputes from the checkpoint rather than reading a cache.",
         "> **The capability column has a target-clustered noise floor of ~0.09.** Only `wd0_on`'s lead "
         "is significant (offoff - wd0_on = +0.093 [+0.004, +0.217]); the other three are statistically "
         "indistinguishable (pairwise CIs all cover 0). Do NOT read the 4-dp ordering as a ranking.",
         "> `crps_oracle_scaled` is an **in-sample upper bound**: `c*` is fitted on the same 12 targets "
         "it scores, and `scale_error` can be slightly negative (subsample fit, full-pool evaluation).", "",
         "| checkpoint | macro CRPS (old anchor) | macro CRPS (new) | **oracle-scaled (capability)** | "
         "scale_error | macro Sp | imp Sp (pooled) | ECE | beats-marg (honest) | beats-marg (legacy) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, r in res.items():
        m1 = r["M1"]
        pa = m1["imp_per_assay"]
        legacy = sum(v["crps"] <= v["marg_crps_legacy_median"] for v in pa.values())
        L.append(
            f"| `{tag}` | {OLD[tag]['macro_crps']:.4f} | {_fmt(m1['imp_macro_crps'])} | "
            f"**{_fmt(m1['imp_macro_crps_oracle_scaled'])}** | {_fmt(m1['imp_macro_scale_error'])} | "
            f"{_fmt(m1['imp_macro_spearman'])} | {_fmt(m1['imp'].get('spearman'))} | "
            f"{_fmt(m1['imp'].get('ece'))} | {m1['imp_beats_marginal_n']}/8 | {legacy}/8 |")
    raw = [r["M1"]["imp_macro_crps"] for r in res.values()]
    orc = [r["M1"]["imp_macro_crps_oracle_scaled"] for r in res.values()]
    L += ["", f"Spread across the four arms: raw macro CRPS **{max(raw) - min(raw):.4f}** → "
              f"oracle-scaled **{max(orc) - min(orc):.4f}** "
              f"({100 * (1 - (max(orc) - min(orc)) / (max(raw) - min(raw))):.0f}% compression). "
              "This is the audit §6.5 question: most of the apparent ON/OFF magnitude frontier is a "
              "per-assay scale-calibration difference, not a capability difference.", ""]

    L += ["## M1 — per-assay, with CORRECTED labels (S6)", "",
          "h5 column order is the CANDI handler's `experiment_aliases` order, not `SANDBOX_ASSAYS`; "
          "every previously reported per-assay label was permuted (38/38 vs 5/38 on a metadata join). "
          "Training saw only integer assay ids, so the numbers stand — only the biology reading changes.",
          "", "| checkpoint | " + " | ".join(MR.H5_ASSAY_ORDER) + " |",
          "|---" * (len(MR.H5_ASSAY_ORDER) + 1) + "|"]
    for tag, r in res.items():
        pa = r["M1"]["imp_per_assay"]
        L.append(f"| `{tag}` CRPS | " + " | ".join(
            _fmt(pa[a]["crps"], 3) if a in pa else "n/a" for a in MR.H5_ASSAY_ORDER) + " |")
        L.append(f"| `{tag}` c* | " + " | ".join(
            f"{pa[a]['c_star']:+.2f}" if a in pa else "n/a" for a in MR.H5_ASSAY_ORDER) + " |")

    L += ["", "## M2 — steering, target-clustered (S4) and sentinel-free (S1)", "",
          "CIs are now bootstrapped over the **12 targets** (n_fg-weighted), not the ~893k positions; "
          "`supports_direction` is sign-aware. `assay Δη` is the cross-target, sentinel-free ablation.",
          "", "| checkpoint | total depth slope | eta_slope (diagnostic) | assay mean\\|Δη\\| | "
          "assay max\\|Δη\\| | run_type clustered CI | supports dir? | sign-test p |",
          "|---|---|---|---|---|---|---|---|"]
    for tag, r in res.items():
        d, rt = r["M2"]["depth"], r["M2"]["run_type"]["overall_clustered"]
        ab = r["M2"]["ablation"]["assay_id"]
        L.append(f"| `{tag}` | {_fmt(d['median_total_slope'])} | {_fmt(d['median_eta_slope'])} | "
                 f"{ab['mean_abs_d_eta']:.3e} | {ab['max_abs_d_eta']:.3e} | "
                 f"[{rt['lo']:+.4f}, {rt['hi']:+.4f}] (n_cl={rt['n_clusters']}) | "
                 f"{'YES' if rt['supports_direction'] else 'no'} | {_fmt(rt['sign_test_p'], 3)} |")
    L += ["", "`sign-test p` drops exact ties (a bit-exactly unresponsive arm has no signs to test and "
              "reads `n/a`, not a significant p-value). `total depth slope` should be read next to the "
              "clamp tail below — the median clamp fraction is 0 on all four arms, but a minority of "
              "targets do read the slope through the saturating `log2_mu` floor:", "",
          "| checkpoint | targets with any clamp | p90 clamp frac | max clamp frac |", "|---|---|---|---|"]
    for tag, r in res.items():
        d = r["M2"]["depth"]
        L.append(f"| `{tag}` | {_fmt(d.get('frac_targets_any_clamp'), 4)} | "
                 f"{_fmt(d.get('p90_frac_log2mu_at_clamp'), 3)} | "
                 f"{_fmt(d.get('max_frac_log2mu_at_clamp'), 3)} |")

    L += ["", "## S14 — real depth counterfactual", "",
          "Each told depth is scored against its OWN `counts_dsf{k}` ground truth (the old sweep scored "
          "every told depth against the fixed dsf1 target, which any mu-decreasing model passes).", "",
          "> **0.25 is NOT a chance level** for `frac_min_at_true` -- it is the deterministic value of "
          "\"argmin always at told=1\". And because the foreground is the top 2% of the level-k "
          "realization being scored, an exactly-correct model reads as 2.2x under-predicting at dsf8, "
          "capping `frac_min_at_true` at **~0.73** (verified positive control: a perfectly depth-scaled "
          "oracle scores 0.7292 / 0.9167).",
          "> **This is a LEVEL failure, not a depth-steering failure.** Correcting ONE per-target constant "
          "fitted at (GT=1, told=1) -- leaving the told-depth response untouched -- flips all four arms to "
          "passing (`frac_beats_told1` main 1.000, wd0_on 0.972, wd0_off 1.000, offoff 0.778). It is the "
          "same per-assay scale error the M1 table quantifies.", "",
          "| checkpoint | frac_min_at_true (ceiling ~0.73) | frac beats told=1 |", "|---|---|---|"]
    for tag, r in res.items():
        d = r["dsf_counterfactual"]
        L.append(f"| `{tag}` | {_fmt(d['frac_min_at_true'])} | {_fmt(d['frac_beats_told1'])} |")

    L += ["", "## S23 -- condition recoverability :warning: WITHDRAWN, DO NOT CITE", "",
          "> **Not a validated instrument: its ordering is INVERTED against every other measurement.** "
          "`offoff_s0_perassay` -- one of the two arms with real assay steering, carrying ~5900x more "
          "feature energy than `wd0_on_s0` -- scores BELOW the 0.125 chance level, while `wd0_on_s0` "
          "scores ~2.5x higher on essentially no signal. Cause: leave-one-target-out nearest centroid on "
          "within-target-centred features penalises a target-ADAPTIVE response whose direction flips sign "
          "across targets, and can reward a deterministic near-zero one. De-prefixing the probe did not "
          "fix it. Reliable ONLY as a bit-exactly-dead detector (`main_s0_perassay` correctly reads exact "
          "chance at ~0 energy). Retained for the record.", "",
          "| checkpoint | assay acc | assay energy | run_type acc | rt energy | depth acc | depth energy |",
          "|---|---|---|---|---|---|---|"]
    for tag, r in res.items():
        p = r["recoverability"]
        L.append(f"| `{tag}` | {_fmt(p['assay']['accuracy'])} | {p['assay']['feature_energy']:.2e} | "
                 f"{_fmt(p['run_type']['accuracy'])} | {p['run_type']['feature_energy']:.2e} | "
                 f"{_fmt(p['depth']['accuracy'])} | {p['depth']['feature_energy']:.2e} |")
    L += ["", "Chance: assay 0.125, run_type 0.500, depth 0.250. "
              "**Accuracy without feature energy is meaningless here.**", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=50_000_000)
    ap.add_argument("--fg-frac", type=float, default=0.02)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = {}
    for tag in CKPTS:
        print(f"[{tag}] re-scoring...", flush=True)
        r = rescore(tag, budget=a.budget, fg_frac=a.fg_frac, n_boot=a.n_boot, seed=a.seed,
                    max_batches=(a.max_batches or None))
        res[tag] = r
        Path(f"{RES}/{tag}_h48rescored.json").write_text(
            json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print(f"[{tag}] macro CRPS {r['M1']['imp_macro_crps']:.4f} "
              f"(oracle-scaled {r['M1']['imp_macro_crps_oracle_scaled']:.4f})", flush=True)
    Path(f"{RES}/H48_SCORECARD.md").write_text(scorecard(res))
    print(f"\nwrote {RES}/H48_SCORECARD.md")


if __name__ == "__main__":
    main()
