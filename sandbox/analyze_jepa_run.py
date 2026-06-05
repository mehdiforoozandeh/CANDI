"""Quick analysis of a train_jepa metrics.jsonl run.  Usage: python sandbox/analyze_jepa_run.py <run_dir>"""
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sandbox/runs/e19_jepa_stage1_39046758")
path = run_dir / "metrics.jsonl"

steps, epochs = [], []
for line in path.read_text().splitlines():
    r = json.loads(line)
    if r.get("kind") == "training_step":
        d = r.get("lejepa", {})
        steps.append({
            "step": r["global_step"], "epoch": r["epoch"],
            "pred":    d.get("lejepa/pred_loss"),
            "sigreg":  d.get("lejepa/sigreg_loss"),
            "total":   d.get("lejepa/total_loss"),
            "grad":    d.get("lejepa/grad_norm_pre_clip"),
            "clip_frac": d.get("lejepa/grad_clipped_frac_running"),
            "eff_rank":  d.get("lejepa/latent_eff_rank"),
            "std_mean":  d.get("lejepa/latent_std_mean"),
            "std_min":   d.get("lejepa/latent_std_min"),
            "mean_abs":  d.get("lejepa/latent_mean_abs"),
        })
    elif r.get("kind") == "epoch":
        epochs.append(r)

print(f"Training steps logged : {len(steps)}  ({steps[0]['step']} -> {steps[-1]['step']})")
print(f"Epochs complete       : {len(epochs)}  (last ep={epochs[-1]['epoch'] if epochs else None})")
esec = [e["epoch_seconds"] for e in epochs if "epoch_seconds" in e]
print(f"Total elapsed         : {sum(esec):.0f}s = {sum(esec)/3600:.2f}h")
print(f"sec/epoch             : mean={sum(esec)/len(esec):.1f}  min={min(esec):.1f}  max={max(esec):.1f}")
print()

def stats(key):
    vals = [(i, s[key]) for i, s in enumerate(steps) if s.get(key) is not None]
    if not vals:
        return "N/A"
    vs = [v for _, v in vals]
    best_idx = min(range(len(vals)), key=lambda i: vals[i][1])
    return (f"first={vs[0]:.5f}  last={vs[-1]:.5f}  "
            f"best={min(vs):.5f}@step{steps[vals[best_idx][0]]['step']}")

print("pred_loss   :", stats("pred"))
print("sigreg_loss :", stats("sigreg"))
print("total_loss  :", stats("total"))
print("grad_norm   :", stats("grad"))

clipped = [s["clip_frac"] for s in steps if s.get("clip_frac") is not None]
if clipped:
    print("clip_frac   : first={:.3f}  last={:.3f}".format(clipped[0], clipped[-1]))
print()

geo = [s for s in steps if s.get("eff_rank") is not None]
print(f"Geometry snapshots : {len(geo)}")
if geo:
    ranks = [s["eff_rank"] for s in geo]
    stds  = [s["std_mean"] for s in geo]
    stdmn = [s["std_min"]  for s in geo]
    mabs  = [s["mean_abs"] for s in geo]
    print("eff_rank  : first={:.2f}  last={:.2f}  min={:.2f}  max={:.2f}".format(
        ranks[0], ranks[-1], min(ranks), max(ranks)))
    print("std_mean  : first={:.4f}  last={:.4f}".format(stds[0], stds[-1]))
    print("std_min   : first={:.4f}  last={:.4f}".format(stdmn[0], stdmn[-1]))
    print("mean_abs  : first={:.4f}  last={:.4f}".format(mabs[0], mabs[-1]))
print()

totals = [s["total"] for s in steps if s.get("total") is not None]
best_t, last_t = min(totals), totals[-1]
ratio = last_t / best_t if best_t > 0 else float("nan")
flag = "  *** DIVERGED ***" if ratio > 1.5 else ""
print(f"Divergence check : last={last_t:.5f}  best={best_t:.5f}  ratio={ratio:.2f}{flag}")
print()

# 10-point trajectory
n = len(steps)
indices = [int(i * (n - 1) / 9) for i in range(10)]
print("Trajectory (total | pred | sigreg | eff_rank):")
for i in indices:
    s = steps[i]
    er = "{:.1f}".format(s["eff_rank"]) if s.get("eff_rank") is not None else "  - "
    print("  step={:5d} ep={:3d}  total={:.5f}  pred={:.5f}  sigreg={:.5f}  eff_rank={}".format(
        s["step"], s["epoch"], s["total"] or 0, s["pred"] or 0, s["sigreg"] or 0, er))
