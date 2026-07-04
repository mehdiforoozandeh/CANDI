"""Train/eval covariate probes. Run: python -m sandbox.diagnostics.covariate_probes.run --assay H3K4me3
Validation: python -m sandbox.diagnostics.covariate_probes.run --validate --assay H3K4me3
"""
from __future__ import annotations
import argparse, os, csv, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_absolute_error

from sandbox.diagnostics.covariate_probes import data as D
from sandbox.diagnostics.covariate_probes.model import CovariateProbe

# probe -> (target, norm_mode, dna_arms).  norm: raw | scaled (sum-norm) | thin (depth-controlled)
PROBES = {
    "depth_raw":    ("depth", "raw", [False]),       # validator: depth present (R2 high)
    "depth_scaled": ("depth", "scaled", [False]),    # FINDING: sum-norm does NOT strip depth
    "depth_thin":   ("depth", "thin", [False]),      # validator: thinning removes depth (R2~0)
    "run_type":     ("run_type", "thin", [False, True]),
    "read_length":  ("read_length", "thin", [False, True]),
}

def _metric(target, yt, yp):
    if target == "run_type":
        try: return dict(auroc=roc_auc_score(yt, yp), auprc=average_precision_score(yt, yp))
        except ValueError: return dict(auroc=float("nan"), auprc=float("nan"))
    return dict(r2=r2_score(yt, yp), mae=mean_absolute_error(yt, yp))

def _prep(arrs, chrom, use_dna, count_off, device):
    """GPU-resident tensors: X[N,1,768], Y[N], + optional resident DNA (unique [U,4,L], idx[N])."""
    counts, starts, y, inst = arrs
    x = (np.zeros((len(counts), 1, D.WIN_BINS), np.float32) if count_off
         else np.arcsinh(counts)[:, None, :].astype(np.float32))
    X = torch.from_numpy(x).to(device)
    Y = torch.from_numpy(y.astype(np.float32)).to(device)
    dna, didx = (D.dna_gpu(starts, chrom, device) if use_dna else (None, None))
    return X, Y, dna, didx, inst

def train_eval(pool, insts, target, norm, use_dna, seed, epochs, device,
               count_off=False, shuffle=False, bs=None, tiny=False):
    train_g, test_g = D.stratified_group_split(insts, target, seed)
    y_by_group = None
    if shuffle:  # negative control: random group->label map (same map train+test)
        labels = D._group_labels(insts, target)
        vals = list(labels.values()); np.random.default_rng(seed).shuffle(vals)
        y_by_group = dict(zip(labels.keys(), vals))
        if target == "run_type":
            y_by_group = {g: (1 if v == 'P' else 0) for g, v in y_by_group.items()}
    tr = D.assemble(pool, train_g, D.TRAIN_CHR, target, norm=norm, y_by_group=y_by_group)
    te = D.assemble(pool, test_g, D.TEST_CHR, target, norm=norm, y_by_group=y_by_group)
    if tr is None or te is None:
        return None
    if tiny:  # overfit a handful of windows (balanced for cls); eval on train; many small steps
        if target == "run_type":
            i1 = np.where(tr[2] == 1)[0][:64]; i0 = np.where(tr[2] == 0)[0][:64]
            idx = np.concatenate([i0, i1])
        else:
            idx = np.arange(min(128, len(tr[0])))
        tr = tuple(a[idx] for a in tr); te = tr
    bs = bs or (32 if tiny else (128 if use_dna else 512))  # 512 safe at SEQ=192
    is_cls = (target == "run_type")
    mu, sd = (0.0, 1.0)
    if not is_cls:
        mu, sd = float(tr[2].mean()), float(tr[2].std() + 1e-6)

    Xtr, Ytr, Dtr, Itr, _ = _prep(tr, D.TRAIN_CHR, use_dna, count_off, device)
    Xte, Yte, Dte, Ite, inst_te = _prep(te, D.TEST_CHR, use_dna, count_off, device)
    model = CovariateProbe(use_dna).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    if is_cls:
        npos = float((tr[2] == 1).sum()); nneg = float((tr[2] == 0).sum())
        lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([nneg / max(npos, 1.0)], device=device))
    else:
        lossfn = torch.nn.MSELoss()

    N = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            db = Dtr[Itr[b]] if use_dna else None
            yb = Ytr[b]; yt = yb if is_cls else (yb - mu) / sd
            loss = lossfn(model(Xtr[b], db), yt)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), bs):
            db = Dte[Ite[i:i + bs]] if use_dna else None
            out = model(Xte[i:i + bs], db)
            preds.append((torch.sigmoid(out) if is_cls else out * sd + mu).cpu().numpy())
    preds = np.concatenate(preds); ys = te[2]; insts_id = te[3]

    res = {"window": _metric(target, ys, preds)}
    # per-biosample aggregation
    bp, by, bi = [], [], []
    for u in np.unique(insts_id):
        m = insts_id == u; bp.append(float(preds[m].mean())); by.append(float(ys[m][0])); bi.append(int(u))
    res["biosample"] = _metric(target, np.array(by), np.array(bp))
    res["_pred"] = list(zip(bi, by, bp))
    res["_n"] = dict(n_test_inst=len(np.unique(insts_id)), n_test_win=len(preds),
                     n_train_inst=len(np.unique(tr[3])), n_train_win=len(tr[0]))
    del Xtr, Ytr, Xte, Yte, model, opt
    if use_dna: del Dtr, Itr, Dte, Ite
    if device == "cuda": torch.cuda.empty_cache()
    return res

def applicable_probes(insts):
    rv = D.runtype_viable(insts); lv = D.readlength_viable(insts)
    p = []
    if rv or lv: p += ["depth_raw", "depth_scaled", "depth_thin"]
    if rv: p += ["run_type"]
    if lv: p += ["read_length"]
    return p

def sweep(assay, out, k, epochs, device):
    insts = D.census(assays=[assay])
    pool = D.build_pool(insts)
    pcols = ["assay", "probe", "dna", "seed", "inst", "y_true", "y_pred"]
    for probe in applicable_probes(insts):           # write per-probe so a walltime kill keeps done work
        target, norm, arms = PROBES[probe]
        rows, preds = [], []
        for use_dna in arms:
            for seed in range(k):
                r = train_eval(pool, insts, target, norm, use_dna, seed, epochs, device)
                if r is None: continue
                for level in ("window", "biosample"):
                    m = r[level]
                    rows.append(dict(assay=assay, probe=probe, dna=int(use_dna), seed=seed, level=level,
                                     **r["_n"], **{k2: m.get(k2, "") for k2 in ("auroc", "auprc", "r2", "mae")}))
                    print(f"  {assay:10s} {probe:14s} dna={int(use_dna)} s{seed} {level:9s} "
                          + " ".join(f"{k2}={m[k2]:.3f}" for k2 in m), flush=True)
                for inst, yt, yp in r["_pred"]:
                    preds.append(dict(assay=assay, probe=probe, dna=int(use_dna), seed=seed,
                                      inst=inst, y_true=yt, y_pred=yp))
                if device == "cuda":
                    torch.cuda.empty_cache()
        _write(out, rows)
        _write(out.replace(".tsv", "_preds.tsv"), preds, cols=pcols)

def _write(out, rows, cols=None):
    if not rows: return
    new = not os.path.exists(out)
    cols = cols or ["assay", "probe", "dna", "seed", "level", "n_train_inst", "n_test_inst",
                    "n_train_win", "n_test_win", "auroc", "auprc", "r2", "mae"]
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        if new: w.writeheader()
        for r in rows: w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--out", default="sandbox/diagnostics/covariate_probes/results.tsv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} assay={a.assay}")
    if a.validate:
        import sys
        from sandbox.diagnostics.covariate_probes.validate import run_validation
        ok = run_validation(a.assay, device)
        sys.exit(0 if ok else 1)   # nonzero -> afterok dependency blocks the sweep
    else:
        t0 = time.time()
        sweep(a.assay, a.out, a.k, a.epochs, device)
        if device == "cuda":
            print(f"[profile] peak GPU mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        print(f"done {a.assay} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
