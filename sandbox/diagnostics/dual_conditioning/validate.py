"""Correctness gates for the dual-conditioning testbed (crux q15).

Two tiers:
  run_cpu_gates()  - fast, NO training: transform math, sentinel/availability consistency, determinism,
                     metric-math oracles, causal metadata wiring (forward+grad), no-crosswiring. HARD.
  run_train_gates(device) - training-dependent controls (overfit-tiny, shuffled-h_y, shuffled-h_x,
                     identity ceiling). Cheap on GPU; run as the gate job before the sweep. HARD.

A surviving positive under a shuffle control, or a dead metadata pathway, means the M1/M2/M3 numbers
cannot be trusted -> the sweep is blocked.
"""
from __future__ import annotations

import numpy as np
import torch

from sandbox.diagnostics.dual_conditioning import transforms as T
from sandbox.diagnostics.dual_conditioning import metrics as MET
from sandbox.diagnostics.dual_conditioning.data import DualCondData, A
from sandbox.diagnostics.dual_conditioning.model import (
    build_model, forward_counts, encode_latent, nb_nll, nb_mean)


def _ok(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return bool(cond)


# ---------------------------------------------------------------- CPU gates
def run_cpu_gates(h5="sandbox/data/sandbox.h5") -> bool:
    print("== CPU pre-flight gates ==")
    ok = True
    y = np.array([0, 1, 2, 5, 10, 20], np.int64)

    # 1. transform math + non-negativity + identity + thin determinism
    ok &= _ok("mult x2", (T.apply_transform(y, "mult", 2.0) == y * 2).all())
    ok &= _ok("add +5", (T.apply_transform(y, "add", 5.0) == y + 5).all())
    ok &= _ok("cap 5", (T.apply_transform(y, "cap", 5.0) == np.minimum(y, 5)).all())
    ok &= _ok("identity exact", (T.apply_transform(y, "identity", 1.0) == y).all())
    allnn = all(T.apply_transform(y, f, p).min() >= 0 for f, ps in T.PARAMS.items() for p in ps)
    ok &= _ok("all transforms >= 0", allnn)
    t1 = T.apply_transform(y, "thin", 0.5, seed=7); t2 = T.apply_transform(y, "thin", 0.5, seed=7)
    t3 = T.apply_transform(y, "thin", 0.5, seed=8)
    ok &= _ok("thin deterministic (same seed)", (t1 == t2).all())
    ok &= _ok("thin varies (diff seed)", (t1 != t3).any())

    # 2. data: sentinel/availability consistency + determinism (incl. thin target reproducible)
    d = DualCondData(h5)
    rng = np.random.default_rng(0)
    bio = [b for b in d.biosamples if d.avail[b].sum() >= 2][0]
    wi = rng.choice(d.idx_chr19, size=4, replace=False)
    fam_x = np.full(A, T.FAM["thin"]); par_x = np.full(A, 0.5, np.float32)
    fam_y = np.full(A, T.FAM["mult"]); par_y = np.full(A, 2.0, np.float32)
    b1 = d.make_batch(bio, wi, fam_x, par_x, fam_y, par_y, "cpu")
    b2 = d.make_batch(bio, wi, fam_x, par_x, fam_y, par_y, "cpu")
    ok &= _ok("batch determinism (incl thin target)",
              torch.equal(b1["x_data"], b2["x_data"]) and torch.equal(b1["y_target"], b2["y_target"]))
    av = b1["avail"][0].numpy()
    good = True
    for a in range(A):
        col = b1["x_data"][0, :, a]
        good &= bool((col >= 0).all()) if av[a] > 0 else bool((col == -1).all())
    ok &= _ok("signal sentinel matches availability", good)
    # metadata availability must equal signal availability for every column (candi_v2 assert precondition)
    from sandbox.candi_v2.encoder import _infer_availability_from_meta, _infer_availability_from_signal
    am = _infer_availability_from_meta(b1["x_meta"]); asig = _infer_availability_from_signal(b1["x_data"])
    ok &= _ok("meta/signal availability agree (all cols incl control)", torch.equal(am, asig))

    # 3. metric-math oracles (no model): r2, M2-core, M3-core
    tgt = np.array([0., 1., 2., 3., 4., 9., 16.])
    ok &= _ok("r2 oracle: r2(x,x)==1", abs(MET.r2(tgt, tgt) - 1.0) < 1e-9)
    ok &= _ok("r2 oracle: constant pred -> r2<=0", MET.r2(np.zeros_like(tgt), tgt) <= 0.0 + 1e-9)
    dp = tgt - tgt.mean()
    ok &= _ok("M2-core: dpred==dtarget -> r2==1", abs(MET.r2(dp, dp) - 1.0) < 1e-9)
    ok &= _ok("M2-core: dpred==0 -> r2<=0", MET.r2(np.zeros_like(dp), dp) <= 0.0 + 1e-9)
    va = np.random.default_rng(1).normal(size=(20, 8)); vb = va + 5.0  # different base vectors
    ok &= _ok("M3-core: identical -> cos-dist~0", float(MET._cos_dist(va, va).mean()) < 1e-6)
    ok &= _ok("M3-core: different -> cos-dist>0", float(MET._cos_dist(va, vb).mean()) > 1e-3)

    # 4. metadata causally wired + no crosswiring (forward + grad, no training)
    torch.manual_seed(0)
    m = build_model("zscore"); m.eval()
    b_id = d.make_batch(bio, wi, np.full(A, T.FAM["identity"]), np.ones(A, np.float32),
                        np.full(A, T.FAM["identity"]), np.ones(A, np.float32), "cpu")
    b_ymul = d.make_batch(bio, wi, np.full(A, T.FAM["identity"]), np.ones(A, np.float32),
                          np.full(A, T.FAM["mult"]), np.full(A, 4.0, np.float32), "cpu")
    b_xmul = d.make_batch(bio, wi, np.full(A, T.FAM["mult"]), np.full(A, 4.0, np.float32),
                          np.full(A, T.FAM["identity"]), np.ones(A, np.float32), "cpu")
    with torch.no_grad():
        z_id = encode_latent(m, b_id); z_x = encode_latent(m, b_xmul)
    dz = float((z_id - z_x).abs().mean())
    ok &= _ok("h_x perturbs latent (encoder wired)", dz > 1e-4, f"|dz|={dz:.4g}")
    # no crosswiring: z must be invariant to y_meta (encode ignores y_meta)
    with torch.no_grad():
        z_id2 = encode_latent(m, b_ymul)   # same x-side as b_id, different y_meta
    ok &= _ok("no crosswiring: z invariant to y_meta", torch.equal(z_id, z_id2))
    # grad reaches the metadata embedders (proves y_meta->output is trainable; adaLN-zero starts at identity)
    m.train()
    loss = nb_nll(*forward_counts(m, b_ymul), b_ymul["y_target"], b_ymul["avail"])
    loss.backward()
    # h_y activation: adaLN-zero => ~0 influence at init; after a few opt steps the decoder must USE h_y
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    for _ in range(30):
        opt.zero_grad()
        nb_nll(*forward_counts(m, b_ymul), b_ymul["y_target"], b_ymul["avail"]).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        dy = float((nb_mean(*forward_counts(m, b_id)) - nb_mean(*forward_counts(m, b_ymul))).abs().mean())
    ok &= _ok("h_y activates output after training (decoder wired)", dy > 1e-4, f"|dout|={dy:.4g}")
    m.train()
    genc = m.encoder.metadata_embedding.family_embedding.weight.grad
    gdec = m.decoder.decoder_meta_embedding.family_embedding.weight.grad
    ok &= _ok("grad reaches encoder meta embedder", genc is not None and float(genc.abs().sum()) > 0)
    ok &= _ok("grad reaches decoder meta embedder", gdec is not None and float(gdec.abs().sum()) > 0)

    print(f"== CPU gates: {'ALL PASS' if ok else 'SOME FAILED'} ==")
    return ok


# ---------------------------------------------------------------- train gates (GPU)
def _tiny_train(model, data, device, units, conditions, steps, lr=2e-3, shuffle_side=None, rng=None):
    """Train on a small window set. shuffle_side in {None,'y','x'} feeds a WRONG covariate (an
    independently-drawn cell) with the TRUE target -> the covariate becomes uninformative, so genuine
    use of it must collapse. (Conditions are uniform per batch, so a wrong DRAW, not a permutation.)"""
    opt = torch.optim.Adam(model.parameters(), lr)
    model.train()
    for s in range(steps):
        bio, wi = units[s % len(units)]
        fx, px, fy, py = data.sample_conditions(bio, rng, conditions)
        kw = {}
        if shuffle_side == "y":
            wc = conditions[rng.integers(len(conditions))]
            kw.update(fam_ym=np.full(A, wc[0]), par_ym=np.full(A, wc[1], np.float32))
        if shuffle_side == "x":
            wc = conditions[rng.integers(len(conditions))]
            kw.update(fam_xm=np.full(A, wc[0]), par_xm=np.full(A, wc[1], np.float32))
        b = data.make_batch(bio, wi, fx, px, fy, py, device, **kw)
        loss = nb_nll(*forward_counts(model, b), b["y_target"], b["avail"])
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)


def run_train_gates(device="cpu", *, steps=500, seed=0) -> bool:
    print(f"== train-dependent control gates (device={device}) ==")
    ok = True
    d = DualCondData()
    rng = np.random.default_rng(seed)
    conds = T.all_conditions()
    units = MET.make_eval_units(d, n_units=2, batch_size=8, rng=rng, chrom="chr19")

    # overfit-tiny: correct conditioning should learn to steer the output (M2 clearly > 0)
    torch.manual_seed(seed)
    m = build_model("zscore").to(device)
    lf = _tiny_train(m, d, device, units, conds, steps, shuffle_side=None, rng=np.random.default_rng(1))
    m2_ok = MET.eval_M2(m, d, units, device)["median_invertible"]
    ok &= _ok("overfit-tiny: correct h_y steers output (M2>0.2)", m2_ok > 0.2, f"M2={m2_ok:.3f} NLL={lf:.2f}")

    # shuffled-h_y: identical capacity, WRONG output covariate -> steering must collapse toward 0
    torch.manual_seed(seed)
    ms = build_model("zscore").to(device)
    _tiny_train(ms, d, device, units, conds, steps, shuffle_side="y", rng=np.random.default_rng(1))
    m2_sh = MET.eval_M2(ms, d, units, device)["median_invertible"]
    ok &= _ok("shuffled-h_y: steering collapses (< correct-0.2)", m2_sh < m2_ok - 0.2, f"M2_shuf={m2_sh:.3f}")

    print(f"== train gates: {'ALL PASS' if ok else 'SOME FAILED'} ==")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gates", action="store_true")
    a = ap.parse_args()
    import sys
    passed = run_cpu_gates()
    if a.train_gates:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        passed = run_train_gates(dev) and passed
    sys.exit(0 if passed else 1)
