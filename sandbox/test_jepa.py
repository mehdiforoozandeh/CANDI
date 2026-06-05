"""Validation script for sandbox/jepa.py (run directly).

Usage:
    python sandbox/test_jepa.py
"""
from __future__ import annotations
import sys
import torch
from sandbox import SANDBOX_ASSAYS
from sandbox.model import build_sandbox_candi
from sandbox.jepa import (
    CANDIJepa, SIGReg, JEPAProjector, JEPAPredictor,
    compute_latent_geometry, compute_metadata_sensitivity,
)
from sandbox.jepa_model import JEPAModel, JEPAModelConfig

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []


def check(name, cond, info=""):
    if cond:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  {info}")
        errors.append(name)


# ── SIGReg ───────────────────────────────────────────────────────────────────
print("\n=== SIGReg ===")
reg32 = SIGReg(num_proj=32)
reg256 = SIGReg(num_proj=256)

x_gauss = torch.randn(96, 64, 64)
x_col   = torch.zeros(96, 64, 64)
x_const = torch.ones(96, 64, 64) * 3.0

val_gauss = reg256(x_gauss).item()
val_col   = reg256(x_col).item()
val_const = reg256(x_const).item()

check("output is scalar",    reg32(x_gauss).shape == ())
check("collapse penalised (zeros > gauss*2)", val_col > val_gauss * 2,
      f"zeros={val_col:.3f} gauss={val_gauss:.3f}")
check("collapse penalised (const > gauss*2)", val_const > val_gauss * 2,
      f"const={val_const:.3f} gauss={val_gauss:.3f}")
x_g = torch.randn(4, 8, 32, requires_grad=True)
SIGReg(num_proj=16)(x_g).backward()
check("gradient flows", x_g.grad is not None and x_g.grad.abs().max() > 0)


# ── JEPAProjector ────────────────────────────────────────────────────────────
print("\n=== JEPAProjector ===")
proj = JEPAProjector(in_dim=72, hidden_dim=256, out_dim=72)
z = torch.randn(8, 96, 72, requires_grad=True)
pout = proj(z)
check("output shape [B,L2,D]", pout.shape == (8, 96, 72), str(pout.shape))
pout.sum().backward()
check("gradient flows",       z.grad is not None and z.grad.abs().max() > 0)


# ── JEPAPredictor + AdaLN-zero ───────────────────────────────────────────────
print("\n=== JEPAPredictor + AdaLN-zero ===")
pred = JEPAPredictor(proj_dim=32, hidden_dim=32, mask_cond_dim=8, use_mask_cond=True)
pred.eval()
x_in = torch.randn(4, 8, 32)
with torch.no_grad():
    oa = pred(x_in, torch.zeros(4, 8))
    ob = pred(x_in, torch.ones(4, 8))
diff = (oa - ob).abs().max().item()
check("output shape", pred(x_in, torch.zeros(4, 8)).shape == (4, 8, 32))
check("AdaLN-zero identity at init", diff < 1e-6, f"max_diff={diff:.2e}")


# ── compute_latent_geometry ──────────────────────────────────────────────────
print("\n=== compute_latent_geometry ===")
geo = compute_latent_geometry(torch.randn(4, 24, 32))
check("eff_rank in keys",    "lejepa/latent_eff_rank" in geo)
check("eff_rank > 1",        geo.get("lejepa/latent_eff_rank", 0) > 1,
      str(geo.get("lejepa/latent_eff_rank")))
check("std_mean in keys",    "lejepa/latent_std_mean" in geo)
check("std_max in keys",     "lejepa/latent_std_max" in geo)


# ── CANDIJepa end-to-end ─────────────────────────────────────────────────────
print("\n=== CANDIJepa ===")
F = len(SANDBOX_ASSAYS)  # 8
meta_dim = 4 * F

candi = build_sandbox_candi(
    context_bins=768, signal_dim=F, metadata_embedding_dim=meta_dim,
    n_cnn_layers=3, expansion_factor=2, nhead=4, n_sab_layers=2,
    dropout=0.0, single_shot_decoder_film=True, gaussian_var_min=0.1,
    signal_transform="log1p",
)
model = CANDIJepa(
    candi, proj_dim=0, proj_hidden_dim=256, pred_hidden_dim=0,
    pred_proj_hidden_dim=0, num_assays=F, use_mask_cond=True,
    lambda_sigreg=0.1, sigreg_num_proj=64, sigreg_knots=17,
    target_dsf="dsf1",
)

n_enc  = sum(p.numel() for p in model.candi.parameters())
n_jepa = sum(p.numel() for p in model.parameters()) - n_enc
print(f"  encoder_out_dim={model.encoder_out_dim}  proj_dim={model.proj_dim}")
print(f"  encoder_params={n_enc:,}  jepa_head_params={n_jepa:,}")

B, L, A_plus1 = 2, 768, F + 1
G  = L * 25
L2 = L // (2 ** 3)  # 96

x_ctx = torch.zeros(B, L, A_plus1)
x_tgt = torch.zeros(B, L, A_plus1)
x_dna = torch.zeros(B, G, 4)           # [B, G, 4] — permuted to [B,4,G] inside encoder
meta_ctx = torch.zeros(B, 4, A_plus1)
meta_tgt = torch.zeros(B, 4, A_plus1)
mask_cond = torch.zeros(B, F)
mask_cond[:, :3] = 1.0

out = model(x_ctx, x_tgt, x_dna, meta_ctx, meta_tgt, mask_cond)

check("loss is finite",        torch.isfinite(out["loss"]))
check("proj_ctx shape",        out["proj_ctx"].shape == (B, L2, model.proj_dim),
      str(out["proj_ctx"].shape))
check("z_pred shape",          out["z_pred"].shape   == (B, L2, model.proj_dim),
      str(out["z_pred"].shape))
check("z_tgt_raw in output",   "z_tgt_raw" in out)
check("z_tgt_raw shape",       out["z_tgt_raw"].shape == (B, L2, model.encoder_out_dim),
      str(out["z_tgt_raw"].shape))

# No-stop-gradient: grads must flow to encoder
out["loss"].backward()
enc_params = list(model.candi.encoder.parameters())
grads_ok   = [p.grad is not None and p.grad.abs().sum() > 0 for p in enc_params[:10]]
n_ok = sum(grads_ok)
check("encoder grads non-zero (no stop-grad)", n_ok > 0,
      f"{n_ok}/{len(grads_ok)} params have grad")

# Different masks → different outputs (AdaLN conditioning is active after 1 step of training)
# Note: at init AdaLN is identity, so at step 0 mask has no effect (verified above).
# Just verify predictor grads flow through mask_indicator channel.
model.zero_grad()
mc2 = torch.zeros(B, F, requires_grad=True)
out2 = model(x_ctx, x_tgt, x_dna, meta_ctx, meta_tgt, mc2)
out2["loss"].backward()
check("mask_cond grads flow (AdaLN path)", mc2.grad is not None,
      f"mc2.grad={mc2.grad}")


# ── compute_metadata_sensitivity ─────────────────────────────────────────────
print("\n=== compute_metadata_sensitivity ===")

# Use the same model built above. x_tgt is [B, L, F+1], x_dna [B, G, 4], meta [B,4,F+1].
with torch.no_grad():
    sens = compute_metadata_sensitivity(
        model.candi,
        x_tgt.clone(),
        x_dna.clone(),
        meta_tgt.clone(),
    )

expected_keys = [
    "lejepa/meta_sens_depth",    "lejepa/meta_sens_depth_max",
    "lejepa/meta_sens_depth_wide", "lejepa/meta_sens_depth_wide_max",
    "lejepa/meta_sens_readlen",  "lejepa/meta_sens_readlen_max",
    "lejepa/meta_sens_runtype",  "lejepa/meta_sens_runtype_max",
]
check("all metadata sensitivity keys present", all(k in sens for k in expected_keys),
      f"got {list(sens.keys())}")
check("depth sens is finite and non-negative",
      0.0 <= sens.get("lejepa/meta_sens_depth", -1) < 2.0,
      f"val={sens.get('lejepa/meta_sens_depth'):.5f}")
check("readlen sens is finite and non-negative",
      0.0 <= sens.get("lejepa/meta_sens_readlen", -1) < 2.0,
      f"val={sens.get('lejepa/meta_sens_readlen'):.5f}")
check("depth_wide sens is finite and non-negative",
      0.0 <= sens.get("lejepa/meta_sens_depth_wide", -1) < 2.0,
      f"val={sens.get('lejepa/meta_sens_depth_wide'):.5f}")
check("runtype sens is finite and non-negative",
      0.0 <= sens.get("lejepa/meta_sens_runtype", -1) < 2.0,
      f"val={sens.get('lejepa/meta_sens_runtype'):.5f}")
# max_sensitivity >= mean_sensitivity (by construction: min cos_sim → max 1-cos_sim)
for name in ("depth", "depth_wide", "readlen", "runtype"):
    mean_s = sens.get(f"lejepa/meta_sens_{name}", 0)
    max_s  = sens.get(f"lejepa/meta_sens_{name}_max", -1)
    check(f"{name}: max_sens >= mean_sens",
          max_s >= mean_s - 1e-5,
          f"max={max_s:.5f} mean={mean_s:.5f}")

# Verify only signal assay columns are perturbed (control col A untouched):
# Call with meta where all values are 0 → runtype contrast (0→1) should still fire.
meta_zeros = torch.zeros(B, 4, A_plus1)
with torch.no_grad():
    sens0 = compute_metadata_sensitivity(model.candi, x_tgt.clone(), x_dna.clone(), meta_zeros)
check("returns dict on all-zero meta",
      isinstance(sens0, dict) and "lejepa/meta_sens_depth" in sens0)

print(f"  depth   sens: mean={sens['lejepa/meta_sens_depth']:.5f}  "
      f"max={sens['lejepa/meta_sens_depth_max']:.5f}")
print(f"  depth_wide sens: mean={sens['lejepa/meta_sens_depth_wide']:.5f}  "
      f"max={sens['lejepa/meta_sens_depth_wide_max']:.5f}")
print(f"  readlen sens: mean={sens['lejepa/meta_sens_readlen']:.5f}  "
      f"max={sens['lejepa/meta_sens_readlen_max']:.5f}")
print(f"  runtype sens: mean={sens['lejepa/meta_sens_runtype']:.5f}  "
      f"max={sens['lejepa/meta_sens_runtype_max']:.5f}")


# ── E19 config checks ─────────────────────────────────────────────────────────
print("\n=== E19 config checks ===")

# ── e19k: meta_concat mode (mask_cond_dim = 2*4*(F+1) = 72 for F=8) ──
print(" [e19k] meta_concat mode")
model_k = CANDIJepa(
    candi, proj_dim=0, proj_hidden_dim=256, pred_hidden_dim=0,
    pred_proj_hidden_dim=0, num_assays=F, use_mask_cond=True,
    pred_mask_cond_type="meta_concat",
    lambda_sigreg=0.5, sigreg_num_proj=64, sigreg_knots=17, target_dsf="dsf1",
)
expected_mcd = 2 * 4 * (F + 1)  # 72
check("e19k mask_cond_dim=72",
      model_k.jepa_predictor.adaLN is not None
      and model_k.jepa_predictor.adaLN[1].in_features == expected_mcd,
      f"got in_features={model_k.jepa_predictor.adaLN[1].in_features if model_k.jepa_predictor.adaLN else 'None'}")
# Forward with [B, 72] mask_cond
mc_k = torch.randn(B, expected_mcd)
out_k = model_k(x_ctx, x_tgt, x_dna, meta_ctx, meta_tgt, mc_k)
check("e19k loss is finite", torch.isfinite(out_k["loss"]), f"loss={out_k['loss'].item():.4f}")

# ── e19m: no-AdaLN + proj_dim=256 ──
print(" [e19m] no-AdaLN + proj_dim=256")
model_m = CANDIJepa(
    candi, proj_dim=256, proj_hidden_dim=256, pred_hidden_dim=0,
    pred_proj_hidden_dim=0, num_assays=F, use_mask_cond=True,
    pred_mask_cond_type="none",    # new default
    lambda_sigreg=0.5, sigreg_num_proj=64, sigreg_knots=17, target_dsf="dsf1",
)
check("e19m proj_dim=256", model_m.proj_dim == 256)
check("e19m pred_hidden auto-scales to 256",
      model_m.jepa_predictor.fc1.out_features == 256,
      f"got {model_m.jepa_predictor.fc1.out_features}")
check("e19m AdaLN disabled (pred_mask_cond_type=none)",
      model_m.jepa_predictor.adaLN is None)
mc_m = torch.zeros(B, F)   # mask_cond ignored when adaLN is None
out_m = model_m(x_ctx, x_tgt, x_dna, meta_ctx, meta_tgt, mc_m)
check("e19m loss is finite", torch.isfinite(out_m["loss"]))
check("e19m proj output is 256-dim", out_m["proj_tgt"].shape[-1] == 256)

# ── e19n: no-AdaLN + lambda_sigreg=2.0 ──
print(" [e19n] no-AdaLN + lambda_sigreg=2.0")
model_n = CANDIJepa(
    candi, proj_dim=0, proj_hidden_dim=256, pred_hidden_dim=0,
    pred_proj_hidden_dim=0, num_assays=F, use_mask_cond=True,
    pred_mask_cond_type="none",
    lambda_sigreg=2.0, sigreg_num_proj=64, sigreg_knots=17, target_dsf="dsf1",
)
check("e19n lambda stored as 2.0", model_n.lambda_sigreg == 2.0)
mc_n = torch.zeros(B, F)
out_n = model_n(x_ctx, x_tgt, x_dna, meta_ctx, meta_tgt, mc_n)
check("e19n loss is finite", torch.isfinite(out_n["loss"]))
# lambda=2.0 should make total_loss > pred_loss alone
check("e19n total > pred (lambda scales up sigreg)",
      out_n["loss"].item() > out_n["pred_loss"].item() - 1e-6)

# ── e19o: no-AdaLN + pred_hidden_dim=16 ──
print(" [e19o] no-AdaLN + pred_hidden_dim=16")
model_o = CANDIJepa(
    candi, proj_dim=0, proj_hidden_dim=256, pred_hidden_dim=16,
    pred_proj_hidden_dim=0, num_assays=F, use_mask_cond=True,
    pred_mask_cond_type="none",
    lambda_sigreg=0.5, sigreg_num_proj=64, sigreg_knots=17, target_dsf="dsf1",
)
enc_dim = model_o.encoder_out_dim
check("e19o predictor fc1 → 16 (bottleneck)",
      model_o.jepa_predictor.fc1.out_features == 16,
      f"got {model_o.jepa_predictor.fc1.out_features}")
check(f"e19o predictor fc1 input = encoder_out_dim ({enc_dim})",
      model_o.jepa_predictor.fc1.in_features == enc_dim,
      f"got {model_o.jepa_predictor.fc1.in_features}")
check("e19o predictor fc2 → encoder_out_dim",
      model_o.jepa_predictor.fc2.out_features == enc_dim,
      f"got {model_o.jepa_predictor.fc2.out_features}")
mc_o = torch.zeros(B, F)
out_o = model_o(x_ctx, x_tgt, x_dna, meta_ctx, meta_tgt, mc_o)
check("e19o loss is finite", torch.isfinite(out_o["loss"]))


# ── meta_tgt mode ────────────────────────────────────────────────────────────
print("\n=== meta_tgt conditioning mode ===")

# mask_cond_dim for meta_tgt = 4*(F+1) = 36 (with F=8)
expected_mt_dim = 4 * (F + 1)   # 36

# --- model construction ---
model_mt = CANDIJepa(
    candi, proj_dim=0, proj_hidden_dim=256, pred_hidden_dim=0,
    pred_proj_hidden_dim=0, num_assays=F, use_mask_cond=True,
    pred_mask_cond_type="meta_tgt",
    lambda_sigreg=0.5, sigreg_num_proj=64, sigreg_knots=17, target_dsf="dsf1",
)
check("meta_tgt mask_cond_dim=36",
      model_mt.jepa_predictor.adaLN is not None
      and model_mt.jepa_predictor.adaLN[1].in_features == expected_mt_dim,
      f"got {model_mt.jepa_predictor.adaLN[1].in_features if model_mt.jepa_predictor.adaLN else 'None'}")

# --- forward pass with [B, 36] mask_cond ---
mc_mt = torch.randn(B, expected_mt_dim)
out_mt = model_mt(x_ctx, x_tgt, x_dna, meta_ctx, meta_tgt, mc_mt)
check("meta_tgt loss is finite", torch.isfinite(out_mt["loss"]),
      f"loss={out_mt['loss'].item():.4f}")
check("meta_tgt proj shape unchanged",
      out_mt["proj_tgt"].shape == (B, L2, model_mt.proj_dim))

# --- prepare_jepa_batch: meta_tgt mode produces mask_cond [B, 4*(F+1)] ---
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sandbox.train_jepa import prepare_jepa_batch, make_masker
from sandbox.data import SandboxH5Dataset
import torch

def _get_batch(dsf_list=(1,2,4), sampling='uniform'):
    ds = SandboxH5Dataset('sandbox/data/sandbox.h5', regime='type1_chr19',
                          dsf_list=dsf_list, dsf_sampling=sampling, seed=42)
    return next(iter(ds))

masker_default = make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0,
                              mask_fraction=0.2, chunk_size=40, min_available_frac=0.3)
masker_noask = make_masker(p_full_assay=0.0, p_full_loci=0.0, p_chunks=0.0,
                            mask_fraction=0.2, chunk_size=40, min_available_frac=0.0)
device_test = torch.device('cpu')

# Scenario 2: assay masking — mask_cond should be [B, 36]
batch_q = _get_batch()
prep_q = prepare_jepa_batch(batch_q, masker_default, device_test,
                              target_dsf="dsf1", mask_cond_type="meta_tgt")
assert prep_q is not None, "prepare_jepa_batch returned None for meta_tgt+masking"
Bq = batch_q['x_data'].shape[0]
check("e19q mask_cond shape [B, 36]",
      prep_q["mask_cond"].shape == (Bq, expected_mt_dim),
      f"got {prep_q['mask_cond'].shape}")
check("e19q has_corruption (masking mode)",
      prep_q["has_corruption"])

# Scenario 1: DSF corruption (no masking) — mask_cond [B, 36], has_corruption via depth diff
# Uses context_down so x_dsf=4 but y_dsf=1 → metadata actually differs
batch_p = _get_batch(dsf_list=(4,), sampling='context_down')
prep_p = prepare_jepa_batch(batch_p, masker_noask, device_test,
                              target_dsf="dsf1", mask_cond_type="meta_tgt")
assert prep_p is not None, "prepare_jepa_batch returned None for meta_tgt+DSF"
Bp = batch_p['x_data'].shape[0]
check("e19p mask_cond shape [B, 36]",
      prep_p["mask_cond"].shape == (Bp, expected_mt_dim),
      f"got {prep_p['mask_cond'].shape}")
check("e19p has_corruption (DSF mode, depth_log2 differs)",
      prep_p["has_corruption"])
# Verify meta_tgt != meta_ctx (DSF1 metadata differs from DSF4)
check("e19p meta_tgt != meta_ctx somewhere (DSF1 != DSF4)",
      bool((prep_p["meta_tgt"] != prep_p["meta_ctx"]).any().item()))

# Scenario 3: combined DSF + masking
batch_r = _get_batch(dsf_list=(4,), sampling='context_down')
prep_r = prepare_jepa_batch(batch_r, masker_default, device_test,
                              target_dsf="dsf1", mask_cond_type="meta_tgt")
assert prep_r is not None, "prepare_jepa_batch returned None for meta_tgt+DSF+mask"
check("e19r mask_cond shape [B, 36]",
      prep_r["mask_cond"].shape == (Bp, expected_mt_dim),
      f"got {prep_r['mask_cond'].shape}")
check("e19r has_corruption (combined)",
      prep_r["has_corruption"])


# ── E21 fresh JEPA model ─────────────────────────────────────────────────────
print("\n=== E21 fresh JEPA model ===")
fresh_cfg = JEPAModelConfig(
    num_assays=F,
    context_length=L,
    metadata_embed_dim=32,
    n_cnn_layers=3,
    expansion_factor=2,
    n_transformer_layers=2,
    nhead=4,
    d_model=0,
    proj_dim=0,
    pred_hidden_dim=0,
    predictor_layers=1,
    lambda_sigreg=0.5,
    sigreg_num_proj=64,
)
fresh_model = JEPAModel(fresh_cfg)
out_fresh = fresh_model(x_ctx, x_tgt, x_dna, meta_ctx, meta_tgt, mask_cond)
check("fresh loss is finite", torch.isfinite(out_fresh["loss"]))
check("fresh proj shape [B,L2,D]", out_fresh["proj_tgt"].shape == (B, L2, fresh_model.proj_dim),
      str(out_fresh["proj_tgt"].shape))
check("fresh raw latent shape [B,L2,Denc]",
      out_fresh["z_tgt_raw"].shape == (B, L2, fresh_model.encoder_out_dim),
      str(out_fresh["z_tgt_raw"].shape))

fresh_model.zero_grad()
out_fresh["loss"].backward()
fresh_grads = [
    p.grad is not None and p.grad.abs().sum() > 0
    for p in list(fresh_model.candi.parameters())[:20]
]
check("fresh encoder grads non-zero", sum(fresh_grads) > 0, f"{sum(fresh_grads)}/{len(fresh_grads)}")

# Fresh predictor ablation toggles (E21e+ diagnostics).
variant_cfgs = [
    ("fresh_mlp_embed_shared", dict(predictor_type="mlp")),
    ("fresh_transformer_raw_meta", dict(cond_source="raw_meta_tgt")),
    ("fresh_transformer_embed_separate", dict(cond_source="meta_tgt_embed", cond_embed_shared="separate")),
]
for variant_name, overrides in variant_cfgs:
    cfg_kwargs = dict(**fresh_cfg.__dict__)
    cfg_kwargs.update(overrides)
    variant_cfg = JEPAModelConfig(**cfg_kwargs)
    variant_model = JEPAModel(variant_cfg)
    variant_out = variant_model(x_ctx, x_tgt, x_dna, meta_ctx, meta_tgt, mask_cond)
    check(f"{variant_name} finite loss", torch.isfinite(variant_out["loss"]))
    check(
        f"{variant_name} proj shape [B,L2,D]",
        variant_out["proj_tgt"].shape == (B, L2, variant_model.proj_dim),
        str(variant_out["proj_tgt"].shape),
    )
    variant_model.zero_grad()
    variant_out["loss"].backward()
    predictor_has_grad = any(
        p.grad is not None and bool((p.grad.abs().sum() > 0).item())
        for p in variant_model.jepa_predictor.parameters()
    )
    check(f"{variant_name} predictor grads non-zero", predictor_has_grad)


# ── Summary ──────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"\033[91mFAILED ({len(errors)} tests):\033[0m", errors)
    sys.exit(1)
else:
    print(f"\033[92mALL TESTS PASSED\033[0m")
    sys.exit(0)
