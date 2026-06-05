"""Smoke tests for JEPA Stage 2 decoder training.

Usage:
    python -m sandbox.test_jepa_decoder --device cuda
    python -m sandbox.test_jepa_decoder --device cuda --h5 sandbox/data/sandbox.h5 --run-eval
"""
from __future__ import annotations

import argparse
import copy
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch

from candi_loss import CANDI_LOSS
from sandbox import SANDBOX_ASSAYS
from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.config import deep_merge, load_yaml
from sandbox.config_types import config_from_dict
from sandbox.data import SandboxH5Dataset
from sandbox.jepa_config import JEPADecoderConfig
from sandbox.jepa_decoder import JEPADecoderModel, JEPADecoderTower, _build_stage1_model
from sandbox.losses import SandboxCompositeLoss
from sandbox.train import run_eval_pass
from sandbox.train_jepa_decoder import _make_criterion, train_one_epoch


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors: list[str] = []


def check(name: str, cond: bool, info: str = "") -> None:
    if cond:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  {info}")
        errors.append(name)


def tiny_cfg(tmp_path: Path, device: str) -> JEPADecoderConfig:
    cfg_dict: Dict = asdict(JEPADecoderConfig())
    cfg_dict = deep_merge(
        cfg_dict,
        {
            "model_type": "fresh",
            "data": {"context_length": 64, "h5_path": "sandbox/data/sandbox.h5", "h5_cache_ram": False},
            "model": {"encode_input_transform": "log1p"},
            "fresh": {
                "context_length": 64,
                "n_cnn_layers": 2,
                "expansion_factor": 2,
                "pool_size": 2,
                "dna_pool_size": 5,
                "n_transformer_layers": 1,
                "nhead": 2,
                "dropout": 0.0,
                "proj_hidden_dim": 64,
                "predictor_heads": 2,
                "predictor_dim_head": 16,
                "predictor_ff_mult": 2,
                "metadata_embed_dim": 16,
            },
            "decoder": {
                "n_cnn_layers": 2,
                "expansion_factor": 2,
                "pool_size": 2,
                "decoder_input_dim": 32,
                "checkpoint_path": str(tmp_path / "dummy_jepa.pt"),
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "device": device,
                "amp": False,
                "max_train_batches": 2,
                "eval_max_batches": 1,
                "save_checkpoint": False,
                "save_best_checkpoint": False,
                "training_stats_jsonl_every_n_steps": 1,
                "optimizer": {"name": "adamax", "adamax": {"lr": 1.0e-3}},
                "grad": {"clip_norm": 2.0},
                "masking": {"p_full_assay": 1.0, "p_full_loci": 0.0, "p_chunks": 0.0, "min_available_frac": 0.3, "preserve_assay_id": True},
                "dsf": {"dsf_list": [1, 2, 4], "sampling": "uniform"},
            },
            "eval": {"eval_every_n_epochs": 1, "meta_sensitivity_probe_every_n_steps": 0},
            "wandb": {"mode": "disabled"},
            "hpo": {"disable": True},
        },
    )
    return config_from_dict(JEPADecoderConfig, cfg_dict)


def write_dummy_checkpoint(cfg: JEPADecoderConfig, device: torch.device) -> None:
    model = _build_stage1_model(cfg, device)
    torch.save({"model_state_dict": model.state_dict(), "global_step": 0}, cfg.decoder.checkpoint_path)


def synthetic_batch(cfg: JEPADecoderConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    bsz = 2
    length = int(cfg.data.context_length)
    assays = len(SANDBOX_ASSAYS)
    dna_len = length * int(cfg.data.resolution)
    x_data = torch.rand(bsz, length, assays, device=device) * 4.0
    y_data = torch.rand(bsz, length, assays, device=device) * 4.0
    y_pval = torch.rand(bsz, length, assays, device=device) * 3.0
    y_peaks = torch.randint(0, 2, (bsz, length, assays), device=device).float()
    x_meta = torch.zeros(bsz, 4, assays, device=device)
    y_meta = torch.zeros(bsz, 4, assays, device=device)
    assay_ids = torch.arange(assays, device=device).float()
    x_meta[:, 1, :] = assay_ids
    y_meta[:, 1, :] = assay_ids
    x_meta[:, 0, :] = 23.0
    y_meta[:, 0, :] = 25.0
    x_meta[:, 2, :] = 36.0
    y_meta[:, 2, :] = 36.0
    x_meta[:, 3, :] = 0.0
    y_meta[:, 3, :] = 1.0
    return {
        "x_data": x_data.cpu(),
        "x_meta": x_meta.cpu(),
        "x_avail": torch.ones(bsz, assays),
        "x_dna": torch.zeros(bsz, dna_len, 4),
        "y_data": y_data.cpu(),
        "y_meta": y_meta.cpu(),
        "y_avail": torch.ones(bsz, assays),
        "y_pval": y_pval.cpu(),
        "y_peaks": y_peaks.cpu(),
        "control_data": torch.rand(bsz, length, 1),
        "control_meta": torch.zeros(bsz, 4, 1),
        "control_avail": torch.ones(bsz, 1),
        "y_dsf": torch.ones(bsz, assays, dtype=torch.long),
    }


def run_unit_tests(cfg: JEPADecoderConfig, device: torch.device) -> None:
    print("\n=== Unit tests ===")
    assays = len(SANDBOX_ASSAYS)
    tower = JEPADecoderTower(
        proj_dim=32,
        signal_dim=assays,
        decoder_input_dim=32,
        n_cnn_layers=2,
        expansion_factor=2,
        pool_size=2,
        grouped=False,
    ).to(device)
    z = torch.randn(2, 16, 32, device=device)
    out = tower(z)
    check("non-grouped tower shape", out.shape == (2, 64, assays), str(out.shape))
    grouped = copy.deepcopy(tower)
    grouped = JEPADecoderTower(
        proj_dim=32,
        signal_dim=assays,
        decoder_input_dim=32,
        n_cnn_layers=2,
        expansion_factor=2,
        pool_size=2,
        grouped=True,
    ).to(device)
    gout = grouped(z)
    check("grouped tower shape", gout.shape == (2, 64, assays), str(gout.shape))
    check("config checkpoint path parsed", bool(cfg.decoder.checkpoint_path))


def run_integration_tests(cfg: JEPADecoderConfig, device: torch.device) -> None:
    print("\n=== Integration tests ===")
    write_dummy_checkpoint(cfg, device)
    model = JEPADecoderModel.from_checkpoint(cfg, device)
    batch = synthetic_batch(cfg, device)
    masker = make_masker(
        p_full_assay=1.0,
        p_full_loci=0.0,
        p_chunks=0.0,
        min_available_frac=0.3,
        preserve_assay_id=True,
    )
    prep = prepare_masked_batch(batch, masker, device)
    assert prep is not None
    p, n, mu, var, df, peak = model(prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"], prep["query_mask"], prep["query_mask_signal"])
    expected = (2, int(cfg.data.context_length), len(SANDBOX_ASSAYS))
    check("forward p shape", p.shape == expected, str(p.shape))
    check("forward n shape", n.shape == expected, str(n.shape))
    check("forward mu shape", mu.shape == expected, str(mu.shape))
    check("forward var shape", var.shape == expected, str(var.shape))
    check("forward peak shape", peak.shape == expected, str(peak.shape))
    check("df is None", df is None, str(df))
    check("valid NB p", bool(((p > 0.0) & (p < 1.0)).all().item()))
    check("valid Gaussian var", bool((var >= float(cfg.decoder.gaussian_var_min)).all().item()))
    check("valid peak prob", bool(((peak >= 0.0) & (peak <= 1.0)).all().item()))

    crit = SandboxCompositeLoss(CANDI_LOSS()).to(device)
    loss, stats, _ = crit.forward_with_terms(
        p,
        n,
        mu,
        var,
        df,
        peak,
        prep["y_data"],
        prep["y_pval"],
        prep["y_peaks"],
        prep["observed_map"],
        prep["masked_map"],
        prep["signal_observed_map"],
        prep["signal_masked_map"],
        fallback_imp_to_observed_when_no_masked=False,
    )
    check("loss finite", bool(torch.isfinite(loss).item()), str(stats))
    loss.backward()
    enc_grads = [p.grad for m in model.encoder_modules() for p in m.parameters()]
    pred_grads = [p.grad for m in model.predictor_modules() for p in m.parameters()]
    dec_grads = [p.grad for m in model.decoder_modules() for p in m.parameters() if p.requires_grad]
    check("encoder frozen grads absent", all(g is None for g in enc_grads))
    check("predictor frozen grads absent", all(g is None for g in pred_grads))
    check("decoder grads present", any(g is not None and float(g.abs().sum()) > 0.0 for g in dec_grads))

    for mode in ("decoder_only", "predictor_decoder", "encoder_decoder", "all"):
        model.apply_freeze(mode)
        train_enc = any(p.requires_grad for m in model.encoder_modules() for p in m.parameters())
        train_pred = any(p.requires_grad for m in model.predictor_modules() for p in m.parameters())
        check(f"freeze mode {mode}", train_enc == (mode in {"encoder_decoder", "all"}) and train_pred == (mode in {"predictor_decoder", "all"}))


def run_training_smoke(cfg: JEPADecoderConfig, device: torch.device, h5_path: Optional[Path]) -> None:
    print("\n=== Training smoke ===")
    if h5_path is None or not h5_path.exists():
        print("  SKIP  dataset training/eval smoke (no --h5)")
        return
    cfg.data.h5_path = str(h5_path)
    cfg.training.max_train_batches = 2
    cfg.training.eval_max_batches = 1
    cfg.training.batch_size = 2
    model = JEPADecoderModel.from_checkpoint(cfg, device)
    ds = SandboxH5Dataset(
        h5_path,
        str(cfg.data.regime),
        train=True,
        batch_size=2,
        biosample_prefix="T_",
        dsf_list=list(cfg.training.dsf.dsf_list),
        dsf_sampling=str(cfg.training.dsf.sampling),
        seed=int(cfg.training.seed),
        shuffle=True,
        h5_cache_ram=False,
    )
    masker = make_masker(
        p_full_assay=1.0,
        p_full_loci=0.0,
        p_chunks=0.0,
        min_available_frac=0.3,
        preserve_assay_id=True,
    )
    crit = _make_criterion(cfg).to(device)
    opt = torch.optim.Adamax(list(model.trainable_parameters()), lr=1.0e-3)
    gstep = train_one_epoch(
        model,
        ds,
        device,
        masker,
        crit,
        opt,
        cfg,
        use_amp=False,
        global_step=0,
        max_batches=2,
    )
    check("two-step training ran", gstep == 2, str(gstep))
    metrics = run_eval_pass(
        model,
        _make_criterion(cfg, eval_mode=True).to(device),
        h5_path,
        str(cfg.data.regime),
        device,
        masker,
        batch_size=2,
        seed=42,
        max_batches=1,
        h5_cache_ram=False,
    )
    required = ["eval_metrics/imp_count_pearson_gw", "eval_metrics/imp_peak_auroc_gw", "eval_losses/total_loss"]
    check("eval key format", all(k in metrics for k in required), str(sorted(metrics.keys())[:8]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--h5", type=Path, default=None)
    parser.add_argument("--run-eval", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    with tempfile.TemporaryDirectory() as td:
        cfg = tiny_cfg(Path(td), str(device))
        run_unit_tests(cfg, device)
        run_integration_tests(cfg, device)
        if args.run_eval:
            run_training_smoke(cfg, device, args.h5)
    if errors:
        print("\nFAILED:", ", ".join(errors))
        return 1
    print("\nAll JEPA decoder smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
