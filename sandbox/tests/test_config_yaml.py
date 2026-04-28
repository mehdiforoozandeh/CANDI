from pathlib import Path

import pytest
import torch

from dataclasses import asdict

from sandbox.config import deep_merge, load_yaml, merge_dict
from sandbox.config_types import SandboxConfig, config_from_dict
from sandbox.data import build_canonical_meta
from sandbox import SANDBOX_ASSAYS


def test_merge_dict_order():
    base = {"a": 1, "b": 2}
    assert merge_dict(base, {"b": 3}) == {"a": 1, "b": 3}


def test_load_default_config_nested():
    pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = load_yaml(root)
    # default.yaml no longer overrides wandb.mode; code default is "online"
    assert "wandb" not in cfg or "mode" not in cfg.get("wandb", {})
    merged = deep_merge(asdict(SandboxConfig()), cfg)
    obj = config_from_dict(SandboxConfig, merged)
    assert obj.wandb.mode == "online"
    # default.yaml must still set h5_cache_ram
    assert obj.data.h5_cache_ram is True


def test_build_canonical_meta_shape_and_values():
    """build_canonical_meta must return [4, F] tensor with finite values for SANDBOX_ASSAYS."""
    import math
    eic_path = Path(__file__).resolve().parents[2] / "data" / "eic_metadata.csv"
    if not eic_path.exists():
        pytest.skip("eic_metadata.csv not found")
    meta = build_canonical_meta(str(eic_path), SANDBOX_ASSAYS)
    assert meta.shape == (4, len(SANDBOX_ASSAYS))
    assert meta.dtype == torch.float32
    # Row 0 = depth_log2 must be positive for all SANDBOX_ASSAYS (all are in CSV).
    assert (meta[0] > 0).all(), f"Some assays have non-positive depth_log2: {meta[0]}"
    # Row 1 = assay indices must be 0..F-1.
    assert list(meta[1].long().tolist()) == list(range(len(SANDBOX_ASSAYS)))
    # Row 2 = read_length must be positive.
    assert (meta[2] > 0).all()
    # Row 3 = run_type must be 0 or 1.
    assert ((meta[3] == 0.0) | (meta[3] == 1.0)).all()


def test_build_canonical_meta_missing_file_returns_neg_ones():
    """build_canonical_meta with non-existent path returns all-(-1) tensor."""
    meta = build_canonical_meta("/nonexistent/path.csv", SANDBOX_ASSAYS)
    assert meta.shape == (4, len(SANDBOX_ASSAYS))
    assert (meta == -1.0).all()


def test_eval_config_new_fields_defaults():
    """New EvalConfig fields must have correct defaults."""
    cfg = SandboxConfig()
    assert cfg.eval.use_canonical_missing_meta is True
    assert cfg.eval.eic_metadata_path == "data/eic_metadata.csv"
    assert cfg.eval.meta_sensitivity_probe_every_n_steps == 200
    assert cfg.eval.eval_every_n_epochs == 5
    assert cfg.training.training_stats_jsonl_every_n_steps == 200
    assert cfg.training.save_checkpoint is False
    assert cfg.training.early_stop_enabled is False
    assert cfg.training.early_stop_patience == 5


def test_gate_d_unknown_top_level_yaml_key_rejected():
    pytest.importorskip("yaml")
    d = asdict(SandboxConfig())
    d = deep_merge(d, {"not_a_real_key": 1})
    with pytest.raises(ValueError, match="unknown config keys"):
        config_from_dict(SandboxConfig, d)
