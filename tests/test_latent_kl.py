import os
import sys
import unittest

import torch


# Ensure project root importability
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import CANDI


def _build_tiny_model(enable_latent_kl=False, latent_reparam_mode="clamp"):
    return CANDI(
        signal_dim=2,
        metadata_embedding_dim=8,
        conv_kernel_size=3,
        n_cnn_layers=1,
        nhead=1,
        n_sab_layers=1,
        pool_size=2,
        dropout=0.0,
        context_length=8,
        pos_enc="relative",
        expansion_factor=2,
        separate_decoders=True,
        num_assays=3,
        num_runtypes=2,
        norm="layer",
        attention_type="xtransformers",
        output_ff=False,
        dist_type="gaussian",
        xl_dna=False,
        mask_stem=False,
        signal_transform="none",
        decoder_type="fixed",
        enable_latent_kl=enable_latent_kl,
        latent_std_min=0.01,
        latent_std_max=1.0,
        latent_reparam_mode=latent_reparam_mode,
        latent_sample_train_only=True,
    )


def _copy_shared_state(src_model, dst_model):
    dst_state = dst_model.state_dict()
    for key, value in src_model.state_dict().items():
        if key in dst_state and dst_state[key].shape == value.shape:
            dst_state[key] = value.clone()
    dst_model.load_state_dict(dst_state, strict=False)


def _tiny_inputs():
    # src: [B, L, F+1] ; seq: [B, L*25, 4] ; x_meta: [B, 4, F+1] ; y_meta: [B, 4, F]
    B, L, F = 2, 8, 2
    src = torch.rand(B, L, F + 1)
    seq = torch.rand(B, L * 25, 4)

    x_meta = torch.zeros(B, 4, F + 1)
    y_meta = torch.zeros(B, 4, F)
    # assay IDs in-range (including control id=num_assays for X metadata final lane)
    x_meta[:, 1, :] = torch.tensor([0, 1, 3], dtype=torch.float32)
    y_meta[:, 1, :] = torch.tensor([0, 1], dtype=torch.float32)
    # run_type in-range
    x_meta[:, 3, :] = 1.0
    y_meta[:, 3, :] = 1.0
    # depth/read_length > 0
    x_meta[:, 0, :] = 10.0
    x_meta[:, 2, :] = 50.0
    y_meta[:, 0, :] = 10.0
    y_meta[:, 2, :] = 50.0
    return src, seq, x_meta, y_meta


class TestCandiLatentKl(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_clamp_reparam_respects_std_bounds(self):
        model = _build_tiny_model(enable_latent_kl=True, latent_reparam_mode="clamp")
        z = torch.randn(2, 4, model.latent_dim)

        with torch.no_grad():
            model.latent_logvar_head.weight.zero_()
            model.latent_logvar_head.bias.fill_(100.0)
        _, _ = model._apply_latent_regularization(z)
        stats = model.get_last_latent_stats()
        self.assertLessEqual(stats["latent_std_max_seen"], 1.000001)

        with torch.no_grad():
            model.latent_logvar_head.bias.fill_(-100.0)
        _, _ = model._apply_latent_regularization(z)
        stats = model.get_last_latent_stats()
        self.assertGreaterEqual(stats["latent_std_min_seen"], 0.009999)

    def test_softplus_reparam_respects_std_bounds(self):
        model = _build_tiny_model(enable_latent_kl=True, latent_reparam_mode="softplus")
        z = torch.randn(2, 4, model.latent_dim)

        with torch.no_grad():
            model.latent_logvar_head.weight.zero_()
            model.latent_logvar_head.bias.fill_(100.0)
        _, _ = model._apply_latent_regularization(z)
        stats = model.get_last_latent_stats()
        self.assertLessEqual(stats["latent_std_max_seen"], 1.000001)
        self.assertGreaterEqual(stats["latent_std_min_seen"], 0.009999)

    def test_forward_signature_parity_flag_off_and_on(self):
        src, seq, x_meta, y_meta = _tiny_inputs()

        model_off = _build_tiny_model(enable_latent_kl=False)
        out_off = model_off(src, seq, x_meta, y_meta, return_z=False)
        out_off_z = model_off(src, seq, x_meta, y_meta, return_z=True)
        self.assertEqual(len(out_off), 6)
        self.assertEqual(len(out_off_z), 7)

        model_on = _build_tiny_model(enable_latent_kl=True, latent_reparam_mode="clamp")
        out_on = model_on(src, seq, x_meta, y_meta, return_z=False)
        out_on_z = model_on(src, seq, x_meta, y_meta, return_z=True)
        self.assertEqual(len(out_on), 6)
        self.assertEqual(len(out_on_z), 7)

        for idx in range(6):
            if out_off[idx] is None or out_on[idx] is None:
                self.assertIs(out_off[idx], out_on[idx])
            else:
                self.assertEqual(out_off[idx].shape, out_on[idx].shape)

    def test_enable_latent_kl_preserves_shared_initialization(self):
        torch.manual_seed(123)
        model_off = _build_tiny_model(enable_latent_kl=False)

        torch.manual_seed(123)
        model_on = _build_tiny_model(enable_latent_kl=True, latent_reparam_mode="softplus")

        for key, value in model_off.state_dict().items():
            self.assertIn(key, model_on.state_dict())
            self.assertTrue(
                torch.equal(value, model_on.state_dict()[key]),
                msg=f"shared parameter changed when latent KL enabled: {key}",
            )

    def test_deterministic_latent_path_matches_baseline_with_shared_weights(self):
        src, seq, x_meta, y_meta = _tiny_inputs()

        torch.manual_seed(321)
        model_off = _build_tiny_model(enable_latent_kl=False)

        torch.manual_seed(999)
        model_on = _build_tiny_model(enable_latent_kl=True, latent_reparam_mode="softplus")
        _copy_shared_state(model_off, model_on)
        model_on.set_latent_train_controls(
            global_step=0,
            force_deterministic_train=True,
            freeze_posterior_heads=True,
            blend_alpha_train=0.0,
            enable_sampling_train=False,
        )

        model_off.train()
        model_on.train()
        out_off = model_off(src, seq, x_meta, y_meta, return_z=True)
        out_on = model_on(src, seq, x_meta, y_meta, return_z=True)

        for off_tensor, on_tensor in zip(out_off, out_on):
            if off_tensor is None or on_tensor is None:
                self.assertIs(off_tensor, on_tensor)
            else:
                self.assertTrue(torch.equal(off_tensor, on_tensor))


if __name__ == "__main__":
    unittest.main()
