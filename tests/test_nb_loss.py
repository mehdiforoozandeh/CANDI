import unittest
import torch

import os
import sys

# Ensure project root is importable even if runner resets cwd (e.g., cluster env hooks).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from _utils import negative_binomial_loss, negative_binomial_loss_stable


class TestNegativeBinomialLossStable(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _rand_params(self, shape, device="cpu"):
        # Moderate ranges that should stay finite for torch.distributions implementation.
        y = torch.randint(low=0, high=200, size=shape, device=device, dtype=torch.int64).float()
        n = torch.rand(shape, device=device) * 50.0 + 0.1          # (0.1, 50.1)
        p = torch.rand(shape, device=device) * 0.98 + 0.01         # (0.01, 0.99)
        return y, n, p

    def test_matches_baseline_value_finite_regime(self):
        """
        In the regime where torch.distributions.NegativeBinomial.log_prob stays finite,
        stable implementation should match the current baseline negative_binomial_loss
        (which wraps the torch distribution).
        """
        for shape in [(64,), (8, 16), (2, 4, 8)]:
            y, n, p = self._rand_params(shape)

            baseline = negative_binomial_loss(y, n, p)
            stable = negative_binomial_loss_stable(y, n, p)

            # Baseline may zero-out non-finite entries; in this test we expect everything finite.
            self.assertTrue(torch.isfinite(baseline).all().item())
            self.assertTrue(torch.isfinite(stable).all().item())

            max_abs = (baseline - stable).abs().max().item()
            # Small numerical differences are expected because:
            # - baseline uses F.logsigmoid internally
            # - stable uses log/log1p
            # Both are mathematically identical but can differ at ~1e-5..1e-4 in float32.
            self.assertLess(max_abs, 1e-4)

    def test_matches_baseline_gradients_finite_regime(self):
        """
        Compare autograd gradients w.r.t. n_pred and p_pred for a finite regime.
        This ensures the closed-form is not just value-correct but gradient-consistent.
        """
        y, n, p = self._rand_params((128,))
        n = n.clone().requires_grad_(True)
        p = p.clone().requires_grad_(True)

        baseline = negative_binomial_loss(y, n, p).mean()
        baseline.backward()
        grad_n_baseline = n.grad.detach().clone()
        grad_p_baseline = p.grad.detach().clone()

        n.grad = None
        p.grad = None

        stable = negative_binomial_loss_stable(y, n, p).mean()
        stable.backward()
        grad_n_stable = n.grad.detach().clone()
        grad_p_stable = p.grad.detach().clone()

        self.assertTrue(torch.isfinite(grad_n_baseline).all().item())
        self.assertTrue(torch.isfinite(grad_p_baseline).all().item())
        self.assertTrue(torch.isfinite(grad_n_stable).all().item())
        self.assertTrue(torch.isfinite(grad_p_stable).all().item())

        # Gradients may differ slightly due to clamp boundary effects; keep a small tolerance.
        self.assertLess((grad_n_baseline - grad_n_stable).abs().max().item(), 1e-4)
        self.assertLess((grad_p_baseline - grad_p_stable).abs().max().item(), 1e-4)

    def test_stability_extreme_regime_reduces_zero_replacements(self):
        """
        Deterministic stability test reproducing the AMP/FP16 failure mode:

        In FP16, lgamma(total_count + value) can overflow to inf for moderately large counts,
        which can make log_prob non-finite. The baseline implementation then maps those
        entries to zero loss (zero gradient), while the stable implementation forces FP32
        internally and stays finite.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available; this stability test targets AMP/FP16 behavior on GPU.")

        device = "cuda"
        shape = (4096,)

        # Pick values that are valid but likely to overflow lgamma in float16.
        # Use a representable float16 value close to the max (65504) so that
        # lgamma(total_count + value) overflows to inf in fp16.
        # (The *input* must be representable in fp16, but lgamma(output) will overflow.)
        y = torch.full(shape, 65000.0, dtype=torch.float16, device=device)
        n = torch.full(shape, 10.0, dtype=torch.float16, device=device)
        p = torch.full(shape, 0.5, dtype=torch.float16, device=device)  # your convention

        # Baseline under autocast fp16 (common training setting)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            baseline = negative_binomial_loss(y, n, p)

        # Stable always forces fp32 internally
        stable = negative_binomial_loss_stable(y, n, p, invalid_penalty=1e6)

        # Baseline maps non-finite to 0, so overflow should manifest as many exact zeros.
        self.assertGreater((baseline == 0).sum().item(), 0)

        # Stable should remain finite and not collapse to zeros.
        self.assertTrue(torch.isfinite(stable).all().item())
        self.assertEqual((stable == 0).sum().item(), 0)


if __name__ == "__main__":
    unittest.main()


