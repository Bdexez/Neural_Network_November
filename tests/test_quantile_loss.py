import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.losses import quantile_pinball_loss


class QuantileLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fp16_accumulates_large_batches_without_overflow(self):
        predictions = torch.zeros(70000, 1, 1, 3, dtype=torch.float16, requires_grad=True)
        targets = torch.ones(70000, 1, 1, dtype=torch.float16)
        loss = quantile_pinball_loss(predictions, targets, (0.1, 0.5, 0.9))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(loss.item(), 0.5, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        self.assertTrue((predictions.grad != 0).all())

    def test_bfloat16_keeps_quantiles_in_float32_and_ignores_missing_targets(self):
        predictions = torch.zeros(4, 1, 1, 3, dtype=torch.bfloat16, requires_grad=True)
        targets = torch.tensor([1, -2, float("nan"), float("inf")], dtype=torch.bfloat16).view(4, 1, 1)
        loss = quantile_pinball_loss(predictions, targets, (0.1, 0.5, 0.9))
        self.assertAlmostEqual(loss.item(), 0.75, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        self.assertEqual(predictions.grad[2:].count_nonzero().item(), 0)

    def test_regular_values_keep_standard_loss_and_gradients(self):
        generator = torch.Generator().manual_seed(1881)
        targets = torch.randn(8, 4, 3, generator=generator)
        targets[1, 2] = float('nan')
        observed = torch.isfinite(targets)
        first = torch.randn(8, 4, 3, 3, generator=generator, requires_grad=True)
        second = first.detach().clone().requires_grad_()
        quantiles = (0.1, 0.5, 0.9)
        actual = quantile_pinball_loss(first, targets, quantiles)
        residual = torch.where(observed, targets, 0).unsqueeze(-1) - second
        q = torch.tensor(quantiles)
        expected = (torch.maximum(q * residual, (q - 1) * residual) * observed.unsqueeze(-1)).sum()
        expected = expected / (observed.sum() * len(quantiles))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)

    def test_extreme_float64_mean_and_gradient_remain_representable(self):
        predictions = torch.full((32, 1, 1, 3), -1e308, dtype=torch.float64, requires_grad=True)
        targets = torch.full((32, 1, 1), 1e308, dtype=torch.float64)
        loss = quantile_pinball_loss(predictions, targets, (0.1, 0.5, 0.9))
        torch.testing.assert_close(loss, torch.tensor(1e308, dtype=torch.float64), rtol=1e-15, atol=0)
        loss.backward()
        expected = torch.tensor([-0.1, -0.5, -0.9], dtype=torch.float64) / (32 * 3)
        torch.testing.assert_close(predictions.grad, expected.expand_as(predictions), rtol=1e-15, atol=0)

    def test_extreme_equal_values_do_not_hide_small_residuals(self):
        predictions = torch.tensor([[1e308] * 3, [0.] * 3], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1e308, 1e-100], dtype=torch.float64)
        loss = quantile_pinball_loss(predictions, targets, (0.1, 0.5, 0.9))
        torch.testing.assert_close(loss, torch.tensor(2.5e-101, dtype=torch.float64), rtol=1e-15, atol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        self.assertTrue((predictions.grad[1] != 0).all())

    def test_nonfinite_predictions_at_missing_targets_are_ignored(self):
        predictions = torch.tensor([[1., 1., 1.], [float('nan'), float('inf'), float('-inf')]],
                                   dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([2., float('nan')], dtype=torch.float64)
        loss = quantile_pinball_loss(predictions, targets, (0.1, 0.5, 0.9))
        self.assertEqual(loss.item(), 0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        self.assertEqual(predictions.grad[1].count_nonzero().item(), 0)
        missing = torch.full_like(targets, float('nan'))
        predictions.grad = None
        zero = quantile_pinball_loss(predictions, missing, (0.1, 0.5, 0.9))
        zero.backward()
        self.assertEqual(zero.item(), 0)
        self.assertEqual(predictions.grad.count_nonzero().item(), 0)

    def test_autocast_forecasting_uses_unrounded_quantiles(self):
        config = ChimeraConfig.preset("tiny")
        model = Chimera(config).train()
        images = torch.rand(2, config.image_channels, config.image_size, config.image_size)
        targets = torch.ones(2, config.forecast_horizon, config.series_features)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(images=images, forecast_targets=targets)
        self.assertEqual(output.forecast.dtype, torch.bfloat16)
        expected = quantile_pinball_loss(output.forecast.detach().float(), targets, config.quantiles)
        torch.testing.assert_close(output.losses["forecast"], expected, rtol=1e-6, atol=1e-7)
        output.loss.backward()
        self.assertTrue(torch.isfinite(model.forecaster.proj[-1].weight.grad).all())

    def test_float64_and_entirely_missing_targets(self):
        predictions = torch.zeros(2, 1, 1, 3, dtype=torch.float64, requires_grad=True)
        missing = torch.full((2, 1, 1), float("nan"), dtype=torch.float64)
        loss = quantile_pinball_loss(predictions, missing, (0.1, 0.5, 0.9))
        self.assertEqual(loss.dtype, torch.float64)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(predictions.grad.count_nonzero().item(), 0)

    def test_large_float32_values_do_not_overflow_residual_or_sum(self):
        predictions = torch.full((20, 1, 1, 3), -3e38, dtype=torch.float32, requires_grad=True)
        targets = torch.full((20, 1, 1), 3e38, dtype=torch.float32)
        loss = quantile_pinball_loss(predictions, targets, (0.1, 0.5, 0.9))
        expected = (targets.double().mean() - predictions.detach().double().mean()) / 2
        self.assertEqual(loss.dtype, torch.float64)
        torch.testing.assert_close(loss, expected, rtol=1e-12, atol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        self.assertTrue((predictions.grad != 0).all())


if __name__ == "__main__":
    unittest.main()
