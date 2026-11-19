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
