from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from chimera import Chimera, ChimeraConfig
from chimera.data import synthetic_batch
from chimera.temporal_precision import temporal_work_dtype


class TemporalPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_large_finite_values_keep_encoder_and_adapter_finite(self):
        config = replace(ChimeraConfig.preset("tiny"), temporal_statistics=True,
                         forecast_adapter_context=8, forecast_adapter_relative=True)
        model = Chimera(config).eval()
        series = torch.tensor([[[3e38, float("nan"), 1.], [-3e38, float("nan"), 2.],
                                [3e38, float("nan"), 3.]]], dtype=torch.float32)
        valid = torch.ones(1, 3, dtype=torch.bool)
        tokens, _, mean, scale, baseline = model.temporal(series, valid)
        observed = series[0, :, 0].double()
        torch.testing.assert_close(mean[0, 0], observed.mean(), rtol=1e-12, atol=0)
        torch.testing.assert_close(scale[0, 0], (observed.var(unbiased=False) + 1e-5).sqrt(), rtol=1e-12, atol=0)
        self.assertEqual(mean.dtype, torch.float64)
        self.assertEqual(scale[0, 1].item(), 1)
        self.assertTrue(torch.isfinite(tokens).all() and torch.isfinite(baseline).all())
        context = model.forecast_adapter.prepare(series, valid, mean, scale)
        self.assertTrue(torch.isfinite(context.history).all())
        output = model(series=series, series_mask=valid, class_labels=torch.tensor([1]))
        self.assertTrue(torch.isfinite(output.forecast).all())
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(torch.isfinite(model.temporal.proj.weight.grad).all())
        with torch.inference_mode():
            first = model(torch.tensor([[1, 4]]), series=series, series_mask=valid, use_cache=True)
            second = model(torch.tensor([[5]]), cache=first.cache, use_cache=True)
        self.assertTrue(torch.isfinite(second.forecast).all())
        self.assertEqual(second.forecast.dtype, torch.float64)

    def test_dtype_selection_ignores_masked_values_and_rejects_unrepresentable_variance(self):
        valid = torch.tensor([[True, False]])
        series = torch.tensor([[[1.], [1e200]]], dtype=torch.float64)
        self.assertEqual(temporal_work_dtype(series, valid), torch.float32)
        self.assertEqual(temporal_work_dtype(series.float(), valid), torch.float32)
        with self.assertRaisesRegex(ValueError, "variance"):
            temporal_work_dtype(series, torch.ones_like(valid))

    def test_normal_inputs_keep_float32_outputs_exactly(self):
        config = replace(ChimeraConfig.preset("tiny"), temporal_statistics=True, forecast_adapter_context=8)
        model = Chimera(config).eval()
        batch = synthetic_batch(config, batch_size=2, seed=714)
        with torch.inference_mode():
            new = model(**batch)
            with patch("chimera.modules.temporal_work_dtype", return_value=torch.float32):
                original_precision = model(**batch)
        for name in ("logits", "classification_logits", "forecast", "loss"):
            torch.testing.assert_close(getattr(new, name), getattr(original_precision, name), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
