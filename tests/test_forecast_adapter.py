from dataclasses import replace
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.data import synthetic_batch


class ForecastAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_initialization_preserves_outputs_and_active_head_only_changes_short_forecasts(self):
        config = ChimeraConfig.preset("tiny")
        original = Chimera(config).eval()
        adapted = Chimera(replace(config, forecast_adapter_context=8)).eval()
        missing = adapted.load_state_dict(original.state_dict(), strict=False)
        self.assertEqual(set(missing.missing_keys), {"forecast_adapter.proj.weight", "forecast_adapter.proj.bias"})
        batch = synthetic_batch(config, batch_size=2, seed=1)
        inputs = {key: value for key, value in batch.items() if key not in ("labels", "class_labels", "forecast_targets")}
        inputs["series"] = inputs["series"][:, -8:]
        inputs["series_mask"] = inputs["series_mask"][:, -8:]
        with torch.no_grad():
            before, initial = original(**inputs), adapted(**inputs)
        for name in ("logits", "classification_logits", "forecast"):
            torch.testing.assert_close(getattr(before, name), getattr(initial, name), rtol=0, atol=0)
        torch.nn.init.normal_(adapted.forecast_adapter.proj.weight, std=0.05)
        with torch.no_grad():
            after = adapted(**inputs)
            full_before, full_after = original(**batch), adapted(**batch)
        torch.testing.assert_close(before.logits, after.logits, rtol=0, atol=0)
        torch.testing.assert_close(before.classification_logits, after.classification_logits, rtol=0, atol=0)
        self.assertFalse(torch.allclose(before.forecast, after.forecast))
        self.assertTrue((after.forecast[..., 1:] >= after.forecast[..., :-1]).all())
        torch.testing.assert_close(full_before.forecast, full_after.forecast, rtol=0, atol=0)

    def test_prepared_context_ignores_padding_holes_and_missing_features(self):
        config = replace(ChimeraConfig.preset("tiny"), forecast_adapter_context=8)
        model = Chimera(config).eval()
        torch.nn.init.normal_(model.forecast_adapter.proj.weight, std=0.02)
        batch = synthetic_batch(config, batch_size=2, seed=2)
        series = batch["series"][:, -6:].clone()
        series[0, :, 1] = float("nan")
        series[1, -2:, 0] = float("nan")
        valid = torch.ones(series.shape[:2], dtype=torch.bool)
        valid[1, 2] = False
        with torch.no_grad():
            before = model(series=series, series_mask=valid)
            padded = torch.nn.functional.pad(series, (0, 0, 3, 5), value=9999)
            mask = torch.nn.functional.pad(valid, (3, 5), value=False)
            after = model(series=padded, series_mask=mask)
        self.assertTrue(torch.isfinite(after.forecast).all())
        torch.testing.assert_close(before.forecast, after.forecast, rtol=2e-5, atol=2e-6)
        for value in (-1, True, 33):
            with self.assertRaises(ValueError):
                replace(config, forecast_adapter_context=value)

    def test_cached_generation_retains_adapter_context(self):
        config = replace(ChimeraConfig.preset("tiny"), forecast_adapter_context=8, temporal_statistics=True)
        model = Chimera(config).eval()
        torch.nn.init.normal_(model.forecast_adapter.proj.weight, std=0.1)
        batch = synthetic_batch(config, batch_size=2, seed=3)
        ids, series = batch["input_ids"][:, :9], batch["series"][:, -8:]
        with torch.no_grad():
            full = model(ids, images=batch["images"], series=series)
            first = model(ids[:, :4], images=batch["images"], series=series, use_cache=True)
            second = model(ids[:, 4:], cache=first.cache, use_cache=True)
        self.assertIsNotNone(first.cache.forecast_adapter_context)
        torch.testing.assert_close(full.forecast, second.forecast, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(full.logits[:, 4:], second.logits, rtol=2e-5, atol=2e-6)
