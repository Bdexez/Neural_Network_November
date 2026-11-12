from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.data import synthetic_batch
from chimera.forecast_adapter import fit_forecast_adapter
from chimera.training import load_checkpoint, save_checkpoint


class ForecastAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(39)

    def test_relative_context_is_translation_invariant_and_tracks_missing_value_age(self):
        config = replace(ChimeraConfig.preset("tiny"), forecast_adapter_context=8, forecast_adapter_relative=True)
        model = Chimera(config).eval()
        batch = synthetic_batch(config, batch_size=2, seed=704)
        series = batch["series"][:, -8:].clone()
        valid = batch["series_mask"][:, -8:]
        series[0, :, 1] = float("nan")
        series[1, -2:, 0] = float("nan")
        with torch.inference_mode():
            _, _, mean, scale, _ = model.temporal(series, valid)
            before = model.forecast_adapter.prepare(series, valid, mean, scale)
            after = model.forecast_adapter.prepare(series + 20, valid, mean + 20, scale)
        torch.testing.assert_close(before.history, after.history, rtol=2e-5, atol=1e-5)
        self.assertEqual(before.history[0, 1, -3].item(), 0)  # Fraction observée.
        self.assertEqual(before.history[0, 1, -2].item(), 1)  # Variable entièrement absente.
        self.assertEqual(before.history[1, 0, -2].item(), 0.25)
        with self.assertRaises(ValueError):
            replace(config, forecast_adapter_relative=1)

    def test_changing_feature_semantics_resets_only_adapter_and_roundtrips_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "absolute.pt"
            config = replace(ChimeraConfig.preset("tiny"), forecast_adapter_context=8)
            original = Chimera(config).eval()
            torch.nn.init.normal_(original.forecast_adapter.proj.weight, std=0.02)
            save_checkpoint(path, original)
            relative, _ = load_checkpoint(path, overrides={"forecast_adapter_relative": True})
            self.assertEqual(relative.forecast_adapter.proj.weight.count_nonzero().item(), 0)
            for name, value in original.state_dict().items():
                if not name.startswith("forecast_adapter."):
                    torch.testing.assert_close(value, relative.state_dict()[name], rtol=0, atol=0)
            torch.nn.init.normal_(relative.forecast_adapter.proj.weight, std=0.02)
            batch = synthetic_batch(config, batch_size=2, seed=705)
            ids, series = batch["input_ids"][:, :8], batch["series"][:, -8:]
            relative.eval()
            with torch.inference_mode():
                full = relative(ids, series=series)
                prefix = relative(ids[:, :3], series=series, use_cache=True)
                cached = relative(ids[:, 3:], cache=prefix.cache, use_cache=True)
            torch.testing.assert_close(full.forecast, cached.forecast, rtol=2e-5, atol=2e-6)
            save_checkpoint(Path(directory) / "relative.pt", relative)
            restored, _ = load_checkpoint(Path(directory) / "relative.pt")
            self.assertTrue(restored.forecast_adapter.relative)
            torch.testing.assert_close(restored.forecast_adapter.proj.weight, relative.forecast_adapter.proj.weight)

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

    def test_ridge_fit_improves_held_out_forecasts_changes_only_adapter_and_is_idempotent(self):
        config = replace(ChimeraConfig.preset("tiny"), forecast_adapter_context=8, dropout=0.1)
        model = Chimera(config)
        training = synthetic_batch(config, batch_size=128, seed=6001)
        validation = synthetic_batch(config, batch_size=64, seed=6002)
        for batch in (training, validation):
            batch["series"] = batch["series"][:, -8:]
            batch["series_mask"] = batch["series_mask"][:, -8:]
        training["forecast_targets"][:10, 2, 1] = float("nan")
        state = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.inference_mode():
            before = model.eval()(series=validation["series"])
        model.train()
        rng = torch.get_rng_state().clone()
        report = fit_forecast_adapter(model, [training])
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(report["active_examples"], 128)
        self.assertEqual(report["observed_targets_by_horizon"][2], 374)
        self.assertEqual(report["updated_parameters"], 396)
        for name, value in model.state_dict().items():
            if not name.startswith("forecast_adapter."):
                torch.testing.assert_close(value, state[name], rtol=0, atol=0, msg=name)
        with torch.inference_mode():
            after = model.eval()(series=validation["series"])
        target = validation["forecast_targets"]
        before_mae = (before.forecast[..., 1] - target).abs().mean()
        after_mae = (after.forecast[..., 1] - target).abs().mean()
        self.assertLess(after_mae.item(), before_mae.item() * 0.7)
        torch.testing.assert_close(before.classification_logits, after.classification_logits, rtol=0, atol=0)
        fitted = {name: value.clone() for name, value in model.forecast_adapter.state_dict().items()}
        fit_forecast_adapter(model, [training])
        for name, value in model.forecast_adapter.state_dict().items():
            torch.testing.assert_close(value, fitted[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "plus court"):
            fit_forecast_adapter(model, [synthetic_batch(config, batch_size=2)])
        for name, value in model.forecast_adapter.state_dict().items():
            torch.testing.assert_close(value, fitted[name], rtol=0, atol=0)
        self.assertFalse(model.forecast_adapter._forward_pre_hooks)

    def test_transfer_materializes_only_new_adapter_parameters_and_supports_bf16(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.pt"
            original = Chimera(ChimeraConfig.preset("tiny"))
            save_checkpoint(path, original)
            loaded, _ = load_checkpoint(path, overrides={"forecast_adapter_context": 8})
            self.assertFalse(any(p.is_meta for p in loaded.parameters()))
            self.assertEqual(loaded.forecast_adapter.proj.weight.count_nonzero().item(), 0)
            batch = synthetic_batch(loaded.config, batch_size=16, seed=6003)
            batch["series"], batch["series_mask"] = batch["series"][:, -8:], batch["series_mask"][:, -8:]
            fit_forecast_adapter(loaded, [batch], amp="bf16")
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                forecast = loaded.eval()(series=batch["series"]).forecast
            self.assertTrue(torch.isfinite(forecast).all())
            self.assertTrue((forecast[..., 1:] >= forecast[..., :-1]).all())
            save_checkpoint(Path(directory) / "fitted.pt", loaded)
            restored, _ = load_checkpoint(Path(directory) / "fitted.pt")
            torch.testing.assert_close(restored.forecast_adapter.proj.weight, loaded.forecast_adapter.proj.weight)
            replaced, _ = load_checkpoint(Path(directory) / "fitted.pt", overrides={"forecast_adapter_context": 4})
            self.assertEqual(replaced.forecast_adapter.proj.weight.count_nonzero().item(), 0)
