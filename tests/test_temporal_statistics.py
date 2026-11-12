from dataclasses import replace
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.data import synthetic_batch


class TemporalStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_projection_preserves_existing_outputs_and_receives_gradients(self):
        config = ChimeraConfig.preset("tiny")
        original = Chimera(config).eval()
        extended = Chimera(replace(config, temporal_statistics=True)).eval()
        result = extended.load_state_dict(original.state_dict(), strict=False)
        self.assertEqual(result.missing_keys, ["temporal.statistics_proj.weight"])
        self.assertEqual(result.unexpected_keys, [])
        batch = synthetic_batch(config, batch_size=3, seed=51)
        with torch.no_grad():
            before, after = original(**batch), extended(**batch)
        for name in ("logits", "classification_logits", "forecast"):
            torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)
        extended.train()(**batch).loss.backward()
        gradient = extended.temporal.statistics_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_statistics_ignore_padding_and_handle_an_entire_missing_feature(self):
        config = replace(ChimeraConfig.preset("tiny"), temporal_statistics=True)
        model = Chimera(config).eval()
        torch.nn.init.normal_(model.temporal.statistics_proj.weight, std=0.02)
        batch = synthetic_batch(config, batch_size=2, seed=52)
        series = batch["series"][:, -8:].clone()
        series[0, :, 1] = float("nan")
        series[1, -2:, 0] = float("nan")
        valid = torch.ones(series.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            original = model(series=series, series_mask=valid)
            padded = torch.nn.functional.pad(series, (0, 0, 5, 7), value=99999)
            mask = torch.nn.functional.pad(valid, (5, 7), value=False)
            restored = model(series=padded, series_mask=mask)
        self.assertTrue(torch.isfinite(restored.forecast).all())
        torch.testing.assert_close(original.forecast, restored.forecast, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(original.classification_logits, restored.classification_logits, rtol=2e-5, atol=2e-6)

    def test_raw_level_and_scale_become_visible_to_the_encoder(self):
        model = Chimera(replace(ChimeraConfig.preset("tiny"), temporal_statistics=True)).eval()
        torch.nn.init.normal_(model.temporal.statistics_proj.weight, std=0.02)
        series = torch.linspace(-1, 1, 8)[None, :, None].expand(2, -1, 3).clone()
        series[1] = series[0] * 4 + 10
        captured = []
        hook = model.temporal.statistics_proj.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
        try:
            with torch.no_grad():
                tokens = model.temporal(series, torch.ones(2, 8, dtype=torch.bool))[0]
        finally:
            hook.remove()
        self.assertFalse(torch.allclose(captured[0][0], captured[0][1]))
        self.assertFalse(torch.allclose(tokens[0], tokens[1]))
        with self.assertRaises(ValueError):
            replace(model.config, temporal_statistics=1)


if __name__ == "__main__":
    unittest.main()
