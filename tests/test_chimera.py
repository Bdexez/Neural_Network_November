from contextlib import redirect_stdout
from dataclasses import replace
import io
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import torch

from chimera import ByteTokenizer, Chimera, ChimeraConfig
from chimera.cli import inspect_config, main
from chimera.data import JointDataset, collate_records, synthetic_batch
from chimera.training import load_checkpoint, save_checkpoint


class ChimeraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = replace(ChimeraConfig.preset("tiny"), max_seq_len=128)

    def setUp(self):
        torch.manual_seed(123)
        self.model = Chimera(self.config)
        self.batch = synthetic_batch(self.config, batch_size=3, seed=4)

    def inputs(self):
        return {key: value for key, value in self.batch.items()
                if key not in ("labels", "class_labels", "forecast_targets")}

    def test_utf8_roundtrip(self):
        tokenizer = ByteTokenizer()
        text = "Français, 中文, العربية, 🧠\n"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_invalid_configuration(self):
        for changes in (dict(n_heads=3), dict(n_kv_heads=3), dict(top_k=9),
                        dict(patch_size=7), dict(quantiles=(0.5, 0.1)),
                        dict(dropout=1), dict(n_experts=0), dict(rope_theta=float("nan"))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)
        with self.assertRaises(ValueError):
            ChimeraConfig.from_dict({"typo": 1})

    def test_all_seven_modality_combinations(self):
        self.model.eval()
        for count in range(1, 4):
            for names in itertools.combinations(("input_ids", "images", "series"), count):
                with self.subTest(names=names), torch.no_grad():
                    output = self.model(**{name: self.batch[name] for name in names})
                    self.assertEqual(output.classification_logits.shape, (3, self.config.num_classes))
                    self.assertEqual(output.forecast.shape, (3, self.config.forecast_horizon, 3, 3))
                    self.assertEqual(output.logits is not None, "input_ids" in names)
                    self.assertTrue(torch.isfinite(output.forecast).all())
                    self.assertIsNone(output.loss)

    def test_joint_backward_reaches_every_branch(self):
        output = self.model(**self.batch)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for name in ("embedding", "vision", "temporal", "resampler", "blocks",
                     "future_heads", "classifier", "forecaster", "alignment"):
            module = getattr(self.model, name)
            norm = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
            self.assertGreater(norm, 0, name)
        for block in self.model.blocks:
            self.assertGreater(block.moe.router.weight.grad.abs().sum().item(), 0)
            self.assertGreater(block.moe.shared.down.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_future_tokens_cannot_change_past_logits(self):
        self.model.eval()
        ids = self.batch["input_ids"].clone()
        changed = ids.clone()
        changed[:, 10:] = torch.randint(4, 260, changed[:, 10:].shape)
        with torch.no_grad():
            first = self.model(ids, images=self.batch["images"], series=self.batch["series"]).logits
            second = self.model(changed, images=self.batch["images"], series=self.batch["series"]).logits
        torch.testing.assert_close(first[:, :10], second[:, :10], rtol=1e-5, atol=1e-6)

    def test_cache_matches_full_across_sliding_window(self):
        self.model.eval()
        ids = torch.randint(4, 260, (2, 49))
        images, series = self.batch["images"][:2], self.batch["series"][:2]
        with torch.no_grad():
            full = self.model(ids, images=images, series=series).logits
            first = self.model(ids[:, :35], images=images, series=series, use_cache=True)
            second = self.model(ids[:, 35:42], cache=first.cache, use_cache=True)
            third = self.model(ids[:, 42:], cache=second.cache, use_cache=True)
        joined = torch.cat((first.logits, second.logits, third.logits), 1)
        torch.testing.assert_close(joined, full, rtol=2e-5, atol=2e-6)
        self.assertEqual(third.cache.offset, 49)
        with torch.no_grad():
            expected = self.model(ids, images=images, series=series)
        torch.testing.assert_close(third.forecast, expected.forecast, rtol=2e-5, atol=2e-6)

    def test_cached_left_padding(self):
        self.model.eval()
        ids = torch.randint(4, 260, (2, 12))
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[0, :3] = False
        with torch.no_grad():
            full = self.model(ids, attention_mask=mask).logits
            first = self.model(ids[:, :9], attention_mask=mask[:, :9], use_cache=True)
            second = self.model(ids[:, 9:], cache=first.cache, use_cache=True)
        torch.testing.assert_close(second.logits, full[:, 9:], rtol=2e-5, atol=2e-6)

    def test_text_padding_does_not_change_losses(self):
        self.model.eval()
        batch = self.batch.copy()
        with torch.no_grad():
            first = self.model(**batch)
            batch["input_ids"] = torch.nn.functional.pad(batch["input_ids"], (0, 7), value=0)
            batch["labels"] = batch["input_ids"].clone()
            batch["attention_mask"] = torch.nn.functional.pad(batch["attention_mask"], (0, 7), value=False)
            second = self.model(**batch)
        torch.testing.assert_close(first.loss, second.loss, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(first.classification_logits, second.classification_logits, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(first.expert_usage, second.expert_usage)

    def test_series_padding_and_missing_values(self):
        self.model.eval()
        batch = self.inputs()
        batch["series"] = batch["series"].clone()
        batch["series"][0, 4, 1] = float("nan")
        with torch.no_grad():
            first = self.model(**batch)
            batch["series"] = torch.nn.functional.pad(batch["series"], (0, 0, 0, 9), value=123456)
            batch["series_mask"] = torch.nn.functional.pad(batch["series_mask"], (0, 9), value=False)
            second = self.model(**batch)
        torch.testing.assert_close(first.forecast, second.forecast, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.isfinite(second.logits).all())

    def test_entire_missing_feature_has_neutral_scale(self):
        series = self.batch["series"].clone()
        series[:, :, 1] = float("nan")
        valid = torch.ones(series.shape[:2], dtype=torch.bool)
        tokens, _, mean, scale, baseline = self.model.temporal(series, valid)
        self.assertTrue(torch.isfinite(tokens).all())
        torch.testing.assert_close(mean[:, 1], torch.zeros(3))
        torch.testing.assert_close(scale[:, 1], torch.ones(3))
        torch.testing.assert_close(baseline[:, :, 1], torch.zeros(3, self.config.forecast_horizon))

    def test_forecast_connection_starts_at_last_observation(self):
        series = self.batch["series"].clone()
        series[0, -1, 0] = float("nan")
        valid = torch.ones(series.shape[:2], dtype=torch.bool)
        baseline = self.model.temporal(series, valid)[-1]
        last = series[:, -1].clone()
        last[0, 0] = series[0, -2, 0]
        torch.testing.assert_close(baseline, last[:, None].expand_as(baseline))

    def test_temporal_connection_learns_beyond_persistence(self):
        training = synthetic_batch(self.config, batch_size=32, seed=700)
        validation = synthetic_batch(self.config, batch_size=32, seed=701)
        optimizer = torch.optim.Adam(self.model.temporal.forecast_skip.parameters(), lr=0.02)
        for _ in range(50):
            optimizer.zero_grad(set_to_none=True)
            predicted = self.model.temporal(training["series"], training["series_mask"])[-1]
            loss = (predicted - training["forecast_targets"]).square().mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predicted = self.model.temporal(validation["series"], validation["series_mask"])[-1]
            learned_mae = (predicted - validation["forecast_targets"]).abs().mean()
            persistence_mae = (validation["series"][:, -1:] - validation["forecast_targets"]).abs().mean()
        self.assertLess(learned_mae.item(), persistence_mae.item() * 0.6)
