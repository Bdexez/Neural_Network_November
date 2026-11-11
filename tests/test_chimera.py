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

    def test_quantiles_never_cross(self):
        config = replace(self.config, quantiles=(0.01, 0.1, 0.5, 0.9, 0.99))
        output = Chimera(config)(**self.batch)
        self.assertTrue((output.forecast[..., 1:] >= output.forecast[..., :-1]).all())

    def test_modalities_change_text_and_prediction(self):
        self.model.eval()
        inputs = self.inputs()
        with torch.no_grad():
            original = self.model(**inputs)
            image_changed = self.model(**dict(inputs, images=torch.zeros_like(inputs["images"])))
            series_changed = self.model(**dict(inputs, series=-inputs["series"]))
        for other in (image_changed, series_changed):
            self.assertGreater((original.logits - other.logits).abs().max().item(), 1e-7)
            self.assertGreater((original.forecast - other.forecast).abs().max().item(), 1e-7)

    def test_ignored_targets_are_finite(self):
        batch = dict(self.batch, labels=torch.full_like(self.batch["labels"], -100),
                     class_labels=torch.full_like(self.batch["class_labels"], -100),
                     forecast_targets=torch.full_like(self.batch["forecast_targets"], float("nan")))
        output = self.model(**batch)
        for name in ("text", "multi_token", "classification", "forecast"):
            self.assertEqual(output.losses[name].item(), 0)
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))

    def test_checkpointed_gradients_equal_regular(self):
        other = Chimera(replace(self.config, gradient_checkpointing=True))
        other.load_state_dict(self.model.state_dict())
        first = self.model(**self.batch)
        second = other(**self.batch)
        first.loss.backward()
        second.loss.backward()
        torch.testing.assert_close(first.loss, second.loss)
        for (name, p), (_, q) in zip(self.model.named_parameters(), other.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6, msg=name)

    def test_cpu_bf16_backward(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = self.model(**self.batch)
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_optimizer_learns_all_three_losses(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002)
        initial = self.model(**self.batch)
        baseline = {key: initial.losses[key].item() for key in ("text", "classification", "forecast")}
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            output = self.model(**self.batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            optimizer.step()
        final = self.model(**self.batch)
        for name, value in baseline.items():
            self.assertLess(final.losses[name].item(), value, name)

    def test_atomic_checkpoint_roundtrip(self):
        self.model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "model.pt"
            save_checkpoint(path, self.model, step=7)
            restored, payload = load_checkpoint(path)
            restored.eval()
            self.assertEqual(payload["step"], 7)
            self.assertIs(restored.embedding.weight, restored.lm_head.weight)
            with torch.no_grad():
                a, b = self.model(**self.batch), restored(**self.batch)
            torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
            self.assertFalse(path.with_name("model.pt.tmp").exists())

    def test_resume_matches_uninterrupted_with_dropout(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config_path = Path(directory) / "config.json"
            replace(self.config, dropout=0.1).save(config_path)
            data_path = Path(directory) / "data.jsonl"
            records = [dict(text=f"Signal {i}", series=[[i, 1, 2], [i + 1, 2, 3]], class_label=i)
                       for i in range(5)]
            data_path.write_text("\n".join(json.dumps(record) for record in records))
            for mode in ("synthetic", "jsonl"):
                with self.subTest(mode=mode):
                    common = ["--threads", "2", "train", "--config", str(config_path), "--steps", "4",
                              "--warmup", "1", "--batch-size", "2", "--accumulation", "2"]
                    if mode == "jsonl":
                        common.extend(("--data", str(data_path)))
                    full, split = Path(directory) / (mode + "-full"), Path(directory) / (mode + "-split")
                    main(common + ["--output", str(full)])
                    main(common + ["--output", str(split), "--stop-after", "2"])
                    main(common + ["--output", str(split), "--resume", str(split / "latest.pt")])
                    a, _ = load_checkpoint(full / "latest.pt")
                    b, _ = load_checkpoint(split / "latest.pt")
                    for key, value in a.state_dict().items():
                        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0, msg=key)

    def test_jsonl_loader_variable_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            records = [dict(text="Bonjour", series=[[1, 2, 3], [2, 3, 4]], class_label=0),
                       dict(text="Salut !", series=[[4, 5, 6]], class_label=1)]
            path.write_text("\n".join(json.dumps(record) for record in records))
            dataset = JointDataset(path, self.config)
            batch = collate_records([dataset[0], dataset[1]])
            self.assertFalse(batch["series_mask"][1, -1])
            output = self.model(**batch)
            output.loss.backward()
            self.assertTrue(torch.isfinite(output.loss))

    def test_jsonl_null_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(dict(series=[[1, None, 3]], class_label=1)))
            item = JointDataset(path, self.config)[0]
            self.assertTrue(torch.isnan(item["series"][0, 1]))
            self.assertTrue(torch.isfinite(self.model(**collate_records([item])).loss))

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Installer l'extra vision pour tester PNG/JPEG")
    def test_png_loading_resizes_and_normalizes(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            Image.new("RGB", (17, 11), color=(255, 0, 0)).save(image_path)
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(dict(image_path="sample.png", class_label=1)))
            item = JointDataset(path, self.config)[0]
            self.assertEqual(item["images"].shape, (3, 32, 32))
            torch.testing.assert_close(item["images"][0], torch.ones(32, 32))
            self.assertEqual(item["images"][1:].abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(self.model(**collate_records([item])).loss))

    def test_inspection_counts_tied_weights_once(self):
        report = inspect_config(self.config)
        self.assertEqual(report["parameters"], sum(p.numel() for p in self.model.parameters()))
        self.assertGreater(inspect_config(ChimeraConfig.preset("ultra"))["parameters"], 500_000_000_000)

    def test_generation_restores_mode_and_uses_modalities(self):
        self.model.train()
        prompt = self.batch["input_ids"][:1, :4]
        result = self.model.generate(prompt, images=self.batch["images"][:1],
                                     series=self.batch["series"][:1], max_new_tokens=5,
                                     temperature=0, eos_id=None)
        self.assertEqual(result.shape, (1, 9))
        self.assertTrue(self.model.training)
        torch.testing.assert_close(result[:, :4], prompt)

    def test_invalid_inputs_have_explicit_errors(self):
        with self.assertRaises(ValueError):
            self.model()
        with self.assertRaises(ValueError):
            self.model(self.batch["input_ids"], use_cache=True)
        with self.assertRaises(ValueError):
            self.model(self.batch["input_ids"], attention_mask=torch.zeros_like(self.batch["input_ids"]))
        with self.assertRaises(ValueError):
            self.model(images=torch.zeros(2, 3, 15, 15))
        with self.assertRaises(ValueError):
            self.model.generate(self.batch["input_ids"], temperature=-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
