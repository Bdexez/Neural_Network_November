from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from chimera import Chimera, ChimeraConfig
from chimera.cli import main, prediction_stream
from chimera.data import JointDataset, collate_records, synthetic_batch
from chimera.training import evaluate, load_checkpoint, save_checkpoint, step_optimizer


class ImprovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = ChimeraConfig.preset("tiny")

    def setUp(self):
        torch.manual_seed(23)

    def test_bounded_cache_matches_irregular_chunks_with_padding(self):
        model = Chimera(self.config).eval()
        ids = torch.randint(4, 260, (2, 97))
        valid = torch.ones_like(ids, dtype=torch.bool)
        valid[0, :5] = False
        with torch.inference_mode():
            full = model(ids, attention_mask=valid)
            cache, logits, start = None, [], 0
            for size in (9, 40, 1, 47):
                out = model(ids[:, start:start + size], attention_mask=valid[:, start:start + size],
                            cache=cache, use_cache=True)
                cache = out.cache
                logits.append(out.logits)
                start += size
                for block, layer in zip(model.blocks, cache.layers):
                    expected = min(start, block.attn.window - 1) if block.attn.window else start
                    self.assertEqual(layer.key.size(-2), expected)
                    self.assertEqual(layer.start_pos + expected, start)
                    if block.attn.window:
                        self.assertEqual(layer.key.untyped_storage().nbytes(),
                                         layer.key.numel() * layer.key.element_size())
            torch.testing.assert_close(torch.cat(logits, 1), full.logits, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(out.forecast, full.forecast, rtol=2e-5, atol=2e-6)

    def test_window_and_convolution_of_one_allow_empty_states(self):
        model = Chimera(replace(self.config, sliding_window=1, conv_kernel=1, n_kv_heads=1)).eval()
        ids = torch.randint(4, 260, (1, 40))
        with torch.inference_mode():
            full = model(ids).logits
            first = model(ids[:, :35], use_cache=True)
            second = model(ids[:, 35:], cache=first.cache, use_cache=True)
        torch.testing.assert_close(second.logits, full[:, 35:], rtol=2e-5, atol=2e-6)
        self.assertEqual(second.cache.layers[0].key.size(-2), 0)
        self.assertEqual(second.cache.layers[0].conv_state.size(-1), 0)

    def test_context_projected_only_once_per_decoder_layer(self):
        model = Chimera(self.config).eval()
        counts = [0] * self.config.n_layers
        hooks = []
        for index, block in enumerate(model.blocks):
            def count(module, args, output, index=index):
                counts[index] += 1
            hooks.append(block.cross.kv.register_forward_hook(count))
        batch = synthetic_batch(self.config, batch_size=1)
        try:
            model.generate(batch["input_ids"][:, :3], images=batch["images"], series=batch["series"],
                           max_new_tokens=8, temperature=0, eos_id=None)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(counts, [1] * self.config.n_layers)

    def test_mapped_checkpoint_materializes_buffers_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            original = Chimera(self.config).eval()
            save_checkpoint(path, original)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded, _ = load_checkpoint(path)
            self.assertIs(loaded.embedding.weight, loaded.lm_head.weight)
            self.assertFalse(any(t.is_meta for t in loaded.buffers()))
            batch = synthetic_batch(self.config, batch_size=2)
            with torch.no_grad():
                torch.testing.assert_close(original(**batch).logits, loaded.eval()(**batch).logits, rtol=0, atol=0)
            optimizer = torch.optim.AdamW(loaded.parameters(), lr=0.001)
            loaded.train()(**batch).loss.backward()
            optimizer.step()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            reloaded, _ = load_checkpoint(path)
            torch.testing.assert_close(reloaded.embedding.weight, original.embedding.weight, rtol=0, atol=0)

    def test_indexed_utf8_dataset_is_picklable_and_detects_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            rows = [dict(text="été 🧠", class_label=1), dict(text="中文", class_label=2)]
            path.write_text("\n" + "\n\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
            dataset = JointDataset(path, self.config)
            self.assertEqual(len(dataset), 2)
            self.assertFalse(hasattr(dataset, "records"))
            self.assertEqual(dataset.fingerprint, hashlib.sha256(path.read_bytes()).hexdigest())
            restored = pickle.loads(pickle.dumps(dataset))
            torch.testing.assert_close(restored[-1]["input_ids"], dataset[1]["input_ids"])
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changé"):
                dataset[0]

    def test_indexed_dataset_works_with_two_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("\n".join(json.dumps(dict(text=f"Exemple {i}")) for i in range(8)))
            dataset = JointDataset(path, self.config)
            loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_records)
            self.assertEqual(sum(batch["input_ids"].size(0) for batch in loader), 8)

    def test_record_validation_reports_line_and_rejects_fractional_class(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('\n{"text": "x", "class_label": 1.5}\n')
            with self.assertRaisesRegex(ValueError, "data.jsonl:2:.*class_label"):
                JointDataset(path, self.config)[0]

    def test_losses_and_forecast_metrics_ignore_missing_targets_per_batch(self):
        model = Chimera(self.config).eval()
        batch = synthetic_batch(self.config, batch_size=5, seed=900)
        batch["class_labels"][1:4] = -100
        batch["forecast_targets"][1:4] = float("nan")
        batch["forecast_targets"][0, 3:] = float("nan")
        full = evaluate(model, [batch])
        chunks = [{key: value[section] for key, value in batch.items()} for section in (slice(0, 2), slice(2, 5))]
        split = evaluate(model, chunks)
        for name in ("text", "classification", "forecast", "forecast_mae", "forecast_rmse",
                     "forecast_interval_coverage", "forecast_interval_width", "accuracy"):
            self.assertAlmostEqual(full[name], split[name], places=5, msg=name)
        self.assertEqual(full["classified_examples"], 2)
        self.assertEqual(full["forecast_observations"], 45)
        self.assertEqual(len(full["forecast_mae_by_horizon"]), self.config.forecast_horizon)

    def test_no_observed_targets_do_not_report_perfect_quality(self):
        batch = synthetic_batch(self.config, batch_size=2)
        batch["labels"].fill_(-100)
        batch["class_labels"].fill_(-100)
        batch["forecast_targets"].fill_(float("nan"))
        report = evaluate(Chimera(self.config), [batch])
        self.assertIsNone(report["forecast_mae"])
        self.assertIsNone(report["accuracy"])
        self.assertIsNone(report["byte_perplexity"])

    def test_grad_scaler_skips_overflow_then_recovers(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=16)
        before = model.weight.detach().clone()
        scaler.scale(model(torch.ones(1, 1)).sum() * float("inf")).backward()
        norm, skipped = step_optimizer(model, optimizer, scaler, 1.0)
        self.assertTrue(skipped)
        self.assertIsNone(norm)
        self.assertEqual(scaler.get_scale(), 8)
        torch.testing.assert_close(before, model.weight)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(model(torch.ones(1, 1)).sum()).backward()
        norm, skipped = step_optimizer(model, optimizer, scaler, 1.0)
        self.assertFalse(skipped)
        self.assertNotEqual(before.item(), model.weight.item())

    def test_atomic_prediction_stream_keeps_previous_output_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text("previous\n")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with prediction_stream(path) as stream:
                    stream.write("incomplete\n")
                    raise RuntimeError("interrupted")
            self.assertEqual(path.read_text(), "previous\n")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_best_checkpoint_and_exact_resume_with_early_stopping(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root / "data.jsonl"
            data.write_text("\n".join(json.dumps(dict(text=f"Classe {i}", class_label=i)) for i in range(5)))
            config_path = root / "config.json"
            replace(self.config, dropout=0.1).save(config_path)
            common = ["--threads", "2", "train", "--config", str(config_path), "--steps", "8",
                      "--data", str(data), "--val-data", str(data), "--batch-size", "2", "--accumulation", "2",
                      "--eval-every", "2", "--patience", "1", "--min-delta", "100", "--seed", "6"]
            full, split = root / "full", root / "split"
            main(common + ["--output", str(full)])
            main(common + ["--output", str(split), "--stop-after", "3"])
            main(common + ["--output", str(split), "--resume", str(split / "latest.pt")])
            a, pa = load_checkpoint(full / "latest.pt")
            b, pb = load_checkpoint(split / "latest.pt")
            self.assertEqual(pa["step"], 4)
            self.assertEqual(pb["step"], 4)
            self.assertEqual(pa["trainer"]["validation_state"], pb["trainer"]["validation_state"])
            for name, parameter in a.state_dict().items():
                torch.testing.assert_close(parameter, b.state_dict()[name], rtol=0, atol=0, msg=name)
            best, payload = load_checkpoint(full / "best.pt")
            validations = [json.loads(line) for line in (full / "validation.jsonl").read_text().splitlines()]
            self.assertEqual(payload["step"], min(validations, key=lambda row: row["loss"])["step"])
            self.assertTrue(json.loads((full / "evaluation.json").read_text())["stopped_early"])
            self.assertTrue(pa["trainer"]["shuffle"])
            main(common + ["--output", str(split), "--resume", str(split / "latest.pt")])
            self.assertEqual(load_checkpoint(split / "latest.pt")[1]["step"], 4)

    def test_resume_rejects_modified_dataset_content(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root / "data.jsonl"
            data.write_text('{"text": "original"}\n')
            common = ["train", "--steps", "2", "--data", str(data), "--output", str(root / "run")]
            main(common + ["--stop-after", "1"])
            data.write_text('{"text": "modified"}\n')
            with self.assertRaisesRegex(ValueError, "data_fingerprint"):
                main(common + ["--resume", str(root / "run/latest.pt")])

    def test_cli_prediction_evaluation_and_export_preserve_provenance(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            path = root / "source.pt"
            save_checkpoint(path, Chimera(self.config), step=7, trainer={"seed": 123})
            data = root / "data.jsonl"
            data.write_text('{"text": "Signal", "class_label": 2}\n')
            output = root / "predictions.jsonl"
            main(["predict", "--checkpoint", str(path), "--data", str(data), "--output", str(output),
                  "--max-new-tokens", "3"])
            self.assertIn("generated_text", json.loads(output.read_text()))
            report = root / "evaluation.json"
            main(["evaluate", "--checkpoint", str(path), "--data", str(data), "--output", str(report)])
            self.assertEqual(json.loads(report.read_text())["classified_examples"], 1)
            exported = root / "export.pt"
            main(["export", "--checkpoint", str(path), "--output", str(exported)])
            self.assertEqual(load_checkpoint(exported)[1]["trainer"]["seed"], 123)


if __name__ == "__main__":
    unittest.main(verbosity=2)
