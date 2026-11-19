from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.cli import main
from chimera.training import load_checkpoint, save_checkpoint


class TrainingPolicyTests(unittest.TestCase):
    def test_finetuning_uses_source_weights_but_fresh_optimizer_and_schedule(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = replace(ChimeraConfig.preset("tiny"), num_classes=4)
            torch.manual_seed(71)
            model = Chimera(config)
            source = root / "source.pt"
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.9)
            # État volontairement incompatible avec le nouvel entraînement.
            save_checkpoint(source, model, optimizer=optimizer, step=99,
                            trainer={"steps": 100, "seed": 71, "modality_policy": "balanced",
                                     "validation_state": {"bad_evaluations": 50}})
            initial = model.embedding.weight.detach().clone()
            for name, with_optimizer in (("source", True), ("inference", False)):
                path = source if with_optimizer else root / "inference.pt"
                if not with_optimizer:
                    save_checkpoint(path, model, step=99, trainer={"seed": 71})
                destination = root / name
                main(["--threads", "2", "train", "--init-from", str(path), "--steps", "2",
                      "--batch-size", "2", "--seed", "8", "--lr", "0.001", "--warmup", "0",
                      "--modality-dropout", "0", "--output", str(destination)])
                trained, payload = load_checkpoint(destination / "latest.pt")
                self.assertEqual(trained.config.num_classes, 4)
                self.assertEqual(payload["step"], 2)
                self.assertEqual(payload["trainer"]["seed"], 8)
                self.assertEqual(payload["trainer"]["modality_policy"], "dropout")
                self.assertEqual(payload["trainer"]["initialization"]["step"], 99)
                self.assertTrue(all(int(state["step"]) <= 2 for state in payload["optimizer"]["state"].values()))
                delta = (trained.embedding.weight - initial).abs()
                self.assertGreater(delta.max().item(), 0)
                self.assertLess(delta.max().item(), 0.003)
            a, _ = load_checkpoint(root / "source/latest.pt")
            b, _ = load_checkpoint(root / "inference/latest.pt")
            for name, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0, msg=name)
            original, _ = load_checkpoint(source)
            torch.testing.assert_close(original.embedding.weight, initial, rtol=0, atol=0)

    def test_balanced_augmented_training_resumes_exactly_and_records_all_subsets(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = root / "config.json"
            replace(ChimeraConfig.preset("tiny"), dropout=0.1).save(config)
            common = ["--threads", "2", "train", "--config", str(config), "--steps", "7",
                      "--batch-size", "2", "--accumulation", "2", "--eval-every", "7"]
            full, split = root / "full", root / "split"
            augmentation = ["--series-crop-probability", "0.7", "--series-missing-rate", "0.3"]
            main(common + ["--output", str(full), "--modality-policy", "balanced", *augmentation])
            main(common + ["--output", str(split), "--modality-policy", "balanced", *augmentation, "--stop-after", "3"])
            # Sans répéter l'option : la politique est restaurée du checkpoint.
            main(common + ["--output", str(split), "--resume", str(split / "latest.pt")])
            a, pa = load_checkpoint(full / "latest.pt")
            b, pb = load_checkpoint(split / "latest.pt")
            self.assertEqual(pb["trainer"]["modality_policy"], "balanced")
            self.assertEqual(pb["trainer"]["series_augmentation"]["missing_rate"], 0.3)
            for name, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0, msg=name)
            rows = [json.loads(line) for line in (full / "metrics.jsonl").read_text().splitlines()]
            counts = {}
            for row in rows:
                for name, count in row["modality_batches"].items():
                    counts[name] = counts.get(name, 0) + count
            self.assertEqual(len(counts), 7)
            self.assertEqual(set(counts.values()), {2})
            series = [row["series_statistics"] for row in rows if row["series_statistics"]]
            self.assertLess(sum(row["observed_values"] for row in series), sum(row["possible_values"] for row in series))
            with self.assertRaisesRegex(ValueError, "modality_policy"):
                main(common + ["--output", str(split), "--resume", str(split / "latest.pt"),
                               "--modality-policy", "dropout"])
            with self.assertRaisesRegex(ValueError, "series_missing_rate"):
                main(common + ["--output", str(split), "--resume", str(split / "latest.pt"),
                               "--series-missing-rate", "0"])

    def test_task_metric_selection_resumes_and_maximizes_accuracy(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root / "validation.jsonl"
            records = [dict(series=[[i, i + 1, i + 2], [i + 1, i + 2, i + 3]],
                            class_label=i % 2, future=[[i + step, i + step + 1, i + step + 2]
                                                        for step in range(12)]) for i in range(3)]
            data.write_text("\n".join(json.dumps(record) for record in records))
            common = ["--threads", "2", "train", "--preset", "tiny", "--steps", "4",
                      "--batch-size", "2", "--eval-every", "1", "--val-data", str(data)]
            full, split = root / "full", root / "split"
            main(common + ["--output", str(full), "--monitor", "forecast_mae"])
            main(common + ["--output", str(split), "--monitor", "forecast_mae", "--stop-after", "2"])
            main(common + ["--output", str(split), "--resume", str(split / "latest.pt")])
            a, first = load_checkpoint(full / "latest.pt")
            b, resumed = load_checkpoint(split / "latest.pt")
            self.assertEqual(resumed["trainer"]["monitor"], "forecast_mae")
            self.assertEqual(first["trainer"]["validation_state"], resumed["trainer"]["validation_state"])
            for name, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0, msg=name)
            rows = [json.loads(line) for line in (full / "validation.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["monitor"] == "forecast_mae" and row["monitor_value"] == row["forecast_mae"]
                                for row in rows))
            best = min(rows, key=lambda row: row["forecast_mae"])
            best_checkpoint = load_checkpoint(full / "best.pt")[1]
            self.assertEqual(best_checkpoint["step"], best["step"])
            self.assertEqual(best_checkpoint["trainer"]["validation_state"]["best_loss"], best["forecast_mae"])
            evaluation = json.loads((full / "evaluation.json").read_text())
            self.assertEqual(evaluation["best_monitor_value"], best["forecast_mae"])
            with self.assertRaisesRegex(ValueError, "monitor"):
                main(common + ["--output", str(split), "--resume", str(split / "latest.pt"), "--monitor", "accuracy"])
            accuracy = root / "accuracy"
            main(common + ["--output", str(accuracy), "--monitor", "accuracy"])
            rows = [json.loads(line) for line in (accuracy / "validation.jsonl").read_text().splitlines()]
            best = max(rows, key=lambda row: row["accuracy"])
            self.assertEqual(load_checkpoint(accuracy / "best.pt")[1]["step"], best["step"])

    def test_validation_requires_observed_targets_and_the_monitored_measure(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root / "validation.jsonl"
            common = ["--threads", "2", "train", "--preset", "tiny", "--steps", "1", "--batch-size", "1",
                      "--val-data", str(data), "--output", str(root / "run")]
            data.write_text(json.dumps(dict(series=[[1, 2, 3]], class_label=-100)))
            with self.assertRaisesRegex(ValueError, "aucune cible supervisée observée"):
                main(common)
            data.write_text(json.dumps(dict(series=[[1, 2, 3]], class_label=0)))
            with self.assertRaisesRegex(ValueError, "forecast_mae"):
                main(common + ["--monitor", "forecast_mae"])

    def test_legacy_resume_without_monitor_uses_loss(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            common = ["--threads", "2", "train", "--steps", "2", "--batch-size", "2",
                      "--eval-every", "1", "--output", str(root)]
            main(common + ["--stop-after", "1"])
            path = root / "latest.pt"
            payload = torch.load(path, weights_only=True)
            del payload["trainer"]["monitor"]
            torch.save(payload, path)
            main(common + ["--resume", str(path)])
            self.assertEqual(load_checkpoint(path)[1]["trainer"]["monitor"], "loss")


if __name__ == "__main__":
    unittest.main()
