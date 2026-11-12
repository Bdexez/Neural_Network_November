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


if __name__ == "__main__":
    unittest.main()
