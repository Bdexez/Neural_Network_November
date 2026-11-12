from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from chimera import ChimeraConfig
from chimera.cli import main
from chimera.training import load_checkpoint


class TrainingPolicyTests(unittest.TestCase):
    def test_balanced_training_resumes_exactly_and_records_all_subsets(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = root / "config.json"
            replace(ChimeraConfig.preset("tiny"), dropout=0.1).save(config)
            common = ["--threads", "2", "train", "--config", str(config), "--steps", "7",
                      "--batch-size", "2", "--accumulation", "2", "--eval-every", "7"]
            full, split = root / "full", root / "split"
            main(common + ["--output", str(full), "--modality-policy", "balanced"])
            main(common + ["--output", str(split), "--modality-policy", "balanced", "--stop-after", "3"])
            # Sans répéter l'option : la politique est restaurée du checkpoint.
            main(common + ["--output", str(split), "--resume", str(split / "latest.pt")])
            a, pa = load_checkpoint(full / "latest.pt")
            b, pb = load_checkpoint(split / "latest.pt")
            self.assertEqual(pb["trainer"]["modality_policy"], "balanced")
            for name, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0, msg=name)
            rows = [json.loads(line) for line in (full / "metrics.jsonl").read_text().splitlines()]
            counts = {}
            for row in rows:
                for name, count in row["modality_batches"].items():
                    counts[name] = counts.get(name, 0) + count
            self.assertEqual(len(counts), 7)
            self.assertEqual(set(counts.values()), {2})
            with self.assertRaisesRegex(ValueError, "modality_policy"):
                main(common + ["--output", str(split), "--resume", str(split / "latest.pt"),
                               "--modality-policy", "dropout"])


if __name__ == "__main__":
    unittest.main()
