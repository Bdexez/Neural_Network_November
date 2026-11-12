from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chimera import Chimera, ChimeraConfig
from chimera.cli import main
from chimera.data import synthetic_batch
from chimera.training import load_checkpoint


class AffineTrainingTests(unittest.TestCase):
    def test_balanced_dropout_training_resumes_exactly_with_affine_settings(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = root / "config.json"
            replace(ChimeraConfig.preset("tiny"), temporal_statistics=True, dropout=0.1).save(config)
            common = ["--threads", "2", "train", "--config", str(config), "--steps", "7", "--batch-size", "2",
                      "--accumulation", "2", "--eval-every", "7", "--modality-policy", "balanced"]
            options = ["--series-crop-probability", "0.7", "--series-missing-rate", "0.2",
                       "--series-affine-probability", "0.75", "--series-affine-min-scale", "0.2",
                       "--series-affine-max-scale", "5", "--series-affine-max-shift", "15"]
            full, split = root / "full", root / "split"
            main([*common, *options, "--output", str(full)])
            main([*common, *options, "--output", str(split), "--stop-after", "3"])
            main([*common, "--resume", str(split / "latest.pt"), "--output", str(split)])
            a, pa = load_checkpoint(full / "latest.pt")
            b, pb = load_checkpoint(split / "latest.pt")
            self.assertEqual(pa["trainer"]["series_affine_augmentation"],
                             dict(probability=0.75, min_scale=0.2, max_scale=5.0, max_shift=15.0))
            self.assertEqual(pa["trainer"], pb["trainer"])
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0, msg=name)
            rows = [json.loads(line) for line in (full / "metrics.jsonl").read_text().splitlines()]
            self.assertGreater(sum(row["series_statistics"].get("affine_changed_examples", 0) for row in rows), 0)
            with self.assertRaisesRegex(ValueError, "series_affine_probability"):
                main([*common, "--resume", str(split / "latest.pt"), "--output", str(split),
                      "--series-affine-probability", "0"])

    def test_future_units_follow_retained_series_and_validation_stays_unaugmented(self):
        captured = []
        original_forward = Chimera.forward

        def capture(model, *args, **kwargs):
            captured.append((model.training, "series" in kwargs, kwargs["forecast_targets"].clone()))
            return original_forward(model, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch.object(Chimera, "forward", capture):
            main(["--threads", "2", "train", "--steps", "7", "--batch-size", "2", "--seed", "6081",
                  "--modality-policy", "balanced", "--series-affine-probability", "1",
                  "--series-affine-min-scale", "2", "--series-affine-max-scale", "2", "--series-affine-max-shift", "0",
                  "--eval-every", "7", "--output", directory])
        training = [row for row in captured if row[0]]
        validation = [row for row in captured if not row[0]]
        self.assertEqual(len(training), 7)
        self.assertTrue(any(row[1] for row in training) and any(not row[1] for row in training))
        config = ChimeraConfig.preset("tiny")
        for index, (_, has_series, future) in enumerate(training):
            expected = synthetic_batch(config, batch_size=2, seed=6081 + index)["forecast_targets"]
            torch.testing.assert_close(future, expected * (2 if has_series else 1), rtol=0, atol=0)
        for _, _, future in validation:
            expected = synthetic_batch(config, batch_size=2, seed=1006081)["forecast_targets"]
            torch.testing.assert_close(future, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
