from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chimera import ChimeraConfig
from chimera.cli import main
from chimera.data import JointDataset, collate_records
from chimera.split_audit import audit_splits
from chimera.temporal_splits import prepare_temporal_splits


class TemporalSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = replace(ChimeraConfig.preset("tiny"), series_features=2, forecast_horizon=2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, rows):
        path = self.root / "source.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return path

    def test_chronological_windows_are_disjoint_and_usable(self):
        first = [[i, i * 2] for i in range(45)]
        first[3] = first[4] = [None, None]
        second = [[100 + i, 200 + i] for i in range(45)]
        source = self.source([dict(series_id="a", values=first), dict(series_id="b", values=second)])
        output = self.root / "out"
        manifest = prepare_temporal_splits(source, output, self.config, history=3, train_end=10, validation_end=21,
                                           calibration_end=32, stride=2, gap=1)
        self.assertEqual(manifest["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(manifest["series_count"], 2)
        self.assertEqual(manifest["splits"]["train"]["skipped_unobserved_target"], 1)
        self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
        paths = {name: output / f"{name}.jsonl" for name in ("train", "validation", "calibration", "test")}
        for name, path in paths.items():
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(manifest["splits"][name]["examples"], len(rows))
            self.assertEqual(manifest["splits"][name]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            for row in rows:
                values = first if row["series_id"] == "a" else second
                self.assertEqual(row["series"], values[row["window_start"]:row["forecast_start"]])
                self.assertEqual(row["future"], values[row["forecast_start"]:row["window_end"]])
                self.assertEqual(row["forecast_start"] - row["window_start"], 3)
                if name == "train":
                    self.assertLessEqual(row["window_end"], 10)
                elif name == "validation":
                    self.assertGreaterEqual(row["window_start"], 11)
                    self.assertLessEqual(row["window_end"], 21)
                elif name == "calibration":
                    self.assertGreaterEqual(row["window_start"], 22)
                    self.assertLessEqual(row["window_end"], 32)
                else:
                    self.assertGreaterEqual(row["window_start"], 33)
            dataset = JointDataset(path, self.config)
            batch = collate_records([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["series"].shape), (2, 3, 2))
            self.assertEqual(tuple(batch["forecast_targets"].shape), (2, 2, 2))
        report = audit_splits(paths, fields=["series_id", "window_start", "window_end"],
                              interval_fields=["series_id", "window_start", "window_end"])
        self.assertTrue(all(pair["shared_unique_records"] == 0 and pair["intervals"]["groups_with_overlap"] == 0
                            for pair in report["pairs"]))

    def test_invalid_data_or_boundaries_do_not_publish_outputs(self):
        good = dict(series_id="a", values=[[i, i + 1] for i in range(45)])
        scenarios = [dict(series_id=True, values=good["values"]), dict(series_id="a", values=[[1]] * 45),
                     dict(series_id="a", values=[[float("nan"), 1]] * 45),
                     dict(series_id="a", values=[[1e40, 1]] * 45),
                     dict(series_id="a", values=[[10**400, 1]] * 45),
                     dict(series_id="a", values=[[True, 1]] * 45),
                     dict(series_id="a", values=[[1, 2]] * 30)]
        for index, row in enumerate(scenarios):
            with self.subTest(index=index):
                source = self.source([row])
                output = self.root / f"bad-{index}"
                with self.assertRaises(ValueError):
                    prepare_temporal_splits(source, output, self.config, history=3, train_end=10, validation_end=21,
                                            calibration_end=32, stride=2, gap=1)
                self.assertFalse(output.exists())
        source = self.source([good, good])
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_temporal_splits(source, self.root / "duplicates", self.config, history=3, train_end=10, validation_end=21,
                                    calibration_end=32, gap=1)
        self.assertFalse((self.root / "duplicates").exists())
        for options in (dict(history=0, train_end=10, validation_end=21, calibration_end=32),
                        dict(history=3, train_end=10, validation_end=21, calibration_end=21),
                        dict(history=3, train_end=4, validation_end=21, calibration_end=32),
                        dict(history=3, train_end=10, validation_end=21, calibration_end=32, gap=-1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                prepare_temporal_splits(source, self.root / "invalid", self.config, **options)
        self.assertFalse((self.root / "invalid").exists())

    def test_existing_directory_and_publish_failure_preserve_user_files(self):
        source = self.source([dict(series_id="a", values=[[i, i] for i in range(45)])])
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "train.jsonl").write_text("keep")
        with self.assertRaisesRegex(ValueError, "existe déjà"):
            prepare_temporal_splits(source, existing, self.config, history=3, train_end=10, validation_end=21, calibration_end=32)
        self.assertEqual((existing / "train.jsonl").read_text(), "keep")
        output = self.root / "failed"
        real_link = os.link
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(args)
            if len(calls) == 2:
                raise OSError("simulated failure")
            return real_link(*args, **kwargs)
        with patch("chimera.temporal_splits.os.link", fail_second), self.assertRaisesRegex(OSError, "simulated"):
            prepare_temporal_splits(source, output, self.config, history=3, train_end=10, validation_end=21, calibration_end=32)
        self.assertFalse(output.exists())
        self.assertTrue(source.exists())
        concurrent = self.root / "concurrent"

        def create_conflicting_file(src, dst):
            Path(dst).write_text("created concurrently")
            return real_link(src, dst)

        with patch("chimera.temporal_splits.os.link", create_conflicting_file), self.assertRaises(FileExistsError):
            prepare_temporal_splits(source, concurrent, self.config, history=3, train_end=10,
                                    validation_end=21, calibration_end=32)
        self.assertEqual((concurrent / "train.jsonl").read_text(), "created concurrently")
        self.assertFalse((concurrent / "manifest.json").exists())

    def test_empty_target_windows_skipped_and_group_without_targets_rejected(self):
        values = [[i, i] for i in range(45)]
        values[21:32] = [[None, None] for _ in range(11)]
        source = self.source([dict(series_id="a", values=values)])
        output = self.root / "unobserved"
        with self.assertRaisesRegex(ValueError, "aucune fenêtre supervisée pour calibration"):
            prepare_temporal_splits(source, output, self.config, history=3, train_end=10, validation_end=21, calibration_end=32)
        self.assertFalse(output.exists())

    def test_source_replacement_during_generation_is_rejected(self):
        rows = [dict(series_id="a", values=[[i, i] for i in range(45)]),
                dict(series_id="b", values=[[i + 1, i + 2] for i in range(45)])]
        source = self.source(rows)
        replacement = self.root / "replacement.jsonl"
        replacement.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        original_open = Path.open

        class ReplacingReader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def __iter__(self):
                for line in self.stream:
                    if replacement.exists():
                        os.replace(replacement, source)
                    yield line

        def open_with_replacement(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return ReplacingReader(stream) if path == source else stream

        output = self.root / "changing"
        with patch.object(Path, "open", open_with_replacement):
            with self.assertRaisesRegex(ValueError, "changé pendant sa lecture"):
                prepare_temporal_splits(source, output, self.config, history=3, train_end=10, validation_end=21, calibration_end=32)
        self.assertFalse(output.exists())

    def test_cli_pipeline_can_train_calibrate_and_evaluate(self):
        source = self.source([dict(series_id="a", values=[[i / 50, (50 - i) / 50] for i in range(80)])])
        config_path = self.root / "config.json"
        self.config.save(config_path)
        output = self.root / "splits"
        checkpoint_dir = self.root / "model"
        calibration = self.root / "calibration.json"
        report = self.root / "report.json"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            main(["--threads", "2", "prepare-series", "--input", str(source), "--output-dir", str(output),
                  "--config", str(config_path), "--history", "4", "--train-end", "20", "--validation-end", "40", "--calibration-end", "60"])
            main(["--threads", "2", "train", "--config", str(config_path), "--data", str(output / "train.jsonl"),
                  "--steps", "1", "--batch-size", "4", "--val-data", str(output / "validation.jsonl"),
                  "--output", str(checkpoint_dir)])
            checkpoint = checkpoint_dir / "latest.pt"
            main(["--threads", "2", "calibrate", "--checkpoint", str(checkpoint),
                  "--data", str(output / "calibration.jsonl"), "--output", str(calibration)])
            main(["--threads", "2", "evaluate", "--checkpoint", str(checkpoint),
                  "--data", str(output / "test.jsonl"), "--calibration", str(calibration), "--output", str(report)])
        metrics = json.loads(report.read_text())
        self.assertGreater(metrics["forecast_observations"], 0)
        self.assertIsNotNone(metrics["calibrated_interval_coverage"])
        self.assertFalse(metrics["calibration_jsonl_identical"])


if __name__ == "__main__":
    unittest.main()
