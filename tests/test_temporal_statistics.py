from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.data import evaluation_data_signature, synthetic_batch
from chimera.cli import main
from chimera.training import load_checkpoint, save_checkpoint


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

    def test_checkpoint_transfer_is_explicit_zero_initialized_and_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            original = Chimera(ChimeraConfig.preset("tiny")).eval()
            save_checkpoint(path, original)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            extended, payload = load_checkpoint(path, overrides={"temporal_statistics": True})
            self.assertFalse(payload["config"]["temporal_statistics"])
            self.assertTrue(extended.config.temporal_statistics)
            self.assertIs(extended.embedding.weight, extended.lm_head.weight)
            self.assertFalse(any(p.is_meta for p in extended.parameters()))
            self.assertEqual(extended.temporal.statistics_proj.weight.count_nonzero().item(), 0)
            batch = synthetic_batch(original.config, batch_size=2, seed=70)
            with torch.no_grad():
                torch.testing.assert_close(original(**batch).forecast, extended.eval()(**batch).forecast, rtol=0, atol=0)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            save_checkpoint(Path(directory) / "extended.pt", extended)
            disabled, _ = load_checkpoint(Path(directory) / "extended.pt", overrides={"temporal_statistics": False})
            self.assertIsNone(disabled.temporal.statistics_proj)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path, overrides={"d_model": 128})
            with self.assertRaises(ValueError):
                load_checkpoint(path, overrides={"unknown": True})
            # Un checkpoint déclaré compatible mais réellement incomplet ne reçoit
            # pas d'initialisation de substitution silencieuse.
            broken = dict(payload)
            broken["model"] = dict(payload["model"])
            del broken["model"]["classifier.weight"]
            torch.save(broken, Path(directory) / "broken.pt")
            with self.assertRaises(RuntimeError):
                load_checkpoint(Path(directory) / "broken.pt", overrides={"temporal_statistics": True})

    def test_override_finetuning_and_exact_resume_restore_the_extended_model(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            checkpoint, overrides = root / "source.pt", root / "overrides.json"
            save_checkpoint(checkpoint, Chimera(replace(ChimeraConfig.preset("tiny"), dropout=0.1)))
            overrides.write_text(json.dumps(dict(temporal_statistics=True)))
            common = ["--threads", "2", "train", "--steps", "4", "--batch-size", "2", "--accumulation", "2",
                      "--modality-policy", "balanced", "--series-crop-probability", "0.7", "--eval-every", "4"]
            full, split = root / "full", root / "split"
            transfer = ["--init-from", str(checkpoint), "--model-overrides", str(overrides)]
            main(common + transfer + ["--output", str(full)])
            main(common + transfer + ["--output", str(split), "--stop-after", "2"])
            main(common + ["--resume", str(split / "latest.pt"), "--output", str(split)])
            a, pa = load_checkpoint(full / "latest.pt")
            b, pb = load_checkpoint(split / "latest.pt")
            self.assertTrue(a.config.temporal_statistics)
            self.assertGreater(a.temporal.statistics_proj.weight.abs().sum().item(), 0)
            self.assertEqual(pb["trainer"]["initialization"]["overrides"], dict(temporal_statistics=True))
            for name, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0, msg=name)
            with self.assertRaises(SystemExit):
                main(common + ["--resume", str(split / "latest.pt"), "--model-overrides", str(overrides)])

    def test_statistics_survive_cached_context_and_checkpointed_bf16_backward(self):
        config = replace(ChimeraConfig.preset("tiny"), temporal_statistics=True, gradient_checkpointing=True)
        model = Chimera(config).eval()
        torch.nn.init.normal_(model.temporal.statistics_proj.weight, std=0.01)
        batch = synthetic_batch(config, batch_size=2, seed=71)
        ids = batch["input_ids"][:, :7]
        context = {key: batch[key] for key in ("images", "series")}
        with torch.no_grad():
            full = model(ids, **context)
            first = model(ids[:, :3], **context, use_cache=True)
            second = model(ids[:, 3:], cache=first.cache, use_cache=True)
        torch.testing.assert_close(full.logits[:, 3:], second.logits, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(full.forecast, second.forecast, rtol=2e-5, atol=2e-6)
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        self.assertTrue(torch.isfinite(model.temporal.statistics_proj.weight.grad).all())

    def test_comparison_requires_identical_input_and_target_semantics(self):
        original = ChimeraConfig.preset("tiny")
        extended = replace(original, temporal_statistics=True, forecast_weight=3, n_layers=4)
        self.assertEqual(evaluation_data_signature(original), evaluation_data_signature(extended))
        for change in (dict(num_classes=8), dict(forecast_horizon=8), dict(image_size=64),
                       dict(quantiles=(0.05, 0.5, 0.95)), dict(max_series_len=64)):
            self.assertNotEqual(evaluation_data_signature(original),
                                evaluation_data_signature(replace(original, **change)))

    def test_early_checkpoint_benchmark_serializes_absent_best_score_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, output = root / "early.pt", root / "report.json"
            save_checkpoint(checkpoint, Chimera(ChimeraConfig.preset("tiny")), step=1,
                            trainer={"validation_state": {"best_loss": float("inf"), "best_step": None}})
            script = Path(__file__).resolve().parents[1] / "tools/benchmark_robustness.py"
            result = subprocess.run([sys.executable, str(script), "--reference-checkpoint", str(checkpoint),
                                     "--checkpoint", str(checkpoint), "--batches", "1", "--batch-size", "2",
                                     "--threads", "2", "--cases", "history8", "--output", str(output)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertIsNone(report["checkpoints"]["source"]["trainer"]["validation_state"]["best_loss"])
            _, payload = load_checkpoint(checkpoint)
            self.assertEqual(payload["trainer"]["validation_state"]["best_loss"], float("inf"))


if __name__ == "__main__":
    unittest.main()
