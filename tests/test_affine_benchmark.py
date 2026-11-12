from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from chimera import Chimera, ChimeraConfig
from chimera.training import save_checkpoint
from tools.benchmark_affine_forecast import main


class AffineBenchmarkTests(unittest.TestCase):
    def test_paired_inputs_unit_comparison_compatibility_and_output_protection(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            checkpoint, output = root / "model.pt", root / "report.json"
            save_checkpoint(checkpoint, Chimera(ChimeraConfig.preset("tiny")))
            args = ["--reference-checkpoint", str(checkpoint), "--checkpoint", str(checkpoint),
                    "--output", str(output), "--cases", "history8_missing30", "--transforms", "original", "offset20", "scale10",
                    "--modalities", "series", "--batches", "2", "--batch-size", "8", "--threads", "2"]
            main(args)
            report = json.loads(output.read_text())
            self.assertEqual(report["examples_per_case"], 16)
            self.assertEqual(report["checkpoints"]["source"]["cases"], report["checkpoints"]["candidate"]["cases"])
            baseline = report["persistence_original_unit_mae"]["history8_missing30"]
            self.assertAlmostEqual(baseline["original"], baseline["offset20"], places=6)
            self.assertAlmostEqual(baseline["original"], baseline["scale10"], places=6)
            before = checkpoint.read_bytes()
            with self.assertRaises(SystemExit):
                main([*args, "--output", str(checkpoint)])
            self.assertEqual(checkpoint.read_bytes(), before)
            other = root / "other.pt"
            save_checkpoint(other, Chimera(replace(ChimeraConfig.preset("tiny"), forecast_horizon=2)))
            content = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "mêmes entrées"):
                main([*args, "--checkpoint", str(other)])
            self.assertEqual(output.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
