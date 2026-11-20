import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.training import save_checkpoint


class StableRMSBenchmarkTests(unittest.TestCase):
    def test_cpu_benchmark_and_checkpoint_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, report = root / 'model.pt', root / 'report.json'
            torch.manual_seed(488)
            save_checkpoint(checkpoint, Chimera(ChimeraConfig.preset('tiny')))
            original = checkpoint.read_bytes()
            common = [sys.executable,'tools/benchmark_stable_rms.py','--checkpoint',str(checkpoint),
                      '--batch-size','2','--repeats','1','--min-run-time','.01']
            result = subprocess.run([*common,'--output',str(report)],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            data = json.loads(report.read_text())
            self.assertEqual(data['class_predictions_changed'],0)
            self.assertEqual(data['extreme_example']['historical'],[[0.,0.]])
            self.assertGreater(data['extreme_example']['stable'][0][0],.6)
            self.assertGreater(data['timing_seconds']['stable_over_historical'],0)
            result = subprocess.run([*common,'--output',str(checkpoint)],capture_output=True,text=True,timeout=60)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(checkpoint.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
