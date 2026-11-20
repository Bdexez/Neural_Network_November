import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from chimera import Chimera, ChimeraConfig
from chimera.training import save_checkpoint


class SeriesGateBenchmarkTests(unittest.TestCase):
    def test_seed_ranges_checkpoint_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, report = root / 'model.pt', root / 'report.json'
            save_checkpoint(checkpoint, Chimera(ChimeraConfig.preset('tiny')))
            original = checkpoint.read_bytes()
            common = [sys.executable, 'tools/benchmark_series_gate.py', '--checkpoint', str(checkpoint),
                      '--head-batches','12','--gate-batches','3','--validation-batches','2',
                      '--test-batches','1','--batch-size','16','--shape-context','4','--epochs','2']
            run = subprocess.run([*common,'--output',str(report)], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(report.read_text())
            self.assertEqual(result['gate']['train_examples'],48)
            self.assertEqual(result['gate']['validation_examples'],32)
            self.assertEqual(result['results']['8']['identity']['examples'],16)
            for args in (('--validation-seed','80000001','--output',str(report)),
                         ('--output',str(checkpoint))):
                run = subprocess.run([*common,*args], capture_output=True, text=True, timeout=60)
                self.assertNotEqual(run.returncode, 0)
            self.assertEqual(checkpoint.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
