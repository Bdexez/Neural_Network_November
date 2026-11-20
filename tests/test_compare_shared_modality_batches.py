import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import distributed as dist

from chimera import Chimera, ChimeraConfig
from chimera.training import save_checkpoint


class CompareSharedModalityBatchesTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_study_runs_both_modes_and_reports_real_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, output = root / 'parent.pt', root / 'study'
            torch.manual_seed(31)
            save_checkpoint(checkpoint, Chimera(ChimeraConfig.preset('tiny')))
            script = Path(__file__).resolve().parents[1] / 'tools/compare_shared_modality_batches.py'
            command = [sys.executable, str(script), '--checkpoint', str(checkpoint), '--output-dir', str(output),
                       '--seeds', '17', '--steps', '2', '--train-examples', '8', '--batch-size', '2',
                       '--train-seed', '100', '--test-seed', '2000000', '--test-batches', '1',
                       '--test-batch-size', '4']
            run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout[-8000:])
            report = json.loads((output / 'comparison.json').read_text())
            rows = {row['mode']: row for row in report['runs']}
            self.assertEqual(set(rows), {'shared', 'sharded'})
            self.assertEqual(rows['shared']['exposure']['sample_views'], 8)
            self.assertEqual(rows['shared']['exposure']['unique_examples_per_step_sum'], 4)
            self.assertEqual(rows['sharded']['exposure']['unique_examples_per_step_sum'], 8)
            for row in rows.values():
                self.assertEqual(row['step'], 2)
                self.assertEqual(row['metrics']['examples'], 4)
            before = (output / 'comparison.json').read_bytes()
            rejected = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual((output / 'comparison.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
