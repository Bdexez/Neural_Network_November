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


class CompareUnpairedClassAlignmentTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_study_compares_modes_on_disjoint_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, output = root / 'parent.pt', root / 'study'
            torch.manual_seed(31)
            save_checkpoint(checkpoint, Chimera(ChimeraConfig.preset('tiny')))
            script = Path(__file__).resolve().parents[1] / 'tools/compare_unpaired_class_alignment.py'
            command = [sys.executable, str(script), '--checkpoint', str(checkpoint), '--output-dir', str(output),
                       '--seeds', '17', '--steps', '2', '--train-examples-per-modality', '8', '--batch-size', '2',
                       '--text-seed', '100', '--series-seed', '101', '--test-seed', '2000000',
                       '--test-batches', '1', '--test-batch-size', '4']
            run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout[-8000:])
            report = json.loads((output / 'comparison.json').read_text())
            rows = {row['mode']: row for row in report['runs']}
            self.assertEqual(set(rows), {'control', 'class'})
            self.assertEqual(rows['control']['nonzero_alignment_steps'], 0)
            for row in rows.values():
                self.assertEqual(row['step'], 2)
                self.assertEqual(row['exposure']['sample_views'], 8)
                self.assertEqual(row['metrics']['examples'], 4)
            records = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 16)
            self.assertTrue(all(('text' in row) != ('series' in row) for row in records))
            before = (output / 'comparison.json').read_bytes()
            rejected = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual((output / 'comparison.json').read_bytes(), before)

    def test_rejects_overlapping_data_and_test_seeds(self):
        script = Path(__file__).resolve().parents[1] / 'tools/compare_unpaired_class_alignment.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = subprocess.run([sys.executable, str(script), '--checkpoint', str(root / 'unused.pt'),
                                  '--output-dir', str(root / 'study'), '--text-seed', '100',
                                  '--series-seed', '100'], text=True, capture_output=True, timeout=30)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('doivent être distincts', run.stderr)
            self.assertFalse((root / 'study').exists())


if __name__ == '__main__':
    unittest.main()
