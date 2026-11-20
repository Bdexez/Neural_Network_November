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


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
class CompareTextAlignmentViewTests(unittest.TestCase):
    def test_small_experiment_separates_seeds_and_protects_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, output = root/'source.pt', root/'study'
            torch.manual_seed(41)
            save_checkpoint(checkpoint,Chimera(ChimeraConfig.preset('tiny')))
            original = checkpoint.read_bytes()
            common = [sys.executable,'tools/compare_text_alignment_view.py','--checkpoint',str(checkpoint),
                      '--seeds','111000000','--steps','2','--test-batches','1','--test-batch-size','4']
            process = subprocess.run([*common,'--output-dir',str(output)],capture_output=True,text=True,timeout=90)
            self.assertEqual(process.returncode,0,process.stderr)
            report = json.loads((output/'comparison.json').read_text())
            self.assertEqual([run['alignment_steps'] for run in report['runs']],[2,2])
            self.assertEqual([run['metrics']['examples'] for run in report['runs']],[4,4])
            self.assertIn('text_to_series_class',report['aggregate'])
            self.assertEqual(report['runs'][1]['metrics']['text_view_sensitivity_l2'],0.)
            for options in (('--output-dir',str(output)),
                            ('--test-seed','112000000','--output-dir',str(root/'overlap')),
                            ('--seeds','111000000','111000001','--output-dir',str(root/'overlap2'))):
                process = subprocess.run([*common,*options],capture_output=True,text=True,timeout=90)
                self.assertNotEqual(process.returncode,0)
            self.assertEqual(checkpoint.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
