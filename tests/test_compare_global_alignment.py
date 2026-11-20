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
class CompareGlobalAlignmentTests(unittest.TestCase):
    def test_small_experiment_separates_seeds_and_protects_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, output = root/'source.pt', root/'study'
            torch.manual_seed(41)
            save_checkpoint(checkpoint,Chimera(ChimeraConfig.preset('tiny')))
            original = checkpoint.read_bytes()
            common = [sys.executable,'tools/compare_global_alignment.py','--checkpoint',str(checkpoint),
                      '--seeds','101000000','--steps','2','--test-batches','1','--test-batch-size','4']
            process = subprocess.run([*common,'--output-dir',str(output)],capture_output=True,text=True,timeout=90)
            self.assertEqual(process.returncode,0,process.stderr)
            report = json.loads((output/'comparison.json').read_text())
            self.assertEqual([run['alignment_steps'] for run in report['runs']],[0,2])
            self.assertEqual([run['metrics']['examples'] for run in report['runs']],[4,4])
            self.assertIn('class_retrieval_top1',report['aggregate'])
            for options in (('--output-dir',str(output)),
                            ('--test-seed','102000000','--output-dir',str(root/'overlap')),
                            ('--seeds','101000000','101000001','--output-dir',str(root/'overlap2'))):
                process = subprocess.run([*common,*options],capture_output=True,text=True,timeout=90)
                self.assertNotEqual(process.returncode,0)
            self.assertEqual(checkpoint.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
