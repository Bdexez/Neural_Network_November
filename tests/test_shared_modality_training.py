from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import distributed as dist

from chimera import ChimeraConfig
from chimera.training import load_checkpoint


class SharedModalityTrainingTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_shared_batches_enable_pairs_and_resume_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(ChimeraConfig.preset('tiny'), alignment_scope='global',
                             alignment_text_view='isolated', dropout=.1)
            config_path, data = root / 'config.json', root / 'data.jsonl'
            config.save(config_path)
            rows = [dict(text=f'signal {index}', series=[[float(index), 1., 2.], [2., 1., float(index)]],
                         class_label=index) for index in range(4)]
            data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            base = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                    '-m', 'chimera', '--threads', '1', 'train', '--config', str(config_path), '--data', str(data),
                    '--steps', '4', '--batch-size', '2', '--mixed-modalities', '--modality-policy', 'balanced',
                    '--no-shuffle', '--loss-normalization', 'observed', '--seed', '0', '--log-every', '1']
            env = dict(os.environ)
            for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(key, None)
            def run(extra, *, success=True):
                result = subprocess.run(base + extra, env=env, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=120)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout[-8000:])
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result.stdout
            shared, sharded, resumed = root / 'shared', root / 'sharded', root / 'resumed'
            run(['--shared-modality-batches', '--output', str(shared)])
            run(['--output', str(sharded)])
            shared_metrics = [json.loads(line) for line in (shared / 'metrics.jsonl').read_text().splitlines()]
            sharded_metrics = [json.loads(line) for line in (sharded / 'metrics.jsonl').read_text().splitlines()]
            self.assertGreater(shared_metrics[0]['alignment'], .01)
            self.assertEqual(sharded_metrics[0]['alignment'], 0.)
            run(['--shared-modality-batches', '--output', str(resumed), '--stop-after', '2'])
            run(['--output', str(resumed), '--resume', str(resumed / 'latest.pt')])
            expected, _ = load_checkpoint(shared / 'latest.pt')
            actual, payload = load_checkpoint(resumed / 'latest.pt')
            self.assertTrue(payload['trainer']['shared_modality_batches'])
            for name, tensor in expected.state_dict().items():
                torch.testing.assert_close(tensor, actual.state_dict()[name], rtol=0, atol=0, msg=name)
            error = run(['--no-shared-modality-batches', '--output', str(resumed),
                         '--resume', str(resumed / 'latest.pt')], success=False)
            self.assertIn('shared_modality_batches', error)


if __name__ == '__main__':
    unittest.main()
