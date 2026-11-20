from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
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
from chimera.cli import main
from chimera.training import load_checkpoint


def make_data(root):
    config = replace(ChimeraConfig.preset('tiny'), image_size=8, patch_size=4, dropout=.1,
                     alignment_scope='global', alignment_text_view='isolated')
    path = root / 'config.json'
    config.save(path)
    image = torch.zeros(config.image_channels, config.image_size, config.image_size).tolist()
    records = []
    for signature in range(1, 8):
        for i in range(1 + signature % 2):
            record = {'class_label': (i + signature) % config.num_classes}
            if signature & 1:
                record['text'] = f'texte {signature} {i}'
            if signature & 2:
                record['image'] = image
            if signature & 4:
                record['series'] = [[float(signature), float(i), .5], [1., 2., 3.]]
            records.append(record)
    data = root / 'data.jsonl'
    data.write_text(''.join(json.dumps(record) + '\n' for record in records))
    return path, data


class MixedModalityTrainingTests(unittest.TestCase):
    def test_mixed_training_validation_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            config, data = make_data(root)
            full, split = root / 'full', root / 'split'
            common = ['--threads', '2', 'train', '--config', str(config), '--data', str(data),
                      '--val-data', str(data), '--steps', '5', '--batch-size', '2', '--accumulation', '2',
                      '--eval-every', '5', '--save-every', '5', '--modality-dropout', '0']
            with self.assertRaisesRegex(ValueError, 'modalités'):
                main(common + ['--output', str(root / 'unsupported')])
            main(common + ['--mixed-modalities', '--output', str(full)])
            main(common + ['--mixed-modalities', '--output', str(split), '--stop-after', '2'])
            main(common + ['--output', str(split), '--resume', str(split / 'latest.pt')])
            left, left_payload = load_checkpoint(full / 'latest.pt')
            right, right_payload = load_checkpoint(split / 'latest.pt')
            self.assertTrue(right_payload['trainer']['mixed_modalities'])
            self.assertEqual(left_payload['trainer']['validation_state'], right_payload['trainer']['validation_state'])
            for name, value in left.state_dict().items():
                torch.testing.assert_close(value, right.state_dict()[name], rtol=0, atol=0, msg=name)
            report = json.loads((split / 'evaluation.json').read_text())
            self.assertEqual(report['classified_examples'], 11)
            with self.assertRaisesRegex(ValueError, 'mixed_modalities'):
                main(common + ['--no-mixed-modalities', '--output', str(split), '--resume', str(split / 'latest.pt')])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_distributed_mixed_training_global_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data = make_data(root)
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                       '-m', 'chimera', '--threads', '1', 'train', '--config', str(config), '--data', str(data),
                       '--steps', '4', '--batch-size', '2', '--accumulation', '2', '--loss-normalization', 'observed',
                       '--mixed-modalities', '--no-shuffle', '--modality-dropout', '0', '--output', str(root / 'ddp')]
            env = dict(os.environ)
            for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(name, None)
            result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout[-8000:])
            expected, payload = load_checkpoint(root / 'ddp/latest.pt')
            self.assertEqual(payload['step'], 4)
            self.assertEqual(payload['trainer']['world_size'], 2)
            self.assertTrue(payload['trainer']['mixed_modalities'])
            split = root / 'ddp-split'
            base = command[:-2]
            for extra in (['--stop-after', '2'], ['--resume', str(split / 'latest.pt')]):
                current = base + ['--output', str(split)] + extra
                if '--resume' in extra:
                    current.remove('--mixed-modalities')
                resumed = subprocess.run(current, env=env, text=True, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, timeout=120)
                self.assertEqual(resumed.returncode, 0, resumed.stdout[-8000:])
            actual, resumed_payload = load_checkpoint(split / 'latest.pt')
            self.assertTrue(resumed_payload['trainer']['mixed_modalities'])
            for name, value in expected.state_dict().items():
                torch.testing.assert_close(value, actual.state_dict()[name], rtol=0, atol=0, msg=name)


if __name__ == '__main__':
    unittest.main()
