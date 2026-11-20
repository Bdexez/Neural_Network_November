from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chimera import Chimera, ChimeraConfig
from chimera.cli import main
from chimera.training import save_checkpoint


class ClassRetrievalRelevanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.vectors = self.root / 'data.jsonl', self.root / 'vectors.jsonl'
        self.relevance, self.report = self.root / 'relevance.jsonl', self.root / 'report.json'
        self.checkpoint = self.root / 'model.pt'
        torch.manual_seed(902)
        save_checkpoint(self.checkpoint, Chimera(ChimeraConfig.preset('tiny')).eval())
        series = [[1., 2., 3.], [2., 1., 3.]]
        self.records = [dict(text='zero', series=series, class_label=0),
                        dict(series=series, class_label=0), dict(text='autre zero', class_label=0),
                        dict(text='un', series=series, class_label=1), dict(series=series, class_label=1),
                        dict(text='sans classe'), dict(text='ignoré', class_label=-100)]
        self.data.write_text(''.join(json.dumps(row) + '\n' for row in self.records))
        with redirect_stdout(io.StringIO()):
            main(['--threads', '2', 'embed', '--checkpoint', str(self.checkpoint), '--data', str(self.data),
                  '--output', str(self.vectors)])
        self.args = ['--threads', '2', 'prepare-class-relevance', '--query-data', str(self.data),
                     '--candidate-data', str(self.data), '--queries', str(self.vectors), '--candidates', str(self.vectors),
                     '--query-modality', 'text', '--candidate-modality', 'series', '--output', str(self.relevance)]

    def test_mixed_classes_to_evaluation_and_policy_binding(self):
        with redirect_stdout(io.StringIO()) as stdout:
            main(self.args + ['--exclude-self'])
        stats = json.loads(stdout.getvalue())
        self.assertEqual((stats['queries'], stats['available_queries'], stats['judged_queries']), (7, 5, 3))
        rows = [json.loads(line) for line in self.relevance.read_text().splitlines()]
        self.assertEqual([row['relevant'] for row in rows[1:]], [[1], [], [0, 1], [4], [], [], []])
        self.assertEqual(rows[0]['class_label_relevance'],
                         dict(query_modality='text', candidate_modality='series', exclude_self=True))
        evaluation = ['--threads', '2', 'evaluate-retrieval', '--queries', str(self.vectors),
                      '--candidates', str(self.vectors), '--query-modality', 'text', '--candidate-modality', 'series',
                      '--relevance', str(self.relevance), '--skip-missing', '--top-k', '1', '--output', str(self.report)]
        with redirect_stdout(io.StringIO()):
            main(evaluation + ['--exclude-self'])
            report = json.loads(self.report.read_text())
            self.assertEqual((report['queries'], report['judged_queries'], report['unjudged_queries']), (5, 3, 2))
            with self.assertRaisesRegex(ValueError, 'incompatibles'):
                main(evaluation)
            mismatch = evaluation.copy()
            mismatch[mismatch.index('--query-modality') + 1] = 'series'
            with self.assertRaisesRegex(ValueError, 'incompatibles'):
                main(mismatch + ['--exclude-self'])

    def test_reuse_for_new_checkpoint_but_reject_mixed_checkpoints(self):
        second_checkpoint, second_vectors = self.root / 'second.pt', self.root / 'second.jsonl'
        torch.manual_seed(903)
        save_checkpoint(second_checkpoint, Chimera(ChimeraConfig.preset('tiny')).eval())
        with redirect_stdout(io.StringIO()):
            main(self.args + ['--exclude-self'])
            previous = self.relevance.read_text()
            main(['--threads', '2', 'embed', '--checkpoint', str(second_checkpoint), '--data', str(self.data),
                  '--output', str(second_vectors)])
            args = self.args.copy()
            args[args.index('--candidates') + 1] = str(second_vectors)
            with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
                main(args)
            self.assertEqual(self.relevance.read_text(), previous)
            main(['--threads', '2', 'evaluate-retrieval', '--queries', str(second_vectors),
                  '--candidates', str(second_vectors), '--query-modality', 'text', '--candidate-modality', 'series',
                  '--relevance', str(self.relevance), '--skip-missing', '--exclude-self', '--top-k', '1',
                  '--output', str(self.report)])
        self.assertEqual(json.loads(self.report.read_text())['judged_queries'], 3)

    def test_different_dataset_does_not_exclude_equal_indices(self):
        other, other_vectors = self.root / 'other.jsonl', self.root / 'other-vectors.jsonl'
        other.write_text(json.dumps(dict(series=[[4., 5., 6.]], class_label=0)) + '\n')
        with redirect_stdout(io.StringIO()):
            main(['--threads', '2', 'embed', '--checkpoint', str(self.checkpoint), '--data', str(other),
                  '--output', str(other_vectors)])
            args = self.args.copy()
            args[args.index('--candidate-data') + 1] = str(other)
            args[args.index('--candidates') + 1] = str(other_vectors)
            main(args + ['--exclude-self'])
        rows = [json.loads(line) for line in self.relevance.read_text().splitlines()][1:]
        self.assertEqual(rows[0]['relevant'], [0])
        self.assertEqual(rows[2]['relevant'], [0])
        self.assertEqual(rows[3]['relevant'], [])

    def test_unlabeled_available_candidate_is_rejected(self):
        self.relevance.write_text('précédent')
        for label in (None, -100):
            records = [dict(record) for record in self.records]
            if label is None:
                records[1].pop('class_label')
            else:
                records[1]['class_label'] = label
            self.data.write_text(''.join(json.dumps(row) + '\n' for row in records))
            with redirect_stdout(io.StringIO()):
                main(['--threads', '2', 'embed', '--checkpoint', str(self.checkpoint), '--data', str(self.data),
                      '--output', str(self.vectors)])
                with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'candidat disponible.*classe connue'):
                    main(self.args)
            self.assertEqual(self.relevance.read_text(), 'précédent')

    def test_empty_relevance_fails_without_replacing_output(self):
        other, other_vectors = self.root / 'other.jsonl', self.root / 'other-vectors.jsonl'
        other.write_text(json.dumps(dict(series=[[4., 5., 6.]], class_label=2)) + '\n')
        self.relevance.write_text('précédent')
        with redirect_stdout(io.StringIO()):
            main(['--threads', '2', 'embed', '--checkpoint', str(self.checkpoint), '--data', str(other),
                  '--output', str(other_vectors)])
            args = self.args.copy()
            args[args.index('--candidate-data') + 1] = str(other)
            args[args.index('--candidates') + 1] = str(other_vectors)
            with self.assertRaisesRegex(ValueError, 'Aucune requête'):
                main(args)
        self.assertEqual(self.relevance.read_text(), 'précédent')

    def test_stdout_and_blank_lines_keep_example_indices(self):
        lines = self.data.read_text().splitlines()
        self.data.write_text('\n' + '\n\n'.join(lines) + '\n')
        with redirect_stdout(io.StringIO()):
            main(['--threads', '2', 'embed', '--checkpoint', str(self.checkpoint), '--data', str(self.data),
                  '--output', str(self.vectors)])
        args = self.args[:self.args.index('--output')]
        with redirect_stdout(io.StringIO()) as stdout:
            main(args + ['--exclude-self', '--batch-size', '1'])
        rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual([row['index'] for row in rows[1:]], list(range(7)))
        self.assertEqual(rows[3]['relevant'], [0, 1])

    def test_mismatch_and_late_change_preserve_existing_output(self):
        self.relevance.write_text('précédent')
        altered = self.root / 'altered.jsonl'
        rows = list(self.records)
        rows[0] = dict(rows[0], class_label=2)
        altered.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        args = self.args.copy()
        args[args.index('--query-data') + 1] = str(altered)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'ne correspond pas'):
            main(args)
        self.assertEqual(self.relevance.read_text(), 'précédent')
        with patch('chimera.retrieval_relevance.file_sha256', return_value='changed'):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'modifié'):
                main(self.args)
        self.assertEqual(self.relevance.read_text(), 'précédent')

    def test_invalid_label_and_forged_modality_presence_are_rejected(self):
        self.relevance.write_text('précédent')
        records = list(self.records)
        records[-1] = dict(records[-1], class_label=True)
        self.data.write_text(''.join(json.dumps(row) + '\n' for row in records))
        digest = hashlib.sha256(self.data.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in self.vectors.read_text().splitlines()]
        for row in rows:
            row['dataset_sha256'] = digest
        self.vectors.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'class_label invalide'):
            main(self.args)
        self.assertEqual(self.relevance.read_text(), 'précédent')
        records[-1]['class_label'] = -100
        self.data.write_text(''.join(json.dumps(row) + '\n' for row in records))
        digest = hashlib.sha256(self.data.read_bytes()).hexdigest()
        for row in rows:
            row['dataset_sha256'] = digest
        rows[1]['vectors']['text'] = rows[0]['vectors']['text']
        self.vectors.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Modalités.*incohérentes'):
            main(self.args)
        self.assertEqual(self.relevance.read_text(), 'précédent')

    def test_source_hardlinks_are_protected(self):
        data_link, vector_link = self.root / 'data-link', self.root / 'vector-link'
        os.link(self.data, data_link)
        os.link(self.vectors, vector_link)
        for link, source in ((data_link, self.data), (vector_link, self.vectors)):
            before = source.read_bytes()
            args = self.args.copy()
            args[args.index('--output') + 1] = str(link)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(source.read_bytes(), before)

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow indisponible')
    def test_external_images_and_links_are_protected(self):
        from PIL import Image
        image = self.root / 'photo.png'
        Image.new('RGB', (8, 8), 'red').save(image)
        records = [dict(text='x', image_path=image.name, series=[[1., 2., 3.]], class_label=0),
                   dict(text='y', image_path=image.name, series=[[2., 3., 4.]], class_label=0)]
        self.data.write_text(''.join(json.dumps(row) + '\n' for row in records))
        with redirect_stdout(io.StringIO()):
            main(['--threads', '2', 'embed', '--checkpoint', str(self.checkpoint), '--data', str(self.data),
                  '--output', str(self.vectors)])
        link = self.root / 'photo-hardlink.png'
        os.link(image, link)
        before = image.read_bytes()
        args = self.args.copy()
        args[args.index('--output') + 1] = str(link)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args)
        self.assertEqual(image.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
