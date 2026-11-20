from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chimera.cli import main
from chimera.retrieval_evaluation import evaluate_retrieval
from tests import test_embedding_search


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.queries, self.candidates = root / 'queries.jsonl', root / 'candidates.jsonl'
        self.relevance, self.output = root / 'relevance.jsonl', root / 'report.json'
        test_embedding_search.EmbeddingSearchTests.write(self.queries, [[1, 0], [0, 1], [1, 1]])
        test_embedding_search.EmbeddingSearchTests.write(self.candidates, [[1, 0], [0, 1], [-1, 0]], dataset='c')
        self.write_relevance([[0, 1], [0, 2], []])

    def write_relevance(self, lists):
        query = json.loads(self.queries.read_text().splitlines()[0])
        candidate = json.loads(self.candidates.read_text().splitlines()[0])
        header = {'format_version': 1, 'query_dataset_sha256': query['dataset_sha256'],
                  'query_dataset_images_sha256': query['dataset_images_sha256'],
                  'candidate_dataset_sha256': candidate['dataset_sha256'],
                  'candidate_dataset_images_sha256': candidate['dataset_images_sha256']}
        self.relevance.write_text(json.dumps(header) + '\n' +
                                  ''.join(json.dumps({'index': i, 'relevant': values}) + '\n'
                                          for i, values in enumerate(lists)))

    def evaluate(self, **kwargs):
        return evaluate_retrieval(self.queries, self.candidates, self.relevance, 'text', 'series',
                                  top_k=2, query_batch_size=1, candidate_batch_size=2, **kwargs)

    def test_metrics_multiple_relevant_and_unjudged(self):
        result = self.evaluate()
        self.assertEqual((result['queries'], result['judged_queries'], result['unjudged_queries']), (3, 2, 1))
        self.assertEqual(result['candidates'], 3)
        metrics = result['metrics']
        self.assertEqual(metrics['hit'], 1)
        for name in ('recall', 'precision', 'reciprocal_rank'):
            self.assertAlmostEqual(metrics[name], .75, msg=name)
        expected = (1 + (1 / math.log2(3)) / (1 + 1 / math.log2(3))) / 2
        self.assertAlmostEqual(metrics['ndcg'], expected)
        self.assertEqual(len(result['relevance_sha256']), 64)

    def test_missing_extra_invalid_and_unjudged_relevance(self):
        for relevance in ([[0], [1]], [[0], [1], [2], [0]], [[0], [1, 1], []],
                          [[0], [3], []], [[0], [True], []], [[], [], []]):
            self.write_relevance(relevance)
            with self.subTest(relevance=relevance), self.assertRaises(ValueError):
                self.evaluate()
        self.write_relevance([[0], [1], [2]])
        lines = self.relevance.read_text().splitlines()
        lines[1] = json.dumps({'index': 1, 'relevant': [0]})
        self.relevance.write_text('\n'.join(lines) + '\n')
        with self.assertRaises(ValueError):
            self.evaluate()
        with self.assertRaises(ValueError):
            evaluate_retrieval(self.queries, self.candidates, self.relevance, 'text', 'series', top_k=4)

    def test_relevance_requires_matching_dataset_provenance(self):
        rows = self.relevance.read_text().splitlines()
        header = json.loads(rows[0])
        for key, value in [('query_dataset_sha256', 'f' * 64), ('candidate_dataset_sha256', 'f' * 64),
                           ('query_dataset_images_sha256', 'f' * 64)]:
            bad = dict(header, **{key: value})
            self.relevance.write_text('\n'.join([json.dumps(bad), *rows[1:]]) + '\n')
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'empreinte'):
                self.evaluate()
        del header['candidate_dataset_images_sha256']
        self.relevance.write_text('\n'.join([json.dumps(header), *rows[1:]]) + '\n')
        with self.assertRaisesRegex(ValueError, 'en-tête'):
            self.evaluate()

    def test_self_exclusion_and_cross_dataset(self):
        test_embedding_search.EmbeddingSearchTests.write(self.candidates, [[1, 0], [0, 1], [-1, 0]])
        self.write_relevance([[0, 1], [0, 2], []])
        with self.assertRaisesRegex(ValueError, 'exclue'):
            self.evaluate(exclude_self=True)
        self.write_relevance([[1], [0], [0]])
        result = self.evaluate(exclude_self=True)
        self.assertEqual(result['judged_queries'], 3)
        self.assertEqual(result['metrics']['recall'], 1)

    def test_changed_input_and_atomic_cli(self):
        args = ['--threads', '2', 'evaluate-retrieval', '--queries', str(self.queries),
                '--candidates', str(self.candidates), '--relevance', str(self.relevance),
                '--query-modality', 'text', '--candidate-modality', 'series', '--top-k', '2',
                '--output', str(self.output)]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            main(args)
            original = self.output.read_text()
            self.assertAlmostEqual(json.loads(original)['metrics']['recall'], .75)
            with patch('chimera.retrieval_evaluation.file_sha256', return_value='invalid'):
                with self.assertRaisesRegex(ValueError, 'modifié'):
                    main(args)
            self.assertEqual(self.output.read_text(), original)
            link = self.output.parent / 'relevance-link'
            os.link(self.relevance, link)
            with self.assertRaises(SystemExit):
                main(args[:-1] + [str(link)])
            self.assertEqual(len(self.relevance.read_text().splitlines()), 4)


if __name__ == '__main__':
    unittest.main()
