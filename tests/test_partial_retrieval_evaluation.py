from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from chimera import Chimera, ChimeraConfig
from chimera.cli import main
from chimera.retrieval_evaluation import evaluate_retrieval
from chimera.training import save_checkpoint
from tests import test_embedding_search
from tests.test_partial_embedding_search import remove_modalities
from tests.test_mixed_modality_training import make_data


class PartialRetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.query, self.candidate = self.root / 'query.jsonl', self.root / 'candidate.jsonl'
        self.relevance, self.output = self.root / 'relevance.jsonl', self.root / 'report.json'
        for path in (self.query, self.candidate):
            test_embedding_search.EmbeddingSearchTests.write(path, [[1, 0], [0, 1], [1, 1], [-1, 0]])
        remove_modalities(self.query, text=(1, 3))
        remove_modalities(self.candidate, series=(0, 2))
        self.write_relevance([[1], [0, 2], [3], []])

    def write_relevance(self, values):
        query = json.loads(self.query.read_text().splitlines()[0])
        candidate = json.loads(self.candidate.read_text().splitlines()[0])
        header = {'format_version': 1, 'query_dataset_sha256': query['dataset_sha256'],
                  'query_dataset_images_sha256': query['dataset_images_sha256'],
                  'candidate_dataset_sha256': candidate['dataset_sha256'],
                  'candidate_dataset_images_sha256': candidate['dataset_images_sha256']}
        self.relevance.write_text(json.dumps(header) + '\n' + ''.join(
            json.dumps({'index': index, 'relevant': relevant}) + '\n' for index, relevant in enumerate(values)))

    def evaluate(self, **kwargs):
        return evaluate_retrieval(self.query, self.candidate, self.relevance, 'text', 'series',
                                  top_k=1, skip_missing=True, query_batch_size=1, candidate_batch_size=1, **kwargs)

    def test_original_indices_and_partial_population_metrics(self):
        report = self.evaluate()
        self.assertEqual((report['queries_total'], report['queries'], report['queries_without_modality']), (4, 2, 2))
        self.assertEqual((report['candidates_total'], report['candidates'], report['candidates_without_modality']), (4, 2, 2))
        self.assertEqual((report['judged_queries'], report['unjudged_queries']), (2, 0))
        for name in ('hit', 'recall', 'precision', 'reciprocal_rank', 'ndcg'):
            self.assertAlmostEqual(report['metrics'][name], .5, msg=name)
        self.assertTrue(report['skip_missing'])
        with redirect_stdout(io.StringIO()):
            main(['--threads', '2', 'evaluate-retrieval', '--queries', str(self.query), '--query-modality', 'text',
                  '--candidates', str(self.candidate), '--candidate-modality', 'series', '--skip-missing',
                  '--top-k', '1', '--relevance', str(self.relevance), '--output', str(self.output)])
        self.assertEqual(json.loads(self.output.read_text())['metrics'], report['metrics'])

    def test_unjudged_queries_are_distinct_from_missing_modalities(self):
        self.write_relevance([[1], [], [], []])
        report = self.evaluate()
        self.assertEqual(report['queries'], 2)
        self.assertEqual(report['queries_without_modality'], 2)
        self.assertEqual(report['judged_queries'], 1)
        self.assertEqual(report['unjudged_queries'], 1)
        self.assertEqual(report['metrics']['hit'], 1)

    def test_unavailable_relevant_candidate_is_rejected_atomically(self):
        self.write_relevance([[0], [], [1], []])
        self.output.write_text('précédent')
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'candidat pertinent'):
            main(['--threads', '2', 'evaluate-retrieval', '--queries', str(self.query), '--query-modality', 'text',
                  '--candidates', str(self.candidate), '--candidate-modality', 'series', '--skip-missing',
                  '--top-k', '1', '--relevance', str(self.relevance), '--output', str(self.output)])
        self.assertEqual(self.output.read_text(), 'précédent')
        # Les annotations des requêtes écartées restent bien formées et indexées.
        self.write_relevance([[1], [99], [3], []])
        with self.assertRaisesRegex(ValueError, 'indices pertinents'):
            self.evaluate()

    def test_excluding_self_only_reduces_pool_when_overlap_exists(self):
        # Requêtes texte aux indices 0/2 et séries aux indices 1/3 : aucune exclusion effective.
        self.write_relevance([[1, 3], [0], [1, 3], []])
        report = evaluate_retrieval(self.query, self.candidate, self.relevance, 'text', 'series',
                                    top_k=2, skip_missing=True, exclude_self=True)
        self.assertEqual(report['metrics']['recall'], 1)
        # La requête 0 et le candidat 0 coexistent désormais : top-k=2 ne convient plus.
        test_embedding_search.EmbeddingSearchTests.write(self.candidate, [[1, 0], [0, 1], [1, 1], [-1, 0]])
        remove_modalities(self.candidate, series=(2, 3))
        self.write_relevance([[1], [], [0], []])
        with self.assertRaisesRegex(ValueError, 'top_k'):
            evaluate_retrieval(self.query, self.candidate, self.relevance, 'text', 'series',
                               top_k=2, skip_missing=True, exclude_self=True)
        self.write_relevance([[0], [], [1], []])
        with self.assertRaisesRegex(ValueError, 'exclue'):
            self.evaluate(exclude_self=True)

    def test_real_mixed_embeddings_end_to_end(self):
        config_path, data = make_data(self.root)
        model = Chimera(ChimeraConfig.load(config_path)).eval()
        checkpoint, embeddings = self.root / 'model.pt', self.root / 'mixed.jsonl'
        save_checkpoint(checkpoint, model)
        with redirect_stdout(io.StringIO()):
            main(['--threads', '2', 'embed', '--checkpoint', str(checkpoint), '--data', str(data),
                  '--output', str(embeddings)])
        rows = [json.loads(line) for line in embeddings.read_text().splitlines()]
        header = {'format_version': 1, 'query_dataset_sha256': rows[0]['dataset_sha256'],
                  'query_dataset_images_sha256': rows[0]['dataset_images_sha256'],
                  'candidate_dataset_sha256': rows[0]['dataset_sha256'],
                  'candidate_dataset_images_sha256': rows[0]['dataset_images_sha256']}
        candidate_indices = [i for i, row in enumerate(rows) if 'series' in row['vectors']]
        self.relevance.write_text(json.dumps(header) + '\n' + ''.join(
            json.dumps({'index': i, 'relevant': candidate_indices if 'text' in row['vectors'] else []}) + '\n'
            for i, row in enumerate(rows)))
        report = evaluate_retrieval(embeddings, embeddings, self.relevance, 'text', 'series', top_k=2, skip_missing=True)
        self.assertEqual(report['queries_total'], len(rows))
        self.assertEqual(report['queries'], sum('text' in row['vectors'] for row in rows))
        self.assertEqual(report['candidates'], len(candidate_indices))
        self.assertEqual(report['metrics']['precision'], 1)
        self.assertEqual(report['metrics']['hit'], 1)


if __name__ == '__main__':
    unittest.main()
