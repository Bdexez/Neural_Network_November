from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from chimera import Chimera
from chimera.cli import main
from chimera.embedding_search import search_embeddings
from chimera.training import save_checkpoint
from tests import test_embedding_search
from tests.test_mixed_modality_training import make_data


def remove_modalities(path, *, text=(), series=()):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for name, indices in [('text', text), ('series', series)]:
        for index in indices:
            del rows[index]['vectors'][name]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


class PartialEmbeddingSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.query, self.candidate = self.root / 'query.jsonl', self.root / 'candidate.jsonl'
        self.output = self.root / 'results.jsonl'
        test_embedding_search.EmbeddingSearchTests.write(self.query, [[1, 0], [0, 1], [1, 1], [-1, 0]])
        test_embedding_search.EmbeddingSearchTests.write(self.candidate, [[1, 0], [0, 1], [1, 1], [-1, 0]])
        remove_modalities(self.query, text=(1, 3))
        remove_modalities(self.candidate, series=(0, 2))

    def test_sparse_indices_cosine_and_self_exclusion(self):
        with self.assertRaisesRegex(ValueError, 'absente'):
            list(search_embeddings(self.query, self.candidate, 'text', 'series'))
        rows = list(search_embeddings(self.query, self.candidate, 'text', 'series', skip_missing=True,
                                      top_k=3, query_batch_size=1, candidate_batch_size=1, exclude_self=True))
        self.assertEqual([row['index'] for row in rows], [0, 2])
        self.assertEqual([match['index'] for match in rows[0]['matches']], [1, 3])
        self.assertEqual([match['index'] for match in rows[1]['matches']], [1, 3])
        self.assertAlmostEqual(rows[0]['matches'][1]['cosine'], -1.)
        # Les modalités coexistent à l'indice 0 et l'auto-correspondance doit être exclue.
        other = list(search_embeddings(self.query, self.query, 'series', 'text', skip_missing=True,
                                       query_batch_size=1, candidate_batch_size=1, exclude_self=True))
        self.assertEqual([match['index'] for match in other[0]['matches']], [2])

    def test_missing_modality_or_late_invalid_entry_fails_atomically(self):
        # Une absence explicite ne masque pas un vecteur présent mais invalide.
        rows = [json.loads(line) for line in self.candidate.read_text().splitlines()]
        rows[-1]['vectors']['series'] = [0, 0]
        self.candidate.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'nul'):
            list(search_embeddings(self.query, self.candidate, 'text', 'series', skip_missing=True))
        self.output.write_text('précédent')
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'nul'):
            main(['--threads', '2', 'search-embeddings', '--queries', str(self.query), '--query-modality', 'text',
                  '--candidates', str(self.candidate), '--candidate-modality', 'series', '--skip-missing',
                  '--query-batch-size', '1', '--candidate-batch-size', '1', '--output', str(self.output)])
        self.assertEqual(self.output.read_text(), 'précédent')
        for row in rows:
            row['vectors'].pop('series', None)
        self.candidate.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'Aucun vecteur'):
            list(search_embeddings(self.query, self.candidate, 'text', 'series', skip_missing=True))

    def test_real_heterogeneous_extraction(self):
        from chimera import ChimeraConfig
        config_path, data = make_data(self.root)
        model = Chimera(ChimeraConfig.load(config_path)).eval()
        checkpoint, embeddings = self.root / 'model.pt', self.root / 'vectors.jsonl'
        save_checkpoint(checkpoint, model)
        with redirect_stdout(io.StringIO()):
            main(['--threads', '2', 'embed', '--checkpoint', str(checkpoint), '--data', str(data),
                  '--output', str(embeddings)])
        vectors = [json.loads(line)['vectors'] for line in embeddings.read_text().splitlines()]
        rows = list(search_embeddings(embeddings, embeddings, 'text', 'series', top_k=2, skip_missing=True))
        with redirect_stdout(io.StringIO()):
            main(['--threads', '2', 'search-embeddings', '--queries', str(embeddings), '--query-modality', 'text',
                  '--candidates', str(embeddings), '--candidate-modality', 'series', '--skip-missing',
                  '--top-k', '2', '--output', str(self.output)])
        self.assertEqual([json.loads(line) for line in self.output.read_text().splitlines()], rows)
        self.assertEqual([r['index'] for r in rows], [i for i, value in enumerate(vectors) if 'text' in value])
        for row in rows:
            self.assertTrue(all('series' in vectors[match['index']] for match in row['matches']))


if __name__ == '__main__':
    unittest.main()
