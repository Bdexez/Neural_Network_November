from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.cli import main
from chimera.training import save_checkpoint
from chimera.embedding_search import search_embeddings


class EmbeddingSearchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.queries = self.root / 'queries.jsonl'
        self.candidates = self.root / 'candidates.jsonl'
        self.output = self.root / 'out.jsonl'

    @staticmethod
    def write(path, vectors, *, dataset='a', checkpoint='b', amp='none'):
        rows = []
        for index, vector in enumerate(vectors):
            rows.append(dict(format_version=1, index=index, checkpoint_sha256=checkpoint * 64,
                             dataset_sha256=dataset * 64, dataset_images_sha256=None,
                             alignment_text_view='isolated', amp=amp, device='cpu', batch_size=2,
                             torch_version='test', vectors={'text': vector, 'series': vector}))
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_exact_cosine_blocks_ties_and_exclusion(self):
        vectors = [[1, 0], [1, 0], [0, 1], [-1, 0], [1, 1]]
        self.write(self.queries, vectors)
        self.write(self.candidates, vectors)
        result = list(search_embeddings(self.queries, self.candidates, 'text', 'series', top_k=8,
                                        query_batch_size=2, candidate_batch_size=2, exclude_self=True))
        matrix = torch.tensor(vectors, dtype=torch.float64)
        matrix = matrix / matrix.norm(dim=1, keepdim=True)
        scores = matrix @ matrix.T
        for index, row in enumerate(result):
            expected = sorted(((-scores[index, j].item(), j) for j in range(len(vectors)) if j != index))
            self.assertEqual([m['index'] for m in row['matches']], [j for _, j in expected])
            for match, (score, _) in zip(row['matches'], expected):
                self.assertAlmostEqual(match['cosine'], -score)
        self.assertEqual([m['index'] for m in result[2]['matches'] if m['cosine'] == 0], [0, 1, 3])
        self.write(self.candidates, vectors, dataset='c')
        other = list(search_embeddings(self.queries, self.candidates, 'text', 'series', top_k=1, exclude_self=True))
        self.assertEqual(other[0]['matches'][0]['index'], 0)

    def test_end_to_end_with_real_extraction(self):
        config = ChimeraConfig.preset('tiny')
        checkpoint, data = self.root / 'model.pt', self.root / 'data.jsonl'
        save_checkpoint(checkpoint, Chimera(config).eval())
        records = [dict(text=f'signal {i}', series=[[float(i + j)] * config.series_features for j in range(4)])
                   for i in range(3)]
        data.write_text(''.join(json.dumps(row) + '\n' for row in records))
        with redirect_stdout(io.StringIO()):
            main(['--threads', '2', 'embed', '--checkpoint', str(checkpoint), '--data', str(data),
                  '--output', str(self.queries), '--batch-size', '2'])
            main(['--threads', '2', 'search-embeddings', '--queries', str(self.queries),
                  '--candidates', str(self.queries), '--query-modality', 'text', '--candidate-modality', 'series',
                  '--top-k', '2', '--output', str(self.output)])
        embeddings = [json.loads(line)['vectors'] for line in self.queries.read_text().splitlines()]
        found = [json.loads(line) for line in self.output.read_text().splitlines()]
        for query, row in zip(embeddings, found):
            expected = []
            for index, candidate in enumerate(embeddings):
                score = torch.nn.functional.cosine_similarity(torch.tensor(query['text'], dtype=torch.float64),
                                                              torch.tensor(candidate['series'], dtype=torch.float64), dim=0)
                expected.append((-score.item(), index))
            expected.sort()
            self.assertEqual([match['index'] for match in row['matches']], [index for _, index in expected[:2]])
            for match, (score, _) in zip(row['matches'], expected):
                self.assertAlmostEqual(match['cosine'], -score)

    def test_large_and_small_vector_scales(self):
        self.write(self.queries, [[1e-300, 0]])
        self.write(self.candidates, [[1e300, 0], [0, 1e-300]])
        rows = list(search_embeddings(self.queries, self.candidates, 'text', 'series'))
        self.assertEqual([item['cosine'] for item in rows[0]['matches']], [1, 0])

    def test_cli_atomic_failure_invalid_vectors_and_inputs(self):
        self.write(self.queries, [[1, 0], [0, 1], [1, 1]])
        self.write(self.candidates, [[1, 0], [0, 1]])
        args = ['--threads', '2', 'search-embeddings', '--queries', str(self.queries),
                '--candidates', str(self.candidates), '--query-modality', 'text',
                '--candidate-modality', 'series', '--output', str(self.output), '--query-batch-size', '1']
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            main(args)
            original = self.output.read_text()
            self.assertEqual(len(original.splitlines()), 3)
            for kwargs in ({'checkpoint': 'c'}, {'amp': 'bf16'}):
                self.write(self.candidates, [[1, 0]], **kwargs)
                with self.assertRaisesRegex(ValueError, 'incompatibles'):
                    main(args)
                self.assertEqual(self.output.read_text(), original)
            for vectors in ([[0, 0]], [[float('nan'), 0]], [[True, 1]], [[1, 0, 0]]):
                self.write(self.candidates, vectors)
                with self.assertRaises(ValueError):
                    main(args)
                self.assertEqual(self.output.read_text(), original)
            self.write(self.candidates, [[1, 0], [1, 0]])
            rows = self.candidates.read_text().splitlines()
            row = json.loads(rows[-1]); row['index'] = 7
            self.candidates.write_text(rows[0] + '\n' + json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                main(args)
            self.assertEqual(self.output.read_text(), original)
            link = self.root / 'link'
            os.link(self.queries, link)
            with self.assertRaises(SystemExit):
                main(args[:-4] + ['--output', str(link)])


if __name__ == '__main__':
    unittest.main()
