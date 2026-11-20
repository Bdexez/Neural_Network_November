from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.cli import main
from chimera.data import synthetic_batch
from chimera.series_set import SeriesPredictionSet, class_scores, conformal_threshold, fit_series_set
from chimera.series_classifier import fit_series_classifier
from chimera.training import save_checkpoint


class SeriesSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_scores_quantile_ties_and_nonempty(self):
        probs = torch.tensor([[.5, .3, .2], [.25, .5, .25]], dtype=torch.double)
        torch.testing.assert_close(class_scores(probs, 'lac'), 1 - probs)
        torch.testing.assert_close(class_scores(probs, 'aps'), torch.tensor([[.5, .8, 1.], [.75, .5, 1.]], dtype=torch.double))
        self.assertEqual(conformal_threshold(torch.arange(1, 10).double() / 10, .9), .9)
        with self.assertRaisesRegex(ValueError, 'insuffisant'):
            conformal_threshold(torch.ones(8), .9)
        for method in ('lac', 'aps'):
            obj = SeriesPredictionSet(.9, method, 0., 9, 3, 'none', 'a'*64, 'b'*64, None, None)
            self.assertEqual(obj.predict(probs).sum(-1).tolist(), [1, 1])
            self.assertEqual(obj.predict(probs, torch.tensor([2, 2]))[:, 2].tolist(), [True, True])
            self.assertEqual(SeriesPredictionSet.from_dict(obj.to_dict()), obj)
        for invalid in (torch.tensor([[.5, .1]]), torch.tensor([[float('nan'), .5]])):
            with self.assertRaises(ValueError):
                class_scores(invalid, 'aps')

    def test_simulation_coverage_on_independent_examples(self):
        generator = torch.Generator().manual_seed(237)
        probs = torch.randn(12000, 4, generator=generator, dtype=torch.double).softmax(-1)
        labels = torch.multinomial(probs, 1, generator=generator).flatten()
        for method in ('lac', 'aps'):
            scores = class_scores(probs[:2000], method).gather(1, labels[:2000, None]).flatten()
            obj = SeriesPredictionSet(.9, method, conformal_threshold(scores, .9), 2000, 4,
                                      'none', 'a'*64, 'b'*64, None, None)
            mask = obj.predict(probs[2000:])
            coverage = mask.gather(1, labels[2000:, None]).double().mean().item()
            self.assertGreater(coverage, .87)
            self.assertLess(coverage, .95)
            self.assertTrue(mask.any(-1).all())

    def test_fit_state_and_batching(self):
        config = ChimeraConfig.preset('tiny')
        model = Chimera(config).train()
        def batch(seed):
            return {k:v for k,v in synthetic_batch(config, batch_size=64, seed=seed).items()
                    if k in ('series', 'series_mask', 'class_labels')}
        head = fit_series_classifier(model, [batch(311)], checkpoint_sha256='a'*64, dataset_sha256='b'*64)
        data = batch(312)
        data['class_labels'][:4] = -100
        before = {k:v.clone() for k,v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        kwargs = dict(checkpoint_sha256='a'*64, classifier_sha256='c'*64, dataset_sha256='d'*64)
        fitted = fit_series_set(model, head, [data], **kwargs)
        chunked = [{k:v[s] for k,v in data.items()} for s in (slice(0,32), slice(32,64))]
        other = fit_series_set(model, head, chunked, **kwargs)
        self.assertEqual(fitted.examples, 60)
        self.assertAlmostEqual(fitted.threshold, other.threshold, places=6)
        self.assertTrue(model.training)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        for k,v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'identique'):
            fit_series_set(model, head, [data], **dict(kwargs, dataset_sha256='b'*64))
        with self.assertRaisesRegex(ValueError, 'étiquetés'):
            fit_series_set(model, head, [dict(data, class_labels=torch.full((64,), -100))], **kwargs)
        self.assertTrue(model.training)

    def test_cli_complete_pipeline_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            config = replace(ChimeraConfig.preset('tiny'), temporal_statistics=True)
            checkpoint, head, temp, sets = (root / n for n in ('model.pt', 'head.json', 'temp.json', 'sets.json'))
            train, tuning, calib, test = (root / (n + '.jsonl') for n in ('train', 'tuning', 'calib', 'test'))
            report, predictions = root / 'report.json', root / 'predictions.jsonl'
            save_checkpoint(checkpoint, Chimera(config))
            for path, count, seed in ((train,128,521), (tuning,64,522), (calib,64,523), (test,24,524)):
                batch = synthetic_batch(config, batch_size=count, seed=seed)
                path.write_text('\n'.join(json.dumps(dict(series=batch['series'][i].tolist(),
                    class_label=batch['class_labels'][i].item(), future=batch['forecast_targets'][i].tolist())) for i in range(count)))
            main(['--threads','2','refine-series-classifier','--checkpoint',str(checkpoint),'--data',str(train),'--output',str(head)])
            main(['calibrate-series-probability','--checkpoint',str(checkpoint),'--series-classifier',str(head),
                  '--data',str(tuning),'--output',str(temp)])
            fit_args = ['calibrate-series-set','--checkpoint',str(checkpoint),'--series-classifier',str(head),
                        '--series-probability',str(temp),'--data',str(calib),'--output',str(sets)]
            main(fit_args)
            common = ['--checkpoint',str(checkpoint),'--series-classifier',str(head),'--data',str(test)]
            main(['predict',*common,'--output',str(predictions)])
            before = [json.loads(line) for line in predictions.read_text().splitlines()]
            main(['predict',*common,'--series-probability',str(temp),'--series-set',str(sets),'--output',str(predictions)])
            after = [json.loads(line) for line in predictions.read_text().splitlines()]
            for left,right in zip(before,after):
                self.assertEqual(left['forecast'],right['forecast'])
                self.assertEqual(left['series_classifier_class_id'],right['series_classifier_class_id'])
                self.assertIn(left['series_classifier_class_id'],right['series_prediction_set'])
            main(['evaluate',*common,'--series-probability',str(temp),'--series-set',str(sets),'--output',str(report)])
            result = json.loads(report.read_text())
            self.assertEqual(result['series_prediction_set_labeled_examples'],24)
            self.assertEqual(sum(result['series_prediction_set_counts_by_class']),24)
            self.assertGreaterEqual(result['series_prediction_set_mean_size'],1.)
            saved = sets.read_bytes()
            for reused in (train,tuning):
                with self.assertRaisesRegex(ValueError,'identique'):
                    main([*fit_args,'--data',str(reused)])
                self.assertEqual(sets.read_bytes(),saved)
            with self.assertRaises(ValueError):
                main(['predict',*common,'--series-set',str(sets)])
            with self.assertRaises(SystemExit):
                main([*fit_args,'--output',str(head)])
            with self.assertRaises(SystemExit):
                main(['predict','--checkpoint',str(checkpoint),'--data',str(test),'--series-set',str(sets)])
            main(['calibrate-series-set','--checkpoint',str(checkpoint),'--series-classifier',str(head),
                  '--data',str(calib),'--method','lac','--output',str(sets)])
            main(['evaluate',*common,'--series-set',str(sets),'--output',str(report)])
            self.assertEqual(json.loads(report.read_text())['series_prediction_set_method'],'lac')


    def test_benchmark_seed_ranges_and_checkpoint_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, report = root / "model.pt", root / "report.json"
            save_checkpoint(checkpoint, Chimera(ChimeraConfig.preset("tiny")))
            original = checkpoint.read_bytes()
            common = [sys.executable, "tools/benchmark_series_sets.py", "--checkpoint", str(checkpoint),
                      "--train-batches", "12", "--calibration-batches", "2", "--set-batches", "2",
                      "--test-batches", "1", "--batch-size", "16", "--shape-context", "4"]
            run = subprocess.run([*common, "--output", str(report)], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(report.read_text())
            self.assertEqual(result["prediction_sets"]["lac"]["examples"], 32)
            self.assertEqual(result["measurements"]["aps"]["identity"]["series_prediction_set_examples"], 16)
            for args in (("--set-seed", "60000001", "--output", str(report)),
                         ("--test-mask-seed", "63000001", "--output", str(report)),
                         ("--output", str(checkpoint))):
                run = subprocess.run([*common, *args], capture_output=True, text=True, timeout=60)
                self.assertNotEqual(run.returncode, 0)
            self.assertEqual(checkpoint.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
