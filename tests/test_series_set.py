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
from chimera.binomial import wilson_interval
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

    def test_class_conditional_simulation(self):
        generator = torch.Generator().manual_seed(9012)
        labels = (torch.rand(16000, generator=generator) < .12).long()
        scores = torch.randn(16000, 2, generator=generator, dtype=torch.double)
        scores[:, 0] += 1.5
        probs = scores.softmax(-1)
        calibration_scores = class_scores(probs[:4000], 'lac').gather(1, labels[:4000, None]).flatten()
        groups = [calibration_scores[labels[:4000] == i] for i in range(2)]
        fitted = SeriesPredictionSet(.9, 'lac', conformal_threshold(calibration_scores, .9), 4000, 2,
                                     'none', 'a'*64, 'b'*64, None, None,
                                     [conformal_threshold(group, .9) for group in groups],
                                     [group.numel() for group in groups])
        mask = fitted.predict(probs[4000:])
        hits = mask.gather(1, labels[4000:, None]).flatten()
        for class_id in range(2):
            observed = hits[labels[4000:] == class_id].double().mean().item()
            self.assertGreater(observed, .86)
            self.assertLess(observed, .95)

    def test_class_conditional_thresholds_and_rare_class(self):
        class StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = replace(ChimeraConfig.preset('tiny'), num_classes=2, series_features=2)
        class StubHead:
            num_classes, amp = 2, 'none'
            checkpoint_sha256, dataset_sha256 = 'a'*64, 'b'*64
            def validate(self, config, *, amp=None):
                pass
            def scores(self, model, batch):
                return batch['series'][:, 0]
        model, head = StubModel().train(), StubHead()
        scores = torch.tensor([[2.2, 0.]] * 185 + [[2.2, 0.]] * 15)
        labels = torch.tensor([0] * 185 + [1] * 15)
        batch = dict(series=scores[:, None], class_labels=labels)
        kwargs = dict(checkpoint_sha256='a'*64, classifier_sha256='c'*64,
                      dataset_sha256='d'*64, method='lac')
        marginal = fit_series_set(model, head, [batch], **kwargs)
        conditional = fit_series_set(model, head, [batch], scope='class', **kwargs)
        self.assertEqual(conditional.class_examples, [185, 15])
        self.assertEqual(conditional.scope, 'class')
        self.assertGreater(conditional.class_thresholds[1], conditional.class_thresholds[0])
        probs = scores.double().softmax(-1)
        self.assertFalse(marginal.predict(probs)[-1, 1])
        self.assertTrue(conditional.predict(probs)[-1, 1])
        self.assertEqual(SeriesPredictionSet.from_dict(conditional.to_dict()), conditional)
        legacy = marginal.to_dict()
        del legacy['class_examples'], legacy['class_thresholds']
        self.assertEqual(SeriesPredictionSet.from_dict(legacy).scope, 'marginal')
        with self.assertRaisesRegex(ValueError, 'classe 1'):
            fit_series_set(model, head, [dict(series=scores[:193, None], class_labels=labels[:193])],
                           scope='class', **kwargs)
        with self.assertRaises(ValueError):
            SeriesPredictionSet.from_dict(dict(conditional.to_dict(), class_examples=[192, 8]))
        self.assertTrue(model.training)

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
                  '--data',str(tuning),'--classifier-data',str(train),'--output',str(temp)])
            fit_args = ['calibrate-series-set','--checkpoint',str(checkpoint),'--series-classifier',str(head),
                        '--series-probability',str(temp),'--data',str(calib),'--output',str(sets),
                        '--classifier-data',str(train),'--probability-data',str(tuning)]
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
            self.assertNotIn('series_audited_sources', result)
            audited_args = ['evaluate',*common,'--series-probability',str(temp),'--series-set',str(sets),
                            '--classifier-data',str(train),'--probability-data',str(tuning),
                            '--set-data',str(calib),'--output',str(report)]
            main(audited_args)
            self.assertEqual(json.loads(report.read_text())['series_audited_sources'],
                             ['tête', 'température', 'ensemble'])
            good_report = report.read_bytes()
            contaminated_test = root / 'contaminated-test.jsonl'
            rows = [json.loads(line) for line in test.read_text().splitlines()]
            rows[0]['series'] = json.loads(calib.read_text().splitlines()[0])['series']
            rows[0]['class_label'] = (rows[0]['class_label'] + 1) % config.num_classes
            contaminated_test.write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, 'Historiques partagés'):
                main([*audited_args, '--data', str(contaminated_test)])
            self.assertEqual(report.read_bytes(), good_report)
            with self.assertRaisesRegex(ValueError, 'empreinte'):
                main([*audited_args, '--set-data', str(tuning)])
            with self.assertRaises(SystemExit):
                main([*audited_args, '--output', str(calib)])
            with self.assertRaises(SystemExit):
                main(['evaluate',*common,'--set-data',str(calib)])
            self.assertEqual(result['series_prediction_set_labeled_examples'],24)
            self.assertEqual(sum(result['series_prediction_set_counts_by_class']),24)
            self.assertGreaterEqual(result['series_prediction_set_mean_size'],1.)
            self.assertNotIn('iid_classification_confidence_intervals', result)
            main(['evaluate',*common,'--series-probability',str(temp),'--series-set',str(sets),
                  '--iid-intervals','--output',str(report)])
            intervals = json.loads(report.read_text())['iid_classification_confidence_intervals']
            self.assertEqual(intervals['method'], 'wilson')
            low, high = intervals['series_prediction_set_coverage']
            self.assertLessEqual(low, result['series_prediction_set_coverage'])
            self.assertGreaterEqual(high, result['series_prediction_set_coverage'])
            self.assertEqual(len(intervals['series_prediction_set_coverage_by_class']),config.num_classes)
            labels = [json.loads(line)['class_label'] for line in test.read_text().splitlines()]
            hits = sum(label in row['series_prediction_set'] for label, row in zip(labels, after))
            self.assertEqual(intervals['series_prediction_set_coverage'], wilson_interval(hits, len(labels)))
            for class_id in range(config.num_classes):
                group = [(label, row) for label, row in zip(labels, after) if label == class_id]
                successes = sum(label in row['series_prediction_set'] for label, row in group)
                self.assertEqual(intervals['series_prediction_set_coverage_by_class'][class_id],
                                 wilson_interval(successes, len(group)))
            unlabeled = root / 'unlabeled.jsonl'
            rows = [json.loads(line) for line in test.read_text().splitlines()]
            for row in rows:
                del row['class_label']
            unlabeled.write_text('\n'.join(json.dumps(row) for row in rows))
            main(['evaluate',*common,'--data',str(unlabeled),'--series-probability',str(temp),
                  '--series-set',str(sets),'--iid-intervals','--output',str(report)])
            empty_intervals = json.loads(report.read_text())['iid_classification_confidence_intervals']
            self.assertIsNone(empty_intervals['series_prediction_set_coverage'])
            self.assertEqual(empty_intervals['series_prediction_set_coverage_by_class'], [None] * config.num_classes)
            saved = sets.read_bytes()
            for reused in (train,tuning):
                with self.assertRaisesRegex(ValueError,'identique|partagés'):
                    main([*fit_args,'--data',str(reused)])
                self.assertEqual(sets.read_bytes(),saved)
            contaminated = root / 'contaminated.jsonl'
            contaminated_rows = [json.loads(line) for line in calib.read_text().splitlines()]
            contaminated_rows[0]['series'] = json.loads(train.read_text().splitlines()[0])['series']
            contaminated_rows[0]['class_label'] = (contaminated_rows[0]['class_label'] + 1) % config.num_classes
            contaminated.write_text('\n'.join(json.dumps(row) for row in contaminated_rows))
            with self.assertRaisesRegex(ValueError, 'Historiques partagés'):
                main([*fit_args, '--data', str(contaminated)])
            self.assertEqual(sets.read_bytes(), saved)
            with self.assertRaisesRegex(ValueError, 'empreinte'):
                main([*fit_args, '--classifier-data', str(tuning)])
            self.assertEqual(sets.read_bytes(), saved)
            with self.assertRaises(SystemExit):
                main([*fit_args, '--output', str(train)])
            with self.assertRaises(SystemExit):
                main(['calibrate-series-set','--checkpoint',str(checkpoint),'--series-classifier',str(head),
                      '--data',str(calib),'--probability-data',str(tuning),'--output',str(sets)])
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
            main(['calibrate-series-set','--checkpoint',str(checkpoint),'--series-classifier',str(head),
                  '--data',str(calib),'--method','lac','--scope','class','--coverage','0.8','--output',str(sets)])
            self.assertEqual(SeriesPredictionSet.load(sets).scope, 'class')
            main(['evaluate',*common,'--series-set',str(sets),'--output',str(report)])
            self.assertEqual(json.loads(report.read_text())['series_prediction_set_scope'], 'class')
            saved = sets.read_bytes()
            with self.assertRaisesRegex(ValueError, 'classe'):
                main(['calibrate-series-set','--checkpoint',str(checkpoint),'--series-classifier',str(head),
                      '--data',str(calib),'--scope','class','--coverage','0.95','--output',str(sets)])
            self.assertEqual(sets.read_bytes(),saved)


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
