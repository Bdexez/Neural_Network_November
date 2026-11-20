from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.binomial import wilson_interval
from chimera.cli import main
from chimera.data import synthetic_batch
from chimera.training import save_checkpoint
from chimera.series_classifier import fit_series_classifier
from chimera.series_gate import fit_series_gate
from chimera.evaluation import evaluate
from chimera.series_gate_guard import SeriesGateGuard, fit_series_gate_guard, harm_threshold


class StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = replace(ChimeraConfig.preset('tiny'), series_features=1, num_classes=2)
        self.dropout = torch.nn.Dropout().eval()
    def forward(self, *, series, series_mask=None):
        return SimpleNamespace(classification_logits=torch.tensor([1., 0.]).expand(series.size(0), -1))


class StubHead:
    amp, dataset_sha256 = 'none', 'b'*64
    def validate(self, config, *, amp=None):
        if amp != self.amp:
            raise ValueError('précision')
    def scores(self, model, batch):
        return torch.tensor([0., 1.]).expand(batch['series'].size(0), -1)
    def to_dict(self):
        return {'version': getattr(self, 'version', 0)}


class StubGate:
    amp, threshold = 'none', .1
    checkpoint_sha256, classifier_sha256 = 'a'*64, 'c'*64
    dataset_sha256, validation_sha256 = 'd'*64, 'e'*64
    def validate(self, classifier, *, amp=None):
        classifier.validate(None, amp=amp)
    def advantage(self, batch, logits, scores):
        return batch['series'][:, 0, 0]
    def to_dict(self):
        return {'version': getattr(self, 'version', 0)}


class SeriesGateGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_quantile_inclusive_ties_and_insufficient_examples(self):
        advantage = torch.tensor([.2, .4, .6, .8, 1., 1.2, .3, .5], dtype=torch.float64)
        harm = torch.tensor([True]*6 + [False]*2)
        self.assertEqual(harm_threshold(advantage, harm, .25, .1), 1.)
        selected = advantage > harm_threshold(advantage, harm, .25, .1)
        self.assertEqual((selected & harm).sum().item(), 1)
        tied = torch.tensor([.5]*9, dtype=torch.float64)
        self.assertEqual(harm_threshold(tied, torch.ones(9, dtype=torch.bool), .1, .03), .5)
        self.assertFalse((tied > .5).any())
        self.assertIsNone(harm_threshold(tied[:8], torch.ones(8, dtype=torch.bool), .1, .03))
        self.assertEqual(harm_threshold(tied, torch.zeros(9, dtype=torch.bool), .1, .03), .03)
        for alpha in (0., 1., float('nan')):
            with self.assertRaises(ValueError):
                harm_threshold(advantage, harm, alpha, .1)

    def test_exchangeable_rank_bound_by_exhaustive_leave_one_out(self):
        scores = torch.arange(20, dtype=torch.float64) / 20
        harmful = torch.tensor([False]*14 + [True]*6)
        violations = 0
        for held_out in range(20):
            keep = torch.arange(20) != held_out
            threshold = harm_threshold(scores[keep], harmful[keep], .1, .03)
            violations += int(harmful[held_out] and scores[held_out] > threshold)
        self.assertLessEqual(violations / 20, .1)
        self.assertEqual(violations, 2)

    def test_fit_preserves_model_and_provenance(self):
        model, head, gate = StubModel().train(), StubHead(), StubGate()
        model.dropout.eval()
        values = torch.tensor([.2, .4, .6, .8, 1., 1.2, .3, .5])
        targets = torch.tensor([0,0,0,0,0,0,1,1])
        batch = dict(series=values[:,None,None], class_labels=targets)
        kwargs = dict(checkpoint_sha256='a'*64, classifier_sha256='c'*64,
                      gate_sha256='f'*64, dataset_sha256='1'*64, max_harm=.25)
        state = torch.get_rng_state().clone()
        parameters = {name: value.clone() for name, value in model.state_dict().items()}
        guard = fit_series_gate_guard(model, head, gate, [batch], **kwargs)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
        chunks = [{key: value[part] for key, value in batch.items()} for part in (slice(0, 3), slice(3, 8))]
        self.assertEqual(fit_series_gate_guard(model, head, gate, chunks, **kwargs), guard)
        with self.assertRaisesRegex(ValueError, 'taille des lots'):
            fit_series_gate_guard(model, head, gate, [batch], **dict(kwargs, feature_batch_size=4))
        self.assertEqual((guard.examples, guard.unguarded_harms, guard.guarded_harms), (8, 6, 1))
        self.assertAlmostEqual(guard.threshold, 1.)
        self.assertEqual(guard.select(values).tolist(), [False]*5 + [True] + [False]*2)
        restored = SeriesGateGuard.from_dict(json.loads(json.dumps(guard.to_dict())))
        self.assertEqual(restored, guard)
        guard.validate(gate, head, amp='none')
        gate.version = 1
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            guard.validate(gate, head, amp='none')
        gate.version = 0
        head.version = 1
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            guard.validate(gate, head, amp='none')
        head.version = 0
        class MutatingGate(StubGate):
            def advantage(self, batch, logits, scores):
                self.version = 2
                return super().advantage(batch, logits, scores)
        with self.assertRaisesRegex(ValueError, 'changé'):
            fit_series_gate_guard(model, head, MutatingGate(), [batch], **kwargs)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        with self.assertRaisesRegex(ValueError, 'identique'):
            fit_series_gate_guard(model, head, gate, [batch], **dict(kwargs, dataset_sha256='d'*64))
        with self.assertRaisesRegex(ValueError, 'étiquetés'):
            fit_series_gate_guard(model, head, gate, [dict(series=values[:,None,None])], **kwargs)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        disabled = fit_series_gate_guard(model, head, gate, [batch], **dict(kwargs, max_harm=.01))
        self.assertIsNone(disabled.threshold)
        self.assertFalse(disabled.select(values).any())
        tied = dict(series=torch.full((9, 1, 1), .1), class_labels=torch.zeros(9, dtype=torch.long))
        at_gate_threshold = fit_series_gate_guard(model, head, gate, [tied], **dict(kwargs, max_harm=.1))
        self.assertEqual(at_gate_threshold.unguarded_harms, 0)
        self.assertEqual(at_gate_threshold.guarded_harms, 0)
        self.assertFalse(at_gate_threshold.select(tied['series'][:, 0, 0]).any())
        with self.assertRaises(ValueError):
            SeriesGateGuard.from_dict(dict(disabled.to_dict(), threshold=.1))


    def test_bf16_guard_with_missing_labels_and_empty_evaluation(self):
        config = ChimeraConfig.preset('tiny')
        model = Chimera(config).train()
        def batch(seed):
            return {key:value for key,value in synthetic_batch(config, batch_size=48, seed=seed).items()
                    if key in ('series','series_mask','class_labels')}
        head = fit_series_classifier(model, [batch(2101)], amp='bf16', checkpoint_sha256='a'*64)
        gate = fit_series_gate(model, head, [batch(2102)], [batch(2103)], amp='bf16', epochs=2,
                               checkpoint_sha256='a'*64, classifier_sha256='c'*64)
        data = batch(2104)
        data['class_labels'][:5] = -100
        parameters = {name: value.clone() for name, value in model.state_dict().items()}
        state = torch.get_rng_state().clone()
        guard = fit_series_gate_guard(model, head, gate, [data], amp='bf16', max_harm=.1,
                                      checkpoint_sha256='a'*64, classifier_sha256='c'*64, gate_sha256='f'*64)
        self.assertEqual(guard.examples,43)
        self.assertTrue(model.training)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
        result = evaluate(model, [data], amp='bf16', series_classifier=head, series_gate=gate,
                          series_gate_guard=guard, iid_intervals=True)
        self.assertEqual(result['series_gate_labeled_examples'],43)
        self.assertEqual(result['series_gate_guard_max_harm'],.1)
        empty = evaluate(model, [dict(data, class_labels=torch.full((48,),-100))], amp='bf16',
                         series_classifier=head, series_gate=gate, series_gate_guard=guard, iid_intervals=True)
        self.assertIsNone(empty['series_gate_guard_harm_rate'])
        self.assertIsNone(empty['iid_classification_confidence_intervals']['series_gate_guard_harm_rate'])
        with self.assertRaisesRegex(ValueError, 'précision'):
            evaluate(model, [data], series_classifier=head, series_gate=gate, series_gate_guard=guard)

    def test_cli_calibration_prediction_evaluation_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            config = ChimeraConfig.preset('tiny')
            checkpoint, head, gate, guard = (root / name for name in ('model.pt', 'head.json', 'gate.json', 'guard.json'))
            train, fitting, validation, reference, test = (root / (name + '.jsonl') for name in
                                                             ('train','fit','validation','reference','test'))
            report, predictions = root / 'report.json', root / 'predictions.jsonl'
            save_checkpoint(checkpoint, Chimera(config))
            for path, n, seed in ((train,128,1111), (fitting,96,1112), (validation,64,1113),
                                   (reference,96,1114), (test,32,1115)):
                data = synthetic_batch(config, batch_size=n, seed=seed)
                path.write_text('\n'.join(json.dumps(dict(series=data['series'][i].tolist(),
                    class_label=data['class_labels'][i].item(), future=data['forecast_targets'][i].tolist())) for i in range(n)))
            main(['--threads','2','refine-series-classifier','--checkpoint',str(checkpoint),'--data',str(train),'--output',str(head)])
            main(['refine-series-gate','--checkpoint',str(checkpoint),'--series-classifier',str(head),
                  '--data',str(fitting),'--validation-data',str(validation),'--output',str(gate),'--epochs','4'])
            fit = ['calibrate-series-gate-guard','--checkpoint',str(checkpoint),'--series-classifier',str(head),
                   '--series-gate',str(gate),'--data',str(reference),'--output',str(guard),'--max-harm','.1',
                   '--classifier-data',str(train),'--gate-data',str(fitting),'--gate-validation-data',str(validation)]
            main(fit)
            protected = SeriesGateGuard.load(guard)
            self.assertEqual(protected.examples, 96)
            self.assertEqual(protected.feature_batch_size, 32)
            self.assertLessEqual(protected.guarded_harms, protected.unguarded_harms)
            common = ['--checkpoint',str(checkpoint),'--series-classifier',str(head),'--series-gate',str(gate),'--data',str(test)]
            main(['predict',*common,'--output',str(predictions)])
            before = [json.loads(row) for row in predictions.read_text().splitlines()]
            main(['predict',*common,'--series-gate-guard',str(guard),'--output',str(predictions)])
            after = [json.loads(row) for row in predictions.read_text().splitlines()]
            labels = [json.loads(row)['class_label'] for row in test.read_text().splitlines()]
            for raw, safe in zip(before,after):
                self.assertEqual(raw['forecast'],safe['forecast'])
                self.assertEqual(raw['class_id'],safe['class_id'])
                if raw['series_gate_selected'] == 'base':
                    self.assertEqual(safe['series_gate_selected'],'base')
            hits = sum(row['series_gate_class_id'] == label for row,label in zip(after,labels))
            harms = sum(row['class_id'] == label and row['series_gate_class_id'] != label for row,label in zip(after,labels))
            benefits = sum(row['class_id'] != label and row['series_gate_class_id'] == label for row,label in zip(after,labels))
            eval_args = ['evaluate',*common,'--series-gate-guard',str(guard),'--guard-data',str(reference),
                         '--classifier-data',str(train),'--gate-data',str(fitting),
                         '--gate-validation-data',str(validation),'--iid-intervals','--output',str(report)]
            main(eval_args)
            result = json.loads(report.read_text())
            self.assertAlmostEqual(result['series_gate_accuracy'],hits/32)
            self.assertAlmostEqual(result['series_gate_guard_harm_rate'],harms/32)
            self.assertAlmostEqual(result['series_gate_guard_benefit_rate'],benefits/32)
            self.assertAlmostEqual(result['series_gate_gain_vs_base'],(benefits-harms)/32)
            self.assertEqual(result['iid_classification_confidence_intervals']['series_gate_guard_harm_rate'],
                             wilson_interval(harms,32))
            self.assertEqual(result['series_audited_sources'], ['tête','porte','validation porte','protection'])
            saved = guard.read_bytes()
            with self.assertRaisesRegex(ValueError,'identique|partagés'):
                main([*fit,'--data',str(fitting)])
            self.assertEqual(guard.read_bytes(),saved)
            with self.assertRaises(SystemExit):
                main([*fit,'--output',str(reference)])
            with self.assertRaises(SystemExit):
                main(['predict','--checkpoint',str(checkpoint),'--data',str(test),'--series-gate-guard',str(guard)])
            main([*fit,'--max-harm','.001'])
            self.assertIsNone(SeriesGateGuard.load(guard).threshold)
            main(['predict',*common,'--series-gate-guard',str(guard),'--output',str(predictions)])
            self.assertTrue(all(row['series_gate_selected'] == 'base' for row in
                                map(json.loads,predictions.read_text().splitlines())))
            altered = json.loads(gate.read_text()); altered['biases'][2][0] += 1
            gate.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,'checkpoint, la tête ou la porte'):
                main(['predict',*common,'--series-gate-guard',str(guard)])


if __name__ == '__main__':
    unittest.main()
