from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.binomial import wilson_interval
from chimera.cli import main
from chimera.data import synthetic_batch
from chimera.training import save_checkpoint
from chimera.series_gate import SeriesGate, fit_series_gate, gate_features
from chimera.series_classifier import fit_series_classifier
from chimera.evaluation import evaluate


class StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = replace(ChimeraConfig.preset('tiny'), series_features=1, num_classes=2)

    def forward(self, *, series, series_mask=None):
        logits = torch.tensor([1., 0.], device=series.device).expand(series.size(0), -1)
        return SimpleNamespace(classification_logits=logits)


class StubClassifier:
    num_classes, series_features, amp = 2, 1, 'none'
    checkpoint_sha256, dataset_sha256 = 'a'*64, 'b'*64
    def validate(self, config, *, amp=None):
        if amp != self.amp or config.num_classes != 2:
            raise ValueError('incompatible')
    def scores(self, model, batch):
        return torch.tensor([0., 1.], device=batch['series'].device).expand(batch['series'].size(0), -1)


class SeriesGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def batches(seed, count=120):
        rng = torch.Generator().manual_seed(seed)
        sign = (torch.rand(count, generator=rng) > .5).long()
        values = (sign.float() * 2 - 1).unsqueeze(1).unsqueeze(2).expand(-1, 4, 1).clone()
        return [dict(series=values, series_mask=torch.ones(count, 4, dtype=torch.bool), class_labels=sign)]

    def test_fit_learns_complementary_experts_and_preserves_model_and_rng(self):
        model, head = StubModel().train(), StubClassifier()
        model.nested = torch.nn.Dropout().eval()
        original = {key:value.clone() for key,value in model.state_dict().items()}
        state = torch.get_rng_state().clone()
        kwargs = dict(checkpoint_sha256='a'*64, classifier_sha256='c'*64,
                      dataset_sha256='d'*64, validation_sha256='e'*64, epochs=20, seed=42)
        gate = fit_series_gate(model, head, self.batches(1), self.batches(2), **kwargs)
        self.assertTrue(model.training)
        self.assertFalse(model.nested.training)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        self.assertGreater(gate.validation_gain, .35)
        self.assertEqual(gate.train_examples, 120)
        self.assertEqual((gate.epochs, gate.fit_batch_size, gate.seed), (20, 128, 42))
        repeated = fit_series_gate(model, head, self.batches(1), self.batches(2), **kwargs)
        self.assertEqual(repeated.to_dict(), gate.to_dict())
        with self.assertRaisesRegex(ValueError, 'taille des lots'):
            fit_series_gate(model, head, self.batches(1), self.batches(2), **dict(kwargs, feature_batch_size=100))
        self.assertTrue(model.training)
        self.assertFalse(model.nested.training)
        model.eval()
        heldout = self.batches(3)[0]
        inputs = {k:v for k,v in heldout.items() if k != 'class_labels'}
        logits, scores = model(**inputs).classification_logits, head.scores(model, inputs)
        prediction, selected, advantage = gate.choose(inputs, logits, scores)
        changed_labels = dict(inputs, class_labels=1 - heldout['class_labels'])
        torch.testing.assert_close(gate.advantage(changed_labels, logits, scores), advantage, rtol=0, atol=0)
        self.assertGreater((prediction == heldout['class_labels']).float().mean().item(), .95)
        self.assertEqual(selected.tolist(), (advantage > gate.threshold).tolist())
        restored = SeriesGate.from_dict(json.loads(json.dumps(gate.to_dict())))
        torch.testing.assert_close(restored.advantage(inputs, logits, scores), advantage, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            SeriesGate.from_dict(dict(gate.to_dict(), weights=[]))
        with self.assertRaises(ValueError):
            SeriesGate.from_dict(dict(gate.to_dict(), threshold=-1))
        with self.assertRaisesRegex(ValueError, 'distincts'):
            fit_series_gate(model, head, self.batches(1), self.batches(2),
                            **dict(kwargs, validation_sha256='d'*64))
        with self.assertRaisesRegex(ValueError, 'étiquetés'):
            fit_series_gate(model, head, self.batches(1), [dict(series=torch.zeros(3, 4, 1))], **kwargs)

    def test_fallback_on_validation_and_features_ignore_missing_values(self):
        model, head = StubModel(), StubClassifier()
        data = self.batches(1)
        val = self.batches(2)
        val[0]['class_labels'] = torch.zeros(120, dtype=torch.long)
        kwargs = dict(checkpoint_sha256='a'*64, classifier_sha256='c'*64, epochs=5)
        gate = fit_series_gate(model, head, data, val, **kwargs)
        self.assertEqual(gate.validation_gain, 0.)
        self.assertEqual(gate.best_epoch, 0)
        model.eval()
        inp = dict(series=torch.tensor([[[1.], [float('nan')], [999.]]]), series_mask=torch.tensor([[1, 1, 0]]))
        logits, scores = model(**inp).classification_logits, head.scores(model, inp)
        chosen, selected, _ = gate.choose(inp, logits, scores)
        self.assertEqual(chosen.tolist(), [0])
        self.assertFalse(selected.any())
        feat = gate_features(inp, logits, scores)
        self.assertEqual(feat[0,4].item(), 1.)
        inp['series'][0,2,0] = -999
        torch.testing.assert_close(feat, gate_features(inp, logits, scores), rtol=0, atol=0)


    def test_bf16_gate_and_unlabeled_evaluation(self):
        config = ChimeraConfig.preset('tiny')
        model = Chimera(config).train()
        def batch(seed):
            return {key:value for key,value in synthetic_batch(config, batch_size=64, seed=seed).items()
                    if key in ('series','series_mask','class_labels')}
        head = fit_series_classifier(model, [batch(801)], amp='bf16', checkpoint_sha256='a'*64)
        gate = fit_series_gate(model, head, [batch(802)], [batch(803)], amp='bf16', epochs=2,
                               checkpoint_sha256='a'*64, classifier_sha256='c'*64)
        self.assertTrue(model.training)
        data = batch(804)
        report = evaluate(model, [data], amp='bf16', series_classifier=head, series_gate=gate)
        self.assertEqual(report['series_gate_labeled_examples'],64)
        model.eval()
        inputs = {key:value for key,value in data.items() if key != 'class_labels'}
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
            logits = model(**inputs).classification_logits
            scores = head.scores(model, inputs)
            under_amp = gate.advantage(inputs, logits, scores)
        outside_amp = gate.advantage(inputs, logits, scores)
        torch.testing.assert_close(under_amp, outside_amp, rtol=0, atol=0)
        empty = evaluate(model, [dict(data, class_labels=torch.full((64,),-100))], amp='bf16',
                         series_classifier=head, series_gate=gate)
        self.assertEqual(empty['series_gate_examples'],64)
        self.assertIsNone(empty['series_gate_accuracy'])
        self.assertIsNone(empty['series_gate_gain_vs_base'])
        with self.assertRaisesRegex(ValueError, 'précision'):
            evaluate(model, [data], series_classifier=head, series_gate=gate)

    def test_cli_fit_predict_evaluate_and_file_protection(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            config = ChimeraConfig.preset('tiny')
            checkpoint, head, gate = (root / name for name in ('model.pt', 'head.json', 'gate.json'))
            train, fitting, validation, test = (root / (name + '.jsonl') for name in ('train', 'fit', 'validation', 'test'))
            report, predictions = root / 'report.json', root / 'predictions.jsonl'
            save_checkpoint(checkpoint, Chimera(config))
            for path, n, seed in ((train,128,951), (fitting,96,952), (validation,64,953), (test,24,954)):
                data = synthetic_batch(config, batch_size=n, seed=seed)
                path.write_text('\n'.join(json.dumps(dict(series=data['series'][i].tolist(),
                    class_label=data['class_labels'][i].item(), future=data['forecast_targets'][i].tolist())) for i in range(n)))
            main(['--threads','2','refine-series-classifier','--checkpoint',str(checkpoint),'--data',str(train),'--output',str(head)])
            fit = ['refine-series-gate','--checkpoint',str(checkpoint),'--series-classifier',str(head),
                   '--data',str(fitting),'--validation-data',str(validation),'--output',str(gate),
                   '--classifier-data',str(train),'--audit-series','--epochs','5']
            main(fit)
            fitted = SeriesGate.load(gate)
            self.assertEqual(fitted.train_examples,96)
            self.assertEqual(fitted.validation_examples,64)
            common = ['--checkpoint',str(checkpoint),'--series-classifier',str(head),'--data',str(test)]
            main(['predict',*common,'--output',str(predictions)])
            before = [json.loads(row) for row in predictions.read_text().splitlines()]
            main(['predict',*common,'--series-gate',str(gate),'--output',str(predictions)])
            after = [json.loads(row) for row in predictions.read_text().splitlines()]
            for left, right in zip(before,after):
                self.assertEqual(left['forecast'],right['forecast'])
                self.assertEqual(left['class_id'],right['class_id'])
                self.assertEqual(left['series_classifier_class_id'],right['series_classifier_class_id'])
                expected = right['series_classifier_class_id'] if right['series_gate_selected'] == 'series_classifier' else right['class_id']
                self.assertEqual(right['series_gate_class_id'],expected)
            main(['evaluate',*common,'--series-gate',str(gate),'--gate-data',str(fitting),
                  '--gate-validation-data',str(validation),'--classifier-data',str(train),
                  '--iid-intervals','--output',str(report)])
            measured = json.loads(report.read_text())
            truth = [json.loads(row)['class_label'] for row in test.read_text().splitlines()]
            gate_accuracy = sum(row['series_gate_class_id'] == label for row,label in zip(after,truth))/len(truth)
            self.assertEqual(measured['series_gate_labeled_examples'],24)
            self.assertAlmostEqual(measured['series_gate_accuracy'],gate_accuracy)
            hits = sum(row['series_gate_class_id'] == label for row,label in zip(after,truth))
            self.assertEqual(measured['iid_classification_confidence_intervals']['series_gate_accuracy'],
                             wilson_interval(hits, len(truth)))
            self.assertAlmostEqual(measured['series_gate_gain_vs_base'],gate_accuracy-measured['accuracy'])
            self.assertEqual(measured['series_audited_sources'],['tête','porte','validation porte'])
            saved = gate.read_bytes()
            for bad_args in (['--validation-data',str(fitting)], ['--data',str(train)]):
                with self.assertRaisesRegex(ValueError,'distincts|partagés'):
                    main([*fit,*bad_args])
                self.assertEqual(gate.read_bytes(),saved)
            with self.assertRaises(SystemExit):
                main([*fit,'--output',str(validation)])
            with self.assertRaises(SystemExit):
                main(['predict','--checkpoint',str(checkpoint),'--data',str(test),'--series-gate',str(gate)])
            altered = json.loads(head.read_text()); altered['bias'][0] += 1
            head.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,'checkpoint ou à la tête'):
                main(['predict',*common,'--series-gate',str(gate)])


if __name__ == '__main__':
    unittest.main()
