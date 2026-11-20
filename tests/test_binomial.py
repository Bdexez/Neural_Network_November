from dataclasses import replace
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.binomial import wilson_interval
from chimera.data import synthetic_batch
from chimera.evaluation import evaluate


class BinomialIntervalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_known_intervals_and_boundary_cases(self):
        low, high = wilson_interval(5, 10)
        self.assertAlmostEqual(low, .2365930905, places=8)
        self.assertAlmostEqual(high, .7634069095, places=8)
        self.assertAlmostEqual(wilson_interval(0, 1)[0], 0.)
        self.assertAlmostEqual(wilson_interval(0, 1)[1], .793450686, places=8)
        self.assertAlmostEqual(wilson_interval(1, 1)[0], .206549314, places=8)
        self.assertAlmostEqual(wilson_interval(1, 1)[1], 1.)
        self.assertIsNone(wilson_interval(0, 0))
        for args in ((-1, 5), (6, 5), (0, -1), (1., 2), (True, 2), (1, 2, 1.), (1, 2, float('nan'))):
            with self.assertRaises(ValueError):
                wilson_interval(*args)

    def test_base_accuracy_and_no_labels(self):
        config = replace(ChimeraConfig.preset('tiny'), num_classes=3)
        model = Chimera(config)
        batch = synthetic_batch(config, batch_size=24, seed=213)
        batch = {k: v for k, v in batch.items() if k in ('series', 'series_mask', 'class_labels')}
        plain = evaluate(model, [batch])
        result = evaluate(model, [batch], iid_intervals=True)
        self.assertNotIn('iid_classification_confidence_intervals', plain)
        intervals = result['iid_classification_confidence_intervals']
        self.assertEqual(intervals['confidence'], .95)
        self.assertLessEqual(intervals['accuracy'][0], result['accuracy'])
        self.assertGreaterEqual(intervals['accuracy'][1], result['accuracy'])
        empty = dict(batch, class_labels=torch.full((24,), -100))
        no_labels = evaluate(model, [empty], iid_intervals=True)
        self.assertIsNone(no_labels['iid_classification_confidence_intervals']['accuracy'])


if __name__ == '__main__':
    unittest.main()
