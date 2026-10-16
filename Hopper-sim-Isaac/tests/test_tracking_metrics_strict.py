"""CPU-only checks for the strict apex and foot landing joint metrics."""
from pathlib import Path
import runpy
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
TrackingMetrics = runpy.run_path(
    str(ROOT / 'Quadhopper_Planner_Random/tracking_metrics.py')
)['TrackingMetrics']


class StrictTrackingMetricsTests(unittest.TestCase):
    def test_strict_height_and_xy_must_hold_on_the_same_touchdown(self):
        metrics = TrackingMetrics(2, 'cpu')
        ids = torch.tensor([0, 1])
        zeros = torch.zeros(2)
        target = torch.ones(2)
        metrics.observe(ids, torch.tensor([True, True]),
                        torch.tensor([.04, .08]), zeros, zeros,
                        torch.tensor([1., 1.1]), target)
        metrics.observe(ids, torch.tensor([False, False]),
                        torch.tensor([.04, .08]), zeros, zeros,
                        torch.tensor([.9, 1.]), target)
        result = metrics.summary()
        self.assertEqual(result['valid_hit_10cm'], 1.)
        self.assertEqual(result['strict_apex_ge_target_rate'], .75)
        self.assertEqual(result['strict_valid_hit_10cm'], .75)
        self.assertEqual(result['strict_pair_valid_10cm'], .5)
        self.assertEqual(result['first_strict_valid_hit_10cm'], 1.)
        self.assertEqual(result['second_strict_valid_hit_10cm'], .5)
        self.assertEqual(result['pair_valid_10cm'], 1.)
        self.assertEqual(result['strict_valid_hit_5cm'], .25)

    def test_reset_breaks_strict_pair_without_erasing_started_denominator(self):
        metrics = TrackingMetrics(1, 'cpu')
        ids = torch.tensor([0])
        zeros = torch.zeros(1)
        def touchdown(short):
            metrics.observe(ids, torch.tensor([short]), torch.tensor([.04]),
                            zeros, zeros, torch.tensor([1.]), torch.ones(1))
        touchdown(True)
        metrics.reset(ids)
        touchdown(False)
        result = metrics.summary()
        self.assertEqual(result['pairs_started'], 1)
        self.assertEqual(result['pairs_completed'], 0)
        self.assertEqual(result['strict_valid_hit_10cm'], 1.)
        self.assertEqual(result['strict_pair_valid_10cm'], 0.)
        self.assertEqual(result['pair_valid_10cm'], 0.)

    def test_empty_summary_has_zero_joint_rates(self):
        result = TrackingMetrics(1, 'cpu').summary()
        self.assertEqual(result['strict_valid_hit_10cm'], 0.)
        self.assertEqual(result['strict_pair_valid_10cm'], 0.)


if __name__ == '__main__':
    unittest.main()
