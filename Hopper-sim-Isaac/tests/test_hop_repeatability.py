import math
import unittest

import torch

from benchmark37.hop_repeatability import HopRepeatability, straight_pair


CONFIG = {
    'version': 'height_phase_matched_ab_v1',
    'sample_heights_m': [.40, .55, .75, .90],
    'max_cost_per_sample': .4,
}
FLIGHT = (
    (.47, .12), (.64, .21), (.81, .31), (.96, .41), (1.03, .52),
    (.86, .74), (.69, .83), (.48, .92), (.37, 1.02),
)


class HopRepeatabilityTest(unittest.TestCase):
    def setUp(self):
        self.tracker = HopRepeatability(1, 'cpu', CONFIG)

    def sample(self, height, age, *, pitch_deg=5., roll_deg=0., launch=False, free=True,
               contact=False, clean_a=False, record_a=False, eligible_b=False):
        half_pitch = math.radians(pitch_deg) / 2.
        half_roll = math.radians(roll_deg) / 2.
        pitch_cos, pitch_sin = math.cos(half_pitch), math.sin(half_pitch)
        roll_cos, roll_sin = math.cos(half_roll), math.sin(half_roll)
        cost, markers = self.tracker.step(
            torch.tensor([height]), torch.tensor([age]),
            torch.tensor([[pitch_cos * roll_cos, pitch_cos * roll_sin,
                           pitch_sin * roll_cos, -pitch_sin * roll_sin]]),
            torch.tensor([[0., .2, 0.]]), torch.tensor([[.55, .03]]), torch.tensor([[.5, 0.]]),
            torch.tensor([launch]), torch.tensor([free]), torch.tensor([contact]),
            torch.tensor([clean_a]), torch.tensor([record_a]), torch.tensor([eligible_b]),
        )
        return cost.item(), markers.item()

    def record_a(self, clean=True):
        self.sample(.31, .0, launch=True, record_a=True)
        for height, age in FLIGHT:
            self.assertEqual(self.sample(height, age), (0., 0))
        self.sample(.29, 1.08, free=False, contact=True, clean_a=clean)

    def start_b(self, **attitude):
        return self.sample(.31, .0, launch=True, eligible_b=True, **attitude)

    def test_healthy_equal_hops_compare_release_and_eight_height_markers_without_cost(self):
        self.record_a()
        self.assertTrue(self.tracker.ready.item())
        launch_sample = self.start_b()
        self.assertEqual(launch_sample, (0., 1))
        samples = [launch_sample] + [self.sample(height, age, eligible_b=True) for height, age in FLIGHT]
        self.assertEqual(sum(markers for _, markers in samples), 9)
        self.assertTrue(all(cost == 0. for cost, _ in samples))

    def test_mismatched_release_pitch_and_roll_are_both_debited(self):
        for attitude in ({'pitch_deg': 14.}, {'roll_deg': 10.}):
            with self.subTest(attitude=attitude):
                self.tracker.reset(slice(None))
                self.record_a()
                cost, markers = self.start_b(**attitude)
                self.assertLess(cost, 0.)
                self.assertEqual(markers, 1)
                self.assertEqual(self.sample(.35, .04, eligible_b=True, **attitude), (0., 0))

    def test_mismatched_attitude_or_timing_debits_each_marker_only_once(self):
        for mismatch in ('attitude', 'timing'):
            with self.subTest(mismatch=mismatch):
                self.tracker.reset(slice(None))
                self.record_a()
                self.assertEqual(self.start_b(), (0., 1))
                samples = [self.sample(height, age + (.14 if mismatch == 'timing' else 0.),
                                       pitch_deg=14. if mismatch == 'attitude' else 5., eligible_b=True)
                           for height, age in FLIGHT]
                self.assertEqual(sum(markers for _, markers in samples), 8)
                self.assertLess(sum(cost for cost, _ in samples), 0.)
                self.assertEqual(self.sample(.83, 1.10, pitch_deg=18., eligible_b=True), (0., 0))
                self.assertEqual(self.sample(.37, 1.23, pitch_deg=18., eligible_b=True), (0., 0))

    def test_straight_gate_excludes_holds_reverse_turns_and_unequal_steps(self):
        edge_a = torch.tensor([[.5, 0.]] * 7)
        edge_b = torch.tensor([[.5, 0.], [.48, .01], [0., 0.], [-.5, 0.],
                               [0., .5], [.40, 0.], [.42, .27]])
        self.assertEqual(straight_pair(edge_a, edge_b).tolist(),
                         [True, True, False, False, False, False, False])
        self.assertFalse(straight_pair(torch.zeros_like(edge_a), edge_b).any().item())

    def test_unscored_short_b_rebound_cannot_restart_the_template(self):
        self.record_a()
        self.assertEqual(self.start_b(), (0., 1))
        self.sample(.30, .03, free=False, contact=True, eligible_b=True)
        self.assertEqual(self.start_b(pitch_deg=18.), (0., 0))
        self.assertEqual(self.sample(.50, .09, pitch_deg=18., eligible_b=True), (0., 0))

    def test_reset_or_failed_a_prevents_reuse_of_recorded_markers(self):
        for clean in (False, True):
            with self.subTest(clean=clean):
                self.record_a(clean=clean)
                self.assertEqual(self.tracker.ready.item(), clean)
                if clean:
                    self.tracker.reset(torch.tensor([0]))
                    self.assertFalse(self.tracker.seen.any().item())
                    self.assertFalse(self.tracker.ready.item())
                self.assertEqual(self.start_b(), (0., 0))
                for height, age in FLIGHT:
                    self.assertEqual(self.sample(height, age, pitch_deg=14., eligible_b=True), (0., 0))


if __name__ == '__main__':
    unittest.main()
