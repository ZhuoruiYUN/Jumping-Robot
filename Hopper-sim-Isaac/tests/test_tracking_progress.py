import unittest

import torch

from Quadhopper_Planner_Random.tracking_progress import LandingXYProgress, two_hop_pair_success


class LandingXYProgressTest(unittest.TestCase):
    def test_pair_bonus_requires_second_phase_and_two_consecutive_hits(self):
        final = torch.tensor([False, True, True, True, True])
        hit = torch.tensor([True, True, False, True, True])
        streak = torch.tensor([2, 1, 2, 2, 5])
        torch.testing.assert_close(
            two_hop_pair_success(final, hit, streak),
            torch.tensor([False, False, False, True, True]),
        )

    def test_signed_approach_retreat_and_hover(self):
        progress = LandingXYProgress(3, 'cpu')
        active = torch.ones(3, dtype=torch.bool)
        torch.testing.assert_close(progress.update(torch.full((3,), .2), active), torch.zeros(3))
        torch.testing.assert_close(
            progress.update(torch.tensor([.19, .21, .2]), active),
            torch.tensor([.01, -.01, 0.]),
        )

    def test_round_trip_at_different_speeds_has_no_free_reward(self):
        progress = LandingXYProgress(1, 'cpu')
        active = torch.tensor([True])
        # A large approach followed by slow retreat (and vice versa) must cancel.
        # Per-step delta clipping would incorrectly produce a non-zero sum.
        for distances in ([.2, .1, .12, .15, .2], [.2, .15, .12, .1, .2]):
            total = sum(progress.update(torch.tensor([d]), active).item() for d in distances)
            self.assertAlmostEqual(total, 0., places=7)

    def test_replan_cannot_reward_closer_target_or_penalize_farther_target(self):
        progress = LandingXYProgress(2, 'cpu')
        active = torch.ones(2, dtype=torch.bool)
        progress.update(torch.full((2,), .2), active)
        target_distances = torch.tensor([.01, .9])
        progress.reanchor(torch.arange(2), target_distances)
        torch.testing.assert_close(progress.update(target_distances, active), torch.zeros(2))
        torch.testing.assert_close(
            progress.update(target_distances - .005, active), torch.full((2,), .005)
        )

    def test_inactive_drop_and_stance_do_not_leak_into_flight(self):
        progress = LandingXYProgress(1, 'cpu')
        for d in (.5, .4, .3):
            self.assertEqual(progress.update(torch.tensor([d]), torch.tensor([False])).item(), 0.)
        # First airborne sample discards movement during the last inactive step.
        self.assertEqual(progress.update(torch.tensor([.2]), torch.tensor([True])).item(), 0.)
        self.assertAlmostEqual(progress.update(torch.tensor([.19]), torch.tensor([True])).item(), .01)
        self.assertEqual(progress.update(torch.tensor([.15]), torch.tensor([False])).item(), 0.)
        self.assertEqual(progress.update(torch.tensor([.1]), torch.tensor([True])).item(), 0.)

    def test_partial_reset_preserves_other_environments(self):
        progress = LandingXYProgress(2, 'cpu')
        active = torch.ones(2, dtype=torch.bool)
        progress.update(torch.full((2,), .2), active)
        progress.reanchor(torch.tensor([0]), torch.tensor([1.]))
        torch.testing.assert_close(
            progress.update(torch.tensor([1., .19]), active), torch.tensor([0., .01])
        )
        torch.testing.assert_close(
            progress.update(torch.tensor([.99, .18]), active), torch.full((2,), .01)
        )


if __name__ == '__main__':
    unittest.main()
