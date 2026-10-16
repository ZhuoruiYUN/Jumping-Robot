import math
import unittest

import torch

from Quadhopper_Planner_Random.foot_pair_reward_v33 import FootPairRewardState


class FootPairRewardTest(unittest.TestCase):
    def setUp(self):
        self.state = FootPairRewardState(2, 'cpu')
        self.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                  [math.cos(math.radians(10)),
                                   math.sin(math.radians(10)), 0.0, 0.0]])

    def observe(self, *, short, error, apex=(1.0, 1.0), touchdown=(True, True)):
        return self.state.observe(
            touchdown=torch.tensor(touchdown),
            initial_drop=torch.zeros(2, dtype=torch.bool),
            phase_was_short=torch.tensor((short, short)),
            foot_error_m=torch.tensor(error),
            apex_m=torch.tensor(apex),
            target_apex_m=torch.ones(2),
            quat_wxyz=self.quat,
            pair_reward_scale=60.0,
        )

    def test_bonus_requires_both_strict_foot_hits_and_apexes(self):
        first, _, _ = self.observe(short=True, error=(0.04, 0.04))
        self.assertEqual(first.tolist(), [0.0, 0.0])
        second, _, _ = self.observe(short=False, error=(0.04, 0.12))
        self.assertAlmostEqual(second[0].item(), 60.0, places=5)
        self.assertEqual(second[1].item(), 0.0)
        # No retained first-hop credit after the second touchdown.
        third, _, _ = self.observe(short=False, error=(0.04, 0.04))
        self.assertEqual(third.tolist(), [0.0, 0.0])

    def test_posture_reduces_bonus_and_reset_clears_pair(self):
        self.observe(short=True, error=(0.04, 0.04))
        self.state.reset(torch.tensor([0]))
        bonus, _, _ = self.observe(short=False, error=(0.04, 0.04))
        self.assertEqual(bonus[0].item(), 0.0)
        self.assertGreater(bonus[1].item(), 0.0)
        self.assertLess(bonus[1].item(), 60.0)

    def test_low_first_apex_cannot_be_recovered_by_second_hit(self):
        self.observe(short=True, error=(0.04, 0.04), apex=(0.99, 0.99))
        bonus, _, _ = self.observe(short=False, error=(0.04, 0.04))
        self.assertEqual(bonus.tolist(), [0.0, 0.0])

    def test_pair_xy_quality_prefers_center_without_paying_misses(self):
        state = FootPairRewardState(3, 'cpu')
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
        def touchdown(short, errors, apexes=(1.0, 1.0, 1.0)):
            return state.observe(
                touchdown=torch.ones(3, dtype=torch.bool),
                initial_drop=torch.zeros(3, dtype=torch.bool),
                phase_was_short=torch.full((3,), short, dtype=torch.bool),
                foot_error_m=torch.tensor(errors),
                apex_m=torch.tensor(apexes),
                target_apex_m=torch.ones(3),
                quat_wxyz=quat,
                pair_reward_scale=60.0,
                pair_xy_quality_weight=0.5,
            )[0]
        touchdown(True, (0.01, 0.09, 0.01))
        bonus = touchdown(False, (0.01, 0.09, 0.11))
        self.assertGreater(bonus[0].item(), bonus[1].item())
        self.assertGreater(bonus[1].item(), 0.0)
        self.assertEqual(bonus[2].item(), 0.0)
        # Once P2 lands, stale P1 precision cannot pay on another P2.
        self.assertEqual(touchdown(False, (0.01, 0.01, 0.01)).tolist(), [0.0] * 3)


if __name__ == '__main__':
    unittest.main()
