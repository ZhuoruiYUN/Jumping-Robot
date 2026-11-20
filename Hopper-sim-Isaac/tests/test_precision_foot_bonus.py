import unittest

import torch

from benchmark37.obs73_jump_cycle import JumpCycleReward


class FootBonusTest(unittest.TestCase):
    def test_only_selected_strict_foot_landings_receive_the_missing_bonus(self):
        count = 6
        one = torch.ones(count)
        zero = torch.zeros(count)
        mask = torch.tensor([True, False, True, True, True, False])
        root = torch.tensor([.07, .07, .02, .07, .07, .02])
        foot = torch.tensor([.05, .04, .06, .04, .04, .04])
        peak = torch.tensor([1., 1., 1., .99, 1., 1.])
        tilt = torch.tensor([0., 0., 0., 0., 16., 0.])
        scales = dict(energy_tracking=12., distance_to_z=2., apex=8., distance_to_xy=15., goal_bonus=8.)

        def step(selected):
            cycle = JumpCycleReward(count, 'cpu', dict(strict_goal_bonus=True, energy_margin_m=.04, gamma=.99))
            return cycle.step(one.long()*3, one.long()*6, one*.003, one, zero, peak, one, root, foot,
                              tilt, zero.bool(), one.bool(), 9.81, .5, scales,
                              dict(apex_width_m=.15, tilt_width_deg=15., foot_width_m=.05, scale=6.),
                              .05, selected)

        legacy, landing, apex = step(None)
        revised, landing_after, apex_after = step(mask)
        self.assertEqual((revised['goal_bonus']>0).tolist(), [True, False, False, False, False, True])
        self.assertEqual((legacy['goal_bonus']>0).tolist(), [False, False, False, False, False, True])
        torch.testing.assert_close(legacy['goal_bonus'][1:], revised['goal_bonus'][1:])
        for name in legacy.keys()-{'goal_bonus'}:
            torch.testing.assert_close(legacy[name], revised[name])
        torch.testing.assert_close(landing, landing_after)
        torch.testing.assert_close(apex, apex_after)
