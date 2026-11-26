import math
import unittest

import torch

from benchmark37.moving_phase_reward import yaw_contact_jump_cost
from benchmark37.yaw_heading_servo import allocate_heading_servo
from benchmark37.yaw_thrust_feedback import _THRUST_OFFSET, _THRUST_SPAN, _thrust_from_duty


CONFIG = dict(position_gain_s2=36, rate_gain_s=10, max_yaw_mode_n=.14, collective_shift_max_n=.06)


class HeadingServoTests(unittest.TestCase):
    def test_policy_yaw_is_removed_at_the_heading_reference(self):
        actions = torch.tensor([[0.2, 0.8, -0.4, 0.6], [-.9, -.7, .1, -.2]])
        q = torch.tensor([[1., 0., 0., 0.]]).expand(2, -1)
        values = allocate_heading_servo(q, torch.zeros(2, 3), actions, 0., .02, CONFIG)
        thrust = _thrust_from_duty((1. + values) * .5)
        torch.testing.assert_close(thrust @ actions.new_tensor([-1, 1, -1, 1]), actions.new_zeros(2),
                                   atol=2e-7, rtol=0.)

    def test_roll_pitch_and_collective_budget_hold_at_motor_limits(self):
        actions = torch.cartesian_prod(*[torch.tensor([-1., 1.])]*4)
        q = torch.tensor([[math.cos(.15), 0., 0., math.sin(.15)]]).expand(len(actions), -1)
        omega = torch.tensor([[0., 0., 2.0]]).expand(len(actions), -1)
        values = allocate_heading_servo(q, omega, actions, 0., .02, CONFIG)
        self.assertTrue((values.abs() <= 1.).all())
        thrust = _thrust_from_duty((1. + values) * .5)
        raw = _thrust_from_duty((1. + actions) * .5)
        reserved = _THRUST_OFFSET + .02 + (raw - _THRUST_OFFSET) * (1. - .04 / _THRUST_SPAN)
        roll_pitch = actions.new_tensor([[1., -1., -1., 1.], [1., 1., -1., -1.]])
        torch.testing.assert_close(thrust @ roll_pitch.T, reserved @ roll_pitch.T, atol=2e-7, rtol=0.)
        self.assertTrue(((thrust - reserved).mean(-1).abs() <= .060001).all())

    def test_contact_cost_marks_the_compression_edge_not_continuous_yaw(self):
        before = torch.tensor([-.2, -.2, -.2, -.2, -.2])
        after = torch.tensor([-2.45, -2.45, -2.45, -.35, 2.05])
        previously_loaded = torch.tensor([False, True, False, False, False])
        loaded = torch.tensor([True, True, False, True, True])
        result = yaw_contact_jump_cost(before, after, previously_loaded, loaded,
                                       dict(weight=6.0, free_rate_jump_rad_s=.45))
        torch.testing.assert_close(result, torch.tensor([-15.6, 0., 0., 0., -15.6]))


if __name__ == '__main__':
    unittest.main()
