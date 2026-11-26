import math
import unittest

import torch

from benchmark37.yaw_thrust_feedback import apply_yaw_feedback


def obs73(yaw, rate=0.0, roll=0.0):
    obs = torch.zeros(1, 73, dtype=torch.float64)
    yaw_half, roll_half = math.radians(yaw) * 0.5, math.radians(roll) * 0.5
    cy, sy = math.cos(yaw_half), math.sin(yaw_half)
    cr, sr = math.cos(roll_half), math.sin(roll_half)
    obs[0, 9:13] = obs.new_tensor([cy * cr, cy * sr, sy * sr, sy * cr])
    obs[0, 15] = math.radians(rate)
    return obs


def commanded_thrust(actions):
    duty = 0.5 * (actions + 1.0)
    return -0.3647607454907179 * duty.square() + 0.9884448461365533 * duty + 0.0028762422544365922


class YawThrustFeedbackTest(unittest.TestCase):
    def test_priority_removes_opposing_yaw_without_changing_other_mixer_channels(self):
        actions = torch.tensor([[-0.2, 0.2, -0.2, 0.2]], dtype=torch.float64).repeat(2, 1)
        obs = torch.cat([obs73(9.0, 25.0), obs73(9.0, -25.0)])
        priority = dict(horizon_s=0.14, warning_deg=4.0, full_deg=8.0)
        legacy = apply_yaw_feedback(obs, actions, reserve_n=0.02)
        amended = apply_yaw_feedback(obs, actions, reserve_n=0.02, priority=priority)
        delta = commanded_thrust(amended) - commanded_thrust(legacy)
        mixer = actions.new_tensor([[1, 1, 1, 1], [1, -1, -1, 1], [1, 1, -1, -1]])
        torch.testing.assert_close(delta @ mixer.T, actions.new_zeros(2, 3), atol=1e-14, rtol=0)
        self.assertLess(delta[0] @ actions.new_tensor([-1, 1, -1, 1]), 0.0)
        torch.testing.assert_close(amended[1], legacy[1], rtol=0, atol=0)

    def test_preserves_commanded_total_thrust_and_roll_pitch(self):
        actions = torch.tensor([[0.11, -0.35, 0.29, 0.61]], dtype=torch.float64)
        corrected = apply_yaw_feedback(obs73(6.0, 8.0), actions)
        difference = commanded_thrust(corrected) - commanded_thrust(actions)
        force_roll_pitch = actions.new_tensor([[1, 1, 1, 1], [1, -1, -1, 1], [1, 1, -1, -1]])
        torch.testing.assert_close(difference @ force_roll_pitch.T, actions.new_zeros(1, 3), atol=1e-14, rtol=0)
        self.assertLess((difference @ actions.new_tensor([-1, 1, -1, 1])).item(), 0)

    def test_rate_damping_direction_and_soft_position_deadzone(self):
        actions = torch.zeros(3, 4, dtype=torch.float64)
        obs = torch.cat([obs73(0.0, -9.0), obs73(0.0, 9.0), obs73(1.0)])
        corrected = apply_yaw_feedback(obs, actions)
        contrast = commanded_thrust(corrected) @ actions.new_tensor([-1, 1, -1, 1])
        self.assertGreater(contrast[0].item(), 0)
        torch.testing.assert_close(contrast[0], -contrast[1], atol=1e-14, rtol=0)
        torch.testing.assert_close(corrected[2], actions[2], atol=0, rtol=0)

    def test_reference_heading_wraps_across_half_turn(self):
        actions = torch.zeros(1, 4, dtype=torch.float64)
        equivalent = apply_yaw_feedback(obs73(4.0), actions)
        wrapped = apply_yaw_feedback(obs73(-177.0), actions, target_heading_rad=math.radians(179.0))
        torch.testing.assert_close(equivalent, wrapped, atol=1e-14, rtol=0)

    def test_saturation_uses_smallest_headroom_in_both_directions(self):
        actions = torch.tensor([[0.2, 0.98, 0.0, 0.0], [0.0, -0.98, 0.0, 0.3]], dtype=torch.float64)
        obs = torch.cat([obs73(-40.0), obs73(40.0)])
        result = apply_yaw_feedback(obs, actions)
        torch.testing.assert_close(result[:, 1], actions.new_tensor([1.0, -1.0]), atol=1e-14, rtol=0)
        delta = commanded_thrust(result) - commanded_thrust(actions)
        torch.testing.assert_close(delta.abs(), delta[:, :1].abs().expand_as(delta), atol=1e-14, rtol=0)

    def test_per_motor_correction_cap_and_inverted_fade(self):
        actions = torch.zeros(4, 4, dtype=torch.float64)
        obs = torch.cat([obs73(40.0), obs73(40.0, roll=42.0), obs73(40.0, roll=60.0),
                         obs73(40.0, roll=170.0)])
        corrected = apply_yaw_feedback(obs, actions)
        correction = (commanded_thrust(corrected) - commanded_thrust(actions)).abs()
        torch.testing.assert_close(correction[0], actions.new_full((4,), 0.05), atol=1e-14, rtol=0)
        # With this roll and zero rate the correction is attenuated below the cap.
        self.assertTrue(torch.all((0.0 < correction[1]) & (correction[1] < 0.05)))
        torch.testing.assert_close(corrected[2:], actions[2:], atol=0, rtol=0)

    def test_reserve_guarantees_correction_at_every_motor_limit(self):
        limits = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        native = torch.cartesian_prod(limits, limits, limits, limits)
        original = native.clone()
        reserved = commanded_thrust(apply_yaw_feedback(obs73(0).expand(16, -1), native, reserve_n=0.02))
        mode = native.new_tensor([-1, 1, -1, 1])
        for demand in (-90.0, 90.0):
            with self.subTest(rate=demand):
                result = apply_yaw_feedback(obs73(0, demand).expand(16, -1), native, reserve_n=0.02)
                difference = commanded_thrust(result) - reserved
                self.assertTrue(torch.all(result.abs() <= 1.0))
                magnitude = difference[:, :1].abs()
                torch.testing.assert_close(difference, -math.copysign(1, demand) * magnitude * mode,
                                           atol=1e-14, rtol=0)
                self.assertTrue(torch.all(magnitude >= 0.02 - 1e-14))
        torch.testing.assert_close(native, original, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
