import json
import unittest
from pathlib import Path

import torch

from benchmark37.apex_governor import (
    limit_ascent_collective, overheight_debit, projected_landing_margin, validate as validate_apex,
)
from benchmark37.cross_rebound_control import correct_cross_rebound, validate as validate_cross
from benchmark37.yaw_thrust_feedback import _thrust_from_duty


COURSE = Path(__file__).resolve().parents[1] / 'training_config_v51/ab_transverse_apex/course.json'


class HopControlTest(unittest.TestCase):
    def setUp(self):
        self.course = json.loads(COURSE.read_text())

    def test_cross_moment_only_and_gate_passthrough(self):
        config = self.course['cross_rebound_control']
        validate_cross(config)
        actions = torch.tensor([[.2, .1, .4, .5]] * 2, dtype=torch.float64)
        quat = actions.new_tensor([[1., 0., 0., 0.]] * 2)
        angular_velocity = actions.new_tensor([[0., .5, 0.]] * 2)
        cross_axis = actions.new_tensor([[1., 0.]] * 2)
        result, _, strength = correct_cross_rebound(
            actions, quat, angular_velocity, cross_axis, actions.new_zeros(2), actions.new_tensor([1., 0.]), config)
        delta = _thrust_from_duty(.5 * (result + 1.)) - _thrust_from_duty(.5 * (actions + 1.))
        self.assertLess(strength[0].item(), 0.)
        torch.testing.assert_close(delta[0], strength[0] * actions.new_tensor([1., 1., -1., -1.]))
        torch.testing.assert_close(result[1], actions[1], rtol=0., atol=0.)

    def test_apex_trim_preserves_force_differentials_and_leaves_inactive_ticks(self):
        config = self.course['apex_governor']
        validate_apex(config)
        actions = torch.tensor([[.2, .1, .4, .5]] * 4, dtype=torch.float64)
        height = actions.new_tensor([.95, .95, .95, .45])
        vz = actions.new_tensor([1.3, 1.3, -1.3, 1.3])
        gate = actions.new_tensor([1., 0., 1., 1.])
        mass = actions.new_full((4,), .1556)
        gain = actions.new_tensor([.93, 1., 1., 1.])
        result, _, trim = limit_ascent_collective(
            actions, height, vz, torch.ones_like(mass), gate, mass, torch.ones_like(mass), gain, config,
            ascent_drag_n_s2_m2=actions.new_full((4,), .10))
        delta = _thrust_from_duty(.5 * (result[0] + 1.)) - _thrust_from_duty(.5 * (actions[0] + 1.))
        torch.testing.assert_close(delta, delta.mean().expand_as(delta))
        self.assertGreater(trim[0].item(), 0.)
        self.assertLessEqual(trim[0].item(), config['max_collective_trim_n'])
        torch.testing.assert_close(result[1:], actions[1:], rtol=0., atol=0.)

    def test_apex_trim_waits_for_coast_energy_to_cover_minimum_height(self):
        actions = torch.tensor([[.70, .55, .50, .65]], dtype=torch.float64)
        scalar = actions.new_tensor
        result, _, trim = limit_ascent_collective(
            actions, scalar([.85]), scalar([1.25]), scalar([1.]), scalar([1.]), scalar([.1556]),
            scalar([1.]), scalar([1.]), self.course['apex_governor'],
            ascent_drag_n_s2_m2=scalar([.10]))
        torch.testing.assert_close(result, actions, rtol=0., atol=0.)
        self.assertEqual(trim.item(), 0.)

    def test_apex_trim_yields_to_travel_without_landing_margin(self):
        edge = torch.tensor([[.5, 0.], [-.5, 0.], [0., 0.]])
        offset = torch.tensor([[-.30, 0.], [.30, 0.], [0., 0.]])
        velocity = torch.tensor([[.25, 0.], [-.56, 0.], [0., 0.]])
        margin = projected_landing_margin(edge, offset, velocity, torch.full((3,), .75))
        torch.testing.assert_close(margin, torch.tensor([0., 1., 1.]))

    def test_original_landing_margin_keeps_the_same_motor_commands(self):
        cfg = self.course['apex_governor']
        actions = torch.tensor([[.23, .16, .31, .26]])
        one = torch.ones(1)
        height, speed, gate, margin = .96 * one, 1.5 * one, .65 * one, .43 * one
        plant = (.1556 * one, .95 * one, .92 * one)
        before = limit_ascent_collective(actions, height, speed, one, gate * margin, *plant, cfg,
                                         ascent_drag_n_s2_m2=.09 * one)
        after = limit_ascent_collective(actions, height, speed, one, gate, *plant, cfg,
                                        ascent_drag_n_s2_m2=.09 * one, landing_margin=margin)
        for expected, actual in zip(before, after, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)

    def test_severe_overheight_bypasses_landing_margin_only_above_the_lower_energy_gate(self):
        config = dict(self.course['apex_governor'], override_start_apex_m=1.11,
                      override_full_apex_m=1.20)
        validate_apex(config)
        actions = torch.tensor([[.42, .35, .36, .38]] * 3, dtype=torch.float64)
        height = actions.new_tensor([1.04, .84, .96])
        vz = actions.new_tensor([1.65, 1.25, -1.45])
        one = torch.ones_like(height)
        limited, _, trim = limit_ascent_collective(
            actions, height, vz, one, one, .1556 * one, one, one, config,
            ascent_drag_n_s2_m2=.10 * one, landing_margin=0. * one, applied_thrust=1.4 * one)
        self.assertGreater(trim[0].item(), 0.)
        self.assertLessEqual(trim[0].item(), config['max_collective_trim_n'] + 1e-12)
        torch.testing.assert_close(limited[1:], actions[1:], rtol=0., atol=0.)
        change = _thrust_from_duty(.5 * (limited[0] + 1.)) - _thrust_from_duty(.5 * (actions[0] + 1.))
        torch.testing.assert_close(change, change.mean().expand_as(change))

    def test_overheight_debit_telescopes_and_does_not_charge_the_drop(self):
        config = {'upper_m': 1.10, 'soft_band_m': .02, 'cost_per_m': 32.}
        height = torch.tensor([1.12, 1.20, 1.14, 1.21])
        previous_peak = torch.tensor([1.08, 1.12, 1.20, 0.])
        cost = overheight_debit(height, previous_peak, torch.tensor([True, True, True, False]), config)
        single = overheight_debit(height[1], previous_peak[0], torch.tensor(True), config)
        torch.testing.assert_close(cost.sum(), single)
        torch.testing.assert_close(cost[2:], torch.zeros(2))


if __name__ == '__main__':
    unittest.main()
