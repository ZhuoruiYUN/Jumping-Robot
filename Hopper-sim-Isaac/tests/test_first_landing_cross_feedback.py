import unittest

import torch

from benchmark37.first_landing_cross_feedback import first_landing_cross_feedback, validate


CONFIG = {
    'version': 'late_ascent_first_landing_cross_v1',
    'minimum_step_m': .40,
    'start_elapsed_s': .25,
    'full_elapsed_s': .35,
    'minimum_remaining_s': .20,
    'full_remaining_s': .30,
    'ascent_start_vz_m_s': .9,
    'ascent_full_vz_m_s': .3,
    'descent_cutoff_m_s': .55,
    'maximum_acceleration_m_s2': .55,
    'error_width_m': .03,
    'forecast_weight': 1.,
}


class FirstLandingCrossFeedbackTest(unittest.TestCase):
    def test_does_not_shape_early_release_descent_short_steps_or_inactive_lanes(self):
        validate(CONFIG)
        foot = torch.tensor([[.5, .12]] * 9)
        velocity = torch.tensor([[.1, .1, .0]] * 9)
        velocity[4, 2] = -.55
        edge = torch.tensor([[.5, .0]] * 9)
        edge[3, 0] = .40 - 5.e-7
        edge[5] = torch.tensor([.35, .0])
        edge[8] = 0.
        acceleration = torch.tensor([[.0, -.5]] * 9)
        eligible = torch.tensor([True, True, True, True, True, True, False, True, True])
        elapsed = torch.tensor([.1, .25, .30, .35, .5, .5, .5, .5, .5])
        remaining = torch.tensor([.40, .40, .40, .40, .40, .40, .40, .2, .40])
        target = torch.tensor([[.5, .0]] * 9)
        cost, discount, audit = first_landing_cross_feedback(
            foot, velocity, target, edge, elapsed, remaining, eligible, acceleration, .05, .01, CONFIG,
        )
        torch.testing.assert_close(audit['gate'], torch.tensor([0., 0., .5, 1., 0., 0., 0., 0., 0.]))
        torch.testing.assert_close(audit['active'], audit['gate'] > 0)
        torch.testing.assert_close(cost[~audit['active']], torch.zeros(7))
        torch.testing.assert_close(discount[~audit['active']], torch.zeros(7))
        self.assertLessEqual(float(cost[3]), float(cost[2]))
        self.assertLess(float(cost[2]), 0.)
        self.assertGreaterEqual(float(cost[3]), -.01)

    def test_fades_in_only_after_vertical_speed_drops(self):
        velocity = torch.zeros(5, 3)
        velocity[:, 2] = torch.tensor([1.2, .9, .6, .3, -.275])
        target = torch.tensor([[.5, .0]] * 5)
        anchor = target + torch.tensor([.0, -.1])
        _, _, audit = first_landing_cross_feedback(
            anchor, velocity, target, target, torch.full((5,), .42), torch.full((5,), .55),
            torch.ones(5, dtype=torch.bool), torch.zeros(5, 2), .05, .01, CONFIG,
        )
        torch.testing.assert_close(audit['gate'], torch.tensor([0., 0., .5, 1., .5]))

    def test_projection_uses_xy_velocity_and_is_direction_symmetric(self):
        edge = torch.tensor([[.5, .0], [-.5, .0], [.0, .5], [.0, -.5]])
        normal = torch.stack((-edge[:, 1], edge[:, 0]), dim=-1) * 2.
        target = torch.tensor([[10., -5.]]) + edge
        anchor = target + normal * .02
        velocity = torch.cat((edge + normal * .10, torch.zeros(4, 1)), dim=1)
        acceleration = edge + normal * -.25
        cost, discount, audit = first_landing_cross_feedback(
            anchor, velocity, target, edge, torch.full((4,), .40), torch.full((4,), .40),
            torch.ones(4, dtype=torch.bool), acceleration, .05, .01, CONFIG,
        )
        torch.testing.assert_close(audit['projected_cross_error_m'], torch.full((4,), .06))
        torch.testing.assert_close(audit['reference_acceleration_m_s2'], torch.full((4,), -.125))
        torch.testing.assert_close(discount, torch.full((4,), .25**2 - .125**2))
        torch.testing.assert_close(cost, cost[0].expand_as(cost))

    def test_boost_is_bounded_and_never_subsidizes_wrong_direction_or_along(self):
        target = torch.tensor([[.5, .0]] * 5)
        anchor = target + torch.tensor([[.10, .04], [-.12, .05], [.0, .21], [.0, .21], [.0, -.21]])
        acceleration = torch.tensor([[4., -.6], [4., -.6], [4., 2.], [4., -2.], [4., .20]])
        cost, discount, audit = first_landing_cross_feedback(
            anchor, torch.zeros(5, 3), target, target, torch.full((5,), .50), torch.full((5,), .40),
            torch.ones(5, dtype=torch.bool), acceleration, .05, .01, CONFIG,
        )
        torch.testing.assert_close(audit['reference_acceleration_m_s2'], torch.tensor([0., 0., -.55, -.55, .55]))
        torch.testing.assert_close(cost[:2], torch.zeros(2))
        torch.testing.assert_close(discount, torch.tensor([0., 0., 0., 4. - 1.45**2, .20**2]))
        remaining_cost = acceleration.square().sum(-1) - discount
        self.assertTrue(bool((remaining_cost >= 4.**2).all()))

    def test_rejects_invalid_settings(self):
        for invalid in (
            {'version': 'unknown'}, {'start_elapsed_s': .36}, {'minimum_remaining_s': .4},
            {'maximum_acceleration_m_s2': float('nan')}, {'error_width_m': 0.},
            {'ascent_start_vz_m_s': .2},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate(CONFIG | invalid)
