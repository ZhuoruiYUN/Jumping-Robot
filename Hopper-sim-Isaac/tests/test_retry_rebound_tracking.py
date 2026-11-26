import math
import unittest

import torch

from benchmark37.retry_rebound_tracking import tracked_rebound_reference, validate
from benchmark37.signed_retry_rebound import signed_retry_rebound


SIGNED = dict(start_before_touch_s=.14, full_before_touch_s=.06, full_excess_m=.025,
              inward_start_m_s=.025, inward_full_m_s=.12, vertical_impulse_ratio=2.,
              max_reduction_deg=6.)
TRACKING = dict(settling_time_constants=2., lead_ramp_s=.08, max_preview_s=.27,
                early_guard_start_s=.10, early_guard_full_s=.25, early_guard_extra_m=.03,
                axis_prediction_limit_s=.09, spring_centroid_s=.02, support_window_s=.055,
                confirm_window_s=.03)


class RetryReboundTrackingTest(unittest.TestCase):
    def setUp(self):
        self.radius = .05
        self.gravity = 9.81
        self.tilt = 12.
        self.step_dt = .01
        self.edge = self.vector(.5, 0.)
        self.error = self.vector(.11, .02)
        self.velocity = self.vector(-.22, .04, -3.2)
        self.flight = self.scalar(1.1)
        self.tau = self.scalar(.065)
        self.delay = self.scalar(4.507)
        self.eligible = torch.tensor([True])
        self.actual = self.normal(.055, .02)
        self.omega = self.vector(0., .28, 0.)
        self.nominal = self.normal(.15, .05)
        self.set_time(.185)

    def scalar(self, value):
        return torch.tensor([value], dtype=torch.float64)

    def vector(self, *values):
        return torch.tensor([values], dtype=torch.float64)

    def normal(self, x, y):
        return self.vector(x, y, math.sqrt(1. - x * x - y * y))

    def set_time(self, time):
        self.clearance = (-self.velocity[:, 2]) * time + .5 * self.gravity * time * time

    def v76_reference(self):
        return signed_retry_rebound(
            self.nominal, self.edge, self.error, self.velocity, self.clearance,
            self.flight, self.eligible, self.radius, self.gravity, self.tilt, SIGNED)[0]

    def reference(self, v76=None):
        return tracked_rebound_reference(
            self.nominal, self.v76_reference() if v76 is None else v76, self.edge,
            self.error, self.velocity, self.clearance, self.flight, self.eligible,
            self.actual, self.omega, self.step_dt, self.tau, self.delay,
            self.radius, self.gravity, self.tilt, SIGNED, TRACKING)

    def test_predicts_outward_support_before_v76_changes_the_angle(self):
        torch.testing.assert_close(self.v76_reference(), self.nominal, rtol=0., atol=0.)
        normal, audit = self.reference()
        self.assertTrue(audit['active'].item())
        self.assertLess(normal[0, 0], .04)
        self.assertGreater(audit['projected_support_along'].item(), self.actual[0, 0].item() + .02)
        self.assertAlmostEqual(audit['time_to_touch_s'].item(), .185)
        torch.testing.assert_close(normal[:, 1], self.nominal[:, 1], rtol=0., atol=0.)
        torch.testing.assert_close(normal.norm(dim=-1), self.scalar(1.))

        self.tau[0] = .055
        self.delay[0] = 2.
        _, faster = self.reference()
        self.assertGreater(audit['weight'].item(), faster['weight'].item())

    def test_early_guard_keeps_shallow_forecast_outside_the_active_window(self):
        self.set_time(.24)
        self.error[0, 0] = .07
        normal, audit = self.reference()
        self.assertFalse(audit['active'].item())
        torch.testing.assert_close(normal, self.nominal, rtol=0., atol=0.)
        self.error[0, 0] = .11
        _, audit = self.reference()
        self.assertTrue(audit['active'].item())

    def test_fully_active_reference_solves_desired_signed_impulse(self):
        self.set_time(.08)
        normal, audit = self.reference()
        speed = (self.velocity[:, 2].square() + 2. * self.gravity * self.clearance).sqrt()
        predicted_velocity = self.velocity[:, 0] + SIGNED['vertical_impulse_ratio'] * speed * normal[:, 0] / normal[:, 2]
        self.assertTrue(audit['active'].item())
        torch.testing.assert_close(predicted_velocity, -self.error[:, 0] / self.flight)
        torch.testing.assert_close(normal[:, 1], self.nominal[:, 1], rtol=0., atol=0.)

        self.error[0, 0] = 5.
        bounded, _ = self.reference()
        self.assertLessEqual(math.degrees(math.acos(bounded[0, 2].item())), self.tilt + 1.e-10)

    def test_never_relaxes_a_more_restrictive_v76_reference(self):
        self.set_time(.08)
        v76 = self.normal(-.05, .05)
        normal, audit = self.reference(v76)
        self.assertTrue(audit['active'].item())
        torch.testing.assert_close(normal, v76, rtol=0., atol=0.)
        torch.testing.assert_close(audit['reference_along'], v76[:, 0])

    def test_braking_reference_that_already_meets_target_has_no_extra_weight(self):
        self.set_time(.08)
        self.nominal = self.normal(-.05, .05)
        normal, audit = self.reference()
        self.assertFalse(audit['active'])
        self.assertEqual(audit['weight'].item(), 0.)
        torch.testing.assert_close(normal, self.nominal, rtol=0., atol=0.)

    def test_ineligible_outward_and_compressed_lanes_keep_original(self):
        self.eligible[0] = False
        self.assertFalse(self.reference()[1]['active'])
        self.eligible[0] = True
        self.velocity[0, 0] = .25
        self.assertFalse(self.reference()[1]['active'])
        self.velocity[0, 0] = -.22
        self.clearance[0] = -.001
        self.assertFalse(self.reference()[1]['active'])
        self.set_time(.10)
        self.edge.zero_()
        normal, audit = self.reference()
        self.assertFalse(audit['active'])
        torch.testing.assert_close(normal, self.nominal, rtol=0., atol=0.)
        self.assertTrue(torch.isfinite(audit['projected_support_along']).all())

    def test_axes_rotate_with_the_course(self):
        normal, audit = self.reference()
        angle = .7
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = self.edge.new_tensor([[cosine, sine, 0.], [-sine, cosine, 0.], [0., 0., 1.]])
        self.edge = self.edge @ rotation[:2, :2]
        self.error = self.error @ rotation[:2, :2]
        self.velocity = self.velocity @ rotation
        self.actual = self.actual @ rotation
        self.omega = self.omega @ rotation
        self.nominal = self.nominal @ rotation
        turned, rotated_audit = self.reference()
        torch.testing.assert_close(turned, normal @ rotation)
        torch.testing.assert_close(rotated_audit['projected_support_along'], audit['projected_support_along'])

    def test_invalid_configurations(self):
        validate(TRACKING)
        for invalid in (dict(early_guard_full_s=.31), dict(lead_ramp_s=.28),
                        dict(confirm_window_s=.1), dict(spring_centroid_s=.06),
                        dict(settling_time_constants=float('nan'))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate(TRACKING | invalid)
