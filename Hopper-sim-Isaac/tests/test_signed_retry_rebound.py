import math
import unittest

import torch

from benchmark37.signed_retry_rebound import signed_retry_rebound, validate_signed_retry_rebound


CONFIG = dict(start_before_touch_s=.14, full_before_touch_s=.06, full_excess_m=.025,
              inward_start_m_s=.025, inward_full_m_s=.12, vertical_impulse_ratio=2.,
              max_reduction_deg=6.)


class SignedRetryReboundTest(unittest.TestCase):
    def setUp(self):
        self.edge = torch.tensor([[.55, 0.]], dtype=torch.float64)
        self.error = torch.tensor([[.089, .020]], dtype=torch.float64)
        self.velocity = torch.tensor([[-.158, .15, -3.5]], dtype=torch.float64)
        self.clearance = torch.tensor([.10], dtype=torch.float64)
        self.flight_time = torch.tensor([1.10], dtype=torch.float64)
        self.eligible = torch.tensor([True])
        self.normal = torch.tensor([[.087, .03, math.sqrt(1. - .087 ** 2 - .03 ** 2)]],
                                   dtype=torch.float64)

    def reference(self):
        return signed_retry_rebound(self.normal, self.edge, self.error, self.velocity,
                                    self.clearance, self.flight_time, self.eligible,
                                    .05, 9.81, 12., CONFIG)

    def test_repeated_overshoot_matches_inward_target_and_preserves_transverse(self):
        result, audit = self.reference()
        impact_speed = (self.velocity[:, 2].square() + 2. * 9.81 * self.clearance).sqrt()
        initial_outgoing = self.velocity[:, 0] + 2. * impact_speed * self.normal[:, 0] / self.normal[:, 2]
        corrected_outgoing = self.velocity[:, 0] + 2. * impact_speed * result[:, 0] / result[:, 2]
        self.assertTrue(audit['active'].item())
        self.assertGreater(initial_outgoing.item(), 0.)
        self.assertLess(corrected_outgoing.item(), 0.)
        torch.testing.assert_close(corrected_outgoing, -self.error[:, 0] / self.flight_time)
        torch.testing.assert_close(result[:, 1], self.normal[:, 1], rtol=0., atol=0.)
        torch.testing.assert_close(result.norm(dim=-1), self.flight_time.new_ones(1))

    def test_eligibility_and_contact_gates_preserve_original(self):
        self.normal = self.normal.repeat(8, 1)
        self.edge = self.edge.repeat(8, 1)
        self.error = self.error.repeat(8, 1)
        self.velocity = self.velocity.repeat(8, 1)
        self.clearance = self.clearance.repeat(8)
        self.flight_time = self.flight_time.repeat(8)
        self.eligible = self.eligible.repeat(8)
        self.error[0] = torch.tensor([.04, .12])  # large transverse miss, small along-route error
        self.error[1, 0] = -.089
        self.velocity[2, 0] = .158
        self.velocity[3, 2] = .1
        self.clearance[4] = .8
        self.clearance[5] = 0.
        self.eligible[6] = False
        self.edge[7] = 0.
        result, audit = self.reference()
        self.assertFalse(audit['active'].any().item())
        torch.testing.assert_close(result, self.normal, rtol=0., atol=0.)

    def test_ramps_smoothly_near_hit_radius(self):
        self.error[0, 0] = .05
        result, audit = self.reference()
        self.assertEqual(audit['weight'].item(), 0.)
        torch.testing.assert_close(result, self.normal, rtol=0., atol=0.)
        self.error[0, 0] = .0625
        result, audit = self.reference()
        self.assertAlmostEqual(audit['weight'].item(), .5)
        self.assertTrue(audit['active'].item())

    def test_large_correction_stays_inside_tilt_and_change_limits(self):
        self.error[0, 0] = .6
        self.normal[0] = torch.tensor([.155, .055, math.sqrt(1. - .155 ** 2 - .055 ** 2)],
                                      dtype=torch.float64)
        result, _ = self.reference()
        self.assertLessEqual(math.degrees(math.acos(result[0, 2].item())), 12.)
        self.assertLessEqual(math.degrees(math.acos((result * self.normal).sum().item())),
                             CONFIG['max_reduction_deg'] + 1.e-8)

    def test_reference_rotates_with_course(self):
        baseline, ours = self.reference()
        axes = self.edge.new_tensor(((-1., 0.), (0., -1.), (1., 0.), (0., 1.)))
        for axis in axes:
            # Rotation rather than reflection preserves the actual vehicle's transverse side.
            rotation = torch.stack((axis, torch.stack((-axis[1], axis[0]))))
            normal = torch.cat((self.normal[:, :2] @ rotation, self.normal[:, 2:]), dim=-1)
            velocity = torch.cat((self.velocity[:, :2] @ rotation, self.velocity[:, 2:]), dim=-1)
            result, turned = signed_retry_rebound(normal, self.edge @ rotation,
                                                 self.error @ rotation, velocity,
                                                 self.clearance, self.flight_time,
                                                 self.eligible, .05, 9.81, 12., CONFIG)
            torch.testing.assert_close(result[:, :2], baseline[:, :2] @ rotation)
            torch.testing.assert_close(result[:, 2:], baseline[:, 2:])
            torch.testing.assert_close(turned['corrected_along'], ours['corrected_along'])

    def test_invalid_configuration_rejected(self):
        validate_signed_retry_rebound(CONFIG)
        for invalid in (dict(start_before_touch_s=.01), dict(max_reduction_deg=11.),
                        dict(inward_full_m_s=.01), dict(vertical_impulse_ratio=float('nan'))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_signed_retry_rebound(CONFIG | invalid)
