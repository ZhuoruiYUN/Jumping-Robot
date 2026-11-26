import math
import unittest

import torch

from benchmark37.moving_phase_reward import consistent_outgoing_velocity, retry_midflight_reference, safe_retry_cruise_residual


class RetryMidflightTest(unittest.TestCase):
    def test_direction_disk_cap_window_and_retry_gate(self):
        # At half flight with zero initial velocity, 10 cm axial miss requires
        # 0.4 m/s² to reach the edge of the 5 cm disk in the remaining 0.5 s.
        target = torch.tensor([[.1, 0.], [0., -.6], [.05, 0.], [.1, 0.], [.1, 0.], [.1, 0.]])
        elapsed = torch.tensor([.5, .5, .5, .2, .75, .5])
        allowed = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
        cfg = dict(start_fraction=.35, end_fraction=.75, max_acceleration_m_s2=.8)
        zeros = torch.zeros_like(target)
        reference, active = retry_midflight_reference(zeros, zeros, target, elapsed,
                                                       torch.ones_like(elapsed), allowed, .05, cfg)
        expected = torch.zeros_like(target)
        expected[0, 0], expected[1, 1] = .4, -.8
        torch.testing.assert_close(reference, expected)
        torch.testing.assert_close(active, torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool))
        # Correct current velocity already closes the miss; translating the
        # world frame and retaining velocity should require no correction.
        ref, active = retry_midflight_reference(zeros + 2., 2*target, target + 2., elapsed.fill_(.5),
                                                 torch.ones_like(elapsed), allowed, .05, cfg)
        self.assertFalse(active.any())
        torch.testing.assert_close(ref, zeros)

    def test_safe_assist_only_discounts_existing_aligned_thrust(self):
        actual = torch.tensor([[0., 0.], [-.5, .2], [.3, .4], [1.5, .2],
                               [.4, .1], [.4, .1], [.4, .1], [.4, .1], [.4, .1], [.4, .1]])
        ref = torch.zeros_like(actual)
        ref[:, 0] = .8
        ref[-1] = 0.
        q = torch.zeros(len(actual), 4)
        q[:, 0] = 1.
        omega = torch.zeros_like(actual[:, :1]).expand(-1, 3).clone()
        allowed = torch.ones(len(actual), dtype=torch.bool)
        allowed[4] = False
        omega[5, 1] = 2.  # Predicted tilt exceeds 15 degrees before motor response.
        omega[6, 2] = 1.5  # Predicted yaw exceeds 10 degrees.
        q[7, 0], q[7, 3] = math.cos(math.radians(4)), math.sin(math.radians(4))  # 8° yaw
        q[8, 0], q[8, 2] = math.cos(math.radians(10)), math.sin(math.radians(10))  # 20° tilt
        cfg = dict(tilt_warning_deg=8., tilt_limit_deg=15.)
        yaw = dict(target_heading_rad=0., response_horizon_s=.14,
                   all_phase_envelope=dict(warning_deg=6.), peak_heading_goal_deg=10.)
        residual, discount, _ = safe_retry_cruise_residual(actual, ref, allowed, q, omega, cfg, yaw)
        expected = torch.tensor([0., 0., .3, .8, 0., 0., 0., .2, 0., 0.])
        torch.testing.assert_close(discount, expected)
        torch.testing.assert_close(residual[:, 0], actual[:, 0] - expected)
        torch.testing.assert_close(residual[:, 1], actual[:, 1])

    def test_late_preview_tracks_miss_before_contact(self):
        foot = torch.tensor([[.04, 0.], [.055, 0.], [.065, 0.], [.055, 0.], [.055, 0.]])
        current = torch.zeros_like(foot)
        preview = torch.tensor([[.5, 0.]]).expand_as(foot)
        velocity = torch.zeros(5, 3)
        velocity[:, 2] = -1.
        t = torch.tensor([.08, .08, .08, .12, .16])
        height = t + .5 * 9.81 * t.square()
        args = (foot, height, velocity, current, preview, torch.ones(5), torch.zeros(5), 9.81, 2.)
        cfg = dict(position_blend_width_m=.025, tilt_blend_width_deg=3.)
        near = dict(position_blend_width_m=.008, full_within_s=.1, start_within_s=.14)
        orig, base, xy = consistent_outgoing_velocity(*args, cfg, blend_after_gate=True, foot_limit_m=.05)
        out, conf, forecast = consistent_outgoing_velocity(*args, dict(cfg, near_touchdown_preview=near),
                                                          blend_after_gate=True, foot_limit_m=.05)
        torch.testing.assert_close(base, torch.tensor([1., .8, .4, .8, .8]))
        torch.testing.assert_close(conf, torch.tensor([1., .375, 0., .5875, .8]))
        torch.testing.assert_close(out[[0,4]], orig[[0,4]])
        torch.testing.assert_close(out, conf[:, None]*preview - foot)
        torch.testing.assert_close(forecast, xy)
