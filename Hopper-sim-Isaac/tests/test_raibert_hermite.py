import unittest

import torch

from Quadhopper_Planner_Random.raibert_hermite import raibert_hermite_curve


class RaibertHermiteTest(unittest.TestCase):
    def setUp(self):
        self.start = torch.tensor([[0.0, 0.0, 0.38]])
        self.landing = torch.tensor([[0.30, 0.0, 0.38]])
        self.following = torch.tensor([[0.60, 0.0, 0.38]])
        self.apex = torch.tensor([1.0])
        half_flight = (2.0 * (1.0 - 0.38) / 9.81) ** 0.5
        self.flight_time = 2.0 * half_flight
        self.desired_velocity = torch.tensor([[0.30 / self.flight_time, 0.0]])

    def test_matching_step_velocity_keeps_commanded_foothold(self):
        curve, info = raibert_hermite_curve(
            self.start, self.landing, self.following, self.apex,
            self.desired_velocity,
        )
        torch.testing.assert_close(info['foothold_shift_xy'], torch.zeros(1, 2))
        torch.testing.assert_close(curve[:, 0], self.start)
        torch.testing.assert_close(curve[:, -1], self.landing)
        torch.testing.assert_close(curve[:, 16, 2], self.apex)
        self.assertAlmostEqual(info['flight_time_s'].item(), self.flight_time, places=5)

    def test_velocity_error_moves_foothold_and_respects_cap(self):
        fast = self.desired_velocity + torch.tensor([[10.0, 0.0]])
        curve, info = raibert_hermite_curve(
            self.start, self.landing, self.following, self.apex, fast,
            max_foothold_shift_m=0.04,
        )
        self.assertGreater(info['foothold_shift_xy'][0, 0].item(), 0.0)
        self.assertAlmostEqual(
            torch.linalg.vector_norm(info['foothold_shift_xy'], dim=1).item(),
            0.04,
            places=6,
        )
        torch.testing.assert_close(curve[:, -1], info['guidance_foothold'])
        self.assertAlmostEqual(curve[0, -1, 0].item(), 0.34, places=6)

    def test_reverse_preview_brakes_touchdown_velocity(self):
        following = torch.tensor([[0.0, 0.0, 0.38]])
        _, info = raibert_hermite_curve(
            self.start, self.landing, following, self.apex,
            self.desired_velocity, preview_blend=0.5,
        )
        torch.testing.assert_close(
            info['desired_next_velocity_xy'], -info['desired_takeoff_velocity_xy']
        )
        torch.testing.assert_close(
            info['touchdown_velocity_xy'], torch.zeros(1, 2), atol=1e-7, rtol=0
        )

    def test_stationary_hop_is_finite_and_hits_apex(self):
        curve, info = raibert_hermite_curve(
            self.start, self.start, self.start, self.apex, torch.zeros(1, 2)
        )
        self.assertTrue(torch.isfinite(curve).all())
        for value in info.values():
            self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(curve[:, 0], self.start)
        torch.testing.assert_close(curve[:, -1], self.start)
        torch.testing.assert_close(curve[:, 16, 2], self.apex)
        self.assertLessEqual(curve[..., 2].max().item(), 1.0 + 1e-6)


if __name__ == '__main__':
    unittest.main()
