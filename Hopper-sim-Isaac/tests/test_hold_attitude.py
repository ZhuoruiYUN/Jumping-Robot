import math
import unittest

import torch

from benchmark37.flight_plane import hold_attitude_cost, validate_hold_attitude


CONFIG = dict(weight=180., free_tilt_deg=3., prediction_s=.12, support_weight=.30)


class HoldAttitudeTest(unittest.TestCase):
    def test_roll_and_pitch_pay_the_same_outward_excess(self):
        theta = math.radians(7.)
        sine, cosine = math.sin(theta), math.cos(theta)
        up = torch.tensor([[0., -sine, cosine], [sine, 0., cosine],
                           [0., -sine, cosine], [0., -sine, cosine], [0., -sine, cosine]])
        omega = torch.tensor([[.5, 0., 0.], [0., .5, 0.], [-.5, 0., 0.],
                              [0., -sine, cosine], [0., 0., 5.]])
        active = torch.ones(5, dtype=torch.bool)
        cost, predicted = hold_attitude_cost(up, omega, torch.full((5,), theta),
                                            active, active, .01, CONFIG)
        self.assertLess(cost[0], cost[2])
        torch.testing.assert_close(cost[0], cost[1])
        torch.testing.assert_close(cost[2], cost[3])
        torch.testing.assert_close(cost[2], cost[4])
        self.assertGreater(predicted[0], theta)
        self.assertLess(predicted[2], theta)

    def test_support_weight_and_moving_mask(self):
        theta = math.radians(9.)
        up = torch.tensor([[0., -math.sin(theta), math.cos(theta)]]).expand(3, -1)
        lean = torch.full((3,), theta)
        active = torch.tensor([True, True, False])
        airborne = torch.tensor([True, False, True])
        cost, predicted = hold_attitude_cost(up, torch.zeros_like(up), lean, active,
                                            airborne, .01, CONFIG)
        torch.testing.assert_close(cost[1], cost[0] * CONFIG['support_weight'])
        self.assertEqual(cost[2].item(), 0.)
        self.assertEqual(predicted[2].item(), 0.)

    def test_reward_is_zero_inside_the_free_cone(self):
        config = dict(CONFIG)
        validate_hold_attitude(config, 10.)
        for key, value in [('free_tilt_deg', 10.), ('prediction_s', .21), ('support_weight', 1.2)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_hold_attitude(dict(config, **{key: value}), 10.)
        up = torch.tensor([[0., 0., 1.]])
        active = torch.tensor([True])
        cost, _ = hold_attitude_cost(up, torch.tensor([[0., 0., 2.]]), torch.zeros(1),
                                     active, active, .01, config)
        torch.testing.assert_close(cost, torch.zeros(1))


if __name__ == '__main__':
    unittest.main()
