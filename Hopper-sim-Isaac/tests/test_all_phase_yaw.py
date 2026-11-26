import math
import unittest

import torch

from benchmark37.all_phase_yaw import YawEvaluation, yaw_envelope_cost
from benchmark37.moving_phase_reward import heading_motion


class AllPhaseYawTest(unittest.TestCase):
    prediction = dict(response_horizon_s=.14, rms_heading_goal_deg=3., peak_heading_goal_deg=10.)
    config = dict(warning_deg=6., heading_weight=20., forecast_excess_weight=120., body_z_rate_weight=2.)

    def test_current_forecast_and_body_spin(self):
        error = torch.tensor([0., 2., 7., 7., -7., 0.]).deg2rad()
        rate = torch.tensor([0., 0., -30., 30., -30., 0.]).deg2rad()
        body = torch.tensor([0., 0., 0., 0., 0., .6])
        cost = yaw_envelope_cost(error, rate, body, .01, self.prediction, self.config)
        self.assertEqual(cost[0], 0.)
        self.assertTrue((cost[1:] < 0).all())
        self.assertLess(cost[3], cost[2])
        torch.testing.assert_close(cost[3], cost[4])

    def test_horizontal_heading_and_body_z_differ_during_tilt(self):
        # At 60 degrees roll the body-Y rate also rotates the forward axis in world XY.
        q = torch.tensor([[math.cos(math.pi/6), math.sin(math.pi/6), 0., 0.]])
        omega = torch.tensor([[0., 2., 0.]])
        angle, rate = heading_motion(q, omega)
        self.assertGreater(abs(float(rate)), 1.)
        self.assertLess(yaw_envelope_cost(angle, rate, omega[:,2], .01, self.prediction, self.config), 0.)

    def test_control_band_charges_earlier_and_has_linear_excess_beyond_limit(self):
        config = dict(self.config, control_band=dict(warning_deg=3., weight=240.))
        present = torch.tensor([0., 0., 5., 0., 0., 0., 0.], dtype=torch.float64).deg2rad()
        future = torch.tensor([0., 3., 2., 5., 10., 15., -20.], dtype=torch.float64).deg2rad()
        rate = (future - present) / .14
        base = yaw_envelope_cost(present, rate, torch.zeros_like(rate), .01, self.prediction, self.config)
        new = yaw_envelope_cost(present, rate, torch.zeros_like(rate), .01, self.prediction, config)
        extra = base - new
        self.assertEqual(float(extra[0]), 0.)
        self.assertEqual(float(extra[1]), 0.)
        torch.testing.assert_close(extra[2], extra[3])
        self.assertGreater(float(extra[3]), 0.)
        torch.testing.assert_close(extra[5] - extra[4], extra[6] - extra[5])

    def test_acceptance_counts_terminal_once_and_rejects_forecast(self):
        stats = YawEvaluation(2, 'cpu', self.prediction)
        def sample(errors, rates, done):
            stats.sample(dict(error=torch.tensor(errors).deg2rad(), rate=torch.tensor(rates).deg2rad()),
                         torch.tensor(done))
        sample([1., 0.], [0., 0.], [True, False])
        sample([90., 0.], [999., 80.], [False, True])
        report = stats.report()
        self.assertEqual(report['samples'], 3)
        self.assertEqual(report['episodes_outside_current_or_predicted'], 1)
        self.assertEqual(report['outside_current_or_predicted_ticks'], 1)
        self.assertAlmostEqual(report['peak_deg'], 1., places=5)
        self.assertAlmostEqual(report['predicted_peak_deg'], 11.2, places=4)
        self.assertFalse(report['passed'])
