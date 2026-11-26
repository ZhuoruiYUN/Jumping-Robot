import json
import unittest
from pathlib import Path

import torch

from benchmark37.moving_phase_reward import landing_guidance


class FootPrecisionTest(unittest.TestCase):
    def test_descent_blends_toward_measured_foot_and_preserves_touchdown_scoring(self):
        root = Path(__file__).resolve().parents[1]
        course = json.loads((root/'training_config_v51/ab_foot_precision/course.json').read_text())
        cfg = course['moving_landing_guidance']
        # Same 5 cm measured miss in each lane. The reference leg predicts zero miss.
        foot = torch.tensor([[.05, 0.]]).repeat(5, 1)
        zero = torch.zeros_like(foot)
        velocity = torch.tensor([[0., 0., -2.]]).repeat(5, 1)
        velocity[1, 2] = 2.
        moving = torch.tensor([True, True, False, True, True])
        valid = torch.tensor([False, False, False, True, False])
        clearance = torch.tensor([.01, .01, .01, 0., 2.])
        def run(config):
            return landing_guidance(foot, clearance, velocity, clearance + .2, zero,
                                    torch.ones(5), torch.full((5,), 3), moving, valid,
                                    9.81, .01, config, projection_xy=zero)
        old, _ = run(dict(cfg, precision_to_contact=False, measured_foot_blend_s=0.))
        new, audit = run(cfg)
        self.assertGreater(audit['projected_error_m'][0].item(), .049)
        self.assertGreater(audit['flight_weight'][0].item(), .99)
        self.assertLess(new['moving_velocity_guidance'][0].item(), -.015)
        torch.testing.assert_close(new['moving_velocity_guidance'][1:], old['moving_velocity_guidance'][1:])
        expected = torch.where(valid & moving, -6 * (torch.sqrt(torch.tensor(2.)) - 1), 0.)
        torch.testing.assert_close(new['moving_landing_error'], expected)
