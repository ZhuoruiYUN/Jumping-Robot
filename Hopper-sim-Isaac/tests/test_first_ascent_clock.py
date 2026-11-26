import json
import unittest
from pathlib import Path

import torch

from benchmark37.moving_phase_reward import landing_guidance


class FirstAscentClockTest(unittest.TestCase):
    def test_only_eligible_first_ascent_uses_the_bounded_travel_clock(self):
        course = Path(__file__).resolve().parents[1] / 'training_config_v51/ab_ascent_control/course.json'
        config = json.loads(course.read_text())['moving_landing_guidance']
        feet = torch.zeros(3, 2)
        clearance = torch.full((3,), .65)
        velocity = torch.tensor([[.42, 0., 1.4], [.42, 0., 1.4], [.42, 0., -1.4]])
        root = torch.full((3,), .92)
        goal = torch.tensor([[.42, .015]] * 3)
        target = torch.ones(3)
        phase = torch.full((3,), 3)
        moving = torch.ones(3, dtype=torch.bool)
        valid = torch.zeros_like(moving)
        args = feet, clearance, velocity, root, goal, target, phase, moving, valid, 9.81, .01, config
        # Force an extended inherited clock to distinguish the cap from the
        # same projection evaluated for a retained target or during descent.
        clock_floor = torch.full((3,), .90)
        raw_terms, raw = landing_guidance(*args, remaining_time_floor=clock_floor)
        terms, guide = landing_guidance(
            *args, remaining_time_floor=clock_floor,
            first_clock_cap=(torch.full((3,), .72), torch.tensor([1., 0., 0.])))
        self.assertAlmostEqual(guide['remaining_time_s'][0].item(), .72, places=6)
        self.assertGreater(guide['desired_velocity_xy'][0, 0], raw['desired_velocity_xy'][0, 0])
        torch.testing.assert_close(guide['remaining_time_s'][1:], raw['remaining_time_s'][1:])
        torch.testing.assert_close(terms['moving_landing_error'], raw_terms['moving_landing_error'])


if __name__ == '__main__':
    unittest.main()
