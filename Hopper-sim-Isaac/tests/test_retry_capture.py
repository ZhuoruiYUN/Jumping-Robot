import json
from pathlib import Path
import unittest

import torch

from benchmark37.moving_phase_reward import retry_capture_cost


class RetryCaptureTest(unittest.TestCase):
    def test_contact_velocity_ramp_release_and_attitude(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] /
                          'training_config_v51/ab_retry_capture/course.json').read_text())['retry_capture']
        # Lateral incoming motion and a retained target in the opposite direction.
        incoming = torch.tensor([[1., .4]]).repeat(6, 1)
        desired = torch.tensor([[-.3, .2]]).repeat(6, 1)
        actual = torch.stack((incoming[0], (incoming[1] + desired[1]) / 2, desired[2],
                              incoming[3], desired[4], incoming[5]))
        age = torch.tensor([0., .03, .02, .02, .06, .06])
        released = torch.tensor([False, False, True, True, False, False])
        active = torch.tensor([True, True, True, True, True, False])
        q = torch.tensor([[1., 0., 0., 0.]]).repeat(6, 1)
        omega = torch.zeros(6, 3)
        omega[3, 2] = 4.  # Pure spin around upright body z belongs to the existing yaw terms.
        omega[4:, 1] = 5.
        v, tilt = retry_capture_cost(actual, incoming, desired, age, torch.full_like(age, .06),
                                     released, active, q, omega, .01, cfg)
        torch.testing.assert_close(v[[0, 1, 2, 4, 5]], torch.zeros(5))
        self.assertLess(v[3], 0.)
        torch.testing.assert_close(tilt[[0, 1, 2, 3, 5]], torch.zeros(5))
        self.assertLess(tilt[4], 0.)
