import unittest
import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / 'Quadhopper_Stable/actuator_contract.py'
SPEC = importlib.util.spec_from_file_location('quadhopper_actuator_contract', MODULE_PATH)
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)
corrected_delay_history_index = CONTRACT.corrected_delay_history_index
shape_deployment_motor_commands = CONTRACT.shape_deployment_motor_commands


def shape(target_u, contact, root_z, root_vz):
    return shape_deployment_motor_commands(
        target_u,
        is_contact=contact,
        root_z=root_z,
        root_vz=root_vz,
        target_z=torch.full_like(root_z, 1.095),
        max_pwm=1000.0,
        max_pwm_spread=600.0,
        max_airborne_yaw_diagonal_pwm_diff=140.0,
        max_grounded_yaw_diagonal_pwm_diff=220.0,
        height_brake_start_m=0.03,
        height_brake_full_m=0.23,
        height_brake_vz_min=0.05,
        height_brake_pwm_mean_soft=720.0,
        height_brake_pwm_mean_hard=520.0,
    )


class ActuatorContractTest(unittest.TestCase):
    def test_zero_delay_selects_newest_command(self):
        history = list(range(5))
        self.assertEqual(history[corrected_delay_history_index(0, len(history))], 4)
        self.assertEqual(history[corrected_delay_history_index(1, len(history))], 3)
        self.assertEqual(history[corrected_delay_history_index(2, len(history))], 2)

    def test_deployment_shaping_quantizes_and_enforces_limits(self):
        shaped = shape(
            torch.tensor([[0.0, 1.0, 0.2, 0.9], [1.0, 0.0, 1.0, 0.0]]),
            torch.tensor([False, True]),
            torch.tensor([0.8, 0.28]),
            torch.tensor([-0.2, 0.0]),
        )
        pwm = shaped * 1000.0
        self.assertTrue(torch.all(pwm == torch.floor(pwm)))
        self.assertTrue(torch.all((pwm >= 0.0) & (pwm <= 1000.0)))
        self.assertTrue(torch.all(pwm.max(1).values - pwm.min(1).values <= 600.0))
        diag_diff = torch.abs(0.5 * (pwm[:, 0] + pwm[:, 2] - pwm[:, 1] - pwm[:, 3]))
        self.assertLessEqual(diag_diff[0].item(), 140.0)
        self.assertLessEqual(diag_diff[1].item(), 220.0)

    def test_height_brake_reduces_rising_overshoot_collective(self):
        target_u = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
        shaped = shape(
            target_u,
            torch.tensor([False]),
            torch.tensor([1.20]),
            torch.tensor([0.5]),
        )
        self.assertLess(shaped.mean().item(), target_u.mean().item())


if __name__ == '__main__':
    unittest.main()
