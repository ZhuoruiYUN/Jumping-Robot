"""Static checks for the isolated mounted rigid-leg SysID equations."""
import math
import json
from pathlib import Path
import unittest

import torch

from Quadhopper_Planner_Random.sysid_20261011 import (
    NOMINAL, body_wrench_from_forces, motor_forces_n, motor_state_step,
)


class SysID20261011Tests(unittest.TestCase):
    def test_candidate_matches_packaged_handoff_values(self):
        path = (Path(__file__).resolve().parents[1]
                / 'Quadhopper_Planner_Random/assets/domain_randomization_20261011_source.json')
        source = json.loads(path.read_text())['physical']
        motor = json.loads(path.read_text())['motor_model']
        self.assertEqual(NOMINAL.total_mass_kg, source['mass_mounted']['nominal'])
        self.assertEqual(NOMINAL.inertia_kg_m2, tuple(
            source[key]['nominal'] for key in ('Ixx_mounted', 'Iyy_mounted', 'Izz_mounted')
        ))
        self.assertAlmostEqual(NOMINAL.thrust_quadratic_n, motor['k2_N'], places=9)
        self.assertAlmostEqual(NOMINAL.thrust_linear_n, motor['k1_N'], places=9)

    def test_mass_inertia_and_yaw_uncertainty(self):
        ix, iy, iz = NOMINAL.inertia_kg_m2
        self.assertEqual(NOMINAL.total_mass_kg, 0.194)
        self.assertTrue(ix + iy > iz and ix + iz > iy and iy + iz > ix)
        self.assertFalse(NOMINAL.yaw_ratio_validated)

    def test_motor_lag_and_force_match_report(self):
        state = torch.zeros(1, 4)
        compare = torch.full((1, 4), 999.0)
        after_tau = motor_state_step(state, compare, 3.7, NOMINAL.motor_time_constant_s)
        self.assertAlmostEqual(after_tau[0, 0].item(), 1.0 - math.exp(-1.0), places=6)
        steady = motor_forces_n(torch.ones(1, 4))
        self.assertAlmostEqual(steady[0, 0].item(), 0.5492495448, places=6)
        collective, torque = body_wrench_from_forces(steady)
        self.assertAlmostEqual(collective[0].item(), 4 * 0.5492495448, places=5)
        torch.testing.assert_close(torque, torch.zeros_like(torque))

    def test_yaw_uses_force_not_pwm_squared(self):
        forces = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
        collective, torque = body_wrench_from_forces(forces)
        self.assertAlmostEqual(collective.item(), 1.4, places=6)
        self.assertAlmostEqual(torque[0, 2].item(), 0.002, places=6)

    def test_batched_bus_voltage_and_invalid_compare(self):
        state = torch.zeros(2, 4)
        compare = torch.full((2, 4), 500.0)
        next_state = motor_state_step(state, compare, torch.tensor([3.5, 4.0]), 0.01)
        self.assertTrue(torch.all(next_state[1] > next_state[0]))
        with self.assertRaises(ValueError):
            motor_state_step(state, torch.full((2, 4), 1000.0), 3.7, 0.01)


if __name__ == '__main__':
    unittest.main()
