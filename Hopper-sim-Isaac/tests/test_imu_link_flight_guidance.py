import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from benchmark37.imu_link_flight import (
    ascent_gate, bounded_clock, collective_reference, predict_powered_flight, validate,
)
from benchmark37.imu_link_plant import ImuLinkPlant


ROOT = Path(__file__).resolve().parents[1]
COURSE_DIR = ROOT / "training_config_v51"
COURSE = json.loads((COURSE_DIR / "ab_calibrated_flight/course.json").read_text())
GUIDANCE = COURSE["imu_link_flight_guidance"]


class ImuLinkFlightTest(unittest.TestCase):
    def test_disabled_clock_cannot_change_descent_or_stance(self):
        phase = torch.tensor([3, 3, 1, 3])
        apex_seen = torch.tensor([False, True, False, False])
        vz = torch.tensor([1.2, -.2, 1.2, 1.2])
        elapsed = torch.tensor([.15, .40, .0, .20])
        clearance = torch.tensor([.4, .5, .0, .001])
        gate = ascent_gate(phase, apex_seen, clearance, vz, elapsed, GUIDANCE)
        self.assertGreater(float(gate[0]), 0.)
        current = torch.tensor([.95, .45, .95, .95])
        revised = bounded_clock(current, torch.tensor([1.2] * 4), gate,
                                .15, 1.2, GUIDANCE["maximum_clock_change_s"])
        self.assertGreater(float(revised[0]), float(current[0]))
        self.assertLessEqual(float(revised[0] - current[0]), GUIDANCE["maximum_clock_change_s"])
        torch.testing.assert_close(revised[1:], current[1:])

    def test_predicted_height_incorporates_applied_thrust(self):
        validate(GUIDANCE)
        peak, rise_time, land_time = predict_powered_flight(
            torch.tensor([.8, .8]), torch.tensor([1., 1.]), torch.tensor([2., 7.]),
            GUIDANCE["predictor_coefficients"],
        )
        self.assertGreater(float(peak[1]), float(peak[0]))
        self.assertTrue((land_time > rise_time).all())

    def test_collective_ceiling_yaw_and_no_deficit(self):
        duty = torch.full((5, 4), .40)
        duty[1, 0] = .99
        apex = torch.tensor([.91, .91, 1.03, .91, .91])
        yaw_forecast = torch.tensor([.0, .0, .0, .20, .0])
        reference, extra, spare, _ = collective_reference(
            apex, torch.ones(5), torch.full((5,), .62), torch.tensor([1., 1., 1., 1., .0]), torch.ones(5),
            torch.zeros(5), yaw_forecast, duty, duty, torch.ones(5), torch.ones(5),
            torch.ones(5), GUIDANCE,
        )
        self.assertGreater(float(extra[0]), 0.)
        self.assertTrue((extra <= GUIDANCE["collective_cap_n"]).all())
        self.assertTrue((extra <= spare).all())
        torch.testing.assert_close(extra[1:], torch.zeros(4))
        torch.testing.assert_close(reference - extra, torch.ones(5))

    def test_plant_retains_native_yaw_and_clears_delayed_duties(self):
        env = SimpleNamespace(
            num_envs=2, device="cpu", step_dt=.01, cfg=SimpleNamespace(seed=42),
            _motor_u=torch.full((2, 4), .28), _actions=torch.full((2, 4), .80),
            _thrust=torch.zeros(2, 1, 3), _moment=torch.ones(2, 1, 3),
            dr_mass_multi=torch.ones(2, 1), dr_inertia_multi=torch.ones(2, 3),
            _motor_u_history=torch.zeros(2, 25, 4), _soc_tracker=torch.zeros(2, 1),
        )
        plant = ImuLinkPlant(env, COURSE["imu_link_plant"])
        for _ in range(8):
            plant.step()
        self.assertTrue((plant.collective_u > 0.).all())
        self.assertTrue((env._thrust[:, 0, 2] > 0.).all())
        torch.testing.assert_close(env._moment[:, 0, 2], torch.ones(2))
        torch.testing.assert_close(env._motor_u, torch.full((2, 4), .28))
        plant.reset(torch.tensor([0]))
        self.assertTrue((plant.collective_u[0] == 0.).all())
        self.assertTrue((plant.attitude_u[0] == 0.).all())
        self.assertTrue((plant.collective_u[1] > 0.).all())
        self.assertTrue(all((duty[0] == 0.).all() for duty in plant.history))

    def test_course_keeps_v73_navigation_actor_and_yaw_contracts(self):
        original = json.loads((COURSE_DIR / "ab_yaw_servo/course.json").read_text())
        for key in ("observation_version", "moving_navigation", "yaw_heading_servo",
                    "moving_yaw_prediction", "yaw_thrust_reserve_n", "moving_landing_guidance"):
            self.assertEqual(COURSE[key], original[key])
        updated_omni = COURSE["omni"].copy()
        checkpoint = updated_omni.pop("initial_checkpoint")
        self.assertTrue(checkpoint.endswith("ab_yaw_servo_v73/formal_20261122_214607/model_49.pt"))
        original["omni"].pop("initial_checkpoint")
        self.assertEqual(updated_omni, original["omni"])


if __name__ == "__main__":
    unittest.main()
