import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from benchmark37.continuation_cruise import VERSION, cruise_squared_relief, forward_allowance

from benchmark37.v81_first_landing import first_landing_reference
from benchmark37.v81_ws_transfer import CROSS_REBOUND_TERM, ContinuationForecast, references, terms, validate


class WsCrossReboundTest(unittest.TestCase):
    def setUp(self):
        course = (Path(__file__).resolve().parents[1] / "training_config_v51"
                  / "ab_ws_first_precision" / "course.json")
        self.baseline = json.loads(course.read_text())["ws_transfer_teaching"]
        self.new = {
            **self.baseline,
            "cross_rebound": dict(version="signed_transverse_preimpact_v1", cost_per_s=6.,
                                  minimum_step_m=.40, axis_half_width_deg=15.,
                                  residual_start_m_s=.07, residual_width_m_s=.22,
                                  tilt_prediction_s=.10),
        }

    @staticmethod
    def rollout_inputs(config):
        # Mirrored ±Y, yaw blocked, ±X, retry, opening, turn, B, support,
        # short next edge, and correctly centered preimpact state.
        edge = torch.tensor([[0., .5], [0., -.5], [0., .5], [.5, 0.], [-.5, 0.],
                             [0., .5], [0., 0.], [0., .5], [0., .5], [0., .5],
                             [0., .5], [0., .5]])
        n = len(edge)
        following = edge.clone()
        following[6:8] = torch.tensor([[0., .5], [.5, 0.]])
        following[10, 1] = .3
        one = torch.ones(n)
        zeros = torch.zeros(n)
        index = one.long()
        index[6] = 0
        index[8] = 2
        attempted = zeros.long()
        attempted[5] = 1
        pitch = torch.deg2rad(2. * one)
        pitch[1] *= -1.
        pitch[11] = 0.
        yaw = zeros.clone()
        yaw[2] = math.radians(12.)
        cp, sp = (.5 * pitch).cos(), (.5 * pitch).sin()
        cy, sy = (.5 * yaw).cos(), (.5 * yaw).sin()
        quat = torch.stack((cp * cy, -sp * sy, sp * cy, cp * sy), dim=-1)
        velocity = torch.stack((zeros, zeros, -3. * one), dim=-1)
        phase = 3 * one.long()
        phase[9] = 1
        data = SimpleNamespace(root_link_lin_vel_w=velocity, body_link_pos_w=torch.zeros(n, 1, 3),
                               root_quat_w=quat, root_ang_vel_b=torch.zeros(n, 3), root_link_pos_w=velocity)
        route = SimpleNamespace(stationary_mask=zeros.bool(), index=index,
                                ab_pairs=SimpleNamespace(a_index=one.long()))
        plant = SimpleNamespace(parameters=torch.tensor([[.08, 3., 1., 1., 1., 1., .06, 3.]]),
                                profile=zeros.long(), mass_scale=one, collective_scale=one, voltage_gain=one,
                                collective_u=torch.zeros(n, 4), history=[torch.zeros(n, 4)])
        env = SimpleNamespace(
            ws_transfer_teaching=config, omni_route=route, _imu_link_plant=plant,
            _foot_event=SimpleNamespace(phase=phase, CANDIDATE=2, FLIGHT=3, apex_seen=one.bool(), air_steps=80 * one),
            _robot=SimpleNamespace(data=data), _foot_body_id=0, step_dt=.01,
            _two_ground_pt_xy_w=torch.zeros(n, 2), _two_ground_pt1_xy_w=following,
            _current_route_edge=lambda: edge, last_attempted_target=attempted,
            yaw_prediction_config=dict(target_heading_rad=0., response_horizon_s=.1, peak_heading_goal_deg=10.),
            sim=SimpleNamespace(cfg=SimpleNamespace(gravity=(0., 0., -9.81))),
            first_cross_foot_offset=torch.zeros(n, 2), first_cross_release_step=-one.long(),
            rebound_plan=SimpleNamespace(flight=one),
        )
        return env, torch.zeros(n, 4), .12 * one, .4 * one, pitch.cos(), .04 * one

    def test_opt_in_uses_signed_y_handoff_and_preserves_other_costs(self):
        validate(self.baseline)
        validate(self.new)
        baseline, old_audit = references(*self.rollout_inputs(self.baseline))
        inputs = self.rollout_inputs(self.new)
        actions = inputs[1].clone()
        costs, audit = references(*inputs)
        self.assertEqual(set(baseline), set(terms(self.baseline)))
        self.assertEqual(set(costs), set(terms(self.new)))
        self.assertNotIn("cross_rebound_gate", old_audit)
        for key, value in baseline.items():
            torch.testing.assert_close(costs[key], value, rtol=0., atol=0.)
        torch.testing.assert_close(inputs[1], actions, rtol=0., atol=0.)

        active = audit["cross_rebound_gate"] > 0.
        self.assertEqual(active.nonzero().flatten().tolist(), [0, 1, 11])
        self.assertEqual((costs[CROSS_REBOUND_TERM] < 0.).nonzero().flatten().tolist(), [0, 1])
        forecast = audit["cross_rebound_prediction_m_s"]
        self.assertLess(forecast[0].item(), -.18)
        torch.testing.assert_close(forecast[0], forecast[1])
        torch.testing.assert_close(costs[CROSS_REBOUND_TERM][0], costs[CROSS_REBOUND_TERM][1])

    def test_continuation_credit_requires_a_clean_forward_window_after_a_hit(self):
        config = dict(version="bounded_continuation_acceleration_v1",
                      minimum_step_m=.45, early_credit_m=.06, late_credit_m=.02)
        validate({**self.baseline, "continuation_forecast": config})
        history = ContinuationForecast(9, "cpu", config)
        edge = torch.tensor([[.5, 0.], [-.5, 0.], [.5, 0.], [0., .5], [.5, 0.],
                             [.5, 0.], [.5, 0.], [.5, 0.], [.5, 0.]])
        prior = edge.clone()
        prior[3] = torch.tensor([.5, 0.])
        prior[7] *= .8
        all_lanes = torch.ones(9, dtype=torch.bool)
        hit = all_lanes.clone()
        hit[2] = False
        target = torch.full((9,), 2)
        history.touchdown(all_lanes, hit, target)

        speed = torch.nn.functional.pad(.4 * edge, (0, 1))
        push = torch.zeros_like(all_lanes)
        push[5] = True
        remaining = torch.full((9,), .8)
        ages = torch.full((9,), 19)
        credit, allowed = history.credit(90, .01, target, ages, speed, edge, prior, remaining, all_lanes, push)
        self.assertFalse(allowed.any())
        self.assertEqual(credit.count_nonzero().item(), 0)
        history.reset(torch.tensor([4]))
        new_speed = torch.nn.functional.pad(.6 * edge, (0, 1))
        new_speed[6, 0] = -.1
        first = all_lanes.clone()
        first[8] = False
        credit, allowed = history.credit(98, .01, target, ages + 8, new_speed, edge, prior, remaining,
                                         first, torch.zeros_like(push))
        self.assertEqual(allowed.nonzero().flatten().tolist(), [0, 1])
        torch.testing.assert_close(credit, torch.tensor([.06, .06, 0., 0., 0., 0., 0., 0., 0.]))

    def test_zero_credit_is_identical_and_positive_credit_only_reduces_shortfall(self):
        actions = torch.zeros(2, 4)
        quaternion = torch.tensor([[1., 0., 0., 0.]] * 2)
        omega = torch.zeros(2, 3)
        foot = torch.tensor([[-.315, .16], [-.315, .16]])
        velocity = torch.tensor([[.3, 0., .6]] * 2)
        goal = torch.zeros(2, 2)
        edge = torch.tensor([[.5, 0.]] * 2)
        elapsed, remaining = torch.full((2,), .36), torch.full((2,), .6)
        allow = torch.ones(2, dtype=torch.bool)
        args = (actions, quaternion, omega, foot, velocity, goal, edge, elapsed, remaining, allow,
                self.baseline["landing"])
        raw = first_landing_reference(*args, undertravel_allowed=allow)
        changed = first_landing_reference(*args, undertravel_allowed=allow,
                                          undertravel_credit_m=torch.tensor([0., .06]))
        for key, value in raw.items():
            torch.testing.assert_close(value[0], changed[key][0], rtol=0., atol=0.)
        torch.testing.assert_close(raw["projected_cross_m"], changed["projected_cross_m"], rtol=0., atol=0.)
        torch.testing.assert_close(raw["projected_along_m"], changed["projected_along_m"], rtol=0., atol=0.)
        torch.testing.assert_close(raw["reference_cross_accel_m_s2"], changed["reference_cross_accel_m_s2"],
                                   rtol=0., atol=0.)
        torch.testing.assert_close(raw["target_cross_force_n"], changed["target_cross_force_n"],
                                   rtol=0., atol=0.)
        torch.testing.assert_close(raw["axis_weight"][:, 1], changed["axis_weight"][:, 1], rtol=0., atol=0.)
        self.assertGreater(raw["undertravel_gate"][1], changed["undertravel_gate"][1])
        self.assertGreater(raw["reference_along_accel_m_s2"][1], changed["reference_along_accel_m_s2"][1])

    def test_cruise_relief_needs_confirmed_shortfall_height_and_the_correct_direction(self):
        edge = torch.tensor([[.5, 0.], [-.5, 0.]] * 4)
        speed = torch.nn.functional.pad(edge, (0, 1))
        n = len(edge)
        current = dict(projected_along_m=torch.full((n,), -.20),
                       projected_cross_m=torch.zeros(n),
                       undertravel_gate=torch.ones(n),
                       reference_along_accel_m_s2=torch.full((n,), .25))
        current["projected_along_m"][2] = -.06
        current["projected_cross_m"][3] = .12
        current["undertravel_gate"][4] = 0.
        coast, requested = torch.full((n,), 1.03), torch.full((n,), 1.07)
        coast[5], requested[6] = .99, 1.09
        eligible = torch.ones(n, dtype=torch.bool)
        eligible[7] = False
        allowance = forward_allowance(current, speed, edge, torch.full((n,), .40), torch.full((n,), .42),
                                       coast, requested, eligible, .05)
        torch.testing.assert_close(allowance, torch.tensor([.25, .25, 0., 0., 0., 0., 0., 0.]))
        thrust = torch.tensor([[.60, .3], [.60, .3]] * 4)
        relief = cruise_squared_relief(thrust, edge, allowance)
        torch.testing.assert_close(relief, torch.tensor([.60**2 - .35**2, 0., 0., 0., 0., 0., 0., 0.]))
        self.assertTrue(((thrust.square().sum(-1) - relief) >= .09).all())
        with self.assertRaisesRegex(ValueError, "confirmed-handoff"):
            validate({**self.baseline, "continuation_cruise": {"version": VERSION}})


if __name__ == "__main__":
    unittest.main()
