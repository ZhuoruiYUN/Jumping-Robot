import unittest
import importlib.util
from pathlib import Path

import torch
from types import SimpleNamespace

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'Quadhopper_Planner_Random/flight_stability.py'
)
SPEC = importlib.util.spec_from_file_location('quadhopper_flight_stability', MODULE_PATH)
STABILITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STABILITY)
differential_action_rate_cost = STABILITY.differential_action_rate_cost
launch_pitch_command_cost = STABILITY.launch_pitch_command_cost
rotor_wrench_torque_from_com_offset = STABILITY.rotor_wrench_torque_from_com_offset
correlated_launch_wrench = STABILITY.correlated_launch_wrench


class FlightStabilityTests(unittest.TestCase):
    def test_rate_cost_symmetric_deadband(self):
        rate = torch.tensor([-0.5, -0.18, 0.0, 0.18, 0.5])
        cost = STABILITY.angular_rate_excess_cost(rate, 0.18)
        torch.testing.assert_close(cost, torch.tensor([0.1024, 0.0, 0.0, 0.0, 0.1024]))

    def test_phase_masks_exclude_drop_compression_and_stale_timer(self):
        # extension, compression, early flight, later ascent, pre-apex,
        # descent, initial drop, moving target, touchdown-with-stale-timer.
        position = torch.tensor([.01, .01, 0., 0., 0., 0., 0., 0., .01])
        velocity = torch.tensor([-.1, .1, 0., 0., 0., 0., 0., 0., .1])
        active = torch.tensor([False, False, True, True, True, True, True, True, False])
        timer = torch.tensor([0, 0, 15, 0, 0, 0, 15, 15, 12])
        drop = torch.tensor([False] * 6 + [True, False, False])
        stationary = torch.tensor([True] * 7 + [False, True])
        vz = torch.tensor([1., -1., 3., 2., .8, -2., -2., 3., -2.])
        launch, yaw = STABILITY.stability_phase_masks(
            position, velocity, active, timer, drop, stationary, vz, 1.0,
        )
        self.assertEqual(launch.tolist(), [True, False, True, False, False, False, False, False, False])
        self.assertEqual(yaw.tolist(), [False, False, False, False, True, True, False, False, False])

    def test_v9_profile_is_opt_in_and_keeps_nominal_motor_model(self):
        path = MODULE_PATH.with_name('stability_profile.py')
        spec = importlib.util.spec_from_file_location('stability_profile', path)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        cfg = SimpleNamespace(play_motor_time_constant=.0674,
                              deployment_max_airborne_yaw_diagonal_pwm_diff=140.0)
        self.assertEqual(profile.apply_stability_profile(cfg, 'baseline'), {})
        self.assertEqual(cfg.deployment_max_airborne_yaw_diagonal_pwm_diff, 140.0)
        recorded = profile.apply_stability_profile(cfg, 'takeoff_v9')
        self.assertEqual(cfg.play_motor_time_constant, .0674)
        self.assertEqual(cfg.deployment_max_airborne_yaw_diagonal_pwm_diff, 220.0)
        self.assertEqual(recorded['launch_stability_window_s'], .15)
        self.assertEqual(recorded['train_motor_time_constant_max'], .100)
        with self.assertRaises(ValueError):
            profile.apply_stability_profile(cfg, 'typo')

    def test_v10_profile_keeps_v12_yaw_cap_and_nominal_plant(self):
        path = MODULE_PATH.with_name('stability_profile.py')
        spec = importlib.util.spec_from_file_location('stability_profile_v10', path)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        cfg = SimpleNamespace(play_motor_time_constant=.0674,
                              deployment_max_airborne_yaw_diagonal_pwm_diff=140.0)
        settings = profile.apply_stability_profile(cfg, 'launch_robust_v10')
        self.assertEqual(cfg.play_motor_time_constant, .0674)
        self.assertEqual(cfg.deployment_max_airborne_yaw_diagonal_pwm_diff, 140.0)
        self.assertEqual(settings['train_launch_pitch_impulse_max_nm'], .006)
        self.assertEqual(settings['launch_pitch_command_deadband_pwm'], 150.0)

    def test_v11_profile_adds_event_yaw_cost_without_changing_motor_cap(self):
        path = MODULE_PATH.with_name('stability_profile.py')
        spec = importlib.util.spec_from_file_location('stability_profile_v11', path)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        cfg = SimpleNamespace(play_motor_time_constant=.0674,
                              deployment_max_airborne_yaw_diagonal_pwm_diff=140.0)
        settings = profile.apply_stability_profile(cfg, 'yaw_event_v11')
        self.assertEqual(cfg.play_motor_time_constant, .0674)
        self.assertEqual(cfg.deployment_max_airborne_yaw_diagonal_pwm_diff, 140.0)
        self.assertEqual(settings['stationary_yaw_event_penalty_scale'], -260.0)
        self.assertEqual(settings['stationary_yaw_hop_change_penalty_scale'], -90.0)

    def test_v12_profile_is_separate_from_v11_and_preserves_actuator(self):
        path = MODULE_PATH.with_name('stability_profile.py')
        spec = importlib.util.spec_from_file_location('stability_profile_v12', path)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        cfg = SimpleNamespace(play_motor_time_constant=.0674,
                              deployment_max_airborne_yaw_diagonal_pwm_diff=140.0)
        settings = profile.apply_stability_profile(cfg, 'touchdown_robust_v12')
        self.assertEqual(cfg.play_motor_time_constant, .0674)
        self.assertEqual(cfg.deployment_max_airborne_yaw_diagonal_pwm_diff, 140.0)
        self.assertEqual(settings['train_launch_wrench_probability'], .8)
        self.assertEqual(settings['stationary_touchdown_angle_penalty_scale'], -8.0)
        self.assertNotIn('launch_pitch_command_penalty_scale', settings)
        self.assertNotIn('stationary_yaw_hop_change_penalty_scale', settings)

    def test_v13_preserves_v12_attitude_contract_and_rebalances_apex(self):
        path = MODULE_PATH.with_name('stability_profile.py')
        spec = importlib.util.spec_from_file_location('stability_profile_v13', path)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        cfg = SimpleNamespace(play_motor_time_constant=.0674,
                              deployment_max_airborne_yaw_diagonal_pwm_diff=140.0)
        settings = profile.apply_stability_profile(cfg, 'height_robust_v13')
        self.assertEqual(cfg.play_motor_time_constant, .0674)
        self.assertEqual(cfg.deployment_max_airborne_yaw_diagonal_pwm_diff, 140.0)
        self.assertEqual(settings['train_launch_wrench_probability'], .8)
        self.assertEqual(settings['stationary_touchdown_angle_penalty_scale'], -8.0)
        self.assertEqual(settings['stationary_apex_progress_scale'], 22.0)
        self.assertEqual(settings['stationary_predicted_apex_reward_scale'], 36.0)
        self.assertEqual(settings['stationary_apex_shortfall_penalty_scale'], -360.0)

    def test_touchdown_cost_has_deadband_and_bounded_tail(self):
        angle = torch.deg2rad(torch.tensor([[2., -2.], [13., 0.], [90., -90.]]))
        rate = torch.tensor([[.2, -.2], [1.25, .0], [10., -10.]])
        a, r = STABILITY.touchdown_roll_pitch_cost(
            angle, rate, torch.deg2rad(torch.tensor(3.)).item(), .25,
        )
        torch.testing.assert_close(a, torch.tensor([0., 1., 18.]), atol=1e-5, rtol=0)
        torch.testing.assert_close(r, torch.tensor([0., 1., 18.]))

    def test_launch_pitch_cost_excludes_collective_and_yaw(self):
        command = torch.tensor([[.5, .5, .5, .5], [.7, .3, .7, .3],
                                [.65, .65, .35, .35]])
        cost, pitch_pwm = launch_pitch_command_cost(command, 150.0)
        torch.testing.assert_close(pitch_pwm, torch.tensor([0., 0., 300.]))
        torch.testing.assert_close(cost, torch.tensor([0., 0., 2.25]))

    def test_collective_action_change_is_not_penalized(self):
        previous = torch.tensor([[0.1, -0.1, 0.2, -0.2]])
        current = previous + 0.7
        cost, delta = differential_action_rate_cost(current, previous, 0.0)
        torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(cost, torch.zeros_like(cost), atol=1.0e-6, rtol=0.0)

    def test_differential_action_change_is_penalized(self):
        previous = torch.zeros(1, 4)
        current = torch.tensor([[0.2, -0.2, 0.2, -0.2]])
        cost, delta = differential_action_rate_cost(current, previous, 0.025)
        self.assertGreater(float(delta), 0.025)
        self.assertGreater(float(cost), 0.0)

    def test_com_offset_generates_expected_pitch_and_roll_torque(self):
        thrust = torch.tensor([10.0, 20.0])
        offset = torch.tensor([[0.008, 0.006, 0.004], [-0.005, 0.002, -0.003]])
        torque = rotor_wrench_torque_from_com_offset(thrust, offset)
        expected = torch.tensor([[-0.060, 0.080, 0.0], [-0.040, -0.100, 0.0]])
        torch.testing.assert_close(torque, expected)

    def test_persistent_launch_wrench_keeps_first_sample_and_bounds_ar_update(self):
        previous = torch.tensor([[.010, -.010, .001], [.004, .001, -.001]])
        innovation = torch.tensor([[.020, -.020, .010], [.010, -.010, .002]])
        initialized = torch.tensor([False, True])
        bounds = torch.tensor([.012, .010, .003])
        result = correlated_launch_wrench(previous, innovation, initialized, .7, bounds)
        torch.testing.assert_close(result[0], torch.tensor([.012, -.010, .003]))
        expected_second = .7 * previous[1] + (1.0 - .7 ** 2) ** .5 * innovation[1]
        torch.testing.assert_close(result[1], torch.minimum(torch.maximum(expected_second, -bounds), bounds))

    def test_v14_profile_is_task_only_and_has_correlated_launch_wrench(self):
        path = MODULE_PATH.with_name('stability_profile.py')
        spec = importlib.util.spec_from_file_location('stability_profile_v14', path)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        cfg = SimpleNamespace(play_motor_time_constant=.0674,
                              deployment_max_airborne_yaw_diagonal_pwm_diff=140.0)
        settings = profile.apply_stability_profile(cfg, 'cross_hop_v14')
        self.assertEqual(cfg.play_motor_time_constant, .0674)
        self.assertEqual(cfg.deployment_max_airborne_yaw_diagonal_pwm_diff, 140.0)
        self.assertEqual(settings['train_launch_wrench_persistence'], .70)
        self.assertEqual(settings['launch_roll_rate_penalty_scale'], -24.0)

    def test_v15_profile_records_the_post_touchdown_deployment_envelope(self):
        path = MODULE_PATH.with_name('stability_profile.py')
        spec = importlib.util.spec_from_file_location('stability_profile_v15', path)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        cfg = SimpleNamespace(play_motor_time_constant=.0674,
                              deployment_max_airborne_yaw_diagonal_pwm_diff=140.0)
        settings = profile.apply_stability_profile(cfg, 'cross_hop_v15')
        self.assertEqual(cfg.play_motor_time_constant, .0674)
        self.assertEqual(cfg.deployment_max_airborne_yaw_diagonal_pwm_diff, 140.0)
        self.assertEqual(settings['deployment_launch_recovery_window_s'], .16)
        self.assertEqual(settings['deployment_launch_recovery_spread_limit_pwm'], 450.0)
        self.assertEqual(settings['train_launch_wrench_persistence'], .80)
        self.assertEqual(settings['train_launch_yaw_impulse_max_nm'], .006)

    def test_v15_1_is_a_v14_based_ramp_ablation(self):
        path = MODULE_PATH.with_name('stability_profile.py')
        spec = importlib.util.spec_from_file_location('stability_profile_v15_1', path)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        cfg = SimpleNamespace(play_motor_time_constant=.0674,
                              deployment_max_airborne_yaw_diagonal_pwm_diff=140.0)
        settings = profile.apply_stability_profile(cfg, 'cross_hop_v15_1')
        self.assertEqual(cfg.play_motor_time_constant, .0674)
        self.assertEqual(settings['deployment_launch_recovery_window_s'], .16)
        self.assertEqual(settings['deployment_launch_recovery_hold_s'], .08)
        self.assertEqual(settings['deployment_launch_recovery_spread_limit_pwm'], 450.0)
        self.assertEqual(settings['deployment_launch_recovery_spread_limit_end_pwm'], 600.0)
        self.assertEqual(settings['launch_recovery_raw_spread_penalty_scale'], -4.0)
        self.assertEqual(settings['train_launch_wrench_persistence'], .70)
        self.assertEqual(settings['train_launch_yaw_impulse_max_nm'], .003)
        self.assertEqual(settings['stationary_yaw_event_penalty_scale'], -20.0)

    def test_v16_adds_persistent_task_uncertainty_without_actuator_change(self):
        path = MODULE_PATH.with_name('stability_profile.py')
        spec = importlib.util.spec_from_file_location('stability_profile_v16', path)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
        cfg = SimpleNamespace(play_motor_time_constant=.0674,
                              deployment_max_airborne_yaw_diagonal_pwm_diff=140.0)
        settings = profile.apply_stability_profile(cfg, 'cross_hop_v16')
        self.assertEqual(cfg.play_motor_time_constant, .0674)
        self.assertEqual(cfg.deployment_max_airborne_yaw_diagonal_pwm_diff, 140.0)
        self.assertEqual(settings['deployment_launch_recovery_hold_s'], .08)
        self.assertEqual(settings['train_yaw_reaction_imbalance_max'], .14)
        self.assertEqual(settings['train_touchdown_wrench_window_s'], .06)
        self.assertEqual(settings['train_touchdown_wrench_persistence'], .75)
        self.assertEqual(settings['stationary_touchdown_angle_penalty_scale'], -14.0)


if __name__ == '__main__':
    unittest.main()
