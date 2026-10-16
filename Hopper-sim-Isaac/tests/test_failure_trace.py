"""CPU-only trace integration tests; no Isaac or rollout is started."""
import ast
import csv
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/failure_trace.py'))
FailureTrace = MODULE['FailureTrace']
Session = MODULE['EvaluationFailureTrace']
StrictHitReferenceTrace = MODULE['StrictHitReferenceTrace']


class FakeEnv:
    def __init__(self, n=16):
        self.num_envs, self.step_dt, self._spring_joint_id = n, .01, 0
        self.common_step_counter = 100
        self.reset_terminated = torch.zeros(n, dtype=torch.bool)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self._robot = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.zeros(n, 3), root_lin_vel_w=torch.zeros(n, 3),
            root_ang_vel_b=torch.zeros(n, 3),
            root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
            joint_pos=torch.zeros(n, 1), joint_vel=torch.zeros(n, 1)))
        self.target = torch.ones(n, 2)
        self.commands = SimpleNamespace(lookahead=lambda: (self.target, self.target))
        for name in ('_raw_target_u', '_shaped_target_u', '_motor_u'):
            setattr(self, name, torch.full((n, 4), .4))
        self.dr_thrust_curve_shape = torch.ones(n, 4)
        self.dr_thrust_scale = torch.ones(n, 4)
        self.dr_mass_multi = torch.ones(n, 1)
        self.dr_inertia_multi = torch.ones(n, 3)
        self.dr_tm = torch.full((n, 4), .0674)
        self._hardware_global_thrust_scale = torch.full((n, 1), .9)
        self._hardware_spring_stiffness_scale = torch.ones(n)
        self._hardware_spring_damping_scale = torch.ones(n)
        self._touchdown_wrench_remaining = torch.zeros(n)
        self._launch_wrench_impulse_nm = torch.zeros(n, 3)
        self._touchdown_wrench_nm = torch.zeros(n, 3)
        self._hardware_rp_effectiveness = torch.eye(2).repeat(n, 1, 1)
        self._hardware_body_com_offset_b = torch.zeros(n, 3)
        self._thrust = torch.zeros(n, 1, 3)
        self._moment = torch.zeros(n, 1, 3)
        for name in (*Session.COUNTERS, '_hardware_hard_mask', '_cycle_active',
                     '_initial_drop_pending', '_touchdown_event', '_liftoff_event',
                     '_apex_event', '_setup_touchdown_event', '_moving_long_recovery_remaining',
                     '_moving_launch_remaining', '_launch_stability_remaining'):
            setattr(self, name, torch.zeros(n))
        self.calls = 0

    def _get_rewards(self):
        self.calls += 1
        self._touchdown_event[2] = 1
        self.target += 1  # emulate ordinary cycle event/target advance
        return torch.arange(self.num_envs, dtype=torch.float32)


class FailureTraceTests(unittest.TestCase):
    def test_first_death_window_wrap_and_post_reset_label(self):
        trace = FailureTrace(16, .01, .02)
        for step in range(1, 10):
            death = torch.zeros(16, dtype=torch.bool)
            death[2] = step in (5, 8)
            trace.observe(step, {'value': torch.full((16,), float(step))}, death, death)
        event = trace.events[2]
        self.assertEqual([r['step'] for r in event['samples']], [3, 4, 5, 6, 7])
        self.assertEqual([r['same_episode_as_failure'] for r in event['samples']],
                         [True, True, True, False, False])
        self.assertEqual(event['samples'][2]['value'], 5.)
        self.assertTrue(event['samples'][2]['reset_pending'])
        self.assertEqual(len(trace.history), 3)
        self.assertEqual(len(trace.events), 1)

    def test_timeout_excluded_and_pre_window_does_not_cross_reset(self):
        trace = FailureTrace(2, .01, .05)
        for step in range(1, 5):
            death = torch.tensor([step == 3, False])
            reset = torch.tensor([step in (1, 3), step == 2])
            trace.observe(step, {'value': torch.zeros(2)}, death, reset)
        self.assertEqual(list(trace.events), [0])
        self.assertEqual([r['step'] for r in trace.events[0]['samples']], [2, 3, 4])
        with tempfile.TemporaryDirectory() as d:
            report = trace.export(d)
            self.assertFalse(report['events'][0]['complete_pre_window'])
            self.assertFalse(report['events'][0]['complete_post_window'])

    def test_rates_never_cross_reset_and_partial_tail_exported(self):
        trace = FailureTrace(1, .01, .02)
        for step in range(1, 5):
            values = {'policy_action_1': torch.tensor([float(step)]),
                      'motor_pwm_1': torch.tensor([step * 10.])}
            death = torch.tensor([step == 3])
            trace.observe(step, values, death, death)
        with tempfile.TemporaryDirectory() as d:
            trace.export(d)
            with next(Path(d).glob('*.csv')).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[1]['policy_action_1_rate_per_s']), 100.)
            self.assertEqual(rows[-1]['action_rate_valid'], 'False')
            self.assertEqual(rows[-1]['policy_action_1_rate_per_s'], 'nan')
            self.assertEqual(len(rows), 4)

    def test_no_failure_index_and_rng_unchanged(self):
        trace = FailureTrace(16, .01)
        before = torch.get_rng_state().clone()
        trace.observe(1, {'value': torch.ones(16)}, torch.zeros(16, dtype=torch.bool),
                      torch.zeros(16, dtype=torch.bool))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(trace.export(d)['events'], [])
            self.assertTrue((Path(d) / 'index.json').exists())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_snapshot_is_read_only_and_model_force_matches_formula(self):
        env = FakeEnv()
        original = env._motor_u.clone()
        before = torch.get_rng_state().clone()
        values = MODULE['plant_values'](env)
        self.assertTrue(all(v.shape == (16,) for v in values.values()))
        expected = (-.2371*.4**2 + .8130*.4 + .0113)*.9
        torch.testing.assert_close(values['rotor_model_F_N_1'], torch.full((16,), expected))
        self.assertTrue(torch.equal(original, env._motor_u))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_actual_reward_wrapper_terminal_snapshot_and_restore(self):
        env = FakeEnv()
        session = Session(env)
        before_rng = torch.get_rng_state().clone()
        actions = torch.full((16, 4), .3)
        session.begin_step(actions)
        actions.zero_()  # input storage must be independent of caller buffers
        env._physical_death_total[2] = 1
        env._tilt_death_total[2] = 1
        env.reset_terminated[2] = True
        env._robot.data.root_pos_w[2, 2] = .09
        reward = env._get_rewards()
        env._robot.data.root_pos_w[2, 2] = 1.  # emulate subsequent automatic reset
        self.assertEqual(env.calls, 1)
        torch.testing.assert_close(reward, torch.arange(16, dtype=torch.float32))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        row = session.trace.events[2]['samples'][0]
        self.assertAlmostEqual(row['root_pos_w_z'], .09)
        self.assertAlmostEqual(row['policy_action_1'], .3)
        self.assertEqual(row['target_at_action_w_x'], 1.)
        self.assertEqual(row['target_after_events_w_x'], 2.)
        self.assertEqual(row['touchdown'], 1.)
        self.assertEqual(row['trace_episode_hop_id'], 1.)
        with tempfile.TemporaryDirectory() as d:
            session.close(d)
        self.assertNotIn('_get_rewards', vars(env))
        env._get_rewards()
        self.assertEqual(env.calls, 2)

    def test_optional_next_target_is_snapshotted_before_queue_advance(self):
        env = FakeEnv()
        env.commands = SimpleNamespace(lookahead=lambda: (env.target, env.target + 2))
        session = Session(env, record_next_target=True)
        session.begin_step(torch.zeros(16, 4))
        env._get_rewards()
        fields = session.trace.fields
        matrix = session.trace.history[-1][2]
        self.assertAlmostEqual(float(matrix[0, fields.index('target_at_action_w_x')]), 1)
        self.assertAlmostEqual(float(matrix[0, fields.index('next_target_at_action_w_x')]), 3)
        self.assertAlmostEqual(float(matrix[0, fields.index('target_after_events_w_x')]), 2)
        with tempfile.TemporaryDirectory() as tmp:
            session.close(tmp)

    def test_invalid_schema_or_step_rejected(self):
        trace = FailureTrace(2, .01)
        mask = torch.zeros(2, dtype=torch.bool)
        with self.assertRaises(ValueError):
            trace.observe(1, {'bad': torch.zeros(2, 2)}, mask, mask)
        trace = FailureTrace(2, .01)
        trace.observe(1, {'ok': torch.zeros(2)}, mask, mask)
        with self.assertRaises(ValueError):
            trace.observe(1, {'ok': torch.zeros(2)}, mask, mask)

    def test_synthetic_trace_offline_plot(self):
        plotter = runpy.run_path(str(ROOT / 'tools/plot_failure_trace.py'))
        env = FakeEnv()
        session = Session(env)
        for step in range(1, 5):
            session.begin_step(torch.full((16, 4), step*.1))
            env.reset_terminated[:] = False
            if step == 3:
                env._physical_death_total[2] += 1
                env.reset_terminated[2] = True
            env._get_rewards()
        with tempfile.TemporaryDirectory() as directory:
            report = session.close(directory)
            output = plotter['plot_csv'](Path(directory) / report['events'][0]['csv'])
            self.assertGreater(output.stat().st_size, 1000)

    def test_strict_hit_reference_requires_full_pre_window_and_labels_reset_tail(self):
        trace = FailureTrace(2, .01, .02)
        references = StrictHitReferenceTrace(trace)
        for step in range(1, 7):
            reset = torch.tensor([step == 5, False])
            trigger = torch.tensor([step in (1, 3), False])
            death = torch.tensor([step == 5, False])
            trace.observe(step, {'value': torch.full((2,), float(step))}, death, reset)
            references.observe(trigger)
        event = references.events[0]
        self.assertEqual(event['trigger_step'], 3)
        self.assertEqual([r['step'] for r in event['samples']], [1, 2, 3, 4, 5])
        self.assertTrue(event['samples'][-1]['reset_pending'])
        with tempfile.TemporaryDirectory() as directory:
            report = references.export(directory)
            self.assertFalse(report['events'][0]['uninterrupted_post_window'])
            self.assertTrue(report['events'][0]['precedes_first_failure'])
            with (Path(directory) / report['events'][0]['csv']).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[2]['reference_trigger'], 'True')
            self.assertNotIn('failure_trigger', rows[2])

    def test_strict_hit_reference_uses_action_time_target_and_is_read_only(self):
        env = FakeEnv()
        env._touchdown_event = torch.zeros(16, dtype=torch.bool)
        env._initial_drop_pending = torch.zeros(16, dtype=torch.bool)
        env._completed_stationary_command = torch.zeros(16, dtype=torch.bool)
        env._settled_apex_height = torch.ones(16)
        env._settled_apex_target = torch.ones(16)
        env.foot = torch.ones(16, 3)
        env.foot_position_w = lambda: env.foot
        session = Session(env, window_s=.02)
        before_rng = torch.get_rng_state().clone()
        for step in range(1, 6):
            env.target.fill_(1)  # reward advances it after the action snapshot
            env._touchdown_event.zero_()
            if step == 3:
                env._settled_apex_height[2] = .9  # not a strict hit
            else:
                env._settled_apex_height[2] = 1.
            session.begin_step(torch.full((16, 4), .3))
            env._get_rewards()
        self.assertEqual(session.trace.events, {})
        self.assertEqual(session.reference_trace.events[2]['trigger_step'], 4)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        with tempfile.TemporaryDirectory() as directory:
            report = session.close(Path(directory) / 'failure_traces')
            reference = report['strict_hit_references']
            self.assertEqual(reference['captured_environments'], 1)
            self.assertTrue(Path(reference['index']).is_file())

    def test_directional_entrypoint_only_adds_trace_flag(self):
        import run_directional_eval as entry
        manifest = dict(stability_profile='two_circle_v18', observation_layout='history_foot_v19',
                        two_circle_route=dict(short_hop_radius_m=[.12, .3], long_hop_radius_m=[.5, .75]),
                        reverse_turn_halfwidth_deg=30., policy_target_root_z_m=1., target_tolerance_m=.1)
        commands = []
        for extra in ([], ['--failure-trace']):
            with patch.object(sys, 'argv', ['test', '/tmp/fake.pt', '--root', '/tmp/trace-test-unused',
                              '--scenario', 'reverse_only', '--condition', 'robust',
                              '--robust-seed', '383', '--dry-run', *extra]), \
                 patch.object(Path, 'is_file', return_value=True), \
                 patch.object(Path, 'exists', return_value=False), \
                 patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
                 patch.object(entry.subprocess, 'run') as run:
                entry.main()
                run.assert_called_once()
                commands.append(run.call_args.args[0])
        window = commands[1].index('--failure-trace-window-s')
        self.assertEqual(commands[1][window + 1], '0.5')
        del commands[1][window:window + 2]
        commands[1].remove('--failure-trace')
        self.assertEqual(commands[0], commands[1])
        self.assertEqual(commands[0][commands[0].index('--num-envs')+1], '16')
        self.assertEqual(commands[0][commands[0].index('--seed')+1], '383')

    def test_two_second_trace_retains_earlier_touchdown(self):
        trace = FailureTrace(1, .01, 2.0)
        for step in range(1, 101):
            death = torch.tensor([step == 100])
            trace.observe(step, {'touchdown': torch.tensor([step == 25])},
                          death, death)
        rows = trace.events[0]['samples']
        self.assertEqual(rows[0]['step'], 1)
        self.assertTrue(rows[24]['touchdown'])
        self.assertEqual(trace.window_steps, 200)

    def test_long_failure_window_keeps_reference_window_at_two_seconds(self):
        trace = FailureTrace(1, 1.0, 4.0)
        references = StrictHitReferenceTrace(trace)
        for step in range(1, 8):
            no_reset = torch.zeros(1, dtype=torch.bool)
            trace.observe(step, {'value': torch.tensor([float(step)])}, no_reset, no_reset)
            references.observe(torch.tensor([step == 5]))
        self.assertEqual(trace.window_steps, 4)
        self.assertEqual(references.window_steps, 2)
        self.assertEqual([r['step'] for r in references.events[0]['samples']],
                         [3, 4, 5, 6, 7])
        with tempfile.TemporaryDirectory() as directory:
            report = references.export(directory)
            self.assertEqual(report['window_s'], 2.0)
            self.assertTrue(report['events'][0]['uninterrupted_post_window'])

    def test_directional_entrypoint_passes_explicit_trace_window(self):
        import run_directional_eval as entry
        manifest = dict(stability_profile='two_circle_v18', observation_layout='history_foot_v19',
                        two_circle_route=dict(short_hop_radius_m=[.12, .3], long_hop_radius_m=[.5, .75]),
                        reverse_turn_halfwidth_deg=30., policy_target_root_z_m=1., target_tolerance_m=.1)
        with patch.object(sys, 'argv', ['test', '/tmp/fake.pt', '--root', '/tmp/trace-test-unused',
                          '--scenario', 'reverse_only', '--condition', 'robust',
                          '--robust-seed', '383', '--failure-trace', '--failure-trace-window-s',
                          '8', '--dry-run']), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'exists', return_value=False), \
             patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
             patch.object(entry.subprocess, 'run') as run:
            entry.main()
            command = run.call_args.args[0]
        self.assertEqual(command[command.index('--failure-trace-window-s') + 1], '8.0')
        self.assertEqual(command[command.index('--num-envs') + 1], '16')

    def test_real_eval_loop_installs_after_reset_and_captures_action_before_step(self):
        tree = ast.parse((ROOT / 'train_tracking_recovery.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        source = ast.unparse(main)
        self.assertLess(source.index('obs, _ = env.reset()'), source.index('EvaluationFailureTrace('))
        self.assertLess(source.index('failure_trace_session.begin_step(actions)'),
                        source.index('env.step(actions)'))


if __name__ == '__main__':
    unittest.main()
