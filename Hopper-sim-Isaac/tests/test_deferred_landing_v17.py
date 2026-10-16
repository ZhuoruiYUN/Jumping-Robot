"""CPU event-sequence, production wrapper and launch-contract regressions."""
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from Quadhopper_Planner_Random.deferred_landing_v17 import DeferredLandingReward
from Quadhopper_Planner_Random.path_free_stage2 import endpoint_terms, manifest_flags, PROTOCOL_V3

ROOT = Path(__file__).resolve().parents[1]


def step(state, **changes):
    n = len(state.pending)
    values = dict(landing_reward=0., touchdown=False, apex=False, cycle_active=True,
                  initial_drop=False, episode_end=False, physical_death=False,
                  root_z=.3, vertical_speed=0., clearance=.1, next_target_z=1.)
    values.update(changes)
    return state.update(**{k: torch.tensor(v if isinstance(v, list) else [v]*n)
                           for k, v in values.items()})


def ascend(state):
    step(state, root_z=.5, vertical_speed=2.)
    step(state, root_z=.7, vertical_speed=1.)


class EscrowTest(unittest.TestCase):
    def check_accounting(self, state):
        self.assertAlmostEqual(state.summary()['v17_accounting_residual'], 0., places=5)

    def test_full_next_apex_pays_once_and_preserves_undiscounted_budget(self):
        s = DeferredLandingReward(1, 'cpu')
        r = step(s, landing_reward=100., touchdown=True, apex=True, cycle_active=False)
        self.assertEqual(r['correction'].item(), -50.)
        self.assertEqual(r['paid'].item(), 0.)  # TD fallback is the previous hop
        ascend(s)
        r = step(s, apex=True, root_z=1.)
        self.assertEqual(r['paid'].item(), 50.)
        self.assertEqual(r['settled_sequence'].item(), 1)
        self.assertEqual(step(s, apex=True, root_z=1.)['paid'].item(), 0.)
        self.check_accounting(s)

    def test_low_next_apex_is_continuous_and_uses_frozen_next_target(self):
        s = DeferredLandingReward(1, 'cpu')
        step(s, landing_reward=100., touchdown=True, root_z=.2, next_target_z=1.)
        step(s, root_z=.4, vertical_speed=2., next_target_z=2.)
        step(s, root_z=.5, vertical_speed=1., next_target_z=2.)
        r = step(s, apex=True, root_z=.6, next_target_z=2.)
        self.assertAlmostEqual(r['paid'].item(), 25., places=5)
        self.assertAlmostEqual(s.summary()['v17_quality_forfeit_reward'], 25., places=5)
        self.check_accounting(s)

    def test_spring_liftoff_without_upward_clearance_cannot_pay(self):
        s = DeferredLandingReward(1, 'cpu')
        step(s, landing_reward=100., touchdown=True)
        step(s, vertical_speed=-1.5)
        step(s, vertical_speed=2., clearance=.01)
        step(s, vertical_speed=2., clearance=.01)
        self.assertEqual(step(s, apex=True, root_z=1.)['paid'].item(), 0.)
        step(s, touchdown=True, apex=True)
        self.assertEqual(s.summary()['v17_cancelled_reward'], 50.)
        self.check_accounting(s)

    def test_ascent_confirmation_requires_consecutive_samples(self):
        s = DeferredLandingReward(1, 'cpu')
        step(s, touchdown=True, landing_reward=100.)
        step(s, vertical_speed=2.)
        step(s, vertical_speed=0.)
        step(s, vertical_speed=2.)
        self.assertEqual(step(s, apex=True, root_z=1.)['paid'].item(), 0.)

    def test_death_preempts_apex_and_terminal_touchdown_reservation(self):
        s = DeferredLandingReward(2, 'cpu')
        step(s, touchdown=True, landing_reward=100.)
        ascend(s)
        r = step(s, apex=True, root_z=1., episode_end=True, physical_death=True,
                 touchdown=[False, True], landing_reward=[0., 100.])
        self.assertEqual(r['paid'].tolist(), [0., 0.])
        self.assertEqual(s.pending.tolist(), [0., 0.])
        self.assertEqual(s.summary()['v17_death_cancelled_reward'], 150.)
        self.check_accounting(s)

    def test_timeout_cancels_without_counting_death(self):
        s = DeferredLandingReward(1, 'cpu')
        step(s, touchdown=True, landing_reward=100.)
        ascend(s)
        step(s, episode_end=True, apex=True, root_z=1.)
        self.assertEqual(s.summary()['v17_cancelled_reward'], 50.)
        self.assertEqual(s.summary()['v17_death_cancelled_reward'], 0.)
        self.check_accounting(s)

    def test_missed_next_apex_cannot_be_carried_across_touchdowns(self):
        s = DeferredLandingReward(1, 'cpu')
        step(s, touchdown=True, landing_reward=100.)
        ascend(s)
        step(s, touchdown=True, apex=True, landing_reward=80.)
        self.assertEqual(s.summary()['v17_paid_reward'], 0.)
        self.assertEqual(s.summary()['v17_cancelled_reward'], 50.)
        ascend(s)
        r = step(s, apex=True, root_z=1.)
        self.assertEqual(r['paid'].item(), 40.)
        self.assertEqual(r['settled_sequence'].item(), 2)
        self.check_accounting(s)

    def test_partial_reset_preserves_other_env_and_never_reuses_escrow(self):
        s = DeferredLandingReward(2, 'cpu')
        step(s, touchdown=True, landing_reward=100.)
        s.reset(torch.tensor([0]))
        ascend(s)
        r = step(s, apex=True, root_z=1.)
        self.assertEqual(r['paid'].tolist(), [0., 50.])
        self.assertEqual(s.summary()['v17_reset_cancelled_reward'], 50.)
        s.reset(torch.arange(2))
        self.check_accounting(s)

    def test_initial_drop_and_invalid_height_never_create_reward(self):
        s = DeferredLandingReward(1, 'cpu')
        step(s, touchdown=True, initial_drop=True)
        ascend(s)
        self.assertEqual(step(s, apex=True, root_z=1.)['paid'].item(), 0.)
        step(s, touchdown=True, landing_reward=100., root_z=1., next_target_z=1.)
        ascend(s)
        self.assertEqual(step(s, apex=True, root_z=1.2)['paid'].item(), 0.)
        self.check_accounting(s)


def wrapper_class():
    class Parent:
        def _get_dones(self):
            self._v16_physical_death = self.death.clone()
            return self.death, self.timeout

        def _get_rewards(self):
            self.calls += 1
            landing, _, _ = endpoint_terms(self._stage2_foot_error,
                self._stage2_valid_touchdown, self._stage2_power_w, self.step_dt, self.cfg)
            self._stage2_totals[2] += landing.sum()
            return landing + 123.

        def _reset_idx(self, ids):
            pass

        def stage2_summary(self):
            return {'stage2_foot_reward_per_env': self._stage2_totals[2].item()/self.num_envs}

    tree = ast.parse((ROOT/'Quadhopper_Planner_Random/path_free_stage2_v17_env.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    ns = dict(PathFreeStage2V16Env=Parent, endpoint_terms=endpoint_terms,
              DeferredLandingReward=DeferredLandingReward)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<production-v17>', 'exec'), ns)
    return ns['PathFreeStage2V17Env']


class IntegrationTest(unittest.TestCase):
    def test_production_wrapper_reward_totals_dones_and_reset(self):
        env = object.__new__(wrapper_class())
        env.num_envs, env.device, env.step_dt, env.calls = 1, 'cpu', .01, 0
        env.cfg = NS(stage2_foot_precision_scale=60., stage2_foot_precision_width_m=.06,
                     stage2_foot_hit_scale=40., stage2_foot_hit_radius_m=.05,
                     stage2_energy_penalty_scale=0.)
        env._robot = NS(data=NS(root_pos_w=torch.tensor([[0., 0., .3]]),
                               root_lin_vel_w=torch.zeros(1, 3)), _ALL_INDICES=torch.arange(1))
        env.foot_position_w = lambda: torch.tensor([[0., 0., .12]])
        env.foot_geometry = {'sphere_radius_m': .02}
        env._stage2_foot_error = torch.zeros(1)
        env._stage2_power_w = torch.zeros(1)
        env._stage2_totals = torch.zeros(5, dtype=torch.float64)
        env._active_target_height = torch.ones(1)
        env.extras = {'log': {}}
        for name in ('death', 'timeout', '_initial_drop_pending', '_apex_event', '_cycle_active'):
            setattr(env, name, torch.tensor([False]))
        env._stage2_valid_touchdown = env._touchdown_event = torch.tensor([True])
        env._get_dones()
        self.assertEqual(env._get_rewards().item(), 173.)
        env._stage2_valid_touchdown = env._touchdown_event = torch.tensor([False])
        env._cycle_active[:] = True
        env._robot.data.root_lin_vel_w[:, 2] = 1.
        for z in (.5, .7):
            env._robot.data.root_pos_w[:, 2] = z
            env._get_dones()
            self.assertEqual(env._get_rewards().item(), 123.)
        env._robot.data.root_pos_w[:, 2] = 1.
        env._robot.data.root_lin_vel_w[:, 2] = 0.
        env._apex_event[:] = True
        env._get_dones()
        self.assertEqual(env._get_rewards().item(), 173.)
        self.assertEqual(env.calls, 4)
        self.assertEqual(env.stage2_summary()['stage2_foot_reward_per_env'], 100.)
        self.assertEqual(env.stage2_summary()['v17_paid_reward'], 50.)
        env._reset_idx(None)
        self.assertEqual(env._v17_deferred.pending.item(), 0.)

    def test_manifest_and_directional_eval_restore_v17(self):
        manifest = dict(protocol=PROTOCOL_V3, apex_event_fix_v15=True,
                        terminal_penalty_fix_v16=True, deferred_landing_v17=True,
                        stability_profile='two_circle_v18', observation_layout='history_foot_v19',
                        two_circle_route=dict(short_hop_radius_m=[.12,.3], long_hop_radius_m=[.5,.75]),
                        reverse_turn_halfwidth_deg=30, policy_target_root_z_m=1., target_tolerance_m=.1)
        self.assertIn('--deferred-landing-v17', manifest_flags(manifest))
        with self.assertRaises(ValueError):
            manifest_flags({**manifest, 'terminal_penalty_fix_v16': False})
        import run_directional_eval
        with patch('sys.argv', ['test', '/tmp/v17_model.pt', '--dry-run', '--scenario', 'reverse_only']), \
             patch.object(Path, 'is_file', return_value=True), patch.object(Path, 'exists', return_value=False), \
             patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
             patch.object(run_directional_eval.subprocess, 'run') as run:
            run_directional_eval.main()
            self.assertGreater(run.call_count, 0)
            for call in run.call_args_list:
                for flag in ('--deferred-landing-v17', '--terminal-penalty-fix-v16', '--apex-event-fix-v15'):
                    self.assertIn(flag, call.args[0])

    def test_launcher_changes_only_deferred_flag_and_uses_isaac_python(self):
        commands = {}
        for variant in ('control', 'fix'):
            result = subprocess.run(['bash', str(ROOT/'run_train_two_circle_path_free_stage2_v17_deferred_landing_075.sh')],
                cwd=ROOT, capture_output=True, text=True,
                env={**os.environ, 'VARIANT': variant, 'DRY_RUN': '1', 'OUTPUT': '/tmp/v17-test',
                     'ITERATIONS': '100', 'NUM_ENVS': '256'})
            self.assertEqual(result.returncode, 0, result.stderr)
            commands[variant] = shlex.split(result.stdout.strip())
            self.assertTrue(commands[variant][0].endswith('/python.sh'))
            for flag in ('--terminal-penalty-fix-v16', '--apex-event-fix-v15'):
                self.assertIn(flag, commands[variant])
        commands['fix'].remove('--deferred-landing-v17')
        self.assertEqual(commands['fix'], commands['control'])


if __name__ == '__main__':
    unittest.main()
