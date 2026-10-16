"""CPU regressions for the v18 low-hop transition reward and CLI contract."""
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
from Quadhopper_Planner_Random.low_hop_recovery_v18 import LowHopRecoveryDebt, SETTINGS_V18
from Quadhopper_Planner_Random.path_free_stage2 import manifest_flags, PROTOCOL_V3

ROOT = Path(__file__).resolve().parents[1]


def step(state, **changes):
    n = len(state.debt)
    values = dict(touchdown=False, previous_moving=True, previous_apex=1.,
                  previous_target=1., previous_valid=True, next_target=1.,
                  root_z=.3, cycle_active=False, initial_drop=False,
                  episode_end=False, physical_death=False)
    values.update(changes)
    return state.update(**{k: torch.tensor(v if isinstance(v, list) else [v]*n)
                           for k, v in values.items()})


class DebtTest(unittest.TestCase):
    def balanced(self, state):
        self.assertAlmostEqual(state.summary()['v18_accounting_residual'], 0., places=4)

    def test_low_moving_touchdown_charges_even_when_foot_bonus_would_be_zero(self):
        s = LowHopRecoveryDebt(1, 'cpu')
        self.assertAlmostEqual(step(s, touchdown=True, previous_apex=.7).item(), -18., places=5)
        self.assertEqual(s.summary()['v18_low_touchdowns'], 1)
        self.balanced(s)

    def test_next_ascent_repays_proportional_height_without_duplicate_or_overshoot(self):
        s = LowHopRecoveryDebt(1, 'cpu')
        step(s, touchdown=True, previous_apex=.5)
        self.assertAlmostEqual(step(s, cycle_active=True, root_z=.65).item(), 15., places=4)
        self.assertEqual(step(s, cycle_active=True, root_z=.60).item(), 0.)
        self.assertAlmostEqual(step(s, cycle_active=True, root_z=1.05).item(), 15., places=4)
        self.assertEqual(step(s, cycle_active=True, root_z=1.2).item(), 0.)
        self.assertEqual(s.summary()['v18_fully_recovered'], 1)
        self.assertEqual(s.summary()['v18_recovery_hops'], 1)
        self.balanced(s)

    def test_next_apex_stays_low_then_touchdown_forfeits_unrepaid_debt(self):
        s = LowHopRecoveryDebt(1, 'cpu')
        step(s, touchdown=True, previous_apex=.6)  # debt 24
        self.assertAlmostEqual(step(s, cycle_active=True, root_z=.65).item(), 12., places=4)
        step(s, touchdown=True, previous_apex=.65)  # 12 lost, new debt 21
        self.assertAlmostEqual(s.summary()['v18_forfeited'], 12., places=4)
        self.assertAlmostEqual(s.summary()['v18_active_debt'], 21., places=4)
        self.balanced(s)

    def test_initial_stationary_valid_height_and_termination_never_open_debt(self):
        s = LowHopRecoveryDebt(4, 'cpu')
        reward = step(s, touchdown=True, previous_apex=.4,
                      previous_moving=[False,True,True,True],
                      previous_valid=[True,False,True,True],
                      initial_drop=[False,False,True,False],
                      episode_end=[False,False,False,True])
        self.assertEqual(reward.tolist(), [0.]*4)
        self.assertEqual(s.summary()['v18_charged'], 0.)
        self.balanced(s)

    def test_fully_repaid_debt_has_zero_undiscounted_net_return(self):
        s = LowHopRecoveryDebt(1, 'cpu')
        charge = step(s, touchdown=True, previous_apex=.75)
        credit = step(s, cycle_active=True, root_z=1.)
        self.assertAlmostEqual((charge + credit).item(), 0., places=5)
        self.balanced(s)

    def test_death_timeout_reset_forfeit_only_unpaid_balance(self):
        s = LowHopRecoveryDebt(3, 'cpu')
        step(s, touchdown=True, previous_apex=.5)
        step(s, cycle_active=True, root_z=.65)
        step(s, episode_end=[True,True,False], physical_death=[True,False,False])
        s.reset(torch.tensor([2]))
        self.assertAlmostEqual(s.summary()['v18_forfeited'], 45., places=4)
        self.assertAlmostEqual(s.summary()['v18_death_forfeited'], 15., places=4)
        self.assertEqual(s.summary()['v18_active_count'], 0)
        self.balanced(s)

    def test_capped_debt_and_frozen_target_with_no_reward_on_touchdown(self):
        s = LowHopRecoveryDebt(1, 'cpu')
        self.assertEqual(step(s, touchdown=True, previous_apex=.2).item(), -30.)
        self.assertAlmostEqual(step(s, cycle_active=True, root_z=.65,
                                    next_target=2.).item(), 15., places=4)
        step(s, touchdown=True, previous_apex=1.)
        self.assertEqual(s.summary()['v18_active_count'], 0)
        self.balanced(s)


def production_class():
    class Parent:
        def _update_cycle_events(self):
            pass

        def _get_dones(self):
            self._v16_physical_death = self.death.clone()
            return self.death, self.timeout

        def _get_rewards(self):
            self.calls += 1
            return torch.full((self.num_envs,), 123.)

        def _reset_idx(self, ids):
            pass

        def stage2_summary(self):
            return {}

    tree = ast.parse((ROOT/'Quadhopper_Planner_Random/path_free_stage2_v18_env.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    ns = dict(torch=torch, PathFreeStage2V16Env=Parent,
              LowHopRecoveryDebt=LowHopRecoveryDebt, SETTINGS_V18=SETTINGS_V18)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<production-v18>', 'exec'), ns)
    return ns['PathFreeStage2V18Env']


class IntegrationTest(unittest.TestCase):
    def test_wrapper_scores_live_touchdown_then_next_measured_height(self):
        env = object.__new__(production_class())
        env.num_envs, env.device, env.calls = 1, 'cpu', 0
        env._robot = NS(data=NS(root_pos_w=torch.tensor([[0.,0.,.3]])),
                        _ALL_INDICES=torch.arange(1))
        env.extras = {'log': {}}
        env._touchdown_event = torch.tensor([True])
        env.commands = NS(lookahead=lambda: (torch.tensor([[.14, 0.]]), None),
                          anchor_w=torch.zeros(1,2))
        env._completed_stationary_command = torch.tensor([False])
        env._settled_apex_valid = torch.tensor([True])
        env._settled_apex_height = torch.tensor([.7])
        env._settled_apex_target = torch.tensor([1.])
        env._active_target_height = torch.tensor([1.])
        env._cycle_active = torch.tensor([False])
        env._initial_drop_pending = torch.tensor([False])
        env.death = env.timeout = torch.tensor([False])
        env._update_cycle_events()
        env._get_dones()
        self.assertAlmostEqual(env._get_rewards().item(), 105., places=4)
        env._touchdown_event[:] = False
        env._cycle_active[:] = True
        env._robot.data.root_pos_w[:, 2] = .65
        env._get_dones()
        self.assertAlmostEqual(env._get_rewards().item(), 132., places=4)
        self.assertEqual(env.calls, 2)
        self.assertAlmostEqual(env.stage2_summary()['v18_repaid'], 9., places=4)
        env._reset_idx(None)
        self.assertAlmostEqual(env.stage2_summary()['v18_forfeited'], 9., places=4)

    def test_short_command_under_stationary_radius_is_still_recoverable(self):
        env = object.__new__(production_class())
        env.num_envs, env.device = 2, 'cpu'
        env._robot = NS(_ALL_INDICES=torch.arange(2))
        env.commands = NS(lookahead=lambda: (torch.tensor([[0.,0.],[.14,0.]]), None),
                          anchor_w=torch.zeros(2,2))
        env._touchdown_event = torch.tensor([True, True])
        env._update_cycle_events()
        self.assertEqual(env._v18_completed_nonzero_command.tolist(), [False, True])

    def test_manifest_rebuild_and_directional_eval_restore(self):
        manifest = dict(protocol=PROTOCOL_V3, apex_event_fix_v15=True,
                        terminal_penalty_fix_v16=True, low_hop_recovery_v18=True,
                        stability_profile='two_circle_v18', observation_layout='history_foot_v19',
                        two_circle_route=dict(short_hop_radius_m=[.12,.3],long_hop_radius_m=[.5,.75]),
                        reverse_turn_halfwidth_deg=30,policy_target_root_z_m=1.,target_tolerance_m=.1)
        self.assertIn('--low-hop-recovery-v18', manifest_flags(manifest))
        for wrong in ({**manifest,'terminal_penalty_fix_v16':False},
                      {**manifest,'deferred_landing_v17':True}):
            with self.assertRaises(ValueError):
                manifest_flags(wrong)
        import run_directional_eval
        with patch('sys.argv',['test','/tmp/v18_model.pt','--dry-run','--scenario','reverse_only']), \
             patch.object(Path,'is_file',return_value=True), patch.object(Path,'exists',return_value=False), \
             patch.object(Path,'read_text',return_value=json.dumps(manifest)), \
             patch.object(run_directional_eval.subprocess,'run') as run:
            run_directional_eval.main()
            self.assertGreater(run.call_count, 0)
            for call in run.call_args_list:
                self.assertIn('--low-hop-recovery-v18',call.args[0])

    def test_launcher_two_arms_use_isaac_and_differ_only_in_v18(self):
        commands={}
        for variant in ('control','fix'):
            result=subprocess.run(['bash',str(ROOT/'run_train_two_circle_path_free_stage2_v18_low_hop_recovery_075.sh')],
                cwd=ROOT,capture_output=True,text=True,
                env={**os.environ,'VARIANT':variant,'DRY_RUN':'1','OUTPUT':'/tmp/v18-test',
                     'ITERATIONS':'100','NUM_ENVS':'256'})
            self.assertEqual(result.returncode,0,result.stderr)
            commands[variant]=shlex.split(result.stdout.strip())
            self.assertTrue(commands[variant][0].endswith('/python.sh'))
            self.assertIn('--terminal-penalty-fix-v16',commands[variant])
            self.assertIn('--apex-event-fix-v15',commands[variant])
            self.assertNotIn('--deferred-landing-v17',commands[variant])
        commands['fix'].remove('--low-hop-recovery-v18')
        self.assertEqual(commands['fix'],commands['control'])


if __name__=='__main__':
    unittest.main()
