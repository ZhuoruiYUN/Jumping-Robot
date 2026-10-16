"""CPU tests using the actual inherited death mask and terminal expression."""
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
from Quadhopper_Planner_Random.terminal_penalty_fix_v16 import terminal_reward_correction
from Quadhopper_Planner_Random.path_free_stage2 import PROTOCOL_V3, manifest_flags

ROOT = Path(__file__).resolve().parents[1]


def class_node(relative, name):
    tree = ast.parse((ROOT / relative).read_text())
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def load_actual_reward_and_dones():
    stable = class_node('Quadhopper_Stable/quadhopper_env.py', 'QuadhopperEnv')
    method = next(n for n in stable.body if isinstance(n, ast.FunctionDef) and n.name == '_get_rewards')
    # Extract the current production terminal predicate and reward expression,
    # so changing that code makes this regression check fail instead of drifting.
    predicate = next(n.value for n in method.body if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'died_penalty')
    rewards = next(n.value for n in method.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'rewards')
    contribution = next(v for k,v in zip(rewards.keys,rewards.values) if k.value == 'termination_p')
    predicate_code = compile(ast.Expression(predicate), '<actual-base-terminal-mask>', 'eval')
    contribution_code = compile(ast.Expression(contribution), '<actual-base-terminal-reward>', 'eval')

    class StableStub:
        def _get_dones(self):
            return torch.zeros(self.num_envs, dtype=torch.bool), self.timeout

    circular = class_node('Quadhopper_Planner_Circular/planner_circular_env.py', 'PlannerCircularEnv')
    dones = next(n for n in circular.body if isinstance(n, ast.FunctionDef) and n.name == '_get_dones')
    # Preserve __class__ cells for the production method's super() call.
    parent = ast.ClassDef(name='Parent', bases=[ast.Name(id='StableStub',ctx=ast.Load())],
                          keywords=[], body=[dones], decorator_list=[])
    ns = {'torch':torch, 'StableStub':StableStub}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[parent],type_ignores=[])), '<actual-task-dones>', 'exec'), ns)
    parent_cls = ns['Parent']

    def legacy_reward(self):
        self.reward_calls += 1
        q = self._robot.data.root_quat_w
        context = {'self':self, 'torch':torch, 'z_pos':self._robot.data.root_pos_w[:,2],
                   'roll_pitch_sq':q[:,1].square()+q[:,2].square()}
        context['died_penalty'] = eval(predicate_code, context)
        contribution = eval(contribution_code, context)
        self._episode_sums['termination_p'] += contribution
        return torch.full((self.num_envs,),123.) + contribution

    parent_cls._get_rewards = legacy_reward
    parent_cls._reset_idx = lambda self,ids: None
    parent_cls.stage2_summary = lambda self: {}
    fixed = class_node('Quadhopper_Planner_Random/path_free_stage2_v16_env.py','PathFreeStage2V16Env')
    ns.update(PathFreeStage2V15Env=parent_cls, terminal_reward_correction=terminal_reward_correction)
    exec(compile(ast.Module(body=[fixed],type_ignores=[]),'<actual-v16-wrapper>','exec'),ns)
    return parent_cls, ns['PathFreeStage2V16Env']


def fixture(cls):
    env=object.__new__(cls)
    env.num_envs,env.device=8,'cpu'
    data=NS(root_pos_w=torch.zeros(8,3),root_quat_w=torch.zeros(8,4),root_ang_vel_b=torch.zeros(8,3))
    data.root_pos_w[:,2]=.3
    data.root_quat_w[:,0]=1
    data.root_ang_vel_b[[1,4,7],0]=19
    data.root_pos_w[2,0]=3
    angle=torch.deg2rad(torch.tensor(100.))/2
    data.root_quat_w[[3,4,7],0]=torch.cos(angle)
    data.root_quat_w[[3,4,7],1]=torch.sin(angle)
    env._robot=NS(data=data,_ALL_INDICES=torch.arange(8))
    env.cfg=NS(tilt_death_sq_threshold=.5,terminate_far_xy_distance=2.,terminate_angvel_norm=18.,
               terminate_after_two_hop_pair=False,terminate_on_target_miss=False,termination_penalty_scale=-30.)
    env.commands=NS(lookahead=lambda:(torch.zeros(8,2),torch.zeros(8,2)),
                    start_points=lambda:torch.zeros(8,2),steps_per_revolution=100)
    env._initial_drop_pending=torch.zeros(8,dtype=torch.bool)
    env._initial_drop_pending[7]=True
    env._consecutive_hits=torch.zeros(8)
    env._consecutive_hits[6]=100
    env.timeout=torch.zeros(8,dtype=torch.bool)
    env.timeout[5]=True
    for name in ('physical','low_height','tilt','far_xy','high_angvel'):
        setattr(env,f'_{name}_death_total',torch.zeros(8))
    env.extras={'log':{}}
    env._episode_sums={'termination_p':torch.zeros(8)}
    env.reward_calls=0
    return env


class TerminalPenaltyFixTest(unittest.TestCase):
    def test_actual_done_mask_unchanged_and_only_physical_deaths_charged_once(self):
        old,new=load_actual_reward_and_dones()
        legacy,fixed=fixture(old),fixture(new)
        old_done=legacy._get_dones()
        new_done=fixed._get_dones()
        for a,b in zip(old_done,new_done):
            torch.testing.assert_close(a,b)
        self.assertEqual(new_done[0].tolist(),[False,True,True,True,True,False,True,False])
        self.assertEqual(legacy._get_rewards().tolist(),[123,123,123,93,93,123,123,93])
        self.assertEqual(fixed._get_rewards().tolist(),[123,93,93,93,93,123,123,123])
        self.assertEqual(fixed.reward_calls,1)
        self.assertEqual(fixed._episode_sums['termination_p'].tolist(),[0,-30,-30,-30,-30,0,0,0])
        self.assertEqual(fixed._v16_terminal_totals.tolist(),[4,2,1])

    def test_partial_reset_and_next_live_step_do_not_reuse_death_event(self):
        _,cls=load_actual_reward_and_dones()
        env=fixture(cls)
        env._get_dones();env._get_rewards()
        env._reset_idx(torch.tensor([1]))
        self.assertFalse(env._v16_physical_death[1])
        self.assertTrue(env._v16_physical_death[2])
        env._robot.data.root_ang_vel_b.zero_()
        env._robot.data.root_pos_w[:,0]=0
        env._robot.data.root_quat_w.zero_();env._robot.data.root_quat_w[:,0]=1
        env._get_dones()
        self.assertEqual(env._get_rewards().tolist(),[123]*8)
        self.assertEqual(env._v16_terminal_totals.tolist(),[4,2,1])
        env._reset_idx(None)
        self.assertFalse(env._v16_physical_death.any())

    def test_configured_tilt_threshold_and_low_height_use_task_event(self):
        _,cls=load_actual_reward_and_dones()
        env=fixture(cls)
        env.cfg.tilt_death_sq_threshold=.4
        env._robot.data.root_quat_w[0]=torch.tensor([.75,(1-.75**2)**.5,0,0])
        env._robot.data.root_pos_w[5,2]=.04
        env._get_dones()
        reward=env._get_rewards()
        self.assertEqual(reward[0],93)
        self.assertEqual(reward[5],93)  # physical death and timeout together

    def test_manifest_and_directional_eval_restore_both_repairs(self):
        manifest=dict(protocol=PROTOCOL_V3,apex_event_fix_v15=True,terminal_penalty_fix_v16=True,
                      stability_profile='two_circle_v18',observation_layout='history_foot_v19',
                      two_circle_route=dict(short_hop_radius_m=[.12,.3],long_hop_radius_m=[.5,.75]),
                      reverse_turn_halfwidth_deg=30,policy_target_root_z_m=1.,target_tolerance_m=.1)
        self.assertIn('--terminal-penalty-fix-v16',manifest_flags(manifest))
        with self.assertRaises(ValueError):
            manifest_flags({**manifest,'apex_event_fix_v15':False})
        import run_directional_eval
        with patch('sys.argv',['test','/tmp/v16_model.pt','--dry-run','--scenario','reverse_only']), \
             patch.object(Path,'is_file',return_value=True),patch.object(Path,'exists',return_value=False), \
             patch.object(Path,'read_text',return_value=json.dumps(manifest)), \
             patch.object(run_directional_eval.subprocess,'run') as run:
            run_directional_eval.main()
            for call in run.call_args_list:
                self.assertIn('--terminal-penalty-fix-v16',call.args[0])
                self.assertIn('--apex-event-fix-v15',call.args[0])

    def test_launcher_ab_differs_only_by_terminal_repair(self):
        commands={}
        for variant in ('control','fix'):
            result=subprocess.run(['bash',str(ROOT/'run_train_two_circle_path_free_stage2_v16_terminal_penalty_075.sh')],
                cwd=ROOT,capture_output=True,text=True,
                env={**os.environ,'VARIANT':variant,'DRY_RUN':'1','OUTPUT':'/tmp/v16-terminal-test',
                     'ITERATIONS':'100','NUM_ENVS':'256'})
            self.assertEqual(result.returncode,0,result.stderr)
            commands[variant]=shlex.split(result.stdout.strip())
            self.assertTrue(commands[variant][0].endswith('/python.sh'))
            self.assertIn('--apex-event-fix-v15',commands[variant])
        commands['fix'].remove('--terminal-penalty-fix-v16')
        self.assertEqual(commands['fix'],commands['control'])


if __name__=='__main__':
    unittest.main()
