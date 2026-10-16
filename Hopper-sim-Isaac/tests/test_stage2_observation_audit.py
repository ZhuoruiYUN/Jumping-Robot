"""CPU checks of the actual inherited 336-D observation construction methods."""
import ast
from collections import deque
from pathlib import Path
import runpy
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
HISTORY = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/history_foot.py'))


def rotation(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def inverse(q, v):
    return (rotation(q).transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)


def source_class(path, name, parent, methods, namespace):
    tree = ast.parse((ROOT / path).read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    selected = [node for node in original.body
                if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert len(selected) == len(methods)
    cls = ast.ClassDef(name=name, bases=[ast.Name(id=parent, ctx=ast.Load())],
                       keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(ROOT / path), 'exec'), namespace)
    return namespace[name]


def observation_class():
    ns = dict(torch=torch, **{k: HISTORY[k] for k in
              ('ObservationHistory', 'foot_offset_world', 'observation_frame')})
    ns['subtract_frame_transforms'] = lambda p, q, target: (inverse(q, target-p), None)
    ns['quat_apply_inverse'] = inverse
    chain = [
        ('Quadhopper_Stable/quadhopper_env.py', 'QuadhopperEnv', 'object', ['_get_observations']),
        ('Quadhopper_Planner_Circular/planner_circular_env.py', 'PlannerCircularEnv',
         'QuadhopperEnv', ['_get_observations', '_lookahead_error_b']),
        ('Quadhopper_Planner_Random/random_two_hop_env.py', 'PlannerRandomTwoHopEnv',
         'PlannerCircularEnv', ['_lookahead_error_b']),
        ('Quadhopper_Planner_Random/spatial_tracking_env.py', 'SpatialTrackingEnv',
         'PlannerRandomTwoHopEnv', ['_get_observations', '_update_reference']),
        ('Quadhopper_Planner_Random/history_foot_env.py', 'HistoryFootEnv',
         'SpatialTrackingEnv', ['_get_observations', 'foot_position_w']),
        ('Quadhopper_Planner_Random/path_free_stage2_env.py', 'PathFreeStage2Env',
         'HistoryFootEnv', ['_spatial_guidance']),
    ]
    for spec in chain:
        cls = source_class(*spec, ns)
    return cls


def make_env():
    env = observation_class()()
    env.num_envs, env.device, env.common_step_counter, env._spring_joint_id = 2, 'cpu', 0, 0
    env.cfg = NS(observation_noise_std=0., planner_velocity_observation=False,
                 relative_next_hop_observation=True, hop_distance=.2)
    quat = torch.tensor([[.9, .1, -.2, .3], [.8, -.2, .3, .1]])
    quat /= quat.norm(dim=-1, keepdim=True)
    env._robot = NS(data=NS(root_pos_w=torch.tensor([[0., 0., .6], [10., -5., .7]]),
        root_quat_w=quat, root_lin_vel_b=torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
        root_ang_vel_b=torch.tensor([[.1, .2, .3], [.4, .5, .6]]),
        joint_pos=torch.tensor([[.02], [.04]]), joint_vel=torch.tensor([[-1.], [2.]])))
    env.p = env._robot.data.root_pos_w[:, :2] + torch.tensor([.3, -.2])
    env.p_next = env.p + torch.tensor([-.4, .1])
    env.start = env.p - .1
    env.commands = NS(lookahead=lambda: (env.p, env.p_next), start_points=lambda: env.start)
    env._active_target_height = torch.ones(2)
    env._height_commands = lambda: (env._active_target_height, torch.full((2,), 1.2))
    env._initial_drop_pending = torch.zeros(2, dtype=torch.bool)
    env._desired_pos_w = torch.zeros(2, 3)
    env._planned_velocity_w = torch.zeros(2, 3)
    env._action_history = deque(torch.full((2, 4), i/10.) for i in range(5))
    env.foot_geometry = dict(rest_offset_body_m=[0., 0., -.2613543123], spring_travel_m=.08)
    env.observation_history = HISTORY['ObservationHistory'](2, 'cpu')
    return env


class ObservationAuditTests(unittest.TestCase):
    def test_actual_chain_dimensions_frames_and_two_target_semantics(self):
        env = make_env()
        obs = env._get_observations()['policy']
        self.assertEqual(tuple(obs.shape), (2, 336))
        frames = obs.reshape(2, 6, 56)
        for t in range(1, 6):
            torch.testing.assert_close(frames[:, 0], frames[:, t])
        f, data = frames[:, 0], env._robot.data
        torch.testing.assert_close(f[:, :3], data.root_lin_vel_b)
        torch.testing.assert_close(f[:, 3:6], data.root_ang_vel_b)
        torch.testing.assert_close(f[:, 6:10], data.root_quat_w)
        torch.testing.assert_close(f[:, 17:37], torch.cat(list(env._action_history), -1))
        current_w = torch.cat((env.p-data.root_pos_w[:, :2], torch.zeros(2, 1)), -1)
        segment_w = torch.cat((env.p_next-env.p, torch.zeros(2, 1)), -1)
        torch.testing.assert_close(f[:, 37:39], inverse(data.root_quat_w, current_w)[:, :2]/.2)
        torch.testing.assert_close(f[:, 39:41], inverse(data.root_quat_w, segment_w)[:, :2]/.2)
        torch.testing.assert_close(f[:, 41:43], torch.tensor([[.5, .6], [.5, .6]]))
        self.assertEqual(f[:, 43:52].count_nonzero().item(), 0)
        foot = env.foot_position_w()
        torch.testing.assert_close(f[:, 52:54], (foot-data.root_pos_w)[:, :2])
        torch.testing.assert_close(f[:, 54:56], env.p-foot[:, :2])

    def test_next_target_change_does_not_change_current_target_features(self):
        env = make_env()
        before = env._get_observations()['policy'].reshape(2, 6, 56)[:, 0].clone()
        env.p_next += torch.tensor([.1, .2])
        env.common_step_counter += 1
        frames = env._get_observations()['policy'].reshape(2, 6, 56)
        torch.testing.assert_close(frames[:, 1], before)
        delta = frames[:, 0]-before
        self.assertGreater(delta[:, 39:41].abs().max().item(), .01)
        delta[:, 39:41] = 0
        torch.testing.assert_close(delta, torch.zeros_like(delta))

    def test_target_advance_keeps_past_targets_and_cache_is_one_sample_per_tick(self):
        env = make_env()
        before = env._get_observations()['policy'].reshape(2, 6, 56)[:, 0].clone()
        env.p = env.p_next.clone()
        env.p_next += .2
        env.common_step_counter += 1
        obs = env._get_observations()['policy']
        frames = obs.reshape(2, 6, 56)
        torch.testing.assert_close(frames[:, 1], before)
        self.assertFalse(torch.equal(frames[:, 0, 54:56], before[:, 54:56]))
        state = torch.get_rng_state().clone()
        torch.testing.assert_close(env._get_observations()['policy'], obs)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_body_com_foot_relation_and_translation_invariance(self):
        env = make_env()
        data = env._robot.data
        com_offset_b = torch.tensor([[.002, -.002, .001], [-.002, .001, -.001]])
        c_w = (rotation(data.root_quat_w) @ com_offset_b.unsqueeze(-1)).squeeze(-1)
        body_com_w = data.root_pos_w + c_w
        r_b = torch.tensor(env.foot_geometry['rest_offset_body_m']).repeat(2, 1)
        r_b[:, 2] += data.joint_pos[:, 0]
        expected_foot = body_com_w + (rotation(data.root_quat_w) @
                                      (r_b-com_offset_b).unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(env.foot_position_w(), expected_foot)
        before = env._get_observations()['policy']
        shift = torch.tensor([123., -43.])
        data.root_pos_w[:, :2] += shift
        env.p += shift
        env.p_next += shift
        env.common_step_counter += 1
        after = env._get_observations()['policy']
        torch.testing.assert_close(before[:, :56], after[:, :56], atol=5e-5, rtol=1e-5)

    def test_initial_drop_zeroes_preview_segment_and_anchors_current_target(self):
        env = make_env()
        env._initial_drop_pending[:] = True
        f = env._get_observations()['policy'][:, :56]
        torch.testing.assert_close(f[:, 39:41], torch.zeros(2, 2))
        torch.testing.assert_close(f[:, 54:56], env.start-env.foot_position_w()[:, :2])


if __name__ == '__main__':
    unittest.main()
