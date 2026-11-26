"""Regression checks for the opt-in OBS73 training perturbations."""

from types import SimpleNamespace
import unittest

import torch

from benchmark37.physical_robustness import PlanarPhysicalRobustness, VERSION


CONFIG = dict(version=VERSION, com_radius_m=.005, impulse_probability=1.,
              impulse_speed_m_s=[.05, .15], impulse_flight_time_s=[.12, .28])


class FakePhysX:
    def __init__(self):
        self.coms = torch.zeros((64, 2, 7))
        self.coms[..., 6] = 1.
        self.coms[:, 0, :3] = torch.tensor([.01, -.01, .03])
        self.masses = torch.tensor([.1456, .01]).expand(64, -1)

    def get_coms(self):
        return self.coms

    def get_masses(self):
        return self.masses

    def set_coms(self, values, ids):
        self.coms[ids] = values[ids]


def make_env():
    view = FakePhysX()
    event = SimpleNamespace(CANDIDATE=2, FLIGHT=3, phase=torch.zeros(64, dtype=torch.long),
                            air_steps=torch.zeros(64, dtype=torch.long),
                            apex_seen=torch.zeros(64, dtype=torch.bool))
    data = SimpleNamespace(joint_pos=torch.zeros((64, 1)),
                           root_com_vel_w=torch.arange(6.).expand(64, -1).clone())
    for name in ('_body_com_pose_b', '_body_com_pose_w', '_body_link_vel_w', '_body_com_state_w',
                 '_body_link_state_w', '_root_com_pose_w', '_root_link_vel_w', '_root_com_state_w',
                 '_root_link_state_w'):
        setattr(data, name, SimpleNamespace(timestamp=0.))
    data.root_com_lin_vel_w = data.root_com_vel_w[:, :3]

    def write_velocity(velocity, env_ids):
        data.root_com_vel_w[env_ids] = velocity

    robot = SimpleNamespace(root_physx_view=view, body_names=['Body', 'SpringLeg'],
                            data=data, write_root_com_velocity_to_sim=write_velocity)
    return SimpleNamespace(_robot=robot, _foot_event=event, _spring_joint_id=0, num_envs=64,
                           cfg=SimpleNamespace(seed=42), device='cpu', step_dt=.01)


class PhysicalRobustnessTests(unittest.TestCase):
    def test_com_readback_subset_and_no_accumulation(self):
        env = make_env()
        view = env._robot.root_physx_view
        nominal = view.coms.clone()
        perturbation = PlanarPhysicalRobustness(env, CONFIG)
        perturbation.reset(torch.arange(64))
        previous = view.coms.clone()
        for _ in range(8):
            perturbation.reset(torch.arange(0, 64, 2))
            self.assertLessEqual(perturbation.audit()['com_max_physx_readback_m'], .005001)
            torch.testing.assert_close(view.coms[:, 0, 2:], nominal[:, 0, 2:])
            torch.testing.assert_close(view.coms[:, 1], nominal[:, 1])
            torch.testing.assert_close(view.coms[1::2], previous[1::2])
    def test_impulses_are_seeded_one_shot_and_planar(self):
        env, repeat_env = make_env(), make_env()
        first = PlanarPhysicalRobustness(env, CONFIG)
        repeat = PlanarPhysicalRobustness(repeat_env, CONFIG)
        ids = torch.arange(64)
        first.reset(ids)
        repeat.reset(ids)
        torch.testing.assert_close(first.body_shift, repeat.body_shift, rtol=0., atol=0.)
        torch.testing.assert_close(first.push_xy, repeat.push_xy, rtol=0., atol=0.)
        event = env._foot_event
        first.flight_count[:] = first.push_flight
        event.phase[:] = event.FLIGHT
        event.air_steps[:] = first.push_step
        event.apex_seen[0] = True
        env._robot.data.joint_pos[1] = .003
        before = env._robot.data.root_com_vel_w.clone()
        first.maybe_push()
        first.maybe_push()
        delta = env._robot.data.root_com_vel_w - before
        self.assertTrue((delta[:2] == 0.).all())
        self.assertTrue((delta[:, 2:] == 0.).all())
        torch.testing.assert_close(delta[2:, :2], first.push_xy[2:])
        self.assertEqual(first.audit()['impulses'], 62)
        self.assertLessEqual(first.audit()['impulse_max_m_s'], .15)


if __name__ == '__main__':
    unittest.main()
