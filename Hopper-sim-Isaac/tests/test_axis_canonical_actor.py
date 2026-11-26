import unittest

import torch

from benchmark37.axis_canonical_actor import OPPOSITE_MOTORS, choose_basis, rotate_obs73_half_turn
from benchmark37.crossing_obs72 import HISTORY_VERSION, ObservationMemory
from benchmark37.crossing_obs73_quat import assemble
from benchmark37.t_frame import quaternion_from_rpy


class AxisCanonicalActorTest(unittest.TestCase):
    def test_rotating_raw_observations_matches_rotating_source_measurements(self):
        torch.manual_seed(57)
        n = 5
        half_turn = torch.tensor([-1., -1., 1.])
        quat_sign = torch.tensor([1., -1., -1., 1.])
        pose = torch.randn(n, 3)
        attitude = quaternion_from_rpy(torch.randn(n, 3) * .2)
        velocity = torch.randn(n, 3)
        omega = torch.randn(n, 3)
        targets = torch.randn(n, 3, 2)
        reference = torch.randn(n, 3)
        desired = quaternion_from_rpy(torch.randn(n, 3) * .2)
        desired_vel = torch.randn(n, 3)
        desired_omega = torch.randn(n, 3)
        state = torch.tensor([0., .5, 1., 0., .5])
        ground = torch.zeros(n)
        height = torch.ones(n)
        memory = ObservationMemory(n, 'cpu', HISTORY_VERSION)
        raw,snapshot = assemble(pose, attitude, velocity, omega, targets, ground,
                                height, memory, reference, desired, desired_vel, desired_omega, state)
        rotated,rotated_snapshot = assemble(pose*half_turn, attitude*quat_sign, velocity*half_turn,
            omega*half_turn, -targets, ground, height, memory, reference*half_turn,
            desired*quat_sign, desired_vel*half_turn, desired_omega*half_turn, state)
        memory.states[:] = snapshot[:, None]
        memory.actions[:] = torch.randn_like(memory.actions)
        memory.recent_action[:] = torch.randn_like(memory.recent_action)
        raw,_ = assemble(pose, attitude, velocity, omega, targets, ground,
                        height, memory, reference, desired, desired_vel, desired_omega, state)
        memory.states[:] = rotated_snapshot[:, None]
        memory.actions[:] = memory.actions[..., OPPOSITE_MOTORS]
        memory.recent_action[:] = memory.recent_action[..., OPPOSITE_MOTORS]
        rotated,_ = assemble(pose*half_turn, attitude*quat_sign, velocity*half_turn,
            omega*half_turn, -targets, ground, height, memory, reference*half_turn,
            desired*quat_sign, desired_vel*half_turn, desired_omega*half_turn, state)
        torch.testing.assert_close(rotate_obs73_half_turn(raw), rotated)
        torch.testing.assert_close(rotate_obs73_half_turn(rotated), raw)

    def test_waypoint_progress_holds_its_basis_through_a_retry(self):
        current = torch.tensor([[.5,0.],[0.,-.5],[0.,0.],[0.,0.],[-.4,.1],[0.,0.]])
        preview = torch.tensor([[-.5,0.],[0.,.5],[0.,-.5],[.5,0.],[0.,0.],[0.,0.]])
        previous = torch.tensor([False,False,False,True,False,True])
        expected = torch.tensor([False,True,True,False,True,True])
        torch.testing.assert_close(choose_basis(current, preview, previous), expected)

    def test_opposite_motor_map_keeps_yaw_and_thrust(self):
        command = torch.tensor([[.38,.52,.87,.41],[.12,.98,.45,.02]])
        mixer = command.new_tensor([[1,1,1,1],[1,-1,-1,1],[1,1,-1,-1],[-1,1,-1,1]])
        original = command @ mixer.T
        mirrored = command[..., OPPOSITE_MOTORS] @ mixer.T
        torch.testing.assert_close(mirrored, original * command.new_tensor([1,-1,-1,1]))


if __name__ == '__main__':
    unittest.main()
