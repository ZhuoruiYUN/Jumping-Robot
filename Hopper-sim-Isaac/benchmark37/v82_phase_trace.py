"""Record passive motor and contact traces during a native OBS73 evaluation."""

import argparse
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path


def install_trace(evaluation, output, case_ids, trace_steps, *, trace_start=1):
    import numpy as np
    import torch

    initialize = evaluation.__init__
    sample = evaluation.sample
    trace_end = trace_start + trace_steps - 1

    def initialize_trace(self, env):
        initialize(self, env)
        if max(case_ids) >= env.num_envs:
            raise ValueError('Trace case ID is outside the native evaluation batch')
        self.trace_ids = torch.tensor(case_ids, device=env.device)
        self.phase_trace = []
        step = env._pre_physics_step

        def trace_input(actions):
            if trace_start <= env.common_step_counter + 1 <= trace_end:
                self.trace_input = actions[self.trace_ids].clone()
            step(actions)

        env._pre_physics_step = trace_input

    def sample_trace(self):
        env = self.env
        if trace_start <= env.common_step_counter <= trace_end:
            ids = self.trace_ids
            plant = env._imu_link_plant
            data = env._robot.data
            event = env._foot_event
            origin = env._terrain.env_origins[ids]
            root = data.root_link_pos_w[ids] - origin
            foot = data.body_link_pos_w[ids, env._foot_body_id] - origin
            spring = data.joint_pos[ids, env._spring_joint_id]
            spring_speed = data.joint_vel[ids, env._spring_joint_id]
            clock = spring.new_full(spring.shape, env.common_step_counter)
            route = env.omni_route
            self.phase_trace.append(torch.column_stack((
                clock, clock * env.step_dt, route.index[ids], event.phase[ids],
                root, foot, env._two_ground_pt_xy_w[ids] - origin[:, :2],
                data.root_link_lin_vel_w[ids], data.root_com_lin_vel_w[ids], data.root_quat_w[ids],
                data.root_ang_vel_b[ids], spring, spring_speed, env._thrust[ids, 0, 2],
                env._moment[ids, 0], self.trace_input, env._actions[ids],
                plant.collective_u[ids], plant.attitude_u[ids], plant.profile[ids], plant.voltage_gain[ids],
            )))
            if env.common_step_counter == trace_end:
                columns = [
                    'step', 'time_s', 'target_index', 'phase',
                    'root_x', 'root_y', 'root_z', 'foot_x', 'foot_y', 'foot_z', 'goal_x', 'goal_y',
                    'root_vx', 'root_vy', 'root_vz', 'com_vx', 'com_vy', 'com_vz',
                    'q_w', 'q_x', 'q_y', 'q_z', 'omega_roll', 'omega_pitch', 'omega_yaw',
                    'spring_q', 'spring_qdot', 'thrust_z_n', 'torque_roll', 'torque_pitch', 'torque_yaw',
                ]
                for mode in ('actor', 'accepted', 'collective_duty', 'attitude_duty'):
                    columns.extend(f'{mode}_f{i}' for i in range(1, 5))
                columns.extend(('plant_profile', 'voltage_gain'))
                trace = torch.stack(self.phase_trace).cpu().numpy()
                np.savez_compressed(output / 'phase_trace.npz', trace=trace, columns=columns, case_ids=case_ids)
                self.phase_trace.clear()
        sample(self)

    evaluation.__init__ = initialize_trace
    evaluation.sample = sample_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-cases', type=int, nargs='+', required=True)
    parser.add_argument('--trace-start', type=int, default=1, help='First recorded control step')
    parser.add_argument('--trace-steps', type=int, default=1500, help='Number of consecutive samples')
    parser.add_argument('runner', nargs=argparse.REMAINDER, help='Pass native runner flags after --')
    parsed = parser.parse_args()
    argv = parsed.runner
    if (not argv or argv[:2] != ['--', 'eval'] or '--output' not in argv
            or not 0 < parsed.trace_steps <= 1500 or parsed.trace_start < 1 or min(parsed.trace_cases) < 0
            or len(parsed.trace_cases) != len(set(parsed.trace_cases))):
        parser.error('Supply distinct nonnegative trace cases, 1–1500 trace steps, and -- eval ... --output PATH')
    output_index = argv.index('--output') + 1
    if output_index == len(argv):
        parser.error('--output requires a path')
    from . import ring_runner

    if '--stationary-retention-eval' in argv:
        module_name, evaluation_name = 'benchmark37.ring_evaluation', 'RingEvaluation'
    else:
        module_name, evaluation_name = 'benchmark37.obs73_omni_evaluation', 'OmniEvaluation'

    class TraceLoader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            evaluation = getattr(module, evaluation_name)
            install_trace(evaluation, Path(argv[output_index]), parsed.trace_cases, parsed.trace_steps,
                          trace_start=parsed.trace_start)

    class TraceFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != module_name:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                raise ImportError(fullname)
            spec.loader = TraceLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, TraceFinder())  # Isaac must initialize USD before importing the environment.
    sys.argv[1:] = argv[1:]
    ring_runner.main()


if __name__ == '__main__':
    main()
