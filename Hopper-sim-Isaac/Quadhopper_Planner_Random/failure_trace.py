"""Read-only, bounded failure traces. No Isaac import, RNG draws or controller."""
from collections import deque
import csv
import json
import math
from pathlib import Path

import torch


class FailureTrace:
    """First physical death per env, pre-reset samples and labelled post-reset tail."""

    def __init__(self, num_envs, step_dt, window_s=0.5):
        if num_envs < 1 or step_dt <= 0 or not math.isfinite(window_s) or window_s < 0:
            raise ValueError('invalid trace dimensions/window')
        self.num_envs = num_envs
        self.step_dt = step_dt
        self.window_steps = round(window_s / step_dt)
        self.history = deque(maxlen=self.window_steps + 1)
        self.episodes = torch.zeros(num_envs, dtype=torch.long)
        self.events = {}
        self.fields = None
        self.last_step = None

    def observe(self, step, values, physical_death, reset_pending):
        if self.last_step is not None and step != self.last_step + 1:
            raise ValueError('trace requires one consecutive sample per control step')
        names = tuple(values)
        if self.fields is None:
            self.fields = names
        if names != self.fields or any(v.shape != (self.num_envs,) for v in values.values()):
            raise ValueError('trace fields/shapes changed')
        if physical_death.shape != (self.num_envs,) or reset_pending.shape != (self.num_envs,):
            raise ValueError('trace masks must match num_envs')
        matrix = torch.stack(tuple(values.values()), -1).detach().cpu().clone()
        deaths = physical_death.detach().cpu().bool()
        resets = reset_pending.detach().cpu().bool()
        if (deaths & ~resets).any():
            raise ValueError('physical death must trigger reset')
        sample = (step, self.episodes.clone(), matrix, resets.clone())
        self.history.append(sample)
        # Retain the reset tail without pretending it continues the failed hop.
        for env_id, event in self.events.items():
            if 0 < step - event['trigger_step'] <= self.window_steps:
                event['samples'].append(self._row(sample, env_id, event))
        for env_id in deaths.nonzero().flatten().tolist():
            if env_id in self.events:
                continue
            event = dict(env_id=env_id, trigger_step=step,
                         trigger_episode=int(self.episodes[env_id]), samples=[])
            event['samples'] = [self._row(old, env_id, event) for old in self.history
                                if int(old[1][env_id]) == event['trigger_episode']]
            self.events[env_id] = event
        self.episodes += resets.long()
        self.last_step = step

    def _row(self, sample, env_id, event):
        step, episodes, matrix, resets = sample
        row = dict(env_id=env_id, step=step, time_s=step * self.step_dt,
                   relative_time_s=(step - event['trigger_step']) * self.step_dt,
                   episode_id=int(episodes[env_id]),
                   same_episode_as_failure=int(episodes[env_id]) == event['trigger_episode'],
                   reset_pending=bool(resets[env_id]), failure_trigger=step == event['trigger_step'])
        row.update(zip(self.fields, matrix[env_id].tolist()))
        return row

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        report = dict(num_envs=self.num_envs, step_dt_s=self.step_dt,
                      window_s=self.window_steps * self.step_dt, last_step=self.last_step,
                      sample_timing='after reward/cycle events, before automatic reset',
                      action_timing='policy action and applied wrench for the preceding control interval',
                      post_failure='automatic reset is preserved; tail belongs to new episodes',
                      contact='spring q > 0.002 m proxy, not a contact-force sensor',
                      thrust='rotor_model_F is reconstructed scaled model thrust; applied_force/moment '
                             'are the actual body-frame external wrench buffers, excluding contact forces',
                      motor_pwm='1000 times lagged normalized motor state, not measured hardware PWM',
                      frames='world Z-up, quaternion wxyz, angular velocity in body frame',
                      events=[])
        for env_id, event in sorted(self.events.items()):
            rows = event['samples']
            # Finite differences never cross reset boundaries.
            for i, row in enumerate(rows):
                valid = i > 0 and rows[i-1]['episode_id'] == row['episode_id']
                row['action_rate_valid'] = valid
                for prefix in ('policy_action', 'motor_pwm'):
                    for motor in range(1, 5):
                        key = f'{prefix}_{motor}'
                        if key in row:
                            row[f'{key}_rate_per_s'] = (
                                (row[key] - rows[i-1][key]) / self.step_dt if valid else float('nan'))
            name = f'env_{env_id:03d}_step_{event["trigger_step"]:06d}.csv'
            with (directory / name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report['events'].append(dict(
                env_id=env_id, trigger_step=event['trigger_step'], csv=name,
                pre_samples=sum(r['relative_time_s'] < 0 for r in rows),
                post_samples=sum(r['relative_time_s'] > 0 for r in rows),
                complete_pre_window=sum(r['relative_time_s'] < 0 for r in rows) == self.window_steps,
                complete_post_window=sum(r['relative_time_s'] > 0 for r in rows) == self.window_steps,
            ))
        (directory / 'index.json').write_text(json.dumps(report, indent=2) + '\n')
        return report


def plant_values(env):
    """Read existing buffers only. No noisy observation calls or event updates."""
    data = env._robot.data
    values = {'common_step_counter': torch.full(
        (env.num_envs,), float(env.common_step_counter), device=data.root_pos_w.device)}

    def columns(prefix, tensor, labels):
        tensor = tensor.reshape(env.num_envs, -1)
        if tensor.shape[1] != len(labels):
            raise ValueError(f'{prefix}: unexpected shape {tensor.shape}')
        values.update({f'{prefix}_{label}': tensor[:, i] for i, label in enumerate(labels)})

    columns('root_pos_w', data.root_pos_w, 'xyz')
    columns('root_vel_w', data.root_lin_vel_w, 'xyz')
    columns('angvel_b', data.root_ang_vel_b, 'xyz')
    columns('quat_w', data.root_quat_w, 'wxyz')
    w, x, y, z = data.root_quat_w.unbind(-1)
    values['roll_deg'] = torch.rad2deg(torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    values['pitch_deg'] = torch.rad2deg(torch.asin((2*(w*y-z*x)).clamp(-1, 1)))
    values['yaw_deg'] = torch.rad2deg(torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    values['tilt_deg'] = torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1, 1)))
    values['spring_q_m'] = data.joint_pos[:, env._spring_joint_id]
    values['spring_qdot_m_s'] = data.joint_vel[:, env._spring_joint_id]
    values['spring_contact_proxy'] = values['spring_q_m'] > .002
    target, _ = env.commands.lookahead()
    columns('target_after_events_w', target, 'xy')
    if hasattr(env, 'foot_position_w'):
        columns('foot_pos_w', env.foot_position_w(), 'xyz')
    columns('raw_cmd_pwm', env._raw_target_u * 1000, range(1, 5))
    columns('shaped_cmd_pwm', env._shaped_target_u * 1000, range(1, 5))
    columns('motor_pwm', env._motor_u * 1000, range(1, 5))
    columns('applied_force_b_N', env._thrust[:, 0], 'xyz')
    columns('applied_moment_b_Nm', env._moment[:, 0], 'xyz')
    u = env._motor_u
    rotor_f = (-.2371 * env.dr_thrust_curve_shape * u.square() + .8130*u + .0113).clamp_min(0)
    rotor_f = rotor_f * env.dr_thrust_scale / env.dr_mass_multi
    rotor_f = rotor_f * env._hardware_global_thrust_scale
    columns('rotor_model_F_N', rotor_f, range(1, 5))
    columns('thrust_curve_shape', env.dr_thrust_curve_shape, range(1, 5))
    columns('rotor_thrust_scale', env.dr_thrust_scale, range(1, 5))
    columns('motor_tc_s', env.dr_tm, range(1, 5))
    columns('inertia_multiplier', env.dr_inertia_multi, 'xyz')
    columns('rp_effectiveness', env._hardware_rp_effectiveness, ('xx', 'xy', 'yx', 'yy'))
    columns('com_offset_b_m', env._hardware_body_com_offset_b, 'xyz')
    for name, attr in (
        ('global_thrust_scale', '_hardware_global_thrust_scale'),
        ('spring_stiffness_scale', '_hardware_spring_stiffness_scale'),
        ('spring_damping_scale', '_hardware_spring_damping_scale'),
        ('mass_multiplier', 'dr_mass_multi'),
        ('hard_environment', '_hardware_hard_mask'), ('cycle_active', '_cycle_active'),
        ('initial_drop', '_initial_drop_pending'), ('touchdown', '_touchdown_event'),
        ('liftoff', '_liftoff_event'), ('apex', '_apex_event'),
        ('short_touchdown', '_setup_touchdown_event'),
        ('moving_contact_remaining', '_moving_long_recovery_remaining'),
        ('moving_launch_remaining', '_moving_launch_remaining'),
        ('launch_wrench_remaining', '_launch_stability_remaining'),
    ):
        values[name] = getattr(env, attr).reshape(env.num_envs)
    if hasattr(env, '_settled_apex_height'):
        values['settled_apex_height_m'] = env._settled_apex_height
    if hasattr(env, '_settled_apex_target'):
        values['settled_apex_target_m'] = env._settled_apex_target
    if hasattr(env, '_near_ground_recovery'):
        values['v12_recovery_remaining'] = env._near_ground_recovery.remaining
        values['v12_release_confirmation_count'] = env._near_ground_recovery.release_count
        if hasattr(env._near_ground_recovery, 'attitude_safe'):
            values['v14_recovery_active'] = env._near_ground_recovery.active
            values['v14_recovery_attitude_safe'] = env._near_ground_recovery.attitude_safe
    return values


class EvaluationFailureTrace:
    """Wrap only the evaluation instance's reward call to sample before reset."""

    COUNTERS = ('_physical_death_total', '_tilt_death_total', '_low_height_death_total',
                '_far_xy_death_total', '_high_angvel_death_total')

    def __init__(self, env, window_s=0.5, record_next_target=False):
        self.env = env
        self.record_next_target = record_next_target
        self.trace = FailureTrace(env.num_envs, env.step_dt, window_s=window_s)
        self.reference_trace = StrictHitReferenceTrace(self.trace)
        self.episode_hop_id = torch.zeros(
            env.num_envs, dtype=torch.long, device=env._robot.data.root_pos_w.device)
        self.had_instance_reward = '_get_rewards' in vars(env)
        self.original_reward = env._get_rewards
        self.previous_counters = {k: getattr(env, k).clone() for k in self.COUNTERS}
        self.input_values = None
        self.calls = 0
        self.installed_reward = self._reward_and_capture
        env._get_rewards = self.installed_reward

    def begin_step(self, actions):
        target, next_target = self.env.commands.lookahead()
        self.input_values = {f'policy_action_{i+1}': actions[:, i].detach().clone() for i in range(4)}
        self.input_values.update({f'target_at_action_w_{axis}': target[:, i].clone()
                                  for i, axis in enumerate('xy')})
        if self.record_next_target:
            self.input_values.update({f'next_target_at_action_w_{axis}': next_target[:, i].clone()
                                      for i, axis in enumerate('xy')})
        self.input_values['pre_launch_wrench_remaining'] = self.env._launch_stability_remaining.clone()
        self.input_values['pre_touchdown_wrench_remaining'] = self.env._touchdown_wrench_remaining.clone()
        self.input_values['pre_spring_contact_proxy'] = (
            self.env._robot.data.joint_pos[:, self.env._spring_joint_id] > .002)
        for prefix, attr in (('pre_launch_wrench_Nm', '_launch_wrench_impulse_nm'),
                             ('pre_touchdown_wrench_Nm', '_touchdown_wrench_nm')):
            for i, axis in enumerate('xyz'):
                self.input_values[f'{prefix}_{axis}'] = getattr(self.env, attr)[:, i].clone()

    def _reward_and_capture(self):
        reward = self.original_reward()  # exactly once, preserves RNG/order/events
        if self.input_values is None:
            raise RuntimeError('failure trace begin_step must precede env.step')
        values = plant_values(self.env)
        self.episode_hop_id += self.env._touchdown_event.long()
        values['trace_episode_hop_id'] = self.episode_hop_id.clone()
        values.update(self.input_values)
        for name in self.COUNTERS:
            values[name.removeprefix('_')] = getattr(self.env, name) > self.previous_counters[name]
            self.previous_counters[name] = getattr(self.env, name).clone()
        resets = self.env.reset_terminated | self.env.reset_time_outs
        reference = torch.zeros_like(resets)
        if (hasattr(self.env, 'foot_position_w')
                and hasattr(self.env, '_settled_apex_height')
                and hasattr(self.env, '_settled_apex_target')
                and hasattr(self.env, '_completed_stationary_command')):
            foot_error_sq = ((values['foot_pos_w_x'] - values['target_at_action_w_x']).square()
                             + (values['foot_pos_w_y'] - values['target_at_action_w_y']).square())
            apex = self.env._settled_apex_height
            target_height = self.env._settled_apex_target
            reference = (self.env._touchdown_event & ~self.env._completed_stationary_command
                         & ~self.env._initial_drop_pending & ~resets
                         & (foot_error_sq < 0.10 ** 2)
                         & (apex >= target_height)
                         & ((apex - target_height).abs() <= 0.15))
        self.calls += 1
        self.trace.observe(self.calls, values, values['physical_death_total'], resets)
        self.reference_trace.observe(reference)
        self.episode_hop_id[resets] = 0
        self.input_values = None
        return reward

    def close(self, directory):
        if self.env._get_rewards is self.installed_reward:
            if self.had_instance_reward:
                self.env._get_rewards = self.original_reward
            else:
                del self.env._get_rewards  # restore normal class method lookup
        report = self.trace.export(directory)
        reference_directory = Path(directory).parent / 'strict_hit_references'
        references = self.reference_trace.export(reference_directory)
        report['strict_hit_references'] = dict(
            index=str(reference_directory / 'index.json'),
            captured_environments=len(references['events']))
        return report


class StrictHitReferenceTrace:
    """First eligible moving touchdown per env with a complete pre-event window.

    A strict touchdown is only an anchor for comparison. It is not labelled a
    safe transition until the post-event window is observed without a reset.
    """

    def __init__(self, failure_trace):
        self.source = failure_trace
        # Keep reference CSVs bounded when a longer failure lead-in is needed.
        self.window_steps = min(failure_trace.window_steps, round(2.0 / failure_trace.step_dt))
        self.events = {}

    def _row(self, sample, env_id, event):
        row = self.source._row(sample, env_id, event)
        row['same_episode_as_reference'] = row.pop('same_episode_as_failure')
        row['reference_trigger'] = row.pop('failure_trigger')
        return row

    def observe(self, eligible):
        source = self.source
        if eligible.shape != (source.num_envs,):
            raise ValueError('reference mask must match num_envs')
        if not source.history:
            raise RuntimeError('failure trace must observe this step first')
        sample = source.history[-1]
        step, episodes, _, resets = sample
        if (eligible.detach().cpu().bool() & resets).any():
            raise ValueError('reference touchdown cannot reset on the trigger step')
        for env_id, event in self.events.items():
            if 0 < step - event['trigger_step'] <= self.window_steps:
                event['samples'].append(self._row(sample, env_id, event))
        for env_id in eligible.detach().cpu().nonzero().flatten().tolist():
            if env_id in self.events:
                continue
            event = dict(env_id=env_id, trigger_step=step,
                         trigger_episode=int(episodes[env_id]), samples=[])
            rows = [self._row(old, env_id, event)
                    for old in list(source.history)[-self.window_steps - 1:]
                    if int(old[1][env_id]) == event['trigger_episode']]
            # A short window or a recent reset makes the comparison misleading.
            if len(rows) != self.window_steps + 1:
                continue
            event['samples'] = rows
            self.events[env_id] = event

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        source = self.source
        report = dict(
            num_envs=source.num_envs, step_dt_s=source.step_dt,
            window_s=self.window_steps * source.step_dt,
            reference_definition='first moving touchdown per env with foot XY error <0.10 m, '
                                 'apex >= target and abs(apex-target) <=0.15 m, '
                                 'and a complete same-episode pre-window',
            sample_timing='after reward/cycle events, before automatic reset',
            action_timing='policy action and applied wrench for the preceding control interval',
            post_reference='post-window is labelled by episode; uninterrupted_post_window '
                           'requires no reset through the full window',
            events=[],
        )
        for env_id, event in sorted(self.events.items()):
            rows = event['samples']
            for i, row in enumerate(rows):
                valid = i > 0 and rows[i - 1]['episode_id'] == row['episode_id']
                row['action_rate_valid'] = valid
                for prefix in ('policy_action', 'motor_pwm'):
                    for motor in range(1, 5):
                        key = f'{prefix}_{motor}'
                        if key in row:
                            row[f'{key}_rate_per_s'] = (
                                (row[key] - rows[i - 1][key]) / source.step_dt
                                if valid else float('nan'))
            name = f'env_{env_id:03d}_step_{event["trigger_step"]:06d}.csv'
            with (directory / name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            post = [r for r in rows if r['relative_time_s'] > 0]
            complete_post = len(post) == self.window_steps
            report['events'].append(dict(
                env_id=env_id, trigger_step=event['trigger_step'],
                trigger_episode=event['trigger_episode'], csv=name,
                precedes_first_failure=(
                    event['trigger_step'] < source.events[env_id]['trigger_step']
                    if env_id in source.events else None),
                pre_samples=self.window_steps, post_samples=len(post),
                complete_pre_window=True, complete_post_window=complete_post,
                uninterrupted_post_window=complete_post and all(
                    r['same_episode_as_reference'] and not r['reset_pending'] for r in post),
            ))
        (directory / 'index.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
