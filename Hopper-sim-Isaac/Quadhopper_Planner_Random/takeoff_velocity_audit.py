"""Read-only per-hop kinematics from FailureTrace's pre-reset CPU samples."""

import csv
import json
import math
from pathlib import Path


class TakeoffVelocityAudit:
    """Pair liftoff and touchdown within one environment episode.

    The constant-velocity reference uses realized airtime only as a diagnostic.
    It is not a desired controller setpoint or whole-robot COM estimate.
    """

    FIELDS = (
        'env_id', 'episode_id', 'liftoff_step', 'apex_step', 'touchdown_step',
        'airtime_s', 'target_x_m', 'target_y_m', 'next_target_x_m',
        'next_target_y_m', 'liftoff_foot_x_m',
        'liftoff_foot_y_m', 'touchdown_foot_x_m', 'touchdown_foot_y_m',
        'p1_distance_at_liftoff_m', 'next_segment_distance_m', 'turn_angle_deg',
        'launch_body_vx_m_s', 'launch_body_vy_m_s',
        'reference_vx_m_s', 'reference_vy_m_s', 'launch_mismatch_m_s',
        'launch_along_error_m_s', 'launch_cross_error_m_s',
        'foot_displacement_x_m', 'foot_displacement_y_m',
        'apex_foot_error_m', 'settled_apex_m', 'apex_target_m',
        'touchdown_foot_error_m', 'strict_foot_10cm', 'strict_height_window',
        'strict_pair', 'prior_touchdown_foot_error_m', 'prior_foot_miss_10cm',
        'hard_environment', 'global_thrust_scale', 'reset_pending',
    )
    REQUIRED = (
        'initial_drop', 'liftoff', 'apex', 'touchdown',
        'foot_pos_w_x', 'foot_pos_w_y', 'target_at_action_w_x',
        'target_at_action_w_y', 'next_target_at_action_w_x',
        'next_target_at_action_w_y', 'root_vel_w_x', 'root_vel_w_y',
        'settled_apex_height_m', 'settled_apex_target_m',
        'hard_environment', 'global_thrust_scale',
    )
    PHASE_REQUIRED = (
        'root_vel_w_z', 'tilt_deg', 'spring_q_m', 'spring_qdot_m_s',
        'spring_contact_proxy', 'motor_pwm_1', 'motor_pwm_2',
        'motor_pwm_3', 'motor_pwm_4',
    )
    PHASE_FIELDS = (
        'prior_touchdown_step', 'prior_touchdown_body_vx_m_s',
        'prior_touchdown_body_vy_m_s', 'prior_touchdown_body_vz_m_s',
        'prior_touchdown_tilt_deg', 'prior_touchdown_spring_q_m',
        'prior_touchdown_pre_body_vx_m_s', 'prior_touchdown_pre_body_vy_m_s',
        'prior_touchdown_pre_body_vz_m_s', 'prior_touchdown_pre_tilt_deg',
        'prior_touchdown_pre_foot_speed_xy_m_s',
        'contact_dwell_s', 'contact_proxy_samples', 'contact_peak_spring_q_m',
        'contact_min_spring_qdot_m_s', 'contact_max_spring_qdot_m_s',
        'contact_mean_motor_pwm', 'liftoff_body_vz_m_s', 'liftoff_tilt_deg',
        'liftoff_spring_q_m', 'liftoff_foot_speed_xy_m_s',
        'touchdown_body_vx_m_s', 'touchdown_body_vy_m_s',
        'touchdown_body_vz_m_s', 'touchdown_tilt_deg',
        'touchdown_spring_q_m', 'touchdown_pre_body_vx_m_s',
        'touchdown_pre_body_vy_m_s', 'touchdown_pre_body_vz_m_s',
        'touchdown_pre_tilt_deg', 'touchdown_pre_foot_speed_xy_m_s',
    )

    def __init__(self, num_envs, step_dt, contact_phase=False):
        if num_envs < 1 or not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError('invalid audit dimensions')
        self.num_envs = num_envs
        self.step_dt = step_dt
        self.contact_phase = contact_phase
        self.fields = self.FIELDS + (self.PHASE_FIELDS if contact_phase else ())
        self.pending = [None] * num_envs
        self.previous = [None] * num_envs
        self.previous_touchdown = [None] * num_envs
        self.contact = [None] * num_envs
        self.previous_sample = None
        self.rows = []
        self.last_step = None
        self.skipped_no_liftoff = 0
        self.skipped_target_changed = 0
        self.skipped_reset_before_touchdown = 0

    def observe(self, sample, fields):
        step, episodes, matrix, resets = sample
        if self.last_step is not None and step != self.last_step + 1:
            raise ValueError('audit requires consecutive pre-reset samples')
        if matrix.shape != (self.num_envs, len(fields)) or len(episodes) != self.num_envs:
            raise ValueError('invalid trace sample shape')
        index = {name: i for i, name in enumerate(fields)}
        missing = set(self.REQUIRED + (self.PHASE_REQUIRED if self.contact_phase else ())) - index.keys()
        if missing:
            raise ValueError(f'missing trace fields: {sorted(missing)}')

        def read(env_id, name):
            return float(matrix[env_id, index[name]])

        def phase_snapshot(env_id):
            return dict(body_vx_m_s=read(env_id, 'root_vel_w_x'),
                        body_vy_m_s=read(env_id, 'root_vel_w_y'),
                        body_vz_m_s=read(env_id, 'root_vel_w_z'),
                        tilt_deg=read(env_id, 'tilt_deg'),
                        spring_q_m=read(env_id, 'spring_q_m'))

        def pre_snapshot(env_id, episode):
            if self.previous_sample is None:
                return None
            prior_episodes, prior_matrix = self.previous_sample
            if int(prior_episodes[env_id]) != episode:
                return None
            def old(name):
                return float(prior_matrix[env_id, index[name]])
            result = dict(body_vx_m_s=old('root_vel_w_x'),
                          body_vy_m_s=old('root_vel_w_y'),
                          body_vz_m_s=old('root_vel_w_z'),
                          tilt_deg=old('tilt_deg'))
            result['foot_speed_xy_m_s'] = math.hypot(
                read(env_id, 'foot_pos_w_x') - old('foot_pos_w_x'),
                read(env_id, 'foot_pos_w_y') - old('foot_pos_w_y')) / self.step_dt
            return result

        if self.contact_phase:
            for env_id in range(self.num_envs):
                state = self.contact[env_id]
                if state is None or state['episode_id'] != int(episodes[env_id]):
                    continue
                if read(env_id, 'spring_contact_proxy') > .5:
                    state['contact_proxy_samples'] += 1
                state['contact_peak_spring_q_m'] = max(
                    state['contact_peak_spring_q_m'], read(env_id, 'spring_q_m'))
                state['contact_min_spring_qdot_m_s'] = min(
                    state['contact_min_spring_qdot_m_s'], read(env_id, 'spring_qdot_m_s'))
                state['contact_max_spring_qdot_m_s'] = max(
                    state['contact_max_spring_qdot_m_s'], read(env_id, 'spring_qdot_m_s'))
                state['motor_pwm_sum'] += sum(read(env_id, f'motor_pwm_{motor}')
                                              for motor in range(1, 5)) / 4
                state['motor_pwm_samples'] += 1

        event_ids = set()
        for name in ('liftoff', 'apex', 'touchdown'):
            event_ids.update((matrix[:, index[name]] > .5).nonzero().flatten().tolist())
        event_ids.update(resets.nonzero().flatten().tolist())
        for env_id in sorted(event_ids):
            episode = int(episodes[env_id])
            pending = self.pending[env_id]
            if pending is not None and pending['episode_id'] != episode:
                self.skipped_reset_before_touchdown += 1
                self.pending[env_id] = None
            if read(env_id, 'initial_drop') < .5:
                if read(env_id, 'liftoff') > .5:
                    prior = self.previous[env_id]
                    self.pending[env_id] = dict(
                        env_id=env_id, episode_id=episode, liftoff_step=step,
                        target_x_m=read(env_id, 'target_at_action_w_x'),
                        target_y_m=read(env_id, 'target_at_action_w_y'),
                        next_target_x_m=read(env_id, 'next_target_at_action_w_x'),
                        next_target_y_m=read(env_id, 'next_target_at_action_w_y'),
                        liftoff_foot_x_m=read(env_id, 'foot_pos_w_x'),
                        liftoff_foot_y_m=read(env_id, 'foot_pos_w_y'),
                        launch_body_vx_m_s=read(env_id, 'root_vel_w_x'),
                        launch_body_vy_m_s=read(env_id, 'root_vel_w_y'),
                        hard_environment=read(env_id, 'hard_environment') > .5,
                        global_thrust_scale=read(env_id, 'global_thrust_scale'),
                        prior_touchdown_foot_error_m=prior,
                        prior_foot_miss_10cm=(prior is not None and prior >= .10),
                        apex_step=None, apex_foot_error_m=None,
                    )
                    if self.contact_phase:
                        details = {name: None for name in self.PHASE_FIELDS}
                        prior_touchdown = self.previous_touchdown[env_id]
                        if prior_touchdown is not None and prior_touchdown['episode_id'] == episode:
                            details.update({f'prior_touchdown_{key}': value
                                            for key, value in prior_touchdown.items()
                                            if key != 'episode_id'})
                        contact = self.contact[env_id]
                        if contact is not None and contact['episode_id'] == episode:
                            details.update(
                                contact_dwell_s=(step - contact['step']) * self.step_dt,
                                contact_proxy_samples=contact['contact_proxy_samples'],
                                contact_peak_spring_q_m=contact['contact_peak_spring_q_m'],
                                contact_min_spring_qdot_m_s=contact['contact_min_spring_qdot_m_s'],
                                contact_max_spring_qdot_m_s=contact['contact_max_spring_qdot_m_s'],
                                contact_mean_motor_pwm=contact['motor_pwm_sum'] /
                                contact['motor_pwm_samples'])
                        details.update(liftoff_body_vz_m_s=read(env_id, 'root_vel_w_z'),
                                       liftoff_tilt_deg=read(env_id, 'tilt_deg'),
                                       liftoff_spring_q_m=read(env_id, 'spring_q_m'))
                        pre = pre_snapshot(env_id, episode)
                        if pre is not None:
                            details['liftoff_foot_speed_xy_m_s'] = pre['foot_speed_xy_m_s']
                        self.pending[env_id].update(details)
                        self.contact[env_id] = None
                pending = self.pending[env_id]
                if read(env_id, 'apex') > .5 and pending is not None:
                    pending['apex_step'] = step
                    pending['apex_foot_error_m'] = math.hypot(
                        read(env_id, 'foot_pos_w_x') - pending['target_x_m'],
                        read(env_id, 'foot_pos_w_y') - pending['target_y_m'])
                if read(env_id, 'touchdown') > .5:
                    tx = read(env_id, 'target_at_action_w_x')
                    ty = read(env_id, 'target_at_action_w_y')
                    foot_error = math.hypot(read(env_id, 'foot_pos_w_x') - tx,
                                            read(env_id, 'foot_pos_w_y') - ty)
                    if pending is None or pending['liftoff_step'] >= step:
                        self.skipped_no_liftoff += 1
                    elif math.hypot(tx - pending['target_x_m'],
                                    ty - pending['target_y_m']) > 1e-4:
                        self.skipped_target_changed += 1
                    else:
                        phase = None
                        if self.contact_phase:
                            phase = {f'touchdown_{key}': value for key, value in
                                     phase_snapshot(env_id).items()}
                            pre = pre_snapshot(env_id, episode)
                            if pre is not None:
                                phase.update({f'touchdown_pre_{key}': value
                                              for key, value in pre.items()})
                        self._complete(pending, step, matrix, index, env_id,
                                       bool(resets[env_id]), phase)
                    self.previous[env_id] = foot_error
                    self.pending[env_id] = None
                    if self.contact_phase:
                        snapshot = phase_snapshot(env_id)
                        pre = pre_snapshot(env_id, episode)
                        self.previous_touchdown[env_id] = dict(episode_id=episode, step=step, **snapshot)
                        if pre is not None:
                            self.previous_touchdown[env_id].update(
                                {f'pre_{key}': value for key, value in pre.items()})
                        self.contact[env_id] = dict(
                            episode_id=episode, step=step,
                            contact_proxy_samples=int(read(env_id, 'spring_contact_proxy') > .5),
                            contact_peak_spring_q_m=read(env_id, 'spring_q_m'),
                            contact_min_spring_qdot_m_s=read(env_id, 'spring_qdot_m_s'),
                            contact_max_spring_qdot_m_s=read(env_id, 'spring_qdot_m_s'),
                            motor_pwm_sum=sum(read(env_id, f'motor_pwm_{motor}')
                                              for motor in range(1, 5)) / 4,
                            motor_pwm_samples=1)
            if bool(resets[env_id]):
                if self.pending[env_id] is not None:
                    self.skipped_reset_before_touchdown += 1
                self.pending[env_id] = None
                self.previous[env_id] = None
                self.previous_touchdown[env_id] = None
                self.contact[env_id] = None
        if self.contact_phase:
            self.previous_sample = episodes.clone(), matrix.clone()
        self.last_step = step

    def _complete(self, pending, step, matrix, index, env_id, reset_pending, phase=None):
        def read(name):
            return float(matrix[env_id, index[name]])

        dt = (step - pending['liftoff_step']) * self.step_dt
        dx = pending['target_x_m'] - pending['liftoff_foot_x_m']
        dy = pending['target_y_m'] - pending['liftoff_foot_y_m']
        distance = math.hypot(dx, dy)
        next_dx = pending['next_target_x_m'] - pending['target_x_m']
        next_dy = pending['next_target_y_m'] - pending['target_y_m']
        next_distance = math.hypot(next_dx, next_dy)
        turn_angle = (math.degrees(math.atan2(dx * next_dy - dy * next_dx,
                                              dx * next_dx + dy * next_dy))
                      if distance > 1e-9 and next_distance > 1e-9 else None)
        ref_x, ref_y = dx / dt, dy / dt
        error_x = pending['launch_body_vx_m_s'] - ref_x
        error_y = pending['launch_body_vy_m_s'] - ref_y
        foot_x, foot_y = read('foot_pos_w_x'), read('foot_pos_w_y')
        foot_error = math.hypot(foot_x - pending['target_x_m'],
                                foot_y - pending['target_y_m'])
        apex, target_height = read('settled_apex_height_m'), read('settled_apex_target_m')
        height_ok = apex >= target_height and abs(apex - target_height) <= .15
        row = dict(pending)
        row.update(
            touchdown_step=step, airtime_s=dt, p1_distance_at_liftoff_m=distance,
            next_segment_distance_m=next_distance, turn_angle_deg=turn_angle,
            touchdown_foot_x_m=foot_x, touchdown_foot_y_m=foot_y,
            reference_vx_m_s=ref_x, reference_vy_m_s=ref_y,
            launch_mismatch_m_s=math.hypot(error_x, error_y),
            launch_along_error_m_s=(error_x * dx + error_y * dy) / distance if distance > 1e-9 else None,
            launch_cross_error_m_s=(error_x * -dy + error_y * dx) / distance if distance > 1e-9 else None,
            foot_displacement_x_m=foot_x - pending['liftoff_foot_x_m'],
            foot_displacement_y_m=foot_y - pending['liftoff_foot_y_m'],
            settled_apex_m=apex, apex_target_m=target_height,
            touchdown_foot_error_m=foot_error, strict_foot_10cm=foot_error < .10,
            strict_height_window=height_ok, strict_pair=foot_error < .10 and height_ok,
            reset_pending=reset_pending,
        )
        if phase is not None:
            row.update(phase)
        self.rows.append(row)

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'hops.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)
        report = dict(
            rows=len(self.rows), num_envs=self.num_envs, step_dt_s=self.step_dt,
            last_step=self.last_step, skipped_no_liftoff=self.skipped_no_liftoff,
            skipped_target_changed=self.skipped_target_changed,
            skipped_reset_before_touchdown=self.skipped_reset_before_touchdown,
            csv=str(directory / 'hops.csv'),
            sample_timing='after reward/cycle events, before automatic reset',
            target_timing='target_at_action before event target advancement',
            velocity_definition='Body rigid-body COM world XY velocity at liftoff, not whole-robot COM velocity',
            reference_definition='liftoff foot-to-current-P1 displacement divided by realized airtime; diagnostic only',
            turn_definition='angle from actual liftoff-foot-to-P1 vector to planned P1-to-P2 vector',
            success_definition='foot error <0.10m, apex >= target and <= target+0.15m',
            contact_phase_audit=self.contact_phase,
        )
        if self.contact_phase:
            report.update(
                contact_definition='spring q > 0.002 m proxy; no contact force or impulse measured',
                phase_timing='event-sample state after preceding control interval; pre state is one 10 ms sample earlier',
                foot_speed_definition='XY finite difference of foot position between adjacent pre-reset samples',
                dwell_definition='touchdown-to-liftoff event sample time; motor mean covers these samples',
            )
        (directory / 'index.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
