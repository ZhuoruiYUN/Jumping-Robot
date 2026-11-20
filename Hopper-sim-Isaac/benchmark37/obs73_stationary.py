"""Stationary foundation for OBS73; same actuator/physics and task reward base.

Continuous attempts until physical failure or15s. Ring geometry is visible
context only; ring costs, collisions, completion and route advancement are off.
"""
import torch

TASK = 'obs73_stationary_foundation_v1'
CASE_VERSION = 'obs73_stationary_seed7001_initial128_v1'


def settled_summary(rows, config):
    """Post-settling metrics; missing later flights never count as convergence."""
    from .metrics import distribution
    warmup = config['warmup_landings']
    later = [row['events'][warmup:] for row in rows]
    events = [event for group in later for event in group]
    enough = sum(len(group) >= config['minimum_later_landings_per_episode'] for group in later)
    recorded = [row.get('settled_flight_totals', [0.]*5) for row in rows]
    totals = torch.tensor(recorded).sum(0).tolist()
    count = int(totals[0]);denominator = max(1, count)
    coverage = all(int(value[0]) == len(group) for value, group in zip(recorded, later))
    strict_rate = sum(event['strict'] for event in events)/max(1, len(events))
    foot = distribution([event['foot_error_m'] for event in events])
    apex = distribution([event['apex_root_link_world_m'] for event in events])
    deaths = sum(bool(row['physical_death']) for row in rows)
    pitch = dict(completed_flights=count, mean_span_deg=totals[2]/denominator,
                 span_gt10_rate=totals[3]/denominator, mean_rate_rms_rad_s=totals[4]/denominator)
    passed = (bool(events) and count > 0 and coverage and enough == len(rows)
              and deaths <= config['physical_deaths_max']
              and strict_rate >= config['strict_landing_rate_min']
              and foot['p95'] is not None and foot['p95'] <= config['foot_error_p95_m_max']
              and pitch['mean_span_deg'] <= config['mean_pitch_span_deg_max']
              and pitch['span_gt10_rate'] <= config['pitch_span_gt10_rate_max']
              and pitch['mean_rate_rms_rad_s'] <= config['mean_pitch_rate_rms_rad_s_max'])
    return dict(scope=config['scope'], episodes=len(rows), warmup_landings=warmup,
                episodes_with_enough_later_landings=enough, completed_landings=len(events),
                metric_coverage_complete=coverage, strict_landing_rate=strict_rate,
                foot_error_m=foot, apex_world_m=apex, pitch=pitch,
                physical_deaths_entire_episode=deaths, gate=dict(config, passed=passed))


def env_class():
    from .ring_env import RingEnv
    from .ring_post_stability import terms, validate
    from .ring_targeted_adjustments import FlightPitchStats, pitch_work_cost

    class Obs73StationaryEnv(RingEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stationary_config = self.course['stationary_pretraining']
            validate(self.stationary_config)
            self._context_rng = torch.Generator(device=self.device).manual_seed(int(self.cfg.seed) + 290511)
            self.stationary_stats = FlightPitchStats(self.num_envs, self.device)
            self.stationary_episode_pitch = torch.zeros(self.num_envs, 4, device=self.device)
            for name in ('stationary_pitch', 'stationary_pitch_rate', 'stationary_pitch_work'):
                self._episode_sums[name] = torch.zeros(self.num_envs, device=self.device)

        def _reset_task_targets(self, ids):
            self._two_ground_pt_xy_w[ids] = self._robot.data.root_link_pos_w[ids, :2]
            self._two_ground_pt1_xy_w[ids] = self._two_ground_pt_xy_w[ids]
            self._desired_pos_w[ids, 2] = self._terrain.env_origins[ids, 2] + 1.

        def _reset_idx(self, ids):
            super()._reset_idx(ids)
            if hasattr(self, 'stationary_stats'):
                ids = self._robot._ALL_INDICES if ids is None else ids
                self.stationary_stats.reset(ids)
                self.stationary_episode_pitch[ids] = 0
                low, high = self.stationary_config['ring_context_height_range_m']
                self.ring.center[ids, 2] = self._terrain.env_origins[ids, 2] + low + (high - low) * torch.rand(
                    len(ids), device=self.device, generator=self._context_rng)

        def _ring_sample(self):
            pass

        def _ring_eligible(self):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        def _landing_event(self, touch, designated, died):
            pass  # Failed precision landings are training samples, not terminals.

        def _advance_targets(self):
            pass

        def _get_rewards(self):
            d = self._robot.data
            f = self._foot_event
            c = self.stationary_config
            spring = d.joint_pos[:, self._spring_joint_id]
            rate = d.root_ang_vel_b[:, 1]
            active = (f.phase != f.DROP) & ~self.reset_buf
            inherited, pitch = terms(d.root_quat_w, rate, active, self.step_dt, c)
            airborne = active & ((f.phase == f.CANDIDATE) | (f.phase == f.FLIGHT))
            free = airborne & (spring <= c['free_spring_q_max_m']) & (d.joint_vel[:, self._spring_joint_id].abs() <= c['free_spring_qdot_max_m_s'])
            work = torch.where(free, pitch_work_cost(self._moment[:, 0, 1], rate, self.step_dt, c), 0.)
            contact = spring > .002
            settle = airborne & contact & (f.phase == f.FLIGHT) & (f.air_steps >= 5) & (f.apex_seen | (d.root_com_lin_vel_w[:, 2] <= 0))
            completed = settle & (self.stationary_stats.tracking | airborne)
            self.stationary_stats.step(pitch, rate, airborne, contact, settle)
            span = torch.rad2deg(self.stationary_stats.high - self.stationary_stats.low)
            rms = torch.sqrt(self.stationary_stats.rate_sq / self.stationary_stats.samples.clamp(min=1.))
            self.stationary_episode_pitch += torch.stack((completed.float(), torch.where(completed, span, 0.),
                (completed & (span > 10)).float(), torch.where(completed, rms, 0.)), -1)
            reward = super()._get_rewards()
            extra = dict(stationary_pitch=inherited['post_ring_pitch'],
                         stationary_pitch_rate=inherited['post_ring_pitch_rate'], stationary_pitch_work=work)
            for name, value in extra.items():
                reward = reward + value
                self._episode_sums[name] += value
            logs = {key.replace('PostRing/', 'Stationary/'): value for key, value in self.stationary_stats.log().items()}
            logs.update({'Stationary/valid_landings_total': self._foot_event.touchdowns,
                         'Stationary/strict_landings_total': self._foot_event.qualified})
            self.extras.setdefault('log', {}).update(logs)
            return reward

    return Obs73StationaryEnv


def eval_env_class(base):
    from .ring_evaluation import RingEvaluation
    from .metrics import summarize

    class StationaryEvaluation(RingEvaluation):
        def restore_cases(self, path):
            super().restore_cases(path)
            e = self.env
            ids = e._robot._ALL_INDICES
            e._reset_task_targets(ids)
            from .two_ground_targets import _set_active_goal
            _set_active_goal(e, ids)
            import json
            cases = json.loads(path.read_text())['cases']
            e.ring.center[:] = torch.tensor([case['ring_center_world'] for case in cases], device=e.device)
            response = getattr(e, 'response_envelope', None)
            if response is not None:
                response.restore(cases, e.dr_inertia_multi)

        def freeze_cases(self, folder):
            super().freeze_cases(folder)
            import json
            path = folder / 'cases.json'
            data = json.loads(path.read_text())
            data['version'] = CASE_VERSION
            data['task'] = TASK
            if self.env.course.get('vertical_pretraining'):
                data['version']='obs73_vertical_seed7001_initial128_v1'
                data['evaluation_difficulty']='full1m target and0.8..1.2m/5deg release; curriculum disabled'
                if self.env.evaluation_at_training_difficulty:
                    data['version']='obs73_vertical_training_level_seed7001_initial128_v1'
                    data['evaluation_difficulty']=self.env.vertical_curriculum.current
                    for episode in self.episodes:
                        episode.start['target_z']=self.env.vertical_curriculum.current['target_apex_m']
            response = getattr(self.env, 'response_envelope', None)
            if response is not None:
                from .obs73_response_robustness import CASE_VERSION as RESPONSE_CASE_VERSION
                data['version'] = RESPONSE_CASE_VERSION
                data['response_scope'] = 'same128 releases/base DR; cases0..63 nominal,64..127 weaker roll/pitch; not the old all-nominal benchmark'
                response.annotate(data['cases'])
            path.write_text(json.dumps(data, indent=2) + '\n')

        def finish(self):
            start = len(self.rows)
            super().finish()
            for row in self.rows[start:]:
                i = row['case_id']
                response = getattr(self.env, 'response_envelope', None)
                if response is not None:
                    row['response_group'] = 'nominal' if i < 64 else 'weaker'
                    row['response_gain'] = float(response.gain[i])
                if hasattr(self.env,'plane_totals'):
                    row['flight_plane_totals'] = self.env.plane_totals[i].tolist()
                row['first_four_strict'] = len(row['events']) >= 4 and all(event['strict'] for event in row['events'][:4])
                row['stationary_pitch_totals'] = self.env.stationary_episode_pitch[i].cpu().tolist()
                if hasattr(self.env, 'settled_episode_flight'):
                    row['settled_flight_totals'] = self.env.settled_episode_flight[i].cpu().tolist()
                if hasattr(self.env, 'vertical_episode_yaw'):
                    row['vertical_yaw_totals'] = self.env.vertical_episode_yaw[i].cpu().tolist()
                # This curriculum has no ring outcome; don't present false ring
                # success/failure labels in its per-episode artifacts.
                for key in ('attempted_takeoff','confirmed_takeoff','root_crossed','full_crossed','ring_collision',
                            'reverse','bypass','partial_root_only','minimum_signed_clearance_m','crossing_window',
                            'first_ring_contact','joint_success','joint_and_survive_episode','joint_and_three_later_strict'):
                    row.pop(key, None)

        def result(self):
            assert len(self.rows) == 128
            rows = sorted(self.rows, key=lambda row: row['case_id'])
            keys = ('first_strict', 'first_pair_strict', 'first_four_strict', 'physical_death')
            counts = {key: sum(bool(row[key]) for row in rows) for key in keys}
            totals = torch.tensor([row['stationary_pitch_totals'] for row in rows]).sum(0).tolist()
            events = [event for row in rows for event in row['events']]
            count = max(1, totals[0])
            summary = dict(episodes=128, seed=7001, case_version=CASE_VERSION, case_source=self.case_source,
                task=TASK, denominator='all128 first15s episodes; initial drop excluded; first attempts retained, later retries reported separately',
                counts=counts, rates={key: value / 128 for key, value in counts.items()},
                failure_counts_overlapping=dict(no_confirmed_landing=sum(not row['events'] for row in rows),
                    first_four_not_strict=128-counts['first_four_strict']),
                strict_landing_rate=sum(event['strict'] for event in events)/max(1, len(events)),
                pitch=dict(completed_flights=int(totals[0]), mean_span_deg=totals[1]/count,
                    span_gt10_rate=totals[2]/count, mean_rate_rms_rad_s=totals[3]/count),
                continuous_jump_metrics=summarize(rows),
                promotion_gate=dict(first_four_strict_rate_min=.9, physical_deaths_max=0,
                    passed=counts['first_four_strict']/128 >= .9 and counts['physical_death'] == 0))
            if self.env.course.get('vertical_pretraining'):
                summary.update(case_version='obs73_vertical_seed7001_initial128_v1',
                    benchmark_variant=self.env.course['version'],evaluation_curriculum_disabled=True,
                    evaluation_target_apex_m=1.)
                if self.env.vertical_config.get('yaw_refinement'):
                    yaw = torch.tensor([row['vertical_yaw_totals'] for row in rows])
                    duration = yaw[:,2].sum().clamp(min=.01)
                    peaks = torch.rad2deg(yaw[:,3])
                    summary['yaw'] = dict(
                        scope='post-drop alive samples, including ground preparation and flights',
                        target_heading_rad=self.env.yaw_config['target_heading_rad'],
                        heading_error_rms_deg=float(torch.rad2deg(torch.sqrt(yaw[:,0].sum()/duration))),
                        rate_rms_rad_s=float(torch.sqrt(yaw[:,1].sum()/duration)),
                        peak_abs_heading_error_deg=float(peaks.max()),
                        episode_peak_abs_heading_error_p95_deg=float(torch.quantile(peaks,.95)))
                refinement = self.env.vertical_config.get('attitude_refinement')
                limits = self.env.vertical_config.get('attitude_evaluation_targets')
                if limits is None and refinement:
                    limits = refinement['evaluation_targets']
                if limits:
                    stable = (totals[0] > 0
                              and summary['pitch']['mean_span_deg'] <= limits['mean_pitch_span_deg_max']
                              and summary['pitch']['span_gt10_rate'] <= limits['pitch_span_gt10_rate_max'])
                    summary['attitude_gate'] = dict(limits, passed=stable,
                        scope='completed-flight pitch diagnostic; strict1m landing gate is unchanged')
                    summary['foundation_ready'] = stable and summary['promotion_gate']['passed']
                if self.env.evaluation_at_training_difficulty:
                    target=self.env.vertical_curriculum.current['target_apex_m']
                    tolerance=self.env.vertical_config['training_height_tolerance_m']
                    def native(event):
                        return (event['apex_root_link_world_m'] >= target-tolerance
                                and event['foot_error_m'] <= self.env.landing_radius_m and event['touchdown_tilt_deg'] <= 15.)
                    first_four=sum(len(row['events']) >= 4 and all(native(e) for e in row['events'][:4]) for row in rows)
                    first_four_alive=sum(len(row['events']) >= 4 and all(native(e) for e in row['events'][:4])
                                         and not row['physical_death'] for row in rows)
                    summary.update(case_version='obs73_vertical_training_level_seed7001_initial128_v1',
                                   evaluation_target_apex_m=target,
                                   current_target=dict(target_apex_m=target,height_tolerance_m=tolerance,
                                       completed_landings=len(events),qualified_landings=sum(native(e) for e in events),
                                       qualified_landing_rate=sum(native(e) for e in events)/max(1,len(events)),
                                       first_four_qualified=first_four,first_four_qualified_and_survive=first_four_alive,
                                       physical_deaths=counts['physical_death'],
                                       passed=first_four/128>=.9 and counts['physical_death']==0,
                                       scope='current training height only; strict1m promotion_gate is unchanged'))
            settled_config = self.env.course.get('vertical_pretraining', {}).get('settled_evaluation')
            if settled_config:
                summary['legacy_first_four_promotion_gate'] = summary['promotion_gate']
                summary['settled'] = settled_summary(rows, settled_config)
                summary['promotion_gate'] = summary['settled']['gate']
                summary['foundation_ready'] = summary['promotion_gate']['passed']
            if getattr(self.env, 'response_envelope', None) is not None:
                from .obs73_response_robustness import CASE_VERSION as RESPONSE_CASE_VERSION
                summary['case_version'] = RESPONSE_CASE_VERSION
                summary['response_scope'] = 'single128-case evaluation;64 nominal and64 weaker-response cases; not an all-nominal baseline comparison'
                summary['response_strata'] = {}
                for group in ('nominal', 'weaker'):
                    subset = [row for row in rows if row['response_group'] == group]
                    limits = dict(settled_config, scope=f'jump5+ in the64 {group} response cases')
                    summary['response_strata'][group] = settled_summary(subset, limits)
                passed = all(s['gate']['passed'] for s in summary['response_strata'].values())
                summary['promotion_gate'] = dict(summary['promotion_gate'], passed=passed,
                    scope='both response strata must pass; no averaging away nominal precision or weak-response failures')
                summary['foundation_ready'] = passed
            return summary, rows

    class StationaryEvalEnv(base):
        def __init__(self, *args, **kwargs):
            self.evaluation = None
            super().__init__(*args, **kwargs)
            self.evaluation = StationaryEvaluation(self)

        def _get_rewards(self):
            if self.evaluation is not None:
                self.evaluation.sample()
            reward = super()._get_rewards()
            if self.evaluation is not None:
                self.evaluation.finish()
            return reward

    return StationaryEvalEnv
