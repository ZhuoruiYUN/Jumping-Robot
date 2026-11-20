"""Event metrics from pre-reset samples. No policy inputs or simulator imports."""

import math
import numpy as np


def rebound_motion_metrics(rows,yaw):
    values=[row['rebound_totals'] for row in rows]
    if any(len(v)!=9 or any(not math.isfinite(x) or x<0 for x in v) for v in values):
        raise ValueError('Invalid rebound diagnostic totals')
    total=[sum(v[i] for v in values) for i in range(9)]
    peak=max((v[7] for v in values),default=0.)
    coverage=bool(values) and all(v[6]>0 for v in values)
    rms=math.degrees(math.sqrt(total[4]/total[6])) if total[6]>0 else None
    passed=(coverage and rms<=yaw['rms_heading_goal_deg'] and math.degrees(peak)<=yaw['peak_heading_goal_deg'])
    return dict(episodes_with_motion=sum(v[6]>0 for v in values),metric_coverage_complete=coverage,
        active_air_seconds=total[0],mean_air_horizontal_force_N=total[1]/total[0] if total[0]>0 else None,
        launch_events=int(total[3]),launch_velocity_error_rms_m_s=math.sqrt(total[2]/total[3]) if total[3]>0 else None,
        moving_heading_rms_deg=rms,moving_heading_peak_deg=math.degrees(peak) if coverage else None,
        world_heading_rate_rms_rad_s=math.sqrt(total[5]/total[6]) if total[6]>0 else None,
        preparation_weighted_seconds=total[8],yaw_gate=dict(rms_deg_max=yaw['rms_heading_goal_deg'],
            peak_deg_max=yaw['peak_heading_goal_deg'],passed=bool(passed)),
        scope='All first15s cases; conditional motion diagnostics do not replace missing-attempt/death/foot gates')


class Episode:
    def __init__(self, protocol):
        self.p = protocol
        self.phase = "drop"
        self.steps = 0
        self.events = []
        self.first_touch_step = None
        self.first_touch_tilt_deg = None
        self.first_touch_foot_error_m = None
        self.first_touch_lateral_speed_m_s = None
        self.pre_touch_steps = 0
        self.pre_touch_off_steps = 0
        self.pre_touch_duty_max = 0.0
        self.pre_touch_force_max = 0.0
        self.start = None
        self.air_steps = 0
        self.peak = None
        self.apex_seen = False
        self.peak_foot_formula_error = 0.0
        self.drop_min_root_z = math.inf
        self.drop_last_root_z = None

    def sample(self, s):
        """s is post-physics, pre-auto-reset, including the terminating step."""
        self.steps += 1
        if self.start is None:
            self.start = dict(root_z_m=s["z"], joint_q_m=s["q"], target_z_m=s["target_z"])
        self.peak_foot_formula_error = max(self.peak_foot_formula_error, s["foot_formula_error"])
        contact = s["q"] > self.p["contact_q_on_m"]
        if self.phase == "drop":
            self.drop_min_root_z = min(self.drop_min_root_z, s["z"])
            self.drop_last_root_z = s["z"]
            self.pre_touch_steps += 1
            off = s["duty_max"] <= 1e-6 and s["motor_max"] <= 1e-6 and abs(s["force"]) <= 1e-6
            self.pre_touch_off_steps += int(off)
            self.pre_touch_duty_max = max(self.pre_touch_duty_max, s["duty_max"])
            self.pre_touch_force_max = max(self.pre_touch_force_max, abs(s["force"]))
            if contact:
                self.first_touch_step = self.steps
                self.first_touch_tilt_deg = s["tilt"]
                self.first_touch_foot_error_m = s["foot_error"]
                self.first_touch_lateral_speed_m_s = math.hypot(s["velocity"][0], s["velocity"][1])
                self.phase = "ground"
            return
        if self.phase == "ground":
            if s["q"] <= self.p["takeoff_q_max_m"] and s["vz"] > 0:
                self.phase = "candidate"
                self.air_steps = 1
                self.peak = s["z"]
                self.apex_seen = False
            return
        # candidate and confirmed flight
        self.peak = max(self.peak, s["z"])
        self.apex_seen |= s["vz"] <= 0
        if contact:
            if self.phase == "flight":
                valid_flight = self.air_steps >= self.p["minimum_flight_samples"] and self.apex_seen
                event = dict(
                    jump_index=len(self.events) + 1, touchdown_step=self.steps,
                    flight_samples=self.air_steps, valid_flight=valid_flight,
                    apex_root_link_world_m=self.peak,
                    apex_ge_1m=self.peak >= self.p["apex_min_world_m"],
                    apex_ge_native_target=self.peak >= s["target_z"],
                    foot_error_m=s["foot_error"], root_link_error_m=s["root_error"],
                    touchdown_tilt_deg=s["tilt"], root_com_velocity_world=s["velocity"],
                )
                event["strict"] = bool(valid_flight and event["apex_ge_1m"]
                                       and s["foot_error"] <= self.p["foot_error_max_m"]
                                       and s["tilt"] <= self.p["touchdown_tilt_max_deg"]
                                       and not s["died"])
                self.events.append(event)
            self.phase = "ground"
            return
        self.air_steps += 1
        if self.air_steps >= self.p["takeoff_confirm_samples"]:
            self.phase = "flight"

    def finish(self, died, timed_out, dt):
        first = len(self.events) >= 1 and self.events[0]["strict"]
        second = len(self.events) >= 2 and self.events[1]["strict"]
        completed_pairs = len(self.events) // 2
        # Includes an incomplete final pair as a failure, including no liftoff.
        pair_slots = max(1, (len(self.events) + int(self.phase == "flight") + 1) // 2)
        strict_pairs = sum(self.events[2*i]["strict"] and self.events[2*i+1]["strict"]
                           for i in range(completed_pairs))
        streak = best = 0
        for e in self.events:
            streak = streak + 1 if e["strict"] else 0
            best = max(best, streak)
        return dict(
            start=self.start, seconds=self.steps * dt, died=bool(died), timed_out=bool(timed_out),
            completed_jumps=len(self.events), first_strict=bool(first), second_strict=bool(second),
            first_pair_strict=bool(first and second),
            first_pair_strict_and_survive=bool(first and second and not died),
            pair_slots=pair_slots, completed_pairs=completed_pairs, strict_pairs=strict_pairs,
            best_strict_streak=best, pending_flight=self.phase == "flight",
            first_touch_step=self.first_touch_step, pre_touch_steps=self.pre_touch_steps,
            first_touch_time_s=self.first_touch_step * dt if self.first_touch_step is not None else None,
            first_touch_tilt_deg=self.first_touch_tilt_deg,
            first_touch_foot_error_m=self.first_touch_foot_error_m,
            first_touch_lateral_speed_m_s=self.first_touch_lateral_speed_m_s,
            pre_touch_min_root_z_m=self.drop_min_root_z,
            pre_touch_last_root_z_m=self.drop_last_root_z,
            pre_touch_off_steps=self.pre_touch_off_steps,
            pre_touch_duty_max=self.pre_touch_duty_max, pre_touch_force_max_n=self.pre_touch_force_max,
            foot_formula_max_error_m=self.peak_foot_formula_error, events=self.events,
        )


def distribution(values):
    if not values:
        return dict(n=0, mean=None, minimum=None, p05=None, p50=None, p90=None, p95=None, p99=None, maximum=None)
    x = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(x.mean()), minimum=float(x.min()), p05=float(np.quantile(x, .05)),
                p50=float(np.quantile(x, .5)),
                p90=float(np.quantile(x, .9)), p95=float(np.quantile(x, .95)),
                p99=float(np.quantile(x, .99)), maximum=float(x.max()))


def summarize(rows):
    if not rows:
        raise ValueError("No completed quota episodes; cannot report success")
    events = [e for r in rows for e in r["events"]]
    n = len(rows)
    total_time = sum(r["seconds"] for r in rows)
    deaths = sum(r["died"] for r in rows)
    def rate(key):
        return sum(r[key] for r in rows) / n
    by_jump = {}
    for name, selected in (("first", [r["events"][0] for r in rows if len(r["events"]) >= 1]),
                           ("second", [r["events"][1] for r in rows if len(r["events"]) >= 2]),
                           ("all", events)):
        by_jump[name] = dict(
            completed=len(selected), foot_error_m=distribution([e["foot_error_m"] for e in selected]),
            apex_world_m=distribution([e["apex_root_link_world_m"] for e in selected]),
            tilt_deg=distribution([e["touchdown_tilt_deg"] for e in selected]),
            valid_flights=sum(e["valid_flight"] for e in selected),
            apex_ge_1m_count=sum(e["valid_flight"] and e["apex_ge_1m"] for e in selected),
            apex_ge_native_target_count=sum(e["valid_flight"] and e["apex_ge_native_target"] for e in selected),
            apex_ge_1m_rate_all_episodes=(sum(e["valid_flight"] and e["apex_ge_1m"] for e in selected)/n)
                if name != "all" else None,
            apex_ge_native_target_rate_all_episodes=(sum(e["valid_flight"] and e["apex_ge_native_target"] for e in selected)/n)
                if name != "all" else None,
            missing_attempts=(n-len(selected)) if name != "all" else None,
        )
    return dict(
        episodes=n, exposure_robot_seconds=total_time, deaths=deaths,
        first_touch_count=sum(r["first_touch_step"] is not None for r in rows),
        first_touch_fraction=sum(r["first_touch_step"] is not None for r in rows)/n,
        first_touch_time_s=distribution([r["first_touch_time_s"] for r in rows
                                         if r["first_touch_time_s"] is not None]),
        first_touch_tilt_deg=distribution([r["first_touch_tilt_deg"] for r in rows
                                            if r["first_touch_tilt_deg"] is not None]),
        first_touch_foot_error_m=distribution([r["first_touch_foot_error_m"] for r in rows
                                               if r["first_touch_foot_error_m"] is not None]),
        first_touch_lateral_speed_m_s=distribution([r["first_touch_lateral_speed_m_s"] for r in rows
                                                    if r["first_touch_lateral_speed_m_s"] is not None]),
        initial_release_root_z_m=distribution([r["start"]["root_z_m"] for r in rows]),
        reset_joint_q_absmax_m=max(abs(r["start"]["joint_q_m"]) for r in rows),
        reset_joint_qdot_absmax_m_s=max(abs(r["start"]["joint_qdot_m_s"]) for r in rows),
        reset_joint_nonzero_count=sum(abs(r["start"]["joint_q_m"]) > 1e-4 or
                                      abs(r["start"]["joint_qdot_m_s"]) > 1e-2 for r in rows),
        pre_touch_min_root_z_m=distribution([r["pre_touch_min_root_z_m"] for r in rows]),
        no_touch_end_root_z_m=distribution([r["pre_touch_last_root_z_m"] for r in rows
                                            if r["first_touch_step"] is None]),
        survived_to_timeout_fraction=sum(r["timed_out"] and not r["died"] for r in rows)/n,
        death_fraction=deaths/n, deaths_per_robot_hour=deaths / (total_time/3600),
        zero_death_upper95_iid_episode_approx=(1-math.pow(.05, 1/n)) if deaths == 0 else None,
        reliability_note="episodes share simulator/seed; iid interval is descriptive, not deployment assurance",
        first_strict_rate=rate("first_strict"), second_strict_rate=rate("second_strict"),
        strict_first_pair_rate=rate("first_pair_strict"),
        strict_first_pair_and_survive_rate=rate("first_pair_strict_and_survive"),
        strict_nonoverlap_pair_rate=sum(r["strict_pairs"] for r in rows)/sum(r["pair_slots"] for r in rows),
        pair_slots=sum(r["pair_slots"] for r in rows),
        completed_jumps=distribution([r["completed_jumps"] for r in rows]),
        strict_streak=distribution([r["best_strict_streak"] for r in rows]),
        pre_touch_all_three_off_fraction=sum(r["pre_touch_off_steps"] for r in rows)
            / max(1, sum(r["pre_touch_steps"] for r in rows)),
        pre_touch_duty_max=max(r["pre_touch_duty_max"] for r in rows),
        pre_touch_force_max_n=max(r["pre_touch_force_max_n"] for r in rows),
        foot_formula_max_error_m=max(r["foot_formula_max_error_m"] for r in rows),
        by_jump=by_jump,
        conditional_metrics_note="error/height/tilt distributions condition on touchdown; missing attempts separately counted as failures",
    )


def landing_precision_metrics(rows,maximum_error_m=.1):
    """Observed landing bounds, including retries/holds and fixed-case gaps.

    Initial release contact is not a jump event. A finite observed maximum
    describes this saved episode set, not a bound on future trajectories.
    """
    if not rows or not math.isfinite(maximum_error_m) or not 0<maximum_error_m<=.1:
        raise ValueError('Landing precision requires episodes and a goal <=10cm')
    events=[ev for row in rows for ev in row['events']]
    first=[ev for ev in events if ev['moving'] and ev['first_attempt_for_target']]
    def stats(selected):
        errors=[ev['foot_error_m'] for ev in selected]
        if any(not math.isfinite(e) or e<0 for e in errors):
            raise ValueError('Landing errors must be finite and nonnegative')
        outside=sum(e>maximum_error_m for e in errors)
        return dict(completed_landings=len(errors),outside_goal_count=outside,
            within_goal_rate=(len(errors)-outside)/len(errors) if errors else None,
            foot_error_m=distribution(errors))
    all_stats=stats(events);first_stats=stats(first)
    missing=sum(not any(ev['moving'] for ev in row['events']) for row in rows)
    deaths=sum(row['physical_death'] for row in rows)
    return dict(maximum_error_goal_m=maximum_error_m,episodes=len(rows),
        scope='All post-drop jump landings, including initial stationary hop,holds and retries; moving first attempts also reported separately. Missing moving exposure/deaths cannot pass. Existing height/tilt/continuity gates apply separately.',
        moving_first_attempt=first_stats,all_target_landings=all_stats,
        episodes_without_moving_landing=missing,physical_deaths=deaths,
        passed=bool(first) and missing==0 and deaths==0 and all_stats['outside_goal_count']==0)


def continuity_metrics(rows):
    """Fixed episode denominators expose retries, short exposure and deaths.

    Holds do not increase a moving streak, but a failed hold breaks it.
    Missing later landings never count as a successful prefix. Conditional
    landing distributions remain descriptive, alongside coverage counts.
    """
    if not rows:raise ValueError('No episodes for continuous-jump metrics')
    n=len(rows);prefixes=[];best_streaks=[]
    for row in rows:
        prefix=best=streak=0;prefix_open=True
        for ev in row['events']:
            if ev['target_index']==0:continue
            good=ev['strict'] and (not ev['moving'] or ev['first_attempt_for_target'])
            if not good:
                streak=0;prefix_open=False
            elif ev['moving']:
                streak+=1;best=max(best,streak)
                if prefix_open:prefix+=1
        prefixes.append(prefix);best_streaks.append(best)
    def stats(selected):
        return dict(completed_landings=len(selected),strict_rate=sum(e['strict'] for e in selected)/len(selected) if selected else None,
            foot_error_m=distribution([e['foot_error_m'] for e in selected]),
            touchdown_tilt_deg=distribution([e['touchdown_tilt_deg'] for e in selected]))
    result=dict(episodes=n,physical_deaths=sum(row['physical_death'] for row in rows),
        scope='First15s; fixed episode denominator. Count moving first-attempt strict hits; retries or failed holds break streaks; strict holds preserve but do not increase streaks. Landing distributions condition on completed landings.',
        prefix={},best_consecutive_moving_hits=distribution(best_streaks),by_jump_window={})
    for k in (3,5,8,10):
        count=sum(p>=k for p in prefixes)
        survived=sum(p>=k and not row['physical_death'] for p,row in zip(prefixes,rows))
        result['prefix'][str(k)]=dict(strict_episodes=count,strict_rate=count/n,
            strict_and_survived_episodes=survived,strict_and_survived_rate=survived/n,
            missing_or_failed_episodes=n-count)
    for name,low,high in (('jump2_4',2,4),('jump5_8',5,8),('jump9_plus',9,float('inf'))):
        selected=[ev for row in rows for ev in row['events'] if ev['moving'] and low<=ev['jump_index']<=high]
        result['by_jump_window'][name]=dict(
            episodes_with_moving_landings=sum(any(ev['moving'] and low<=ev['jump_index']<=high for ev in row['events']) for row in rows),
            all_attempts=stats(selected),first_attempts=stats([ev for ev in selected if ev['first_attempt_for_target']]))
    return result


def straight_batch_metrics(rows,spec,yaw,foot_limit_m=.1):
    """All128 cases must complete the first requested targets without retries.

    Keep strict height/tilt scoring and the full15s death/error/yaw exposure.
    This fixed-direction gate says nothing about omnidirectional acceptance.
    """
    if len(rows)!=128 or sorted(r['case_id'] for r in rows)!=list(range(128)):
        raise ValueError('Straight evaluation requires all128 unique ordered case IDs')
    requested=spec['requested_moving_targets'];spacing=spec['spacing_m']
    cases=[]
    for row in sorted(rows,key=lambda r:r['case_id']):
        events=row['events'];first={}
        for ev in events:
            index=ev['target_index'];point=ev['target_local_xy_m']
            if (index<0 or index>=spec['route_points'] or len(point)!=2
                    or not all(math.isfinite(v) for v in point)
                    or abs(point[0]-index*spacing)>1e-4 or abs(point[1])>1e-4):
                raise ValueError('Observed route differs from the declared straight route')
            first.setdefault(index,ev)
        selected=[ev for ev in events if 1<=ev['target_index']<=requested]
        prefix=0
        for index in range(1,requested+1):
            ev=first.get(index)
            if not ev or not ev['strict'] or not ev['first_attempt_for_target']:break
            prefix+=1
        # Exact event sequence also rejects retries of the initial hop and
        # skipped/duplicate targets; navigation advancement cannot credit height.
        sequence=(len(events)>=requested+1 and all(
            ev['target_index']==index and ev['strict'] and ev['first_attempt_for_target']
            for index,ev in enumerate(events[:requested+1])))
        cases.append(dict(case_id=row['case_id'],response_group='nominal' if row['case_id']<64 else 'weaker',
            moving_targets_exposed=sum(i in first for i in range(1,requested+1)),
            first_attempt_strict=sum(first[i]['strict'] and first[i]['first_attempt_for_target']
                for i in range(1,requested+1) if i in first),initial_consecutive_moving_hits=prefix,
            retries=sum(not ev['first_attempt_for_target'] for ev in selected),
            strict_sequence=bool(sequence),physical_death=bool(row['physical_death']),
            strict_sequence_and_survived=bool(sequence) and not row['physical_death']))
    precision=landing_precision_metrics(rows,foot_limit_m);motion=rebound_motion_metrics(rows,yaw)
    def group(selected):
        return dict(episodes=len(selected),episodes_with_all_requested_targets=sum(
            c['moving_targets_exposed']==requested for c in selected),
            strict_sequence_and_survived=sum(c['strict_sequence_and_survived'] for c in selected),
            retries=sum(c['retries'] for c in selected),physical_deaths=sum(c['physical_death'] for c in selected))
    checks=dict(all128_first_requested_sequence_strict_and_survived=all(c['strict_sequence_and_survived'] for c in cases),
        **{f'all15s_landing_errors_within{foot_limit_m*100:g}cm':precision['passed']},
        all15s_yaw=motion['yaw_gate']['passed'])
    return dict(**spec,episodes=128,scope=spec['route_scope'],
        counts=group(cases),response_groups={name:group([c for c in cases if c['response_group']==name])
            for name in ('nominal','weaker')},
        failed_sequence_case_ids=[c['case_id'] for c in cases if not c['strict_sequence_and_survived']],
        cases=cases,checks=checks,passed=all(checks.values()),
        acceptance='Simulation gate for this fixed+X route only; original omnidirectional and stationary gates are unchanged. Hardware deployment is not accepted by this result.')
