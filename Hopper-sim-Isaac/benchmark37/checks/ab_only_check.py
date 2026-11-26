"""Check the AB training boundary and long-route evaluation without optimizer updates."""
import torch

from ..stationary_retention import RetentionRoute


def check_routes(course):
    config = course['omni']
    ids = torch.arange(1024)
    options = dict(training_stationary_fraction=course['stationary_retention']['training_stationary_fraction'])
    route = RetentionRoute(len(ids), 'cpu', config, 42, **options)
    route.reset(ids)
    ab = route.ab_pairs
    assert ab.mask.all() and not route.stationary_mask.any()
    assert (route.index == 1).all() and (ab.a_index == 1).all()
    assert route.points[:, :4].diff(dim=1).norm(dim=-1).max() <= .500001
    valid = torch.ones_like(ab.mask)
    strict = ids % 2 == 0
    a, b = route.pair(ids)
    assert not ab.touchdown(route.index, valid, strict).any()
    route.advance(valid, strict, valid)
    torch.testing.assert_close(route.pair(ids)[0], b)
    assert ab.touchdown(route.index, valid, strict).all()
    route.reset(ids[::2])
    assert (route.index[::2] == 1).all() and (route.index[1::2] == 2).all()

    from ..precision_routes import SEEDS, routes
    eval_ids = ids[:128]
    for suite, continuous in [('development', False), ('continuous', True)]:
        seed = SEEDS[suite]
        count = 96 if continuous else 32
        cfg = dict(config, precision_eval_seed=seed, continuous_route_eval=continuous, route_points=count)
        evaluation = RetentionRoute(len(eval_ids), 'cpu', cfg, seed, evaluation=True, **options)
        evaluation.reset(eval_ids)
        assert evaluation.ab_pairs is None and not evaluation.stationary_mask.any()
        torch.testing.assert_close(evaluation.points, torch.tensor(routes(seed, count, continuous=continuous)))
        for step in range(12):
            _, preview = evaluation.pair(eval_ids)
            evaluation.advance(valid[:128], valid[:128])
            torch.testing.assert_close(evaluation.pair(eval_ids)[0], preview)
    return dict(ab_training_lanes=len(ids), eval_routes=len(eval_ids), consecutive_eval_advances=12)


@torch.inference_mode()
def check_simulation(raw, wrapped, runner):
    route = raw.omni_route
    assert route.ab_pairs.mask.all() and not route.stationary_mask.any() and not hasattr(raw,'key_guard')
    obs = wrapped.get_observations()
    policy = runner.get_inference_policy(device=raw.device)
    ids = torch.arange(raw.num_envs, device=raw.device)
    completed = 0
    midflight = raw.course.get('retry_midflight')
    midflight_counts = torch.zeros(2, dtype=torch.long, device=raw.device)
    safe_assist = (midflight or {}).get('safe_assist')
    preview_counts = torch.zeros(2, dtype=torch.long, device=raw.device)
    assist_counts = torch.zeros(3, dtype=torch.long, device=raw.device)
    compression = raw.course.get('retry_compression')
    compression_counts = torch.zeros(4, dtype=torch.long, device=raw.device)
    support_preview = raw.course.get('support_preview')
    support_counts = torch.zeros(5, dtype=torch.long, device=raw.device)
    waypoint_events = raw.course.get('waypoint_event_costs')
    event_counts = torch.zeros(5, dtype=torch.long, device=raw.device)
    capture = raw.course.get('retry_capture')
    capture_counts = torch.zeros(5, dtype=torch.long, device=raw.device)
    recovery = route.ab_pairs.recovery
    recovery_cost_ticks = torch.zeros(2, dtype=torch.long, device=raw.device)
    check_yaw = bool(raw.course.get('moving_yaw_prediction', {}).get('free_air_peak_weight'))
    yaw_events = torch.zeros(2, dtype=torch.long, device=raw.device)
    envelope = raw.course.get('moving_yaw_prediction', {}).get('all_phase_envelope')
    envelope_ticks = torch.zeros(7, dtype=torch.long, device=raw.device)
    precision = raw.course.get('moving_landing_guidance', {}).get('precision_to_contact')
    precision_counts = torch.zeros(3, dtype=torch.long, device=raw.device)
    first_cross = raw.course.get('first_landing_cross_feedback')
    signed_rebound = raw.course.get('signed_retry_rebound')
    compensation_counts = torch.zeros(4, dtype=torch.long, device=raw.device)
    tracking = raw.course.get('retry_rebound_tracking')
    rebound_tracking_counts = torch.zeros(4, dtype=torch.long, device=raw.device)
    paired = route.ab_pairs.paired_axes
    paired_releases_checked = 0
    yaw_reserve = raw.course.get('yaw_thrust_reserve_n')
    cross_control = raw.course.get('cross_rebound_control')
    height_control = raw.course.get('apex_governor')
    flight_control = cross_control or height_control
    if flight_control:
        flight_control_ticks = torch.zeros(2, dtype=torch.long, device=raw.device)
    repeatability = raw.course.get('hop_repeatability')
    overheight = raw.course.get('overheight_band')
    hop_band_checks = torch.zeros(3, dtype=torch.long, device=raw.device)
    new_teaching_checks = torch.zeros(2, dtype=torch.long, device=raw.device)
    ws_teaching = raw.course.get('ws_transfer_teaching')
    if ws_teaching:
        from ..v81_ws_transfer import terms
        ws_terms = terms(ws_teaching)
        ws_cost_ticks = torch.zeros(len(ws_terms), dtype=torch.long, device=raw.device)
        ws_landing_ticks = torch.zeros(2, dtype=torch.long, device=raw.device)
        v82_options = (ws_teaching['landing'].get('undertravel'),
                       ws_teaching.get('opening_release_speed_m_s'),
                       raw.course['vertical_pretraining']['reference_flight_reward'].get('surplus_height_scale'))
        if any(v82_options):
            v82_ticks = torch.zeros(3, dtype=torch.long, device=raw.device)
        if ws_teaching.get('cross_rebound'):
            ws_cross_ticks = torch.zeros(2, dtype=torch.long, device=raw.device)
        if ws_teaching.get('retained_ascent_pulse'):
            retained_pulse_ticks = torch.zeros(2, dtype=torch.long, device=raw.device)
    continuation_forecast = getattr(raw, 'ws_continuation_forecast', None)
    if continuation_forecast is not None:
        continuation_ticks = torch.zeros(2, dtype=torch.long, device=raw.device)
        max_continuation_credit = torch.zeros((), device=raw.device)
    continuation_cruise = getattr(raw, 'continuation_cruise', None)
    if continuation_cruise is not None:
        cruise_ticks = torch.zeros(2, dtype=torch.long, device=raw.device)
        cruise_relief_total = torch.zeros((), device=raw.device)
    heading_servo = raw.course.get('yaw_heading_servo')
    opening_hold = route.ab_pairs.opening_hold_probability
    if opening_hold:
        opening_starts = (route.index == 0).sum()
        opening_advances = torch.zeros_like(opening_starts)
    yaw_contact = raw.course.get('yaw_contact_jump')
    if yaw_reserve is not None:
        from ..yaw_thrust_feedback import apply_yaw_feedback
        allocated_ticks = torch.zeros((), dtype=torch.long, device=raw.device)
    if heading_servo is not None:
        from ..yaw_heading_servo import allocate_heading_servo
    if yaw_contact is not None:
        contact_yaw_ticks = torch.zeros((), dtype=torch.long, device=raw.device)
    response = getattr(raw, 'response_envelope', None)
    response_tail = response is not None and response.tail_fraction > 0.
    if response_tail:
        core_low = response.training_gain_range[0]
        gain_min = torch.ones((), device=raw.device)
        gain_resets = torch.zeros(3, dtype=torch.long, device=raw.device)
        tail_flight_ticks = torch.zeros((2, len(raw._imu_link_plant.parameters)),
                                       dtype=torch.long, device=raw.device)
        profile_ids = torch.arange(tail_flight_ticks.shape[1], device=raw.device)

    def check_spawn(reset_ids):
        nonlocal paired_releases_checked
        if response_tail and len(reset_ids):
            gains = response.gain[reset_ids]
            base = response.base_inertia[reset_ids]
            inertia = raw.dr_inertia_multi[reset_ids]
            torch.testing.assert_close(inertia[:, :2], base[:, :2] / gains[:, None])
            torch.testing.assert_close(inertia[:, 2], base[:, 2])
            assert (gains >= response.config['roll_pitch_gain_range'][0]).all()
            torch.minimum(gain_min, gains.min(), out=gain_min)
            gain_resets.add_(torch.stack(((gains == 1.).sum(),
                                          ((gains >= core_low) & (gains < 1.)).sum(),
                                          (gains < core_low).sum())))
        if route.ab_pairs.repeated_hold_fraction:
            held = reset_ids[route.ab_pairs.repeated_hold[reset_ids]]
            assert not route.points[held, :4].any() and (route.index[held] == route.ab_pairs.a_index[held]).all()
            if raw.repeated_hold_full_response:
                ordinary = reset_ids[~route.ab_pairs.repeated_hold[reset_ids]]
                gain = raw.response_envelope.gain
                assert (gain[ordinary] >= raw.response_envelope.training_gain_range[0]).all()
                weak_repeated_holds.add_((gain[held] < raw.response_envelope.training_gain_range[0]).sum())
        if paired is not None:
            paired_ids = reset_ids[paired.lanes[reset_ids]]
            d = raw._robot.data
            local_xy = d.root_link_pos_w[paired_ids, :2] - raw._terrain.env_origins[paired_ids, :2]
            current, preview = route.pair(paired_ids)
            torch.testing.assert_close(current, local_xy)
            torch.testing.assert_close(current, preview)
            for name, actual in (
                ('release_pose', d.root_link_pose_w),
                ('rigid_scales', raw.rigid_scales.to(raw.device)),
                ('motor_time_constants', raw.dr_tm),
                ('mass_gain', raw.dr_mass_multi),
                ('inertia_gain', raw.dr_inertia_multi),
                ('response_gain', raw.response_envelope.gain),
            ):
                torch.testing.assert_close(actual[paired_ids], paired.cached[name][paired_ids], rtol=0., atol=2e-6)
            paired_releases_checked += int(paired.replaying[paired_ids].sum())
        if not hasattr(raw, 'ab_spawn') or not len(reset_ids):
            return
        mask = raw.ab_spawn.perturbed[reset_ids]
        changed = reset_ids[mask]
        velocity = raw._robot.data.root_com_vel_w[reset_ids]
        c = raw.course['ab_spawn']
        assert (velocity[~mask].abs() < 1e-5).all()
        assert (velocity[mask, :2].norm(dim=-1) <= c['xy_speed_m_s'] + 1e-5).all()
        assert (velocity[mask, 2].abs() <= c['z_speed_m_s'] + 1e-5).all()
        assert (velocity[mask, 3:5].norm(dim=-1) <= c['roll_pitch_rate_rad_s'] + 1e-5).all()
        assert (velocity[mask, 5].abs() <= c['yaw_rate_rad_s'] + 1e-5).all()
        assert (velocity[mask].abs().sum(-1) > 0).all()
        delta = raw._robot.data.root_link_pos_w[changed, :2] - raw._terrain.env_origins[changed, :2]
        assert (delta.norm(dim=-1) <= c['xy_radius_m'] + 1e-5).all()
        assert (delta.norm(dim=-1) > 0).all()
        # A/B stay in the original route frame as the release position varies.
        torch.testing.assert_close(raw._two_ground_pt_xy_w[reset_ids],
                                   raw._terrain.env_origins[reset_ids, :2] + route.pair(reset_ids)[0])
    repeated_holds = route.ab_pairs.repeated_hold_fraction
    if repeated_holds:
        weak_repeated_holds = torch.zeros((), dtype=torch.long, device=raw.device)
        repeated_hold_touches = torch.zeros(2, dtype=torch.long, device=raw.device)
    check_spawn(ids)
    for step in range(1536 if recovery is not None else 384):
        previous = route.index.clone()
        second = route.ab_pairs.seen_a.clone()
        if continuation_forecast is not None:
            is_first = raw.last_attempted_target != previous
            is_b = previous == route.ab_pairs.a_index + 1
            confirmed_continuation = continuation_forecast.target == previous
        if repeated_holds:
            was_hold = route.ab_pairs.repeated_hold.clone()
            was_b = previous == route.ab_pairs.a_index + 1
        current, preview = route.pair(ids)
        if response_tail:
            phase = raw._foot_event
            airborne = (phase.phase == phase.CANDIDATE) | (phase.phase == phase.FLIGHT)
            previous_point = route.points[ids, (previous - 1).clamp_min(0)]
            moving_tail = ((response.gain < core_low) & airborne
                           & (current - previous_point).norm(dim=-1).gt(.10))
            profile = raw._imu_link_plant.profile[:, None] == profile_ids
            tail_flight_ticks[0] += (profile & (moving_tail & ~second)[:, None]).sum(0)
            tail_flight_ticks[1] += (profile & (moving_tail & second)[:, None]).sum(0)
        if recovery is not None:
            before_mode = recovery.active.clone()
        if signed_rebound and recovery is not None:
            assignment = recovery.assignment.clone()
        if capture or midflight:
            edge = current - route.points[ids, (previous - 1).clamp_min(0)]
            normal_long = (before_mode == recovery.CONTINUE) & (edge.norm(dim=-1) >= .4)
        touches_before = int(route.ab_pairs.totals[3])
        actions = policy(obs)
        if yaw_reserve is not None:
            heading = raw.course['moving_yaw_prediction']['target_heading_rad']
            if heading_servo is not None:
                sample = obs['policy']
                allocated = allocate_heading_servo(sample[:,9:13], sample[:,13:16], actions,
                                                   heading, yaw_reserve, heading_servo)
            else:
                allocated = apply_yaw_feedback(obs['policy'], actions, heading, reserve_n=yaw_reserve,
                                               priority=raw.course.get('yaw_feedback_priority'))
        obs, reward, done, extras = wrapped.step(actions)
        for k, course_key, reward_key in (
            (0, 'y_rebound_teaching', 'y_rebound_prediction'),
            (1, 'first_ascent_teaching', 'first_ascent_work'),
        ):
            if raw.course.get(course_key):
                debit = raw._moving_phase_last_audit['deltas'][reward_key]
                assert torch.isfinite(debit).all() and (debit <= 0.).all()
                new_teaching_checks[k] += (debit < 0.).sum()
        if overheight:
            debit = raw._moving_phase_last_audit['deltas']['moving_overheight']
            assert torch.isfinite(debit).all() and (debit <= 0.).all()
            hop_band_checks[0] += (debit < 0.).sum()
        if repeatability:
            debit = raw._moving_phase_last_audit['deltas']['ab_hop_repeatability']
            assert torch.isfinite(debit).all() and (debit <= 0.).all()
            assert not (debit[~second] < 0.).any()
            hop_band_checks[1] += raw._moving_phase_last_audit['repeatability_samples'].sum()
            hop_band_checks[2] += (debit < 0.).sum()
        if first_cross or signed_rebound:
            audit = raw._moving_phase_last_audit
            rebound = audit['rebound']
        if first_cross:
            cross = rebound['first_landing_cross']
            cost = audit['deltas']['first_landing_cross']
            allowed = cross['measured_cross_miss'] | cross['unlanded_terminal']
            assert torch.isfinite(cost).all() and (cost <= 0.).all() and not cost[~allowed].any()
            retries = (before_mode == recovery.RETRY) if recovery is not None else torch.zeros_like(second)
            assert not (cross['active'] & retries).any()
            assert not raw.first_cross_pending_cost[done.bool()].any()
            compensation_counts[0] += cross['active'].sum()
            compensation_counts[1] += (cost < 0.).sum()
        if signed_rebound:
            signed = rebound['signed_retry_rebound']
            active = signed['active']
            if recovery is not None:
                allowed = (~second & (assignment == recovery.RETRY)) | (second & (before_mode == recovery.RETRY))
                assert not (active & ~allowed).any()
            assert not (active & (signed['overshoot_m'] <= raw.landing_radius_m)).any()
            compensation_counts[2] += (active & ~second).sum()
            compensation_counts[3] += (active & second).sum()
        if tracking:
            tracked = rebound['retry_rebound_tracking']
            prepared, confirmed, support = (tracked[k] for k in ('active', 'confirmed_support', 'support_active'))
            assert not (prepared & ~allowed).any()
            assert (tracked['overshoot_m'][prepared] > raw.landing_radius_m).all()
            assert not (confirmed & ~support).any()
            assert not raw.rebound_tracking_in_support[done.bool()].any()
            retained = (~done.bool()) & confirmed
            assert (previous[retained] == route.index[retained]).all()
            assert (recovery.active[retained] == recovery.RETRY).all()
            rebound_tracking_counts += torch.stack(((prepared & ~second).sum(), (prepared & second).sum(),
                                                    confirmed.sum(), support.sum()))
        if yaw_contact is not None:
            cost = raw._moving_phase_last_audit['deltas']['yaw_contact_jump']
            assert torch.isfinite(cost).all() and (cost <= 0.).all()
            contact_yaw_ticks += (cost < 0.).sum()
        if ws_teaching:
            costs = torch.stack(tuple(raw._moving_phase_last_audit['deltas'][key] for key in ws_terms))
            assert torch.isfinite(costs).all() and (costs <= 0.).all()
            audit = raw._ws_teaching_audit
            assert not ((costs[1] < 0.) & (audit['release_gate'] <= 0.)).any()
            assert not ((costs[2] < 0.) & ~(audit['landing_axis_weight'] > 0.).any(dim=-1)).any()
            if ws_teaching.get('cross_rebound'):
                cross_cost = costs[ws_terms.index('ws_cross_rebound')]
                gate = audit['cross_rebound_gate'] > 0.
                signed_error = audit['cross_rebound_prediction_m_s'] - audit['cross_rebound_target_m_s']
                config = ws_teaching['cross_rebound']
                assert not ((cross_cost < 0.) & (~gate | (signed_error.abs() <= config['residual_start_m_s']))).any()
                assert not gate[second | (previous == 0)].any()
                ws_cross_ticks += torch.stack((gate.sum(), (cross_cost < 0.).sum()))
            if ws_teaching.get('retained_ascent_pulse'):
                active = costs[0] < 0.
                gate = audit['retained_pulse_gate'] > 0.
                requested = audit['retained_pulse_requested_apex_m']
                reduced = audit['retained_pulse_reduced_apex_m']
                conditions = ws_teaching['retained_ascent_pulse']
                allowed = (gate & (requested > overheight['upper_m'])
                           & (reduced > conditions['minimum_apex_m'])
                           & (requested - reduced > conditions['min_removable_apex_m']))
                assert not (active & ~allowed).any()
                retained_pulse_ticks += torch.stack((gate.sum(), active.sum()))
            if continuation_forecast is not None:
                credit = audit['continuation_forecast_m']
                native_gate = audit['continuation_native_short_gate']
                applied = credit > 0.
                allowed = is_first & is_b & second & confirmed_continuation
                reduced = audit['undertravel_gate'] < native_gate
                assert torch.isfinite(credit).all() and (credit >= 0.).all()
                assert (credit <= continuation_forecast.config['early_credit_m'] + 1.e-6).all()
                assert not (applied & ~allowed).any()
                assert not (reduced & ~applied).any()
                assert (audit['undertravel_gate'] <= native_gate).all()
                assert (continuation_forecast.target[done.bool()] == -1).all()
                continuation_ticks += torch.stack((applied.sum(), reduced.sum()))
                torch.maximum(max_continuation_credit, credit.amax(), out=max_continuation_credit)
            if continuation_cruise is not None:
                state = raw._moving_phase_last_audit
                cruise = state['rebound']['continuation_cruise']
                allowance, base, relief = (cruise[key] for key in ('allowance_m_s2', 'baseline_cost', 'relief'))
                assert not ((allowance > 0.) & ~(is_first & is_b & second & confirmed_continuation)).any()
                assert torch.isfinite(relief).all() and (relief >= 0.).all()
                assert not ((relief > 0.) & (allowance <= 0.)).any()
                cost = state['deltas']['moving_cruise_acceleration']
                torch.testing.assert_close(cost, base + relief)
                assert (cost <= 1.e-7).all()
                cruise_ticks += torch.stack(((allowance > 0.).sum(), (relief > 0.).sum()))
                cruise_relief_total += relief.sum()
            ws_cost_ticks += (costs < 0.).sum(-1)
            ws_landing_ticks += (audit['landing_axis_weight'] > 0.).sum(0)
            if any(v82_options):
                if v82_options[0]:
                    v82_ticks[0] += (audit['undertravel_gate'] > 0.).sum()
                if v82_options[1]:
                    v82_ticks[1] += (audit['opening_release_gate'] > 0.).sum()
                if v82_options[2]:
                    v82_ticks[2] += raw._surplus_height_scale_active.sum()
        if yaw_reserve is not None:
            alive = ~done.bool()
            if flight_control:
                torch.testing.assert_close(raw._flight_control_input[alive], allocated[alive])
                torch.testing.assert_close(obs['policy'][alive, 53:57], raw._actions[alive])
                if cross_control:
                    cross_force = raw._cross_rebound_control_audit[3]
                    assert (cross_force.abs() <= cross_control['max_mode_n'] + 1e-6).all()
                    flight_control_ticks[0] += (alive & (cross_force.abs() > 1e-5)).sum()
                if height_control:
                    trim = raw._apex_governor_audit[2]
                    assert ((trim >= 0.) & (trim <= height_control['max_collective_trim_n'] + 1e-6)).all()
                    flight_control_ticks[1] += (alive & (trim > 1e-5)).sum()
            else:
                torch.testing.assert_close(obs['policy'][alive, 53:57], allocated[alive])
            allocated_ticks += (alive & ((allocated - actions.clamp(-1, 1)).abs().amax(-1) > 1e-5)).sum()
        if compression:
            audit = raw._moving_phase_last_audit
            rebound = audit['rebound']
            promoted = rebound['compression_promoted']
            delta = audit['deltas']['retry_compression']
            debit = delta < 0
            assert torch.isfinite(delta).all()
            assert not debit[~rebound['compression_tracked'] | (before_mode != recovery.RETRY)].any()
            assert not debit[rebound['compression_error'] <= compression['threshold_m_s']].any()
            assert (recovery.active[promoted] == recovery.RETRY).all()
            assert (recovery.assignment[promoted] == recovery.CONTINUE).all()
            assert not raw.retry_compression.debt[done.bool()].any()
            compression_counts += torch.stack((promoted.sum(), rebound['compression_tracked'].sum(),
                                                debit.sum(), (delta > 0).sum()))
        if support_preview:
            audit = raw._moving_phase_last_audit
            rebound = audit['rebound']
            cost = audit['deltas']['support_preview_release']
            sample = rebound['support_preview_sample']
            assert not sample[~rebound['preparation_active']].any()
            high = rebound['launch_error_squared'] > support_preview['release_threshold_m_s'] ** 2
            assert torch.isfinite(cost).all() and (cost <= 0.).all()
            budget = rebound['support_preview_budget']
            assert not budget[~(high & rebound['launch'])].any()
            assert not cost[~rebound['support_failed']].any()
            assert not raw.support_preview.count[done.bool()].any()
            support_counts += torch.stack((sample.sum(), rebound['launch'].sum(), (budget < 0).sum(), (cost < 0).sum(),
                                           (rebound['launch'] & ~high).sum()))
        if midflight:
            audit = raw._moving_phase_last_audit
            ref = audit['rebound']['retry_reference_acceleration']
            active = audit['rebound']['retry_midflight_active']
            assert not active[before_mode != recovery.RETRY].any()
            assert not ref[~active].any() and torch.isfinite(ref).all()
            assert (ref.norm(dim=-1) <= midflight['max_acceleration_m_s2'] + 1e-6).all()
            assert not raw.retry_latched[done.bool()].any()
            midflight_counts += torch.stack((active.sum(), (normal_long & audit['moving']).sum()))
        if safe_assist:
            rebound = audit['rebound']
            discount = rebound['retry_assist_discount']
            release_ok = rebound['retry_assist_release_ok']
            headroom = rebound['retry_assist_headroom']
            permitted = active & release_ok & (headroom > 0)
            assert not discount[~permitted | normal_long].any()
            assert (discount >= 0).all() and (discount <= ref.norm(dim=-1) + 1e-6).all()
            original = rebound['horizontal_force'] / raw.rebound_config['nominal_mass_kg']
            residual = rebound['cruise_residual'].norm(dim=-1)
            assert torch.isfinite(residual).all() and (residual <= original + 1e-5).all()
            torch.testing.assert_close(residual[~permitted], original[~permitted])
            assert not raw.retry_release_ok[done.bool()].any()
            if raw.omni_handoff_config.get('near_touchdown_preview'):
                mask = rebound['preparation_active'] & ~done.bool() & (route.index == previous)
                goal = current + raw._terrain.env_origins[:, :2]
                distance = (rebound['handoff_predicted_landing_xy'] - goal).norm(dim=-1)
                q = raw._robot.data.root_quat_w
                tilt = torch.rad2deg(torch.acos((1. - 2. * q[:, 1:3].square().sum(-1)).clamp(-1., 1.)))
                handoff = raw.omni_handoff_config
                old_position = (1. + (raw.landing_radius_m - distance) / handoff['position_blend_width_m']).clamp(0., 1.)
                old_conf = old_position * (1. + (15. - tilt) / handoff['tilt_blend_width_deg']).clamp(0., 1.)
                confidence = rebound['handoff_confidence']
                assert (confidence[mask] <= old_conf[mask] + 1e-4).all()
                inside = mask & normal_long & (distance <= raw.landing_radius_m)
                torch.testing.assert_close(confidence[inside], old_conf[inside], atol=1e-4, rtol=0.)
                preview_counts += torch.stack(((mask & (old_conf - confidence > 1e-3)).sum(), inside.sum()))
            assist_counts += torch.stack(((discount > 0).sum(), (active & ~release_ok).sum(),
                                         (active & release_ok & (headroom == 0)).sum()))
        if waypoint_events:
            reached = rebound['first_straight_apex']
            released = rebound['retained_release']
            cross = audit['deltas']['straight_first_apex_cross']
            growth = audit['deltas']['retry_rebound_growth']
            assert not reached[before_mode != recovery.CONTINUE].any()
            assert not released[before_mode != recovery.RETRY].any()
            assert not cross[~rebound['event_miss']].any() and not growth[~rebound['event_miss']].any()
            assert not growth[before_mode != recovery.RETRY].any()
            cleared = rebound['event_settlement'] | done.bool()
            assert not raw.waypoint_outcome.cross[cleared].any() and not raw.waypoint_outcome.growth[cleared].any()
            assert torch.isfinite(cross).all() and (cross <= 0).all()
            assert torch.isfinite(growth).all() and (growth <= 0).all()
            assert (raw.straight_first_apex.release_step[done.bool()] == -1).all()
            event_counts += torch.stack(((reached & ~second).sum(), (reached & second).sum(),
                                         (cross < 0).sum(), released.sum(), (growth < 0).sum()))
        if capture:
            audit = raw._moving_phase_last_audit
            stance = audit['rebound']['retry_capture_stance']
            flight = audit['rebound']['retry_capture_flight']
            active = stance | flight
            assert not active[before_mode != recovery.RETRY].any()
            v = audit['deltas']['retry_capture_velocity']
            tilt = audit['deltas']['retry_capture_tilt']
            for cost in (v, tilt):
                assert torch.isfinite(cost).all() and (cost <= 0).all() and not cost[~active].any()
            capture_counts += torch.stack((stance.sum(), flight.sum(), (v < 0).sum(), (tilt < 0).sum(),
                                          (normal_long & audit['moving']).sum()))
            assert not raw.retry_latched[done.bool()].any()
        if precision:
            audit = raw._moving_phase_last_audit
            guide = audit['guidance']
            costs = audit['deltas']
            active = (audit['moving'] & guide['projection_clock_active']
                      & (guide['remaining_time_s'] < .15) & (audit['clearance'] > 0.)
                      & (costs['moving_landing_error'] == 0.))
            precision_counts[0] += (active & (costs['moving_velocity_guidance'] < 0.)).sum()
            if raw.first_arrival_bonus:
                paid = costs['moving_first_arrival'] > 0.
                assert not paid[second & (previous == 1)].any()
                precision_counts[1] += (paid & ~second).sum()
                precision_counts[2] += (paid & second).sum()
        if check_yaw:
            audit = raw._moving_phase_last_audit
            cost = audit['deltas']['moving_yaw_peak']
            assert torch.isfinite(cost).all() and (cost <= 0).all()
            assert not cost[~audit['rebound']['air']].any()
            for hop, mask in enumerate((~second, second)):
                yaw_events[hop] += (mask & (cost < 0)).sum()
        if envelope:
            audit = raw._moving_phase_last_audit
            yaw = audit['yaw_prediction']
            cost = audit['deltas']['all_phase_yaw']
            assert torch.isfinite(cost).all() and (cost <= 0).all()
            # Even below the warning threshold, both present heading and spin are damped.
            assert (cost[(yaw['error'].abs() > 1e-5) | (yaw['body_wz'].abs() > 1e-5)] < 0).all()
            phases = [(yaw['phase'] == k) for k in range(4)]
            recovery_phases = ([before_mode == recovery.RETRY, before_mode == recovery.LOCAL_HOLD]
                               if recovery is not None else [torch.zeros_like(second)] * 2)
            phases += [*recovery_phases, yaw['terminal_physical']]
            envelope_ticks += (torch.stack(phases) & (cost < 0)).sum(-1)
        runner.alg.policy.reset(done)
        assert obs['policy'].shape == (raw.num_envs, 73)
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        terminal_b = second & extras['time_outs'].bool()
        if repeated_holds:
            advanced = was_hold & ~was_b & ~done.bool() & (route.index == previous + 1)
            finished = was_hold & was_b & terminal_b
            repeated_hold_touches += torch.stack((advanced.sum(), finished.sum()))
        n_b = int(route.ab_pairs.totals[3]) - touches_before
        assert n_b == int(terminal_b.sum()) and done[terminal_b].all()
        reset_index = route.index[done.bool()]
        if opening_hold:
            assert ((reset_index == 0) | (reset_index == 1)).all()
            opening_starts += (reset_index == 0).sum()
            entered_a = (previous == 0) & (route.index == 1) & ~done.bool()
            torch.testing.assert_close(route.pair(ids)[0][entered_a], preview[entered_a])
            opening_advances += entered_a.sum()
        else:
            assert (reset_index == 1).all()
        check_spawn(ids[done.bool()])
        changed = (previous == 1) & (route.index == 2)
        if paired is not None:
            departed = changed & paired.lanes
            delta = route.pair(ids)[0][departed] - preview[departed]
            expected = delta.new_tensor([[.5, 0.], [-.5, 0.], [0., .5], [0., -.5]])
            torch.testing.assert_close(delta, expected[paired.direction[departed]])
            changed &= ~paired.lanes
        torch.testing.assert_close(route.pair(ids)[0][changed], preview[changed])
        completed += n_b
        if recovery is not None:
            entered = (before_mode == recovery.CONTINUE) & (recovery.active != recovery.CONTINUE)
            audit = raw._moving_phase_last_audit
            unloading = audit['rebound']['retry_unloading_active']
            penalty = audit['deltas']['moving_contact_handoff_velocity']
            recovery_cost_ticks += ((before_mode == recovery.modes) & unloading & (penalty < 0)).sum(-1)
            assert not done[entered].any() and (route.index[entered] == 1).all()
            retry = entered & (recovery.active == recovery.RETRY)
            hold = entered & (recovery.active == recovery.LOCAL_HOLD)
            new_current, new_preview = route.pair(ids)
            torch.testing.assert_close(new_current[retry], current[retry])
            torch.testing.assert_close(new_preview[retry], preview[retry])
            foot_world = raw._robot.data.body_link_pos_w[:,raw._foot_body_id,:2]
            foot = foot_world - raw._terrain.env_origins[:,:2]
            torch.testing.assert_close(new_current[hold], foot[hold])
            torch.testing.assert_close(new_preview[hold], foot[hold])
            target = new_current + raw._terrain.env_origins[:,:2]
            torch.testing.assert_close(raw._two_ground_pt_xy_w[entered], target[entered])
            desired = (target[entered]-foot_world[entered])/raw.rebound_plan.flight[entered,None]
            limit = raw.landing_guidance_config['speed_limit_m_s']
            desired *= (limit/desired.norm(dim=-1,keepdim=True).clamp_min(1e-9)).clamp(max=1)
            torch.testing.assert_close(raw.rebound_plan.velocity[entered], desired)
            minimum_steps = 1024 if tracking else (768 if signed_rebound else 384)
            if (step + 1 >= minimum_steps and (recovery.landings >= 4).all()
                    and (paired is None or (paired.landings > 0).all())):
                break
    assert completed > 0 and not runner.alg.optimizer.state and runner.current_learning_iteration == 0
    result = dict(environments=raw.num_envs, control_steps=step+1, b_timeouts=completed)
    if response_tail:
        assert (gain_resets > 0).all() and (tail_flight_ticks > 0).all()
        result['training_response_coverage'] = dict(
            minimum_gain=float(gain_min),
            sampled_resets=dict(zip(('nominal', 'core_weaker', 'tail_weaker'), gain_resets.tolist())),
            moving_tail_flight_ticks=dict(zip(('A', 'B'), tail_flight_ticks.tolist())),
            calibrated_profiles=raw._imu_link_plant.audit['profiles'])
    if opening_hold:
        assert opening_starts > 0 and opening_advances > 0
        result['opening_hold'] = dict(starts=int(opening_starts), advanced_to_a=int(opening_advances))
    if yaw_reserve is not None:
        assert allocated_ticks > 0
        result['yaw_reserve_adjusted_lane_ticks'] = int(allocated_ticks)
    if flight_control:
        if cross_control:
            assert flight_control_ticks[0] > 0
        if height_control:
            assert flight_control_ticks[1] > 0
        result['flight_control_lane_ticks'] = dict(zip(('cross_rebound', 'apex_governor'),
                                                       flight_control_ticks.tolist()))
    if overheight:
        assert hop_band_checks[0] > 0
        result['overheight_debit_ticks'] = int(hop_band_checks[0])
    if repeatability:
        assert (hop_band_checks[1:] > 0).all()
        result['repeatability_samples'], result['repeatability_debit_ticks'] = hop_band_checks[1:].tolist()
    for k, course_key in enumerate(('y_rebound_teaching', 'first_ascent_teaching')):
        if raw.course.get(course_key):
            assert new_teaching_checks[k] > 0
            result[course_key + '_debit_ticks'] = int(new_teaching_checks[k])
    if ws_teaching:
        result['ws_teaching_debit_ticks'] = dict(zip(ws_terms, ws_cost_ticks.tolist()))
        result['ws_first_landing_reference_ticks'] = dict(zip(('along', 'cross'), ws_landing_ticks.tolist()))
        if any(v82_options):
            result['ws_first_precision_v82_ticks'] = dict(
                zip(('shortfall', 'opening_rebound', 'surplus_height_scale'), v82_ticks.tolist()))
        if ws_teaching.get('cross_rebound'):
            result['ws_cross_rebound_ticks'] = dict(zip(('eligible', 'active'), ws_cross_ticks.tolist()))
        if ws_teaching.get('retained_ascent_pulse'):
            result['retained_ascent_pulse_ticks'] = dict(zip(('eligible', 'active'), retained_pulse_ticks.tolist()))
        if continuation_forecast is not None:
            assert (continuation_ticks > 0).all(), continuation_ticks
            result['continuation_forecast'] = dict(
                credit_ticks=int(continuation_ticks[0]), reduced_gate_ticks=int(continuation_ticks[1]),
                max_credit_m=float(max_continuation_credit))
        if continuation_cruise is not None:
            assert (cruise_ticks > 0).all(), cruise_ticks
            result['continuation_cruise'] = dict(
                allowance_ticks=int(cruise_ticks[0]), relieved_ticks=int(cruise_ticks[1]),
                reward_cost_reduction=float(cruise_relief_total))
    if yaw_contact is not None:
        result['yaw_contact_jump_penalty_ticks'] = int(contact_yaw_ticks)
    if compression:
        assert compression_counts[1] > 0
        result['retry_compression_ticks'] = dict(zip(('promotions', 'tracked', 'debits', 'returns'),
                                                       compression_counts.tolist()))
    if support_preview:
        assert (support_counts[[0, 1, 4]] > 0).all(), support_counts
        result['support_preview'] = dict(zip(('descent_samples', 'releases', 'flagged', 'charged', 'below_threshold'),
                                             support_counts.tolist()))
    if midflight:
        assert (midflight_counts > 0).all(), midflight_counts
        result['retry_midflight_ticks'] = dict(zip(('correcting','ordinary_long'),midflight_counts.tolist()))
    if waypoint_events:
        assert (event_counts[:4] > 0).all(), event_counts
        result['waypoint_events'] = dict(zip(('apex_A','apex_B','cross_penalties','retry_releases',
                                               'growth_penalties'), event_counts.tolist()))
    if safe_assist:
        assert assist_counts[0] > 0, assist_counts
        if raw.omni_handoff_config.get('near_touchdown_preview'):
            assert (preview_counts > 0).all(), preview_counts
            result['handoff_preview_ticks'] = dict(zip(('reduced', 'inside_disk_ordinary_long'), preview_counts.tolist()))
        result['safe_assist_ticks'] = dict(zip(('discounted', 'release_blocked', 'attitude_blocked'),
                                               assist_counts.tolist()))
    if capture:
        assert (capture_counts > 0).all(), capture_counts
        result['retry_capture_ticks'] = dict(zip(
            ('stance','early_air','velocity_penalty','tilt_penalty','ordinary_long_step'),
            capture_counts.tolist()))
    if precision:
        assert precision_counts[0] > 0
        if raw.first_arrival_bonus:
            assert (precision_counts[1:] > 0).all()
        result['foot_precision'] = dict(zip(('late_guidance_ticks','first_arrival_A','first_arrival_B'),
                                            precision_counts.tolist()))
    if first_cross or signed_rebound:
        result['precision_compensation_ticks'] = dict(zip(
            ('first_cross_active', 'first_cross_settled', 'rebound_on_A', 'rebound_on_retained_B'),
            compensation_counts.tolist()))
    if tracking:
        result['rebound_tracking_ticks'] = dict(zip(
            ('prepared_A', 'prepared_retained_B', 'confirmed_misses', 'loaded_support'),
            rebound_tracking_counts.tolist()))
    if check_yaw:
        result['yaw_peak_penalized_ticks_by_target'] = dict(zip(('A', 'B'), yaw_events.tolist()))
    if envelope:
        assert (envelope_ticks[:(6 if recovery is not None else 4)] > 0).all()
        result['all_phase_yaw_penalty_ticks'] = dict(zip(
            ('drop','ground','candidate','flight','retry','local_hold','physical_terminal'), envelope_ticks.tolist()))
    if recovery is not None:
        assert (recovery.landings > 0).all() and (recovery.releases >= recovery.landings).all()
        result['recovery'] = {key: value.item() for key,value in recovery.metrics().items()}
        assert (recovery_cost_ticks > 0).all()
        result['recovery_handoff_penalty_ticks'] = dict(zip(recovery.names, recovery_cost_ticks.tolist()))
    if hasattr(raw, 'ab_spawn'):
        counts = raw.ab_spawn.touchdowns.tolist()
        assert all(group[1][0] > 0 for group in counts)
        result['spawn_resets_original_perturbed'] = raw.ab_spawn.resets.tolist()
        result['spawn_touches_A_B_attempts_hits'] = counts
    if paired is not None:
        assert paired_releases_checked > 0 and (paired.landings > 0).all(), paired.metrics()
        result['paired_axes'] = dict(repeated_releases_checked=paired_releases_checked,
                                     **{k: int(v) for k, v in paired.metrics().items()})
    if hasattr(raw, 'opening_release'):
        counts = {k:int(v) for k,v in raw.opening_release.metrics().items()}
        assert 0 < counts['full_reverse'] <= counts['eligible'] <= int(route.ab_pairs.totals[0])
        result['ab_opening_release'] = counts
    if repeated_holds:
        result['ab_repeated_hold_starts'] = int(route.ab_pairs.repeated_hold_starts)
        result['ab_repeated_hold_touches'] = dict(zip(('advanced_A', 'terminal_B'), repeated_hold_touches.tolist()))
        if raw.repeated_hold_full_response:
            result['ab_repeated_hold_below_regular_gain'] = int(weak_repeated_holds)
    return result
