"""Versioned task settings; no Isaac or torch dependency for CLI consumers."""

PROTOCOL = 'physical_1m_history_foot_path_free_stage2_v1'
VERSION = 21
PROTOCOL_V2 = 'physical_1m_history_foot_path_free_stage2_v2'
VERSION_V2 = 22
PROTOCOL_V3 = 'physical_1m_history_foot_path_free_stage2_v3_strict_success'
VERSION_V3 = 23
SETTINGS = {
    'use_raibert_hermite_guidance': False,
    'path_tracking_reward_scale': 0.0,
    'path_penalty_scale': 0.0,
    'spatial_progress_reward_scale': 0.0,
    'landing_hint_penalty_scale': 0.0,
    # Remove ballistic landing projection shaping as well: no preferred
    # airborne curve, while actual waypoint progress and apex remain useful.
    'projected_landing_reward_scale': 0.0,
    'projected_landing_penalty_scale': 0.0,
    'target_hit_reward_scale': 140.0,
    'landing_precision_reward_scale': 50.0,
    'stage2_foot_precision_scale': 60.0,
    'stage2_foot_precision_width_m': 0.06,
    'stage2_foot_hit_scale': 40.0,
    'stage2_foot_hit_radius_m': 0.05,
    # Additional estimated electrical energy penalty, reward units per joule.
    # Existing baseline power/synergy terms are retained.
    'stage2_energy_penalty_scale': -0.01,
    'target_relative_valid_apex': True,
    'require_minimum_apex_for_hit': True,
    'gate_touchdown_rewards_by_apex': True,
    'apex_tolerance': 0.04,
}

SETTINGS_V2 = {
    **SETTINGS,
    # Replace, rather than duplicate, the body-origin progress reward.
    'spatial_xy_progress_reward_scale': 0.0,
    'stage2_foot_progress_scale': 20.0,
    'stage2_energy_penalty_scale': -0.05,
    'stage2_nominal_fraction': 0.20,
    'stage2_robust_ramp_end_fraction': 2.0 / 3.0,
    'stage2_energy_ramp_start_fraction': 0.75,
}

SETTINGS_V3 = {
    **SETTINGS_V2,
    # Success must match the strict evaluation criterion: apex >= command.
    # The existing symmetric apex reward remains responsible for limiting
    # overshoot, while the touchdown and pair bonuses no longer pay below 1 m.
    'require_apex_tolerance_for_hit': False,
    'require_apex_at_or_above_target_for_hit': True,
    'stage2_strict_apex_for_landing_reward': True,
    'two_hop_pair_reward_scale': 60.0,
    'stage2_pair_xy_quality_weight': 0.0,
    # Keep measuring estimated power but remove the added Stage2 energy cost.
    'stage2_energy_penalty_scale': 0.0,
    # Continue from the already robust-trained v2 policy at its actual target
    # mix instead of restarting a nominal-to-robust curriculum.
    'stage2_curriculum_training': False,
}


def apply_settings(cfg, *, v2=False, v3=False):
    if v2 and v3:
        raise ValueError('select at most one path-free refinement version')
    settings = SETTINGS_V3 if v3 else SETTINGS_V2 if v2 else SETTINGS
    for name, value in settings.items():
        setattr(cfg, name, value)


def manifest_flags(manifest):
    """Recreate this task from its protocol, including interactive playback."""
    if manifest.get('protocol') == PROTOCOL_V2:
        return ['--path-free-stage2-v2']
    if manifest.get('protocol') == PROTOCOL_V3:
        flags = ['--path-free-stage2-v3']
        if manifest.get('v3_accounting_fix'):
            if any(manifest.get(name) for name in (
                'moving_handoff_v11', 'landing_recovery_v12', 'contact_readiness_v13',
                'attitude_recovery_v14', 'apex_event_fix_v15',
                'terminal_penalty_fix_v16', 'deferred_landing_v17',
                'low_hop_recovery_v18',
            )):
                raise ValueError('v3_accounting_fix requires the unmodified v3 reward task')
            flags.append('--v3-accounting-fix')
        if manifest.get('foot_pair_v33'):
            if not manifest.get('v3_accounting_fix'):
                raise ValueError('foot_pair_v33 requires v3_accounting_fix')
            flags.append('--foot-pair-v33')
        if manifest.get('sysid_20261011_plant'):
            if not manifest.get('foot_pair_v33'):
                raise ValueError('sysid_20261011_plant requires foot_pair_v33')
            flags.append('--sysid-20261011-plant')
        if manifest.get('moving_launch_rate_v34'):
            if not manifest.get('foot_pair_v33'):
                raise ValueError('moving_launch_rate_v34 requires foot_pair_v33')
            flags.append('--moving-launch-rate-v34')
        foot_scale = manifest.get('path_free_stage2', {}).get(
            'stage2_foot_precision_scale', SETTINGS_V3['stage2_foot_precision_scale'])
        if foot_scale != SETTINGS_V3['stage2_foot_precision_scale']:
            if not manifest.get('foot_pair_v33'):
                raise ValueError('custom stage2 foot precision requires foot_pair_v33')
            flags.extend(('--stage2-foot-precision-scale', str(foot_scale)))
        foot_width = manifest.get('path_free_stage2', {}).get(
            'stage2_foot_precision_width_m', SETTINGS_V3['stage2_foot_precision_width_m'])
        if foot_width != SETTINGS_V3['stage2_foot_precision_width_m']:
            if not manifest.get('foot_pair_v33') or not 0.0 < foot_width <= 0.20:
                raise ValueError('custom stage2 foot width requires foot_pair_v33 and width in (0, 0.20]')
            flags.extend(('--stage2-foot-precision-width-m', str(foot_width)))
        hit_10cm_scale = manifest.get('path_free_stage2', {}).get('stage2_foot_hit_10cm_scale')
        if hit_10cm_scale is not None:
            if (not manifest.get('foot_pair_v33') or not isinstance(hit_10cm_scale, (int, float))
                    or not 0.0 <= hit_10cm_scale <= 60.0):
                raise ValueError('10 cm foot hit reward requires foot_pair_v33 and scale in [0, 60]')
            flags.extend(('--stage2-foot-hit-10cm-scale', str(hit_10cm_scale)))
        pair_xy_weight = manifest.get('path_free_stage2', {}).get(
            'stage2_pair_xy_quality_weight', 0.0)
        if pair_xy_weight:
            if not manifest.get('foot_pair_v33'):
                raise ValueError('pair XY quality requires foot_pair_v33')
            flags.extend(('--pair-xy-quality-weight-v36', str(pair_xy_weight)))
        if manifest.get('low_hop_recovery_v18') and manifest.get('deferred_landing_v17'):
            raise ValueError('low_hop_recovery_v18 and deferred_landing_v17 are separate experiments')
        if manifest.get('low_hop_recovery_v18') and not manifest.get('terminal_penalty_fix_v16'):
            raise ValueError('low_hop_recovery_v18 requires terminal_penalty_fix_v16')
        if manifest.get('deferred_landing_v17') and not manifest.get('terminal_penalty_fix_v16'):
            raise ValueError('deferred_landing_v17 requires terminal_penalty_fix_v16')
        if manifest.get('terminal_penalty_fix_v16') and not manifest.get('apex_event_fix_v15'):
            raise ValueError('terminal_penalty_fix_v16 requires apex_event_fix_v15')
        if manifest.get('moving_handoff_v11', False):
            flags.append('--moving-handoff-v11')
        if manifest.get('landing_recovery_v12', False):
            if '--moving-handoff-v11' not in flags:
                flags.append('--moving-handoff-v11')
            flags.append('--landing-recovery-v12')
        if manifest.get('apex_event_fix_v15', False):
            if manifest.get('contact_readiness_v13') or manifest.get('attitude_recovery_v14'):
                raise ValueError('apex_event_fix_v15 requires the v12 task, not v13/v14')
            for flag in ('--moving-handoff-v11', '--landing-recovery-v12', '--apex-event-fix-v15'):
                if flag not in flags:
                    flags.append(flag)
        if manifest.get('terminal_penalty_fix_v16', False):
            flags.append('--terminal-penalty-fix-v16')
        if manifest.get('deferred_landing_v17', False):
            flags.append('--deferred-landing-v17')
        if manifest.get('low_hop_recovery_v18', False):
            flags.append('--low-hop-recovery-v18')
        if manifest.get('contact_readiness_v13', False):
            for flag in ('--moving-handoff-v11', '--landing-recovery-v12'):
                if flag not in flags:
                    flags.append(flag)
            flags.append('--contact-readiness-v13')
        if manifest.get('attitude_recovery_v14', False):
            for flag in ('--moving-handoff-v11', '--landing-recovery-v12',
                         '--contact-readiness-v13', '--attitude-recovery-v14'):
                if flag not in flags:
                    flags.append(flag)
        return flags
    return ['--path-free-stage2'] if manifest.get('protocol') == PROTOCOL else []


def curriculum_values(cfg, step, *, final_energy_scale=None):
    """Training-only schedule, in vector steps (independent of env count).

    Evaluation always uses its requested nominal/robust probability and the
    final energy coefficient, even when evaluating an early checkpoint.
    """
    if final_energy_scale is None:
        final_energy_scale = cfg.stage2_energy_penalty_scale
    if not cfg.stage2_curriculum_training:
        return cfg.hardware_randomization_probability, final_energy_scale
    fraction = float(step) / (cfg.curriculum_steps_per_iteration
                              * cfg.stage2_training_iterations)
    def ramp(start, end):
        return max(0.0, min(1.0, (fraction - start) / (end - start)))
    probability = cfg.hardware_randomization_probability * ramp(
        cfg.stage2_nominal_fraction, cfg.stage2_robust_ramp_end_fraction)
    energy = final_energy_scale * ramp(
        cfg.stage2_energy_ramp_start_fraction, 1.0)
    return probability, energy


def endpoint_terms(error_m, valid_touchdown, power_w, dt, cfg):
    import torch
    precision = cfg.stage2_foot_precision_scale * torch.exp(
        -(error_m / cfg.stage2_foot_precision_width_m).square())
    hit = cfg.stage2_foot_hit_scale * (error_m <= cfg.stage2_foot_hit_radius_m)
    # Opt-in per-hop credit at the strict foot-pair threshold. The caller's
    # valid_touchdown already excludes the initial drop and low-apex hops.
    hit_10cm = getattr(cfg, 'stage2_foot_hit_10cm_scale', 0.0) * (error_m < 0.10)
    landing = (precision + hit + hit_10cm) * valid_touchdown
    # Predicted electrical power is an estimate, not a physical measurement.
    energy_j = power_w.clamp_min(0.0) * dt
    return landing, cfg.stage2_energy_penalty_scale * energy_j, energy_j
