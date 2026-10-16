"""Align the inherited terminal reward with the task's physical-death event."""
import torch


SETTINGS_V16 = {
    'terminal_reward_event': 'physical_death_counter_delta_before_reward',
    'legacy_reward_replacement': 'subtract_legacy_then_add_task_once',
    'base_task': 'apex_event_fix_v15',
    'penalty_scale': 'existing_cfg_termination_penalty_scale',
}


def terminal_reward_correction(physical_death, root_z, quat_wxyz, scale):
    """Replace only Quadhopper_Stable's hard-coded termination contribution.

    The caller obtains physical_death from the actual task done calculation,
    so completion, time limits and target advancement cannot create deaths.
    No termination threshold is recomputed or changed here.
    """
    legacy_death = (root_z < .05) | (quat_wxyz[:, 1:3].square().sum(-1) > .5)
    correction = (physical_death.float() - legacy_death.float()) * scale
    return correction, legacy_death
