"""Owned velocity history and one apex reward event per physical hop."""
import torch


SETTINGS_V15 = {
    'previous_vertical_velocity': 'owned_snapshot',
    'apex_settlement': 'first_velocity_crossing_or_touchdown_fallback_once_per_hop',
    'base_task': 'landing_recovery_v12',
}


class ApexEventState:
    def __init__(self, num_envs, device):
        self.previous_vz = torch.zeros(num_envs, device=device)
        self.emitted = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, candidate, liftoff, touchdown, initial_drop, vertical_speed):
        # Liftoff starts a new physical cycle, independently for each env.
        self.emitted[liftoff] = False
        eligible = candidate & ~initial_drop
        event = eligible & ~self.emitted
        suppressed = eligible & self.emitted
        self.emitted |= event
        # PhysX get_root_velocities() reuses its backing tensor. Never retain
        # its view as history, and never let reset write through that view.
        self.previous_vz.copy_(vertical_speed)
        return event, event & touchdown, suppressed

    def reset(self, env_ids):
        self.previous_vz[env_ids] = 0.0
        self.emitted[env_ids] = False
