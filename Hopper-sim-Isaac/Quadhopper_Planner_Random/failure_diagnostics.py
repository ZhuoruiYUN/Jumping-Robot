"""Evaluation-only first-physical-death snapshots for vectorized hopping."""
import torch


class FailureDiagnostics:
    """Keep the first physical failure per environment across episode resets."""

    FIELDS = (
        'step', 'root_z_m', 'root_vz_m_s', 'roll_deg', 'pitch_deg',
        'roll_rate_rad_s', 'pitch_rate_rad_s', 'spring_position_m',
        'target_distance_m', 'seconds_since_touchdown',
        'seconds_since_liftoff', 'last_touchdown_was_short',
        'in_contact', 'cycle_active', 'moving_recovery_remaining_steps',
        'hard_environment', 'global_thrust_scale', 'tilt_death',
        'low_height_death', 'far_xy_death', 'high_angvel_death',
    )

    def __init__(self, num_envs: int, device: str):
        self.last_touchdown_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )
        self.last_liftoff_step = torch.full_like(self.last_touchdown_step, -1)
        self.last_touchdown_was_short = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.captured = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.rows = torch.full((num_envs, len(self.FIELDS)), float('nan'), device=device)

    def reset_episode(self, ids: torch.Tensor) -> None:
        self.last_touchdown_step[ids] = -1
        self.last_liftoff_step[ids] = -1
        self.last_touchdown_was_short[ids] = False

    def note_events(self, step: int, touchdown: torch.Tensor,
                    liftoff: torch.Tensor, touchdown_was_short: torch.Tensor) -> None:
        self.last_touchdown_step = torch.where(
            touchdown, step, self.last_touchdown_step
        )
        self.last_liftoff_step = torch.where(
            liftoff, step, self.last_liftoff_step
        )
        self.last_touchdown_was_short = torch.where(
            touchdown, touchdown_was_short, self.last_touchdown_was_short
        )

    def capture(self, physical_death: torch.Tensor, *, step: int,
                step_dt: float, values: dict[str, torch.Tensor]) -> None:
        fresh = physical_death & ~self.captured
        n = len(self.captured)
        def age(last_step):
            return torch.where(
                last_step >= 0,
                (step - last_step).float() * step_dt,
                torch.full((n,), -1.0, device=self.rows.device),
            )
        measurements = {
            'step': torch.full((n,), float(step), device=self.rows.device),
            'seconds_since_touchdown': age(self.last_touchdown_step),
            'seconds_since_liftoff': age(self.last_liftoff_step),
            'last_touchdown_was_short': self.last_touchdown_was_short.float(),
            **values,
        }
        if set(measurements) != set(self.FIELDS):
            raise ValueError('failure snapshot fields do not match the contract')
        current = torch.stack([measurements[name].float() for name in self.FIELDS], dim=-1)
        if current.shape != self.rows.shape:
            raise ValueError('failure snapshot tensors must match the environment count')
        self.rows = torch.where(fresh[:, None], current, self.rows)
        self.captured |= fresh

    def summary(self) -> list[dict[str, float | int | bool | str]]:
        rows = self.rows[self.captured].detach().cpu().tolist()
        ids = self.captured.nonzero(as_tuple=False).flatten().cpu().tolist()
        result = []
        boolean_fields = {
            'last_touchdown_was_short', 'in_contact', 'cycle_active',
            'hard_environment', 'tilt_death', 'low_height_death',
            'far_xy_death', 'high_angvel_death',
        }
        for env_id, row in zip(ids, rows, strict=True):
            record = {
                name: (bool(value) if name in boolean_fields else
                       int(value) if name in ('step', 'moving_recovery_remaining_steps') else
                       value)
                for name, value in zip(self.FIELDS, row, strict=True)
            }
            touchdown_age = record['seconds_since_touchdown']
            liftoff_age = record['seconds_since_liftoff']
            if record['in_contact']:
                phase = (
                    'contact_after_touchdown'
                    if touchdown_age >= 0 and (liftoff_age < 0 or touchdown_age <= liftoff_age)
                    else 'contact_other'
                )
            elif liftoff_age >= 0 and (touchdown_age < 0 or liftoff_age < touchdown_age):
                phase = 'airborne_after_liftoff'
            else:
                phase = 'airborne_other'
            result.append({'env_id': env_id, 'phase': phase, **record})
        return result
