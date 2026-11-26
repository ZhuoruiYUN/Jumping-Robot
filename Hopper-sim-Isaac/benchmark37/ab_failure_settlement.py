"""Settle the remaining AB scoring slots on physical failure before reset."""
import math

import torch


class ABFailureSettlement:
    def __init__(self, config, dt, device):
        seconds = config["window_s"]
        weight = config["missing_tick_cost"]
        decay = config["slot_decay"]
        if not (math.isfinite(seconds) and seconds > 0 and math.isfinite(weight) and weight > 0
                and 0 < decay <= 1 and dt > 0):
            raise ValueError("Invalid AB failure settlement window, decay or tick cost")
        self.window = round(seconds / dt)
        if self.window < 1 or not math.isclose(self.window * dt, seconds):
            raise ValueError("AB settlement window must be an integer number of control ticks")
        # Time is measured from release. The reward for the terminating tick
        # has already been earned. Only unobserved slots carry this debit.
        slots = weight * decay ** torch.arange(self.window, device=device, dtype=torch.float64)
        remaining = slots.flip((0,)).cumsum(0).flip((0,))
        self.debit = -torch.cat((remaining, remaining.new_zeros(1))).float()

    def charge(self, elapsed_ticks, physical_failure):
        return torch.where(physical_failure, self.debit[elapsed_ticks.clamp(0, self.window)], 0.)
