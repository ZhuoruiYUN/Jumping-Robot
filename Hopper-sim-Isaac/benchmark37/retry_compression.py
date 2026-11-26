"""Training signals for a retained waypoint during spring loading."""
import torch

from .ring_geometry import rotate


def projected_release_at_contact(leg, quat, body_rate, velocity, horizon, impulse_ratio):
    """Short contact-frame estimate for sampling; actual route scoring uses measured velocity."""
    axis = torch.nn.functional.normalize(leg, dim=-1)
    axis_rate = torch.linalg.cross(rotate(quat, body_rate), axis)
    loaded_axis = axis + horizon * axis_rate
    vertical_impulse_per_mass = impulse_ratio * (-velocity[:, 2:3]).clamp_min(0.)
    return velocity[:, :2] + vertical_impulse_per_mass * loaded_axis[:, :2] / loaded_axis[:, 2:3].clamp_min(.8)


class RetryCompression:
    def __init__(self, n, device, threshold, width, weight):
        self.threshold = threshold
        self.width = width
        self.weight = weight
        self.tracking = torch.zeros(n, dtype=torch.bool, device=device)
        self.debt = torch.zeros(n, device=device)

    def contact(self, touch, retained, entry_error):
        self.tracking[touch] = (retained & (entry_error <= self.threshold))[touch]

    def step(self, error, flight_time, launch, touch, hit, physical, cancel):
        # Score the change in excess speed on each support tick. Freeze the
        # last balance at liftoff and return it if the hop is recovered.
        excess = (error - self.threshold).clamp_min(0.) * flight_time / self.width
        current = -self.weight * torch.expm1(-.5 * excess.square())
        balance = torch.where(self.tracking & ~physical & ~touch, current, self.debt)
        refund = ~physical & ((touch & hit) | (cancel & ~touch))
        balance = torch.where(refund, 0., balance)
        delta = self.debt - balance
        finished = touch | physical | cancel
        self.debt = torch.where(finished, 0., balance)
        self.tracking &= ~(launch | finished)
        return delta

    def reset(self, ids):
        self.tracking[ids] = False
        self.debt[ids] = 0.
