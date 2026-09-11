"""Signed XY progress within a single, physically observed flight."""
import torch


def two_hop_pair_success(final_touchdown, target_hit, consecutive_hits):
    """True only on the second touchdown of a fully successful two-hop pair."""
    return final_touchdown & target_hit & (consecutive_hits >= 2)


class LandingXYProgress:
    def __init__(self, num_envs, device):
        self.previous_distance = torch.zeros(num_envs, device=device)
        self.previous_active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reanchor(self, env_ids, distance):
        """Break continuity on reset, liftoff, or a changed landing target."""
        self.previous_distance[env_ids] = distance
        self.previous_active[env_ids] = False

    def update(self, distance, active):
        """Return metres approached; inactive steps still refresh the anchor.

        No per-step clipping: approaching and retreating over the same distance
        cancel even at different speeds. This has no clock or desired velocity.
        """
        delta = torch.where(
            active & self.previous_active,
            self.previous_distance - distance,
            torch.zeros_like(distance),
        )
        self.previous_distance.copy_(distance)
        self.previous_active.copy_(active)
        return delta
