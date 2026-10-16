"""One two-hop objective aligned with strict foot landings and upright contact."""
import math

import torch

from .attitude_recovery_v14 import body_tilt


FOOT_RADIUS_M = 0.10
TILT_WIDTH_RAD = math.radians(8.0)
UPRIGHT_WEIGHT = 0.25
PAIR_XY_WIDTH_M = 0.06


class FootPairRewardState:
    def __init__(self, num_envs, device):
        self.first_quality = torch.zeros(num_envs, device=device)
        self.first_error_sq = torch.zeros(num_envs, device=device)

    def reset(self, env_ids):
        self.first_quality[env_ids] = 0.0
        self.first_error_sq[env_ids] = 0.0

    def observe(self, *, touchdown, initial_drop, phase_was_short,
                foot_error_m, apex_m, target_apex_m, quat_wxyz,
                pair_reward_scale, pair_xy_quality_weight=0.0):
        """Pay once on the long touchdown when both feet and apexes succeeded.

        The smooth upright factor rewards posture only for an actual strict
        foot hit. It does not reward skipping P1, low hops, or a root-only hit.
        """
        valid = touchdown & ~initial_drop
        strict_foot = (foot_error_m < FOOT_RADIUS_M) & (apex_m >= target_apex_m)
        tilt = body_tilt(quat_wxyz)
        upright = (1.0 - UPRIGHT_WEIGHT
                   + UPRIGHT_WEIGHT * torch.exp(-(tilt / TILT_WIDTH_RAD).square()))
        quality = strict_foot.float() * upright
        short = valid & phase_was_short
        long = valid & ~phase_was_short
        pair_quality = self.first_quality * quality
        if pair_xy_quality_weight > 0.0:
            # Within a strict two-foot hit, prefer both feet near their targets.
            # The floor preserves credit for hits near the 10 cm boundary;
            # apex and upright requirements remain the V33 contract.
            xy_quality = torch.exp(-(
                self.first_error_sq + foot_error_m.square()
            ) / (PAIR_XY_WIDTH_M ** 2))
            pair_quality = pair_quality * (
                1.0 - pair_xy_quality_weight
                + pair_xy_quality_weight * xy_quality
            )
        bonus = torch.where(
            long, pair_quality * pair_reward_scale,
            torch.zeros_like(quality),
        )
        self.first_quality[short] = quality[short]
        self.first_error_sq[short] = foot_error_m[short].square()
        self.first_quality[long] = 0.0
        self.first_error_sq[long] = 0.0
        return bonus, strict_foot & valid, tilt
