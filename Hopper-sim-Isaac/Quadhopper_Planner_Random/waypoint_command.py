from __future__ import annotations

import math

import torch


class TwoHopRandomCommand:
    """Rolling two-target command with alternating short and long hops.

    At reset, P_t is sampled 0.5--0.8 m from the start and P_(t+1) is
    sampled 0.8--1.0 m from P_t.  Once a target is hit, the queue advances
    and a new look-ahead point is sampled.  Independent uniform headings make
    every horizontal direction equally likely while retaining the policy's
    two-cycle look-ahead contract.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        short_radius_min: float,
        short_radius_max: float,
        long_radius_min: float,
        long_radius_max: float,
        successful_hops_per_episode: int,
        max_turn_angle_rad: float = math.pi,
        hard_radius_split: float | None = None,
        hard_radius_probability: float = 0.0,
        zero_hop_probability: float = 0.0,
        reverse_turn_probability: float = 0.0,
        reverse_turn_halfwidth_rad: float = math.pi / 6.0,
        fixed_initial_heading_rad: float | None = None,
        pair_scenario_groups: bool = False,
    ):
        self._validate_range("short", short_radius_min, short_radius_max)
        self._validate_range("long", long_radius_min, long_radius_max)
        if successful_hops_per_episode <= 0:
            raise ValueError("successful_hops_per_episode must be positive")

        self.num_envs = int(num_envs)
        self.device = device
        self.short_radius_min = float(short_radius_min)
        self.short_radius_max = float(short_radius_max)
        self.long_radius_min = float(long_radius_min)
        self.long_radius_max = float(long_radius_max)
        if not 0.0 < max_turn_angle_rad <= math.pi:
            raise ValueError("max_turn_angle_rad must be in (0, pi]")
        self.max_turn_angle_rad = float(max_turn_angle_rad)
        if not 0.0 <= hard_radius_probability <= 1.0:
            raise ValueError("hard_radius_probability must be in [0, 1]")
        if hard_radius_probability > 0.0:
            if hard_radius_split is None:
                raise ValueError(
                    "hard_radius_split is required when hard-radius sampling is enabled"
                )
            for name, lower, upper in (
                ("short", short_radius_min, short_radius_max),
                ("long", long_radius_min, long_radius_max),
            ):
                if not lower < hard_radius_split < upper:
                    raise ValueError(
                        f"hard_radius_split must be strictly inside the {name} "
                        f"range [{lower}, {upper}]"
                    )
        self.hard_radius_split = (
            None if hard_radius_split is None else float(hard_radius_split)
        )
        self.hard_radius_probability = float(hard_radius_probability)
        if not 0.0 <= zero_hop_probability <= 1.0:
            raise ValueError("zero_hop_probability must be in [0, 1]")
        # A non-zero mass is required to train exact in-place hops.  Merely
        # setting the continuous radius lower bound to zero samples values
        # near zero, but never exactly zero in practice.
        self.zero_hop_probability = float(zero_hop_probability)
        if not 0.0 <= reverse_turn_probability <= 1.0:
            raise ValueError("reverse_turn_probability must be in [0, 1]")
        if not 0.0 <= reverse_turn_halfwidth_rad <= math.pi:
            raise ValueError("reverse_turn_halfwidth_rad must be in [0, pi]")
        if (
            reverse_turn_probability > 0.0
            and max_turn_angle_rad < math.pi - reverse_turn_halfwidth_rad
        ):
            raise ValueError(
                "max_turn_angle_rad must include the configured reverse-turn range"
            )
        self.reverse_turn_probability = float(reverse_turn_probability)
        self.reverse_turn_halfwidth_rad = float(reverse_turn_halfwidth_rad)
        if fixed_initial_heading_rad is not None and not math.isfinite(fixed_initial_heading_rad):
            raise ValueError("fixed_initial_heading_rad must be finite when supplied")
        self.fixed_initial_heading_rad = fixed_initial_heading_rad
        self.pair_scenario_groups = bool(pair_scenario_groups)
        # Compatibility name consumed by the shared completion/reward logic.
        self.steps_per_revolution = int(successful_hops_per_episode)

        self.anchor_w = torch.zeros(self.num_envs, 2, device=device)
        self.targets_w = torch.zeros(self.num_envs, 2, 2, device=device)
        # cycle_index is the public height-command phase and may be randomized
        # by the shared environment. Keep route_index independent so that the
        # geometric sequence always remains short, long, short, long.
        self.cycle_index = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.route_index = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.pair_reset_count = torch.zeros_like(self.route_index)
        # 0: unrestricted, 1: reversal, 2: exact-zero patterns.
        self.pair_scenario = torch.zeros_like(self.route_index)
        self.pair_zero_pattern = torch.full_like(self.route_index, -1)

    @staticmethod
    def _validate_range(name: str, lower: float, upper: float):
        if lower < 0.0 or upper < lower:
            raise ValueError(
                f"{name} hop radius must satisfy 0 <= min <= max, got [{lower}, {upper}]"
            )

    def _sample_offset(
        self, count: int, radius_min: float, radius_max: float,
        zero_probability: float | None = None,
    ) -> torch.Tensor:
        if self.hard_radius_probability > 0.0:
            # The lower interval replays already learned hops while the upper
            # interval receives most samples.  This is decided independently
            # for every appended waypoint, including both alternating phases.
            choose_hard = (
                torch.rand(count, device=self.device)
                < self.hard_radius_probability
            )
            unit = torch.rand(count, device=self.device)
            split = self.hard_radius_split
            replay_radius = radius_min + (split - radius_min) * unit
            hard_radius = split + (radius_max - split) * unit
            radius = torch.where(choose_hard, hard_radius, replay_radius)
        else:
            radius = radius_min + (radius_max - radius_min) * torch.rand(
                count, device=self.device
            )
        zero_probability = (
            self.zero_hop_probability if zero_probability is None else zero_probability
        )
        if zero_probability > 0.0:
            is_zero_hop = torch.rand(count, device=self.device) < zero_probability
            radius = torch.where(is_zero_hop, torch.zeros_like(radius), radius)
        heading = 2.0 * math.pi * torch.rand(count, device=self.device)
        return radius[:, None] * torch.stack(
            (torch.cos(heading), torch.sin(heading)), dim=-1
        )

    def _sample_phase_offsets(
        self, is_short: torch.Tensor, zero_probability: float | None = None
    ) -> torch.Tensor:
        """Sample one offset per environment from its assigned route phase."""
        offsets = torch.empty(len(is_short), 2, device=self.device)
        short_ids = is_short.nonzero(as_tuple=False).flatten()
        long_ids = (~is_short).nonzero(as_tuple=False).flatten()
        if len(short_ids) > 0:
            offsets[short_ids] = self._sample_offset(
                len(short_ids), self.short_radius_min, self.short_radius_max,
                zero_probability,
            )
        if len(long_ids) > 0:
            offsets[long_ids] = self._sample_offset(
                len(long_ids), self.long_radius_min, self.long_radius_max,
                zero_probability,
            )
        return offsets

    def _sample_phase_offsets_around(
        self, is_short: torch.Tensor, reference_offsets: torch.Tensor,
        zero_probability: float | None = None,
        reverse_probability: float | None = None,
    ) -> torch.Tensor:
        """Sample phase radii with headings near the preceding hop."""
        offsets = self._sample_phase_offsets(is_short, zero_probability)
        radii = torch.linalg.norm(offsets, dim=1)
        reference_heading = torch.atan2(reference_offsets[:, 1], reference_offsets[:, 0])
        heading_delta = self.max_turn_angle_rad * (
            2.0 * torch.rand(len(is_short), device=self.device) - 1.0
        )
        reverse_probability = (
            self.reverse_turn_probability if reverse_probability is None
            else reverse_probability
        )
        if reverse_probability > 0.0:
            # Keep the ordinary uniform distribution for broad coverage, but
            # add a controllable point mass around +/- 180 deg. Uniform
            # [-180, 180] otherwise makes near-reversal events too sparse for
            # far-hop learning.
            choose_reverse = (
                torch.rand(len(is_short), device=self.device)
                < reverse_probability
            )
            reverse_magnitude = math.pi - self.reverse_turn_halfwidth_rad * torch.rand(
                len(is_short), device=self.device
            )
            reverse_sign = torch.where(
                torch.rand(len(is_short), device=self.device) < 0.5,
                -torch.ones_like(reverse_magnitude),
                torch.ones_like(reverse_magnitude),
            )
            heading_delta = torch.where(
                choose_reverse, reverse_sign * reverse_magnitude, heading_delta
            )
        heading = reference_heading + heading_delta
        return radii[:, None] * torch.stack(
            (torch.cos(heading), torch.sin(heading)), dim=-1
        )

    def reset(self, env_ids: torch.Tensor, env_origins: torch.Tensor, random_phase: bool):
        count = len(env_ids)
        anchor = env_origins[env_ids, :2]
        route_phase = (
            torch.randint(0, 2, (count,), device=self.device)
            if random_phase
            else torch.zeros(count, dtype=torch.long, device=self.device)
        )
        first_is_short = (route_phase % 2) == 0
        first_offset = self._sample_phase_offsets(
            first_is_short, 0.0 if self.pair_scenario_groups else None
        )
        if self.fixed_initial_heading_rad is not None:
            radii = torch.linalg.norm(first_offset, dim=1)
            direction = torch.tensor(
                (math.cos(self.fixed_initial_heading_rad), math.sin(self.fixed_initial_heading_rad)),
                device=self.device,
                dtype=first_offset.dtype,
            )
            first_offset = radii[:, None] * direction
        second_offset = self._sample_phase_offsets_around(
            ~first_is_short, first_offset,
            zero_probability=0.0 if self.pair_scenario_groups else None,
            reverse_probability=0.0 if self.pair_scenario_groups else None,
        )
        if self.pair_scenario_groups:
            # Fixed environment buckets give 50/25/25 coverage at 1024 envs.
            # Cycling the zero patterns also prevents a reset-heavy subset
            # from seeing only one stationary transition.
            scenario = (env_ids * 4 // self.num_envs).clamp(max=3)
            reverse = scenario == 2
            zero = scenario == 3
            self.pair_scenario[env_ids] = torch.where(
                reverse, 1, torch.where(zero, 2, 0)
            )
            self.pair_zero_pattern[env_ids] = -1
            if reverse.any():
                second_offset[reverse] = self._sample_phase_offsets_around(
                    ~first_is_short[reverse], first_offset[reverse],
                    zero_probability=0.0, reverse_probability=1.0,
                )
            if zero.any():
                zero_ids = env_ids[zero]
                pattern = (zero_ids + self.pair_reset_count[zero_ids]) % 3
                self.pair_zero_pattern[zero_ids] = pattern
                # 0: stationary -> stationary; 1: stationary -> moving;
                # 2: moving -> stationary. A zero predecessor has no heading,
                # so sample the moving leg's world heading uniformly.
                first_offset[zero] = torch.where(
                    (pattern != 2)[:, None], torch.zeros_like(first_offset[zero]),
                    first_offset[zero],
                )
                zero_second = self._sample_phase_offsets(
                    ~first_is_short[zero], zero_probability=0.0
                )
                second_offset[zero] = torch.where(
                    (pattern == 1)[:, None], zero_second,
                    torch.zeros_like(zero_second),
                )
            self.pair_reset_count[env_ids] += 1
        first = anchor + first_offset
        second = first + second_offset
        self.anchor_w[env_ids] = anchor
        self.targets_w[env_ids, 0] = first
        self.targets_w[env_ids, 1] = second
        self.cycle_index[env_ids] = 0
        self.route_index[env_ids] = route_phase

    def start_points(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.anchor_w if env_ids is None else self.anchor_w[env_ids]

    def lookahead(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        targets = self.targets_w if env_ids is None else self.targets_w[env_ids]
        return targets[:, 0], targets[:, 1]

    def current_hop_is_short(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return the geometric phase before the corresponding targets advance."""
        return (self.route_index[env_ids] % 2) == 0

    def advance(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        self.anchor_w[env_ids] = self.targets_w[env_ids, 0]
        self.targets_w[env_ids, 0] = self.targets_w[env_ids, 1]
        self.cycle_index[env_ids] += 1
        self.route_index[env_ids] += 1

        # After jump 1, append jump 3 (short); after jump 2, append jump 4
        # (long). This keeps the infinite sequence short, long, short, long.
        append_short = (self.route_index[env_ids] % 2) == 1
        upcoming_offset = self.targets_w[env_ids, 0] - self.anchor_w[env_ids]
        offsets = self._sample_phase_offsets_around(append_short, upcoming_offset)
        self.targets_w[env_ids, 1] = self.targets_w[env_ids, 0] + offsets

    def restart_pair(self, env_ids: torch.Tensor, current_xy_w: torch.Tensor):
        """Start a new short/long pair without an unconstrained boundary turn."""
        if len(env_ids) == 0:
            return
        first_is_short = torch.ones(
            len(env_ids), dtype=torch.bool, device=self.device
        )
        # Before replacement, anchor -> targets[0] is the direction of the
        # just-completed long hop. Constrain the new pair's first hop around
        # that direction as well; the old independent sampling made every
        # pair boundary an arbitrary 0--360 degree turn even when
        # max_turn_angle_rad was small.
        previous_offset = self.targets_w[env_ids, 0] - self.anchor_w[env_ids]
        first_offset = self._sample_phase_offsets_around(
            first_is_short, previous_offset
        )
        second_offset = self._sample_phase_offsets_around(
            ~first_is_short, first_offset
        )
        self.anchor_w[env_ids] = current_xy_w
        self.targets_w[env_ids, 0] = current_xy_w + first_offset
        self.targets_w[env_ids, 1] = self.targets_w[env_ids, 0] + second_offset
        self.cycle_index[env_ids] += 1
        # The completed hop was the odd (long) phase.  Advancing once returns
        # the route to an even (short) phase without erasing total progress.
        self.route_index[env_ids] += 1
