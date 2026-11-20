# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadhopper direct RL environment with a passive spring leg."""

import gymnasium as gym

from . import agents


gym.register(
    id="quadhopper-ppo-reference-sysid",
    entry_point=f"{__name__}.quadhopper_env:QuadhopperEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadhopper_env:QuadhopperEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadhopperPPORunnerCfg",
    },
)
