# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct Franka cube-lift environment."""

import gymnasium as gym


gym.register(
    id="Isaac-Lift-Cube-Franka-Direct-v0",
    entry_point=f"{__name__}.franka_lift_env:LiftCubeFrankaDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_lift_env_cfg:LiftCubeFrankaDirectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Lift-Cube-Franka-Direct-v1",
    entry_point=f"{__name__}.franka_lift_cube_env:LiftCubeFrankaDirectV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_lift_cube_env_cfg:LiftCubeFrankaDirectV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_v1_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Franka-Cube-Direct-v0",
    entry_point=f"{__name__}.franka_cube_env:FrankaCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_cube_env_cfg:FrankaCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
