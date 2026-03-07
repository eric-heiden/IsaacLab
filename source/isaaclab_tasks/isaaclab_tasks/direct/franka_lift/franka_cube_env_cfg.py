# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class FrankaCubeEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0 # Tune this
    # - spaces definition
    action_space = 8
    observation_space = 32
    state_space = 0

    solver_cfg = MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        njmax=2000,
        nconmax=1000,
        impratio=1000.0,
        cone="elliptic",
        update_data_interval=2,
        iterations=20,
        ls_iterations=100,
        ls_parallel=True
    )

    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=2,
        debug_mode=False,
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=newton_cfg)

    # robot(s)
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.055], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True)

    # Action parameters
    action_scale = 0.5  # Scale factor for joint position actions

    # Goal generation ranges (relative to robot base)
    goal_pos_x_range = (0.4, 0.6)
    goal_pos_y_range = (-0.25, 0.25)
    goal_pos_z_range = (0.25, 0.5)

    # Object reset ranges (relative to default position)
    object_reset_pos_x_range = (-0.1, 0.1)
    object_reset_pos_y_range = (-0.25, 0.25)

    # Reward scales
    reaching_object_scale = 1.0
    reaching_object_std = 0.1

    lifting_object_scale = 15.0
    lifting_object_min_height = 0.15

    object_goal_tracking_scale = 16.0
    object_goal_tracking_std = 0.3

    object_goal_tracking_fine_scale = 5.0
    object_goal_tracking_fine_std = 0.05

    action_penalty_scale = 1e-4
    action_penalty_max = 1e-2
    joint_vel_penalty_scale = 1e-4
    joint_vel_penalty_max = 1e-2
    penalty_curriculum_steps = 50_000_000

    # Termination conditions
    object_drop_height = -0.05  # Terminate if cube falls below this height