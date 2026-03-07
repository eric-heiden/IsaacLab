# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_tasks.utils import PresetCfg


@configclass
class LiftCubeFrankaDirectPhysicsCfg(PresetCfg):
    """Physics presets for direct Franka lift."""

    default: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=16 * 1024,
        friction_correlation_distance=0.00625,
    )
    physx: PhysxCfg = default
    newton: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=256,
            nconmax=100,
            ls_iterations=20,
            cone="pyramidal",
            ls_parallel=True,
            integrator="implicitfast",
            impratio=1.0,
            use_mujoco_contacts=False,
        ),
        num_substeps=4,
        debug_mode=False,
    )


@configclass
class LiftCubeFrankaDirectEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5.0
    decimation = 2
    action_space = 8  # 7 arm deltas + 1 continuous gripper target
    observation_space = 41
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        physics=LiftCubeFrankaDirectPhysicsCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=True)

    # assets
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.42, 0.0, 0.055], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=sim_utils.UsdFileCfg(
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
        ),
    )

    # control & reset
    arm_action_scale = (0.45, 0.70, 0.70, 0.60, 0.45, 0.30, 0.30)
    gripper_open_pos = 0.04
    gripper_closed_pos = 0.0
    reset_arm_noise = 0.1
    object_x_range = (0.34, 0.45)
    object_y_range = (-0.2, 0.2)
    target_x_range = (0.4, 0.6)
    target_y_range = (-0.25, 0.25)
    target_z_range = (0.25, 0.45)

    # termination
    min_object_height = -0.05

    # rewards
    reach_reward_scale = 1.0
    pregrasp_reward_scale = 0.35
    top_down_reward_scale = 0.15
    lift_progress_reward_scale = 18.0
    lift_reward_scale = 15.0
    goal_reward_scale = 16.0
    close_stage_pose_reward_scale = 1.0
    close_stage_open_penalty_scale = 1.0
    lift_stage_hold_reward_scale = 0.5
    lift_upward_velocity_reward_scale = 4.0
    gripper_open_reward_scale = 0.6
    gripper_close_reward_scale = 2.5
    grasp_reward_scale = 6.0
    premature_close_penalty_scale = 0.75
    close_stage_bonus_scale = 1.0
    lift_stage_bonus_scale = 3.0
    action_penalty_scale = 1e-4
    lifted_height = 0.08
    pregrasp_height = 0.06
    pregrasp_xy_std = 0.07
    pregrasp_z_std = 0.05
    pregrasp_xy_gate_thresh = 0.04
    pregrasp_gate_sharpness = 60.0
    descend_reach_std = 0.05
    gripper_reward_distance_thresh = 0.08
    gripper_reward_sharpness = 40.0
    close_phase_gate_thresh = 0.2
    close_phase_sharpness = 20.0
    approach_stage_reach_thresh = 0.05
    approach_stage_pose_thresh = 0.15
    approach_stage_enclosure_thresh = 0.45
    approach_stage_open_fraction_thresh = 0.75
    lift_stage_secure_grasp_thresh = 0.1
    grasp_midpoint_std = 0.04
    grasp_balance_std = 0.02
    grasp_finger_height_std = 0.03
    enclosure_between_std = 0.015
    enclosure_span_thresh = 0.05
    enclosure_span_sharpness = 60.0
    lift_progress_std = 0.03
    lift_upward_velocity_std = 0.05