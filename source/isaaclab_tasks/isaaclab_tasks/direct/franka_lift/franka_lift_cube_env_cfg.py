# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_tasks.utils import PresetCfg


@configclass
class LiftCubeFrankaDirectV1NewtonContactCfg:
    """Task-local Newton contact tuning for the direct Franka lift-cube v1 task."""

    enabled: bool = True
    ke: float | None = 15_000.0
    kd: float | None = 400.0
    kf: float | None = 3_000.0


@configclass
class LiftCubeFrankaDirectV1PhysicsCfg(PresetCfg):
    """Physics presets for the direct Franka lift-cube v1 task."""

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
class LiftCubeFrankaDirectV1EnvCfg(DirectRLEnvCfg):
    """Configuration for the standalone Franka lift-cube direct v1 task."""

    # env
    episode_length_s = 5.0
    decimation = 2
    action_space = 8
    observation_space = 45
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        physics=LiftCubeFrankaDirectV1PhysicsCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # assets
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.055], rot=[0.0, 0.0, 0.0, 1.0]),
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

    # control
    action_scale = 0.5
    newton_contact: LiftCubeFrankaDirectV1NewtonContactCfg = LiftCubeFrankaDirectV1NewtonContactCfg()

    # goal generation ranges in the robot base frame
    goal_pos_x_range = (0.44, 0.56)
    goal_pos_y_range = (-0.16, 0.16)
    goal_pos_z_range = (0.20, 0.32)

    # object reset ranges relative to the default object pose
    object_reset_pos_x_range = (-0.1, 0.1)
    object_reset_pos_y_range = (-0.25, 0.25)

    # rewards
    reaching_object_scale = 1.0
    reaching_object_std = 0.1
    reaching_object_top_down_mix = 0.75
    pregrasp_reward_scale = 0.75
    top_down_reward_scale = 0.75
    pregrasp_height = 0.06
    pregrasp_xy_std = 0.07
    pregrasp_z_std = 0.05
    pregrasp_xy_gate_thresh = 0.04
    pregrasp_gate_sharpness = 60.0
    descend_reach_std = 0.05

    lifting_object_scale = 15.0
    lifting_object_min_height = 0.15
    lift_progress_reward_scale = 12.0
    lift_progress_std = 0.03
    lift_upward_velocity_reward_scale = 4.0
    lift_upward_velocity_std = 0.05
    stalled_grasp_penalty_scale = 1.5
    stalled_grasp_height = 0.02
    grasp_pose_reward_scale = 4.0
    gripper_close_reward_scale = 3.0
    premature_close_penalty_scale = 1.0
    gripper_reward_distance_thresh = 0.08
    gripper_reward_sharpness = 40.0
    grasp_midpoint_std = 0.04
    grasp_balance_std = 0.03
    grasp_finger_height_std = 0.03
    in_gripper_midpoint_std = 0.05
    in_gripper_balance_std = 0.04

    object_goal_tracking_scale = 16.0
    object_goal_tracking_std = 0.3

    object_goal_tracking_fine_scale = 5.0
    object_goal_tracking_fine_std = 0.05

    action_penalty_scale = 1e-4
    action_penalty_max = 1e-2
    joint_vel_penalty_scale = 1e-4
    joint_vel_penalty_max = 1e-2
    penalty_curriculum_steps = 50_000_000

    # termination
    object_drop_height = -0.05
