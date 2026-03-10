# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class FrankaCubeNewtonContactCfg:
    """Task-local Newton contact tuning for the template Franka cube env."""

    enabled: bool = True
    ke: float | None = 250_000.0
    kd: float | None = 5_000.0
    kf: float | None = 10_000.0
    mu: float | None = 1.25
    contact_margin: float | None = 0.005
    geom_solimp: tuple[float, float, float, float, float] | None = (0.985, 0.999, 0.0015, 0.5, 3.0)
    solimp_friction: tuple[float, float, float, float, float] | None = (0.985, 0.999, 0.0015, 0.5, 3.0)
    solref_friction: tuple[float, float] | None = (0.005, 4.0)
    support_pair_solimp: tuple[float, float, float, float, float] | None = (0.992, 0.999, 0.0015, 0.5, 3.0)
    support_pair_solref: tuple[float, float] | None = (0.006, 5.0)
    support_pair_friction: tuple[float, float, float, float, float] | None = (1.5, 1.5, 0.02, 0.002, 0.002)
    support_pair_margin: float | None = 0.003
    support_pair_condim: int | None = 6
    grasp_pair_solimp: tuple[float, float, float, float, float] | None = (0.995, 0.9995, 0.001, 0.5, 3.0)
    grasp_pair_solref: tuple[float, float] | None = (0.003, 5.0)
    grasp_pair_friction: tuple[float, float, float, float, float] | None = (2.0, 2.0, 0.02, 0.002, 0.002)
    grasp_pair_margin: float | None = 0.004
    grasp_pair_condim: int | None = 6


@configclass
class FrankaCubeEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 3.0
    # - spaces definition
    action_space = 8
    observation_space = 41
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
        ccd_iterations=80,
        ls_parallel=True,
        use_mujoco_contacts=False,
    )

    newton_cfg = NewtonCfg(solver_cfg=solver_cfg, num_substeps=5, debug_mode=False)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics=newton_cfg,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # robot(s)
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot_cfg.spawn.rigid_props.max_depenetration_velocity = 8.0
    robot_cfg.actuators["panda_shoulder"].velocity_limit_sim = 2.175
    robot_cfg.actuators["panda_shoulder"].stiffness = 400.0
    robot_cfg.actuators["panda_shoulder"].damping = 80.0
    robot_cfg.actuators["panda_shoulder"].armature = 0.3
    robot_cfg.actuators["panda_forearm"].velocity_limit_sim = 2.61
    robot_cfg.actuators["panda_forearm"].stiffness = 400.0
    robot_cfg.actuators["panda_forearm"].damping = 80.0
    robot_cfg.actuators["panda_forearm"].armature = 0.11
    robot_cfg.actuators["panda_hand"].effort_limit_sim = 100.0
    robot_cfg.actuators["panda_hand"].velocity_limit_sim = 0.04
    robot_cfg.actuators["panda_hand"].stiffness = 7_500.0
    robot_cfg.actuators["panda_hand"].damping = 220.0
    robot_cfg.actuators["panda_hand"].friction = 0.2
    robot_cfg.actuators["panda_hand"].armature = 0.15

    # cube
    cube_size = 0.05
    cube_density = 400.0
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.42, 0.0, 0.5 * cube_size], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=sim_utils.CuboidCfg(
            size=(cube_size, cube_size, cube_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=2,
                linear_damping=0.1,
                angular_damping=0.2,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=8.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=cube_density),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # Action parameters
    action_scale = (0.45, 1.60, 0.70, 2.70, 0.45, 0.80, 0.30)
    newton_contact: FrankaCubeNewtonContactCfg = FrankaCubeNewtonContactCfg()

    # Goal generation ranges (relative to robot base)
    goal_pos_x_range = (0.4, 0.6)
    goal_pos_y_range = (-0.25, 0.25)
    goal_pos_z_range = (0.25, 0.45)

    # Object reset ranges (relative to default position)
    object_reset_pos_x_range = (-0.08, 0.03)
    object_reset_pos_y_range = (-0.2, 0.2)

    # Control & reset shaping
    reset_arm_noise = 0.1

    # Reward scales
    reaching_object_scale = 1.0
    reaching_object_std = 0.1
    pregrasp_reward_scale = 0.35
    top_down_reward_scale = 0.15
    gripper_open_reward_scale = 0.6
    gripper_close_reward_scale = 2.5
    grasp_reward_scale = 6.0
    premature_close_penalty_scale = 0.75
    close_stage_pose_reward_scale = 1.0
    close_stage_open_penalty_scale = 1.0
    lift_stage_hold_reward_scale = 0.5
    close_stage_bonus_scale = 1.0
    lift_stage_bonus_scale = 3.0

    lifting_object_scale = 15.0
    lifting_object_min_height = 0.15
    lift_progress_reward_scale = 18.0
    lift_upward_velocity_reward_scale = 4.0

    object_goal_tracking_scale = 16.0
    object_goal_tracking_std = 0.3

    object_goal_tracking_fine_scale = 5.0
    object_goal_tracking_fine_std = 0.05

    action_penalty_scale = 1e-4
    action_penalty_max = 1e-2
    joint_vel_penalty_scale = 1e-4
    joint_vel_penalty_max = 1e-2
    penalty_curriculum_steps = 50_000_000

    # Grasp/lift shaping thresholds
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
    in_gripper_midpoint_std = 0.05
    in_gripper_balance_std = 0.04
    enclosure_between_std = 0.015
    enclosure_span_thresh = 0.05
    enclosure_span_sharpness = 60.0
    stalled_grasp_height = 0.02
    stalled_grasp_penalty_scale = 1.5
    lift_progress_std = 0.03
    lift_upward_velocity_std = 0.05

    # Termination conditions
    object_drop_height = -0.05  # Terminate if cube falls below this height