# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import combine_frame_transforms, sample_uniform, subtract_frame_transforms

from .franka_cube_env_cfg import FrankaCubeEnvCfg


class FrankaCubeEnv(DirectRLEnv):
    cfg: FrankaCubeEnvCfg

    def __init__(self, cfg: FrankaCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint limits for action clamping
        joint_pos_limits = wp.to_torch(self.robot.data.soft_joint_pos_limits)[0]
        self.robot_dof_lower_limits = joint_pos_limits[:, 0].to(self.device)
        self.robot_dof_upper_limits = joint_pos_limits[:, 1].to(self.device)

        # Store default joint positions for relative observations
        self.robot_default_joint_pos = wp.to_torch(self.robot.data.default_joint_pos).clone()

        # Buffers for actions and targets
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self.robot.num_joints), dtype=torch.float, device=self.device
        )
        self.previous_actions = torch.zeros(
            (self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device
        )

        # Goal position buffer (in robot's local frame)
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Get body indices for reward computation
        self.ee_body_idx = self.robot.body_names.index("panda_hand")
        self.lf_body_idx = self.robot.body_names.index("panda_leftfinger")
        self.rf_body_idx = self.robot.body_names.index("panda_rightfinger")

        # Precompute gripper open/close positions
        self.finger_joint_indices = [
            self.robot.joint_names.index(name) 
            for name in self.robot.joint_names 
            if "panda_finger_joint" in name
        ]
        self.gripper_open_pos = 0.04
        self.gripper_close_pos = 0.0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cube = RigidObject(self.cfg.cube)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cube"] = self.cube
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        current_joint_pos = wp.to_torch(self.robot.data.joint_pos)

        arm_targets = current_joint_pos[:, :7] + self.cfg.action_scale * self.actions[:, :7]

        gripper_open = self.actions[:, 7] > 0.0
        finger_val = torch.where(gripper_open, self.gripper_open_pos, self.gripper_close_pos)
        finger_targets = finger_val.unsqueeze(-1).expand(-1, 2)

        targets = torch.cat([arm_targets, finger_targets], dim=-1)

        self.robot_dof_targets = torch.clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

        self.robot.set_joint_position_target(self.robot_dof_targets)

    def _get_observations(self) -> dict:

        # Joint positions relative to default
        joint_pos = wp.to_torch(self.robot.data.joint_pos)
        joint_pos_rel = joint_pos - self.robot_default_joint_pos

        # Joint velocities
        joint_vel = wp.to_torch(self.robot.data.joint_vel)

        # Object position in robot's local frame
        object_pos_local = self._get_object_pos_in_robot_frame()

        # Goal position (already in robot's local frame)
        goal_pos_local = self.goal_pos

        # Last actions
        last_actions = self.previous_actions

        # Concatenate all observations
        obs = torch.cat(
            (
                joint_pos_rel,       # 9
                joint_vel,           # 9
                object_pos_local,    # 3
                goal_pos_local,      # 3
                last_actions,        # 8
            ),
            dim=-1,
        )

        # Store current actions for next observation
        self.previous_actions = self.actions.clone()

        return {"policy": obs}

    def _get_object_pos_in_robot_frame(self) -> torch.Tensor:
        """Get object position relative to robot base frame."""
        robot_pos = wp.to_torch(self.robot.data.root_pos_w)
        robot_quat = wp.to_torch(self.robot.data.root_quat_w)
        object_pos = wp.to_torch(self.cube.data.root_pos_w)

        object_pos_local, _ = subtract_frame_transforms(robot_pos, robot_quat, object_pos)
        return object_pos_local

    def _get_rewards(self) -> torch.Tensor:
        body_pos = wp.to_torch(self.robot.data.body_pos_w)
        ee_pos = body_pos[:, self.ee_body_idx]
        lf_pos = body_pos[:, self.lf_body_idx]
        rf_pos = body_pos[:, self.rf_body_idx]
        object_pos = wp.to_torch(self.cube.data.root_pos_w)
        robot_pos = wp.to_torch(self.robot.data.root_pos_w)
        robot_quat = wp.to_torch(self.robot.data.root_quat_w)

        goal_pos_world, _ = combine_frame_transforms(robot_pos, robot_quat, self.goal_pos)

        # 1. Three-point reaching reward: EEF + left fingertip + right fingertip
        d_ee = torch.norm(ee_pos - object_pos, dim=-1)
        d_lf = torch.norm(lf_pos - object_pos, dim=-1)
        d_rf = torch.norm(rf_pos - object_pos, dim=-1)
        reaching_reward = 1.0 - torch.tanh((d_ee + d_lf + d_rf) / 3.0 / self.cfg.reaching_object_std)

        # 2. Lifting reward: binary, gated on object height
        object_height = object_pos[:, 2]
        lifting_reward = torch.where(
            object_height > self.cfg.lifting_object_min_height,
            torch.ones_like(object_height),
            torch.zeros_like(object_height),
        )

        # 3. Object-goal tracking, coarse (gated on height)
        object_to_goal_dist = torch.norm(object_pos - goal_pos_world, dim=-1)
        is_lifted = (object_height > self.cfg.lifting_object_min_height).float()
        goal_tracking_reward = (
            is_lifted * (1.0 - torch.tanh(object_to_goal_dist / self.cfg.object_goal_tracking_std))
        )

        # 4. Object-goal tracking, fine (gated on height)
        goal_tracking_fine_reward = (
            is_lifted * (1.0 - torch.tanh(object_to_goal_dist / self.cfg.object_goal_tracking_fine_std))
        )

        # 5. Penalties with curriculum ramp
        progress = min(self.common_step_counter / self.cfg.penalty_curriculum_steps, 1.0)
        action_scale = self.cfg.action_penalty_scale + progress * (
            self.cfg.action_penalty_max - self.cfg.action_penalty_scale
        )
        vel_scale = self.cfg.joint_vel_penalty_scale + progress * (
            self.cfg.joint_vel_penalty_max - self.cfg.joint_vel_penalty_scale
        )
        action_penalty = torch.sum(self.actions**2, dim=-1)
        joint_vel = wp.to_torch(self.robot.data.joint_vel)
        joint_vel_penalty = torch.sum(joint_vel**2, dim=-1)

        total_reward = (
            self.cfg.reaching_object_scale * reaching_reward
            + self.cfg.lifting_object_scale * lifting_reward
            + self.cfg.object_goal_tracking_scale * goal_tracking_reward
            + self.cfg.object_goal_tracking_fine_scale * goal_tracking_fine_reward
            - action_scale * action_penalty
            - vel_scale * joint_vel_penalty
        )

        self.extras["log"] = {
            "reaching_reward": reaching_reward.mean(),
            "lifting_reward": lifting_reward.mean(),
            "goal_tracking_reward": goal_tracking_reward.mean(),
            "goal_tracking_fine_reward": goal_tracking_fine_reward.mean(),
            "action_penalty": action_penalty.mean(),
            "joint_vel_penalty": joint_vel_penalty.mean(),
            "penalty_scale": torch.tensor(action_scale),
        }

        return total_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Get object height
        object_height = wp.to_torch(self.cube.data.root_pos_w)[:, 2]

        # Termination: object fell off table
        terminated = object_height < self.cfg.object_drop_height

        # Truncation: episode timeout
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = wp.to_torch(self.robot._ALL_INDICES)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        # ---- Reset Robot ----
        default_root_pose = wp.to_torch(self.robot.data.default_root_pose)[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_pose, env_ids)

        joint_pos = wp.to_torch(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        self.robot_dof_targets[env_ids] = joint_pos

        # ---- Reset Object (Cube) ----
        object_default_pose = wp.to_torch(self.cube.data.default_root_pose)[env_ids].clone()
        object_default_vel = wp.to_torch(self.cube.data.default_root_vel)[env_ids].clone()

        # Add position noise
        pos_noise_x = sample_uniform(
            self.cfg.object_reset_pos_x_range[0],
            self.cfg.object_reset_pos_x_range[1],
            (num_resets,),
            self.device,
        )
        pos_noise_y = sample_uniform(
            self.cfg.object_reset_pos_y_range[0],
            self.cfg.object_reset_pos_y_range[1],
            (num_resets,),
            self.device,
        )

        object_default_pose[:, 0] += pos_noise_x
        object_default_pose[:, 1] += pos_noise_y
        # Add env origins
        object_default_pose[:, :3] += self.scene.env_origins[env_ids]

        object_default_vel.zero_()

        self.cube.write_root_pose_to_sim(object_default_pose, env_ids)
        self.cube.write_root_velocity_to_sim(object_default_vel, env_ids)

        # ---- Reset Goal ----
        self._reset_goal(env_ids)

        # ---- Reset Buffers ----
        self.previous_actions[env_ids] = 0.0

    def _reset_goal(self, env_ids: Sequence[int]):
        """Generate new random goal positions."""
        num_resets = len(env_ids)

        # Sample random goal positions in robot's local frame
        goal_x = sample_uniform(
            self.cfg.goal_pos_x_range[0],
            self.cfg.goal_pos_x_range[1],
            (num_resets,),
            self.device,
        )
        goal_y = sample_uniform(
            self.cfg.goal_pos_y_range[0],
            self.cfg.goal_pos_y_range[1],
            (num_resets,),
            self.device,
        )
        goal_z = sample_uniform(
            self.cfg.goal_pos_z_range[0],
            self.cfg.goal_pos_z_range[1],
            (num_resets,),
            self.device,
        )

        self.goal_pos[env_ids, 0] = goal_x
        self.goal_pos[env_ids, 1] = goal_y
        self.goal_pos[env_ids, 2] = goal_z