# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import combine_frame_transforms, quat_apply, sample_uniform, subtract_frame_transforms

from .franka_lift_cube_env_cfg import LiftCubeFrankaDirectV1EnvCfg


class LiftCubeFrankaDirectV1Env(DirectRLEnv):
    """Standalone Franka lift-cube direct environment with table-based physics."""

    cfg: LiftCubeFrankaDirectV1EnvCfg

    def __init__(self, cfg: LiftCubeFrankaDirectV1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_joint_ids, _ = self.robot.find_joints("panda_joint[1-7]")
        self._finger_joint_ids, _ = self.robot.find_joints("panda_finger_joint.*")
        self._hand_body_id = self.robot.find_bodies("panda_hand")[0][0]
        self._left_finger_body_id = self.robot.find_bodies("panda_leftfinger")[0][0]
        self._right_finger_body_id = self.robot.find_bodies("panda_rightfinger")[0][0]

        joint_limits = wp.to_torch(self.robot.data.soft_joint_pos_limits)
        self._joint_lower_limits = joint_limits[0, :, 0].to(device=self.device)
        self._joint_upper_limits = joint_limits[0, :, 1].to(device=self.device)

        self._default_joint_pos = wp.to_torch(self.robot.data.default_joint_pos).clone()
        self._default_object_pose = wp.to_torch(self.object.data.default_root_pose).clone()
        self._default_object_vel = wp.to_torch(self.object.data.default_root_vel).clone()
        self._grasp_frame_offset = torch.tensor((0.0, 0.0, 0.1034), device=self.device, dtype=self._default_joint_pos.dtype)
        self._finger_contact_offset = torch.tensor((0.0, 0.0, 0.046), device=self.device, dtype=self._default_joint_pos.dtype)

        self._joint_targets = self._default_joint_pos.clone()
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self._gripper_open_target = 0.04
        self._gripper_closed_target = 0.0
        self._gripper_span = max(self._gripper_open_target - self._gripper_closed_target, 1.0e-6)

        all_env_ids = wp.to_torch(self.robot._ALL_INDICES).to(dtype=torch.long)
        self._sample_goal_positions(all_env_ids)

    def _setup_scene(self):
        self._register_newton_contact_callback()
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        table_cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        table_cfg.func(
            "/World/envs/env_.*/Table",
            table_cfg,
            translation=(0.5, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.70711, 0.70711),
        )

        self.robot = Articulation(self.cfg.robot)
        self.object = RigidObject(self.cfg.object)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def close(self):
        """Cleanup the environment and deregister task-local Newton callbacks."""
        handle = getattr(self, "_newton_model_init_handle", None)
        if handle is not None:
            handle.deregister()
            self._newton_model_init_handle = None
        super().close()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        joint_pos = wp.to_torch(self.robot.data.joint_pos)

        arm_targets = joint_pos[:, self._arm_joint_ids] + self.cfg.action_scale * self._actions[:, : len(self._arm_joint_ids)]
        self._joint_targets[:, self._arm_joint_ids] = arm_targets

        gripper_open = self._actions[:, 7] > 0.0
        finger_target = torch.where(
            gripper_open,
            torch.full_like(self._actions[:, 7], self._gripper_open_target),
            torch.full_like(self._actions[:, 7], self._gripper_closed_target),
        )
        self._joint_targets[:, self._finger_joint_ids] = finger_target.unsqueeze(-1).expand(-1, len(self._finger_joint_ids))

        self._joint_targets = torch.clamp(
            self._joint_targets,
            self._joint_lower_limits.unsqueeze(0),
            self._joint_upper_limits.unsqueeze(0),
        )
        self.robot.set_joint_position_target_index(target=self._joint_targets)

    def _get_observations(self) -> dict:
        joint_pos = wp.to_torch(self.robot.data.joint_pos)
        joint_vel = wp.to_torch(self.robot.data.joint_vel)
        object_pos = self._get_object_pos_in_robot_frame()
        grasp_pos = self._get_grasp_pos()
        object_pos_local = self._get_object_pos()
        left_finger_pos, right_finger_pos = self._get_finger_positions()
        pregrasp_target = object_pos_local + object_pos_local.new_tensor((0.0, 0.0, self.cfg.pregrasp_height))
        pregrasp_delta = grasp_pos - pregrasp_target
        grasp_dir = self._get_grasp_direction()
        finger_open_fraction = self._get_finger_open_fraction(joint_pos).unsqueeze(-1)

        obs = torch.cat(
            (
                joint_pos - self._default_joint_pos,
                joint_vel,
                object_pos,
                self._goal_pos,
                self._previous_actions,
                pregrasp_delta,
                grasp_dir,
                object_pos_local - left_finger_pos,
                object_pos_local - right_finger_pos,
                finger_open_fraction,
            ),
            dim=-1,
        )

        self._previous_actions[:] = self._actions
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        grasp_pos = self._get_grasp_pos()
        grasp_dir = self._get_grasp_direction()
        left_finger_pos, right_finger_pos = self._get_finger_positions()
        object_pos = self._get_object_pos()
        object_pos_world = wp.to_torch(self.object.data.root_pos_w)

        robot_pos = wp.to_torch(self.robot.data.root_pos_w)
        robot_quat = wp.to_torch(self.robot.data.root_quat_w)
        goal_pos_world, _ = combine_frame_transforms(robot_pos, robot_quat, self._goal_pos)

        hand_distance = torch.linalg.norm(grasp_pos - object_pos, dim=-1)
        left_finger_distance = torch.linalg.norm(left_finger_pos - object_pos, dim=-1)
        right_finger_distance = torch.linalg.norm(right_finger_pos - object_pos, dim=-1)
        mean_reach_distance = (hand_distance + left_finger_distance + right_finger_distance) / 3.0
        pregrasp_reward, top_down_reward, top_down_alignment = self._compute_approach_shaping(
            grasp_pos=grasp_pos,
            grasp_dir=grasp_dir,
            object_pos=object_pos,
        )

        object_height = object_pos[:, 2]
        object_start_height = self._default_object_pose[:, 2].to(device=object_height.device)
        lifted_height = torch.clamp(object_height - object_start_height, min=0.0)
        lifted = object_height > self.cfg.lifting_object_min_height
        lifting_reward = lifted.float()
        lift_progress_reward = torch.tanh(lifted_height / self.cfg.lift_progress_std)
        approach_gate = torch.where(
            lifted,
            torch.ones_like(top_down_alignment),
            (1.0 - self.cfg.reaching_object_top_down_mix) + self.cfg.reaching_object_top_down_mix * top_down_alignment,
        )
        reaching_reward = (1.0 - torch.tanh(mean_reach_distance / self.cfg.reaching_object_std)) * approach_gate
        pregrasp_reward = (1.0 - lifting_reward) * pregrasp_reward
        top_down_reward = (1.0 - lifting_reward) * top_down_reward

        joint_pos = wp.to_torch(self.robot.data.joint_pos)
        finger_open_fraction = self._get_finger_open_fraction(joint_pos)
        finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)
        finger_midpoint_dist = torch.linalg.norm(object_pos - finger_midpoint, dim=-1)
        finger_balance = torch.abs(left_finger_distance - right_finger_distance)
        finger_height_error = 0.5 * (
            torch.abs(left_finger_pos[:, 2] - object_pos[:, 2]) + torch.abs(right_finger_pos[:, 2] - object_pos[:, 2])
        )
        near_object_gate = torch.sigmoid(
            self.cfg.gripper_reward_sharpness * (self.cfg.gripper_reward_distance_thresh - hand_distance)
        )
        grasp_pose_reward = near_object_gate * (1.0 - torch.tanh(finger_midpoint_dist / self.cfg.grasp_midpoint_std)) * (
            torch.exp(-finger_balance / self.cfg.grasp_balance_std)
        ) * (1.0 - torch.tanh(finger_height_error / self.cfg.grasp_finger_height_std))
        gripper_close_reward = grasp_pose_reward * (1.0 - finger_open_fraction)
        secure_grasp_gate = torch.clamp(gripper_close_reward, 0.0, 1.0)
        premature_close_penalty = near_object_gate * (1.0 - grasp_pose_reward) * (1.0 - finger_open_fraction)
        in_gripper_gate = (
            (1.0 - finger_open_fraction)
            * (1.0 - torch.tanh(finger_midpoint_dist / self.cfg.in_gripper_midpoint_std))
            * torch.exp(-finger_balance / self.cfg.in_gripper_balance_std)
        )
        in_gripper_gate = torch.clamp(in_gripper_gate, 0.0, 1.0)
        hold_gate = torch.maximum(secure_grasp_gate, in_gripper_gate)
        stalled_grasp_penalty = (
            (1.0 - finger_open_fraction)
            * reaching_reward
            * (1.0 - hold_gate)
            * (lifted_height < self.cfg.stalled_grasp_height).float()
        )

        lift_progress_reward = hold_gate * lift_progress_reward
        lifting_reward = hold_gate * lifting_reward
        upward_velocity = torch.clamp(wp.to_torch(self.object.data.root_lin_vel_w)[:, 2], min=0.0)
        lift_upward_velocity_reward = secure_grasp_gate * torch.tanh(
            upward_velocity / self.cfg.lift_upward_velocity_std
        )

        object_goal_distance = torch.linalg.norm(object_pos_world - goal_pos_world, dim=-1)
        lifted_float = lifted.float()
        goal_tracking_reward = lifted_float * hold_gate * (
            1.0 - torch.tanh(object_goal_distance / self.cfg.object_goal_tracking_std)
        )
        goal_tracking_fine_reward = lifted_float * hold_gate * (
            1.0 - torch.tanh(object_goal_distance / self.cfg.object_goal_tracking_fine_std)
        )

        progress = min(float(self.common_step_counter) / float(self.cfg.penalty_curriculum_steps), 1.0)
        action_penalty_scale = self.cfg.action_penalty_scale + progress * (
            self.cfg.action_penalty_max - self.cfg.action_penalty_scale
        )
        joint_vel_penalty_scale = self.cfg.joint_vel_penalty_scale + progress * (
            self.cfg.joint_vel_penalty_max - self.cfg.joint_vel_penalty_scale
        )
        action_penalty = torch.sum(self._actions.square(), dim=-1)
        joint_vel = wp.to_torch(self.robot.data.joint_vel)
        joint_vel_penalty = torch.sum(joint_vel.square(), dim=-1)

        rewards = (
            self.cfg.reaching_object_scale * reaching_reward
            + self.cfg.pregrasp_reward_scale * pregrasp_reward
            + self.cfg.top_down_reward_scale * top_down_reward
            + self.cfg.grasp_pose_reward_scale * grasp_pose_reward
            + self.cfg.gripper_close_reward_scale * gripper_close_reward
            + self.cfg.lift_progress_reward_scale * lift_progress_reward
            + self.cfg.lift_upward_velocity_reward_scale * lift_upward_velocity_reward
            + self.cfg.lifting_object_scale * lifting_reward
            + self.cfg.object_goal_tracking_scale * goal_tracking_reward
            + self.cfg.object_goal_tracking_fine_scale * goal_tracking_fine_reward
            - self.cfg.premature_close_penalty_scale * premature_close_penalty
            - self.cfg.stalled_grasp_penalty_scale * stalled_grasp_penalty
            - action_penalty_scale * action_penalty
            - joint_vel_penalty_scale * joint_vel_penalty
        )

        self.extras["log"] = {
            "reaching_reward": reaching_reward.mean(),
            "pregrasp_reward": pregrasp_reward.mean(),
            "top_down_reward": top_down_reward.mean(),
            "top_down_alignment": top_down_alignment.mean(),
            "grasp_pose_reward": grasp_pose_reward.mean(),
            "gripper_close_reward": gripper_close_reward.mean(),
            "secure_grasp_gate": secure_grasp_gate.mean(),
            "in_gripper_gate": in_gripper_gate.mean(),
            "hold_gate": hold_gate.mean(),
            "lift_progress_reward": lift_progress_reward.mean(),
            "lift_upward_velocity_reward": lift_upward_velocity_reward.mean(),
            "lifting_reward": lifting_reward.mean(),
            "goal_tracking_reward": goal_tracking_reward.mean(),
            "goal_tracking_fine_reward": goal_tracking_fine_reward.mean(),
            "premature_close_penalty": premature_close_penalty.mean(),
            "stalled_grasp_penalty": stalled_grasp_penalty.mean(),
            "action_penalty": action_penalty.mean(),
            "joint_vel_penalty": joint_vel_penalty.mean(),
            "action_penalty_scale": torch.tensor(action_penalty_scale, device=self.device),
            "joint_vel_penalty_scale": torch.tensor(joint_vel_penalty_scale, device=self.device),
        }
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_height = wp.to_torch(self.object.data.root_pos_w)[:, 2]
        terminated = object_height < self.cfg.object_drop_height
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_tensor = wp.to_torch(self.robot._ALL_INDICES).to(dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids_tensor)

        default_root_pose = wp.to_torch(self.robot.data.default_root_pose)[env_ids_tensor].clone()
        default_root_vel = wp.to_torch(self.robot.data.default_root_vel)[env_ids_tensor].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids_tensor]
        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids_tensor)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids_tensor)

        joint_pos = self._default_joint_pos[env_ids_tensor].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._joint_targets[env_ids_tensor] = joint_pos
        self._actions[env_ids_tensor] = 0.0
        self._previous_actions[env_ids_tensor] = 0.0

        self.robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids_tensor)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids_tensor)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids_tensor)

        object_pose = self._default_object_pose[env_ids_tensor].clone()
        object_vel = self._default_object_vel[env_ids_tensor].clone()
        object_pose[:, 0] += sample_uniform(
            self.cfg.object_reset_pos_x_range[0],
            self.cfg.object_reset_pos_x_range[1],
            (len(env_ids_tensor),),
            self.device,
        )
        object_pose[:, 1] += sample_uniform(
            self.cfg.object_reset_pos_y_range[0],
            self.cfg.object_reset_pos_y_range[1],
            (len(env_ids_tensor),),
            self.device,
        )
        object_pose[:, :3] += self.scene.env_origins[env_ids_tensor]
        object_vel.zero_()

        self.object.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids_tensor)
        self.object.write_root_velocity_to_sim_index(root_velocity=object_vel, env_ids=env_ids_tensor)

        self._sample_goal_positions(env_ids_tensor)

    def _get_object_pos_in_robot_frame(self) -> torch.Tensor:
        robot_pos = wp.to_torch(self.robot.data.root_pos_w)
        robot_quat = wp.to_torch(self.robot.data.root_quat_w)
        object_pos = wp.to_torch(self.object.data.root_pos_w)

        object_pos_local, _ = subtract_frame_transforms(robot_pos, robot_quat, object_pos)
        return object_pos_local

    def _get_object_pos(self) -> torch.Tensor:
        """Return the object root position in environment-local coordinates."""
        return wp.to_torch(self.object.data.root_pos_w) - self.scene.env_origins

    def _get_grasp_pos(self) -> torch.Tensor:
        """Return the Franka grasp frame in environment-local coordinates."""
        hand_pos = wp.to_torch(self.robot.data.body_pos_w)[:, self._hand_body_id]
        hand_quat = wp.to_torch(self.robot.data.body_quat_w)[:, self._hand_body_id]
        grasp_offset = quat_apply(hand_quat, self._grasp_frame_offset.unsqueeze(0).expand(hand_quat.shape[0], -1))
        return hand_pos + grasp_offset - self.scene.env_origins

    def _get_grasp_direction(self) -> torch.Tensor:
        """Return the world-frame approach direction of the Franka grasp frame."""
        hand_quat = wp.to_torch(self.robot.data.body_quat_w)[:, self._hand_body_id]
        grasp_offset = quat_apply(hand_quat, self._grasp_frame_offset.unsqueeze(0).expand(hand_quat.shape[0], -1))
        return grasp_offset / torch.clamp(torch.linalg.norm(grasp_offset, dim=-1, keepdim=True), min=1.0e-6)

    def _get_finger_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return left and right fingertip positions in environment-local coordinates."""
        body_pos = wp.to_torch(self.robot.data.body_pos_w)
        body_quat = wp.to_torch(self.robot.data.body_quat_w)
        left_finger_pos = body_pos[:, self._left_finger_body_id] + quat_apply(
            body_quat[:, self._left_finger_body_id],
            self._finger_contact_offset.unsqueeze(0).expand(body_quat.shape[0], -1),
        )
        right_finger_pos = body_pos[:, self._right_finger_body_id] + quat_apply(
            body_quat[:, self._right_finger_body_id],
            self._finger_contact_offset.unsqueeze(0).expand(body_quat.shape[0], -1),
        )
        left_finger_pos -= self.scene.env_origins
        right_finger_pos -= self.scene.env_origins
        return left_finger_pos, right_finger_pos

    def _get_finger_open_fraction(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Return the mean normalized Franka finger opening in [0, 1]."""
        finger_joint_pos = joint_pos[:, self._finger_joint_ids].mean(dim=-1)
        return torch.clamp(
            (finger_joint_pos - self._gripper_closed_target) / self._gripper_span,
            min=0.0,
            max=1.0,
        )

    def _compute_approach_shaping(
        self, grasp_pos: torch.Tensor, grasp_dir: torch.Tensor, object_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reward approaching the cube from above with a vertical gripper."""
        pregrasp_target = object_pos + object_pos.new_tensor((0.0, 0.0, self.cfg.pregrasp_height))
        pregrasp_delta = grasp_pos - pregrasp_target
        xy_dist = torch.linalg.norm(pregrasp_delta[:, :2], dim=-1)
        z_err = torch.abs(pregrasp_delta[:, 2])
        reach_dist = torch.linalg.norm(grasp_pos - object_pos, dim=-1)

        pregrasp_reward = (1.0 - torch.tanh(xy_dist / self.cfg.pregrasp_xy_std)) * (
            1.0 - torch.tanh(z_err / self.cfg.pregrasp_z_std)
        )
        descend_gate = torch.sigmoid(
            self.cfg.pregrasp_gate_sharpness * (self.cfg.pregrasp_xy_gate_thresh - xy_dist)
        )
        descend_reward = descend_gate * (1.0 - torch.tanh(reach_dist / self.cfg.descend_reach_std))
        pregrasp_reward = (1.0 - descend_gate) * pregrasp_reward + descend_gate * descend_reward

        top_down_alignment = torch.clamp(-grasp_dir[:, 2], 0.0, 1.0)
        top_down_reward = descend_gate * top_down_alignment.square()
        return pregrasp_reward, top_down_reward, top_down_alignment

    def _sample_goal_positions(self, env_ids: torch.Tensor) -> None:
        self._goal_pos[env_ids, 0] = sample_uniform(
            self.cfg.goal_pos_x_range[0],
            self.cfg.goal_pos_x_range[1],
            (len(env_ids),),
            self.device,
        )
        self._goal_pos[env_ids, 1] = sample_uniform(
            self.cfg.goal_pos_y_range[0],
            self.cfg.goal_pos_y_range[1],
            (len(env_ids),),
            self.device,
        )
        self._goal_pos[env_ids, 2] = sample_uniform(
            self.cfg.goal_pos_z_range[0],
            self.cfg.goal_pos_z_range[1],
            (len(env_ids),),
            self.device,
        )

    def _register_newton_contact_callback(self) -> None:
        """Register a Newton-only callback to harden default shape contacts."""
        self._newton_model_init_handle = None
        physics_mgr_cls = self.sim.physics_manager
        if physics_mgr_cls.__name__ != "NewtonManager":
            return

        self._newton_model_init_handle = physics_mgr_cls.register_callback(
            self._apply_newton_contact_tuning,
            PhysicsEvent.MODEL_INIT,
            order=100,
            name=f"{self.__class__.__name__}_newton_contact_tuning",
        )

    def _apply_newton_contact_tuning(self, _event) -> None:
        """Apply task-local Newton contact stiffness/damping before model finalize."""
        if not self.cfg.newton_contact.enabled:
            return

        physics_mgr_cls = self.sim.physics_manager
        builder = getattr(physics_mgr_cls, "_builder", None)
        if builder is None:
            return

        shape_cfg = builder.default_shape_cfg
        if self.cfg.newton_contact.ke is not None:
            shape_cfg.ke = self.cfg.newton_contact.ke
        if self.cfg.newton_contact.kd is not None:
            shape_cfg.kd = self.cfg.newton_contact.kd
        if self.cfg.newton_contact.kf is not None:
            shape_cfg.kf = self.cfg.newton_contact.kf
