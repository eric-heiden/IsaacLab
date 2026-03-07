# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp
from torch.nn import functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply, sample_uniform

from .franka_lift_env_cfg import LiftCubeFrankaDirectEnvCfg


class LiftCubeFrankaDirectEnv(DirectRLEnv):
    """Direct Franka cube-lift task with optional Newton backend."""

    cfg: LiftCubeFrankaDirectEnvCfg

    def __init__(self, cfg: LiftCubeFrankaDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_joint_ids, _ = self.robot.find_joints("panda_joint[1-7]")
        self._finger_joint_ids, _ = self.robot.find_joints("panda_finger_joint.*")
        self._hand_body_id = self.robot.find_bodies("panda_hand")[0][0]
        self._left_finger_body_id = self.robot.find_bodies("panda_leftfinger")[0][0]
        self._right_finger_body_id = self.robot.find_bodies("panda_rightfinger")[0][0]

        self._joint_pos = wp.to_torch(self.robot.data.joint_pos)
        self._joint_vel = wp.to_torch(self.robot.data.joint_vel)

        joint_limits = wp.to_torch(self.robot.data.soft_joint_pos_limits)
        self._joint_lower_limits = joint_limits[0, :, 0].to(device=self.device)
        self._joint_upper_limits = joint_limits[0, :, 1].to(device=self.device)

        self._default_joint_pos = wp.to_torch(self.robot.data.default_joint_pos).clone()
        self._default_object_pose = wp.to_torch(self.object.data.default_root_pose).clone()
        self._default_object_vel = wp.to_torch(self.object.data.default_root_vel).clone()
        self._arm_action_scale = self._resolve_arm_action_scale()
        self._gripper_open_target, self._gripper_closed_target = self._resolve_gripper_targets()
        self._grasp_frame_offset = torch.tensor((0.0, 0.0, 0.1034), device=self.device, dtype=self._default_joint_pos.dtype)
        self._finger_contact_offset = torch.tensor((0.0, 0.0, 0.046), device=self.device, dtype=self._default_joint_pos.dtype)

        self._joint_targets = self._default_joint_pos.clone()
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._reward_stage = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._close_ready = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._sample_targets(wp.to_torch(self.robot._ALL_INDICES).to(torch.long))

    def _setup_scene(self):
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Match the manager-based Franka lift task: actions command absolute offsets
        # around the default arm pose instead of integrating targets across steps.
        arm_targets = self._default_joint_pos[:, self._arm_joint_ids] + self._arm_action_scale.unsqueeze(0) * self._actions[
            :, : len(self._arm_joint_ids)
        ]
        self._joint_targets[:, self._arm_joint_ids] = torch.clamp(
            arm_targets,
            self._joint_lower_limits[self._arm_joint_ids],
            self._joint_upper_limits[self._arm_joint_ids],
        )

        # Keep the gripper open during the explicit approach phase until the
        # previous step confirmed that the cube is actually enclosed.
        allow_close_control = (self._reward_stage > 0) | self._close_ready
        self._actions[:, 7] = torch.where(allow_close_control, self._actions[:, 7], torch.ones_like(self._actions[:, 7]))
        finger_cmd = 0.5 * (self._actions[:, 7] + 1.0)
        finger_target = self._gripper_closed_target + finger_cmd * (self._gripper_open_target - self._gripper_closed_target)
        self._joint_targets[:, self._finger_joint_ids] = finger_target.unsqueeze(-1).repeat(1, len(self._finger_joint_ids))

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target_index(target=self._joint_targets)

    def _get_observations(self) -> dict:
        hand_pos = self._get_grasp_pos()
        object_pos = self._get_object_pos()
        left_finger_pos, right_finger_pos = self._get_finger_positions()

        obs = torch.cat(
            [
                self._joint_pos - self._default_joint_pos,
                self._joint_vel,
                object_pos - hand_pos,
                object_pos - left_finger_pos,
                object_pos - right_finger_pos,
                self._target_pos - object_pos,
                self._actions,
                F.one_hot(self._reward_stage, num_classes=3).to(dtype=self._joint_pos.dtype),
            ],
            dim=-1,
        )
        obs = torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        obs = torch.clamp(obs, -100.0, 100.0)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        hand_pos = self._get_grasp_pos()
        hand_quat = wp.to_torch(self.robot.data.body_quat_w)[:, self._hand_body_id]
        object_pos = self._get_object_pos()
        left_finger_pos, right_finger_pos = self._get_finger_positions()

        reach_dist = torch.linalg.norm(object_pos - hand_pos, dim=-1)
        goal_dist = torch.linalg.norm(self._target_pos - object_pos, dim=-1)
        reach_reward = torch.exp(-4.0 * reach_dist)

        pregrasp_reward, top_down_reward, _ = self._compute_approach_shaping(
            grasp_pos=hand_pos,
            hand_quat=hand_quat,
            object_pos=object_pos,
            reach_dist=reach_dist,
        )
        lifted = (object_pos[:, 2] > self.cfg.lifted_height).float()
        finger_joint_pos = self._joint_pos[:, self._finger_joint_ids].mean(dim=-1)
        (
            gripper_open_reward,
            gripper_close_reward,
            grasp_reward,
            premature_close_penalty,
            secure_grasp_gate,
            grasp_pose_gate,
            finger_open_fraction,
        ) = self._compute_gripper_shaping(
            object_pos=object_pos,
            left_finger_pos=left_finger_pos,
            right_finger_pos=right_finger_pos,
            finger_joint_pos=finger_joint_pos,
            reach_dist=reach_dist,
            lifted=lifted,
        )
        enclosure_gate = self._compute_enclosure_gate(
            object_pos=object_pos,
            left_finger_pos=left_finger_pos,
            right_finger_pos=right_finger_pos,
        )
        close_ready = self._compute_close_ready(
            reach_dist=reach_dist,
            grasp_pose_gate=grasp_pose_gate,
            enclosure_gate=enclosure_gate,
            finger_open_fraction=finger_open_fraction,
        )
        self._close_ready[:] = close_ready
        entered_close_stage, entered_lift_stage = self._update_reward_stage(
            close_ready=close_ready,
            secure_grasp_gate=secure_grasp_gate,
            lifted=lifted,
        )
        approach_stage = (self._reward_stage == 0).float()
        close_stage = (self._reward_stage == 1).float()
        lift_stage = (self._reward_stage == 2).float()
        lift_progress_reward = self._compute_lift_progress_reward(object_pos=object_pos, secure_grasp_gate=secure_grasp_gate)
        lift_upward_velocity_reward = self._compute_lift_upward_velocity_reward(secure_grasp_gate=secure_grasp_gate)
        goal_reward = self._compute_goal_tracking_reward(
            goal_dist=goal_dist, lifted=lifted, secure_grasp_gate=secure_grasp_gate
        )
        action_penalty = torch.sum(self._actions**2, dim=-1)

        rewards = (
            self.cfg.reach_reward_scale * reach_reward
            + approach_stage
            * (
                self.cfg.pregrasp_reward_scale * pregrasp_reward
                + self.cfg.top_down_reward_scale * top_down_reward
                + self.cfg.gripper_open_reward_scale * gripper_open_reward
                - self.cfg.premature_close_penalty_scale * premature_close_penalty
            )
            + close_stage
            * (
                self.cfg.close_stage_pose_reward_scale * grasp_pose_gate
                + self.cfg.gripper_close_reward_scale * gripper_close_reward
                + self.cfg.grasp_reward_scale * grasp_reward
                - self.cfg.close_stage_open_penalty_scale * finger_open_fraction
            )
            + lift_stage
            * (
                self.cfg.lift_stage_hold_reward_scale * gripper_close_reward
                + self.cfg.lift_stage_hold_reward_scale * grasp_reward
                + self.cfg.lift_progress_reward_scale * lift_progress_reward
                + self.cfg.lift_upward_velocity_reward_scale * lift_upward_velocity_reward
                + self.cfg.lift_reward_scale * lifted
                + self.cfg.goal_reward_scale * goal_reward
            )
            + self.cfg.close_stage_bonus_scale * entered_close_stage
            + self.cfg.lift_stage_bonus_scale * entered_lift_stage
            - self.cfg.action_penalty_scale * action_penalty
        )
        return torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_pos = self._get_object_pos()
        terminated = object_pos[:, 2] < self.cfg.min_object_height
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_tensor = wp.to_torch(self.robot._ALL_INDICES)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids_tensor)

        # Robot reset
        joint_pos = self._default_joint_pos[env_ids_tensor].clone()
        arm_noise = sample_uniform(
            -self.cfg.reset_arm_noise,
            self.cfg.reset_arm_noise,
            (len(env_ids_tensor), len(self._arm_joint_ids)),
            self.device,
        )
        joint_pos[:, self._arm_joint_ids] = torch.clamp(
            joint_pos[:, self._arm_joint_ids] + arm_noise,
            self._joint_lower_limits[self._arm_joint_ids],
            self._joint_upper_limits[self._arm_joint_ids],
        )
        joint_vel = torch.zeros_like(joint_pos)

        self._actions[env_ids_tensor] = 0.0
        self._joint_targets[env_ids_tensor] = joint_pos
        self._reward_stage[env_ids_tensor] = 0
        self._close_ready[env_ids_tensor] = False
        self.robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids_tensor)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids_tensor)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids_tensor)

        # Object reset
        object_pose = self._default_object_pose[env_ids_tensor].clone()
        object_vel = self._default_object_vel[env_ids_tensor].clone()
        object_pose[:, 0] = sample_uniform(
            self.cfg.object_x_range[0], self.cfg.object_x_range[1], (len(env_ids_tensor), 1), self.device
        ).squeeze(-1)
        object_pose[:, 1] = sample_uniform(
            self.cfg.object_y_range[0], self.cfg.object_y_range[1], (len(env_ids_tensor), 1), self.device
        ).squeeze(-1)
        object_pose[:, 2] = self._default_object_pose[env_ids_tensor, 2]
        object_pose[:, :3] += self.scene.env_origins[env_ids_tensor]
        object_vel.zero_()

        self.object.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids_tensor)
        self.object.write_root_velocity_to_sim_index(root_velocity=object_vel, env_ids=env_ids_tensor)

        self._sample_targets(env_ids_tensor)

    def _sample_targets(self, env_ids: torch.Tensor) -> None:
        self._target_pos[env_ids, 0] = sample_uniform(
            self.cfg.target_x_range[0], self.cfg.target_x_range[1], (len(env_ids), 1), self.device
        ).squeeze(-1)
        self._target_pos[env_ids, 1] = sample_uniform(
            self.cfg.target_y_range[0], self.cfg.target_y_range[1], (len(env_ids), 1), self.device
        ).squeeze(-1)
        self._target_pos[env_ids, 2] = sample_uniform(
            self.cfg.target_z_range[0], self.cfg.target_z_range[1], (len(env_ids), 1), self.device
        ).squeeze(-1)

    def _get_grasp_pos(self) -> torch.Tensor:
        """Return the Franka grasp frame in environment-local coordinates."""
        hand_pos = wp.to_torch(self.robot.data.body_pos_w)[:, self._hand_body_id]
        hand_quat = wp.to_torch(self.robot.data.body_quat_w)[:, self._hand_body_id]
        grasp_pos = hand_pos + quat_apply(hand_quat, self._grasp_frame_offset.unsqueeze(0).expand(hand_quat.shape[0], -1))
        return grasp_pos - self.scene.env_origins

    def _get_object_pos(self) -> torch.Tensor:
        """Return the object root position in environment-local coordinates."""
        return wp.to_torch(self.object.data.root_pos_w) - self.scene.env_origins

    def _get_finger_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return left/right fingertip positions in environment-local coordinates."""
        body_pos = wp.to_torch(self.robot.data.body_pos_w)
        body_quat = wp.to_torch(self.robot.data.body_quat_w)
        left_finger_pos = body_pos[:, self._left_finger_body_id] + quat_apply(
            body_quat[:, self._left_finger_body_id], self._finger_contact_offset.unsqueeze(0).expand(body_quat.shape[0], -1)
        )
        right_finger_pos = body_pos[:, self._right_finger_body_id] + quat_apply(
            body_quat[:, self._right_finger_body_id], self._finger_contact_offset.unsqueeze(0).expand(body_quat.shape[0], -1)
        )
        left_finger_pos -= self.scene.env_origins
        right_finger_pos -= self.scene.env_origins
        return left_finger_pos, right_finger_pos

    def _resolve_arm_action_scale(self) -> torch.Tensor:
        """Return per-joint arm action scales as a length-7 tensor."""
        arm_action_scale = torch.as_tensor(
            self.cfg.arm_action_scale,
            device=self._default_joint_pos.device,
            dtype=self._default_joint_pos.dtype,
        )
        if arm_action_scale.ndim == 0:
            arm_action_scale = arm_action_scale.repeat(len(self._arm_joint_ids))
        else:
            arm_action_scale = arm_action_scale.flatten()
        if arm_action_scale.numel() != len(self._arm_joint_ids):
            raise ValueError(
                "arm_action_scale must be a scalar or provide one value per Franka arm joint "
                f"({len(self._arm_joint_ids)} values)."
            )
        return arm_action_scale

    def _resolve_gripper_targets(self) -> tuple[float, float]:
        """Return (open_target, closed_target) clamped to finger joint limits."""
        finger_lower_limit = float(self._joint_lower_limits[self._finger_joint_ids].max().item())
        finger_upper_limit = float(self._joint_upper_limits[self._finger_joint_ids].min().item())

        open_target = float(torch.clamp(torch.tensor(self.cfg.gripper_open_pos), finger_lower_limit, finger_upper_limit).item())
        closed_target = float(
            torch.clamp(torch.tensor(self.cfg.gripper_closed_pos), finger_lower_limit, finger_upper_limit).item()
        )

        if closed_target > open_target:
            raise ValueError(
                "gripper_closed_pos must not exceed gripper_open_pos after applying finger joint limits "
                f"(got closed={closed_target}, open={open_target})."
            )
        return open_target, closed_target

    def _compute_gripper_shaping(
        self,
        object_pos: torch.Tensor,
        left_finger_pos: torch.Tensor,
        right_finger_pos: torch.Tensor,
        finger_joint_pos: torch.Tensor,
        reach_dist: torch.Tensor,
        lifted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gripper-specific shaping rewards for approach and grasp."""
        open_span = max(self._gripper_open_target - self._gripper_closed_target, 1.0e-6)
        finger_open_fraction = (finger_joint_pos - self._gripper_closed_target) / open_span
        finger_open_fraction = torch.clamp(finger_open_fraction, 0.0, 1.0)

        near_object_gate = torch.sigmoid(
            self.cfg.gripper_reward_sharpness * (self.cfg.gripper_reward_distance_thresh - reach_dist)
        )
        finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)
        midpoint_dist = torch.linalg.norm(object_pos - finger_midpoint, dim=-1)
        left_dist = torch.linalg.norm(object_pos - left_finger_pos, dim=-1)
        right_dist = torch.linalg.norm(object_pos - right_finger_pos, dim=-1)
        finger_height_err = 0.5 * (
            torch.abs(left_finger_pos[:, 2] - object_pos[:, 2]) + torch.abs(right_finger_pos[:, 2] - object_pos[:, 2])
        )

        midpoint_reward = 1.0 - torch.tanh(midpoint_dist / self.cfg.grasp_midpoint_std)
        balance_reward = torch.exp(-torch.abs(left_dist - right_dist) / self.cfg.grasp_balance_std)
        finger_height_reward = 1.0 - torch.tanh(finger_height_err / self.cfg.grasp_finger_height_std)
        grasp_pose_gate = near_object_gate * midpoint_reward * balance_reward * finger_height_reward

        close_phase_gate = torch.sigmoid(
            self.cfg.close_phase_sharpness * (grasp_pose_gate - self.cfg.close_phase_gate_thresh)
        )
        keep_open_gate = (1.0 - lifted) * (1.0 - close_phase_gate)
        close_gripper_gate = torch.maximum(close_phase_gate, lifted)

        gripper_open_reward = keep_open_gate * finger_open_fraction
        gripper_close_reward = 1.0 - finger_open_fraction
        grasp_reward = grasp_pose_gate * (1.0 - finger_open_fraction)
        premature_close_penalty = keep_open_gate * (1.0 - finger_open_fraction)
        secure_grasp_gate = close_phase_gate * grasp_pose_gate * (1.0 - finger_open_fraction)

        return (
            gripper_open_reward,
            gripper_close_reward,
            grasp_reward,
            premature_close_penalty,
            secure_grasp_gate,
            grasp_pose_gate,
            finger_open_fraction,
        )

    def _compute_enclosure_gate(
        self, object_pos: torch.Tensor, left_finger_pos: torch.Tensor, right_finger_pos: torch.Tensor
    ) -> torch.Tensor:
        """Estimate whether the cube lies between the open fingertips."""
        finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)
        finger_span = left_finger_pos - right_finger_pos
        finger_span_norm = torch.linalg.norm(finger_span, dim=-1, keepdim=True)
        finger_span_dir = finger_span / torch.clamp(finger_span_norm, min=1.0e-6)

        object_offset = object_pos - finger_midpoint
        span_offset = torch.abs(torch.sum(object_offset * finger_span_dir, dim=-1))
        between_fingers_reward = 1.0 - torch.tanh(span_offset / self.cfg.enclosure_between_std)
        span_ready_gate = torch.sigmoid(
            self.cfg.enclosure_span_sharpness * (finger_span_norm.squeeze(-1) - self.cfg.enclosure_span_thresh)
        )
        return between_fingers_reward * span_ready_gate

    def _compute_close_ready(
        self,
        reach_dist: torch.Tensor,
        grasp_pose_gate: torch.Tensor,
        enclosure_gate: torch.Tensor,
        finger_open_fraction: torch.Tensor,
    ) -> torch.Tensor:
        """Return whether the close phase is allowed to start."""
        return (
            (reach_dist <= self.cfg.approach_stage_reach_thresh)
            & (grasp_pose_gate >= self.cfg.approach_stage_pose_thresh)
            & (enclosure_gate >= self.cfg.approach_stage_enclosure_thresh)
            & (finger_open_fraction >= self.cfg.approach_stage_open_fraction_thresh)
        )

    def _update_reward_stage(
        self,
        close_ready: torch.Tensor,
        secure_grasp_gate: torch.Tensor,
        lifted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the staged reward machine: approach -> close -> lift."""
        ready_for_close = (self._reward_stage == 0) & close_ready
        self._reward_stage = torch.where(ready_for_close, torch.ones_like(self._reward_stage), self._reward_stage)

        ready_for_lift = (
            (self._reward_stage == 1)
            & ((secure_grasp_gate >= self.cfg.lift_stage_secure_grasp_thresh) | (lifted > 0.0))
        )
        self._reward_stage = torch.where(
            ready_for_lift, torch.full_like(self._reward_stage, 2), self._reward_stage
        )
        return ready_for_close.float(), ready_for_lift.float()

    def _compute_lift_progress_reward(self, object_pos: torch.Tensor, secure_grasp_gate: torch.Tensor) -> torch.Tensor:
        """Reward incremental object lifting once the fingers have formed a likely grasp."""
        object_start_height = self._default_object_pose[: object_pos.shape[0], 2].to(device=object_pos.device)
        lifted_height = torch.clamp(object_pos[:, 2] - object_start_height, min=0.0)
        return secure_grasp_gate * torch.tanh(lifted_height / self.cfg.lift_progress_std)

    def _compute_lift_upward_velocity_reward(self, secure_grasp_gate: torch.Tensor) -> torch.Tensor:
        """Reward upward cube motion after the gripper has secured the grasp."""
        upward_velocity = torch.clamp(wp.to_torch(self.object.data.root_lin_vel_w)[:, 2], min=0.0)
        return secure_grasp_gate * torch.tanh(upward_velocity / self.cfg.lift_upward_velocity_std)

    def _compute_goal_tracking_reward(
        self, goal_dist: torch.Tensor, lifted: torch.Tensor, secure_grasp_gate: torch.Tensor
    ) -> torch.Tensor:
        """Reward tracking the commanded object goal once the cube is either lifted or securely grasped."""
        goal_gate = torch.maximum(lifted, secure_grasp_gate)
        return torch.exp(-4.0 * goal_dist) * goal_gate

    def _compute_approach_shaping(
        self,
        grasp_pos: torch.Tensor,
        hand_quat: torch.Tensor,
        object_pos: torch.Tensor,
        reach_dist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute above-cube approach shaping and the gate for closing the gripper."""
        pregrasp_target = object_pos + grasp_pos.new_tensor((0.0, 0.0, self.cfg.pregrasp_height))
        pregrasp_delta = grasp_pos - pregrasp_target
        xy_dist = torch.linalg.norm(pregrasp_delta[:, :2], dim=-1)
        z_err = torch.abs(pregrasp_delta[:, 2])

        pregrasp_reward = (1.0 - torch.tanh(xy_dist / self.cfg.pregrasp_xy_std)) * (
            1.0 - torch.tanh(z_err / self.cfg.pregrasp_z_std)
        )
        descend_gate = torch.sigmoid(
            self.cfg.pregrasp_gate_sharpness * (self.cfg.pregrasp_xy_gate_thresh - xy_dist)
        )
        descend_reward = descend_gate * (1.0 - torch.tanh(reach_dist / self.cfg.descend_reach_std))
        approach_reward = (1.0 - descend_gate) * pregrasp_reward + descend_gate * descend_reward

        hand_to_grasp = quat_apply(hand_quat, self._grasp_frame_offset.unsqueeze(0).expand(hand_quat.shape[0], -1))
        grasp_dir = hand_to_grasp / torch.clamp(torch.linalg.norm(hand_to_grasp, dim=-1, keepdim=True), min=1.0e-6)
        top_down_alignment = torch.clamp(-grasp_dir[:, 2], 0.0, 1.0)
        top_down_reward = descend_gate * top_down_alignment.square()

        approach_gate = descend_gate * top_down_alignment
        return approach_reward, top_down_reward, approach_gate
