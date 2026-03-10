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
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply, sample_uniform

from .franka_cube_env_cfg import FrankaCubeEnvCfg


class FrankaCubeEnv(DirectRLEnv):
    cfg: FrankaCubeEnvCfg

    def __init__(self, cfg: FrankaCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint limits for action clamping
        joint_pos_limits = wp.to_torch(self.robot.data.soft_joint_pos_limits)[0]
        self.robot_dof_lower_limits = joint_pos_limits[:, 0].to(self.device)
        self.robot_dof_upper_limits = joint_pos_limits[:, 1].to(self.device)
        self.arm_joint_indices, _ = self.robot.find_joints("panda_joint[1-7]")
        self.arm_action_scale = self._resolve_arm_action_scale()
        self.arm_dof_lower_limits = self.robot_dof_lower_limits[self.arm_joint_indices]
        self.arm_dof_upper_limits = self.robot_dof_upper_limits[self.arm_joint_indices]
        self.arm_dof_velocity_limits = wp.to_torch(self.robot.data.joint_vel_limits)[0, self.arm_joint_indices].to(self.device)

        # Store default joint positions for relative observations
        self.robot_default_joint_pos = wp.to_torch(self.robot.data.default_joint_pos).clone()
        self.cube_default_root_pose = wp.to_torch(self.cube.data.default_root_pose).clone()
        self.cube_default_root_vel = wp.to_torch(self.cube.data.default_root_vel).clone()

        # Buffers for actions and targets
        self.robot_dof_targets = self.robot_default_joint_pos.clone()
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device)
        self.previous_actions = torch.zeros(
            (self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device
        )

        # Goal position buffer (in robot's local frame)
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.reward_stage = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.close_ready = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

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
        self.finger_joint_ids = torch.tensor(self.finger_joint_indices, device=self.device, dtype=torch.long)
        self.finger_dof_lower_limits = self.robot_dof_lower_limits[self.finger_joint_ids]
        self.finger_dof_upper_limits = self.robot_dof_upper_limits[self.finger_joint_ids]

        finger_lower_limit = float(self.finger_dof_lower_limits.max().item())
        finger_upper_limit = float(self.finger_dof_upper_limits.min().item())
        finger_limit_margin = min(5.0e-4, 0.25 * max(finger_upper_limit - finger_lower_limit, 0.0))
        safe_finger_lower_limit = finger_lower_limit + finger_limit_margin
        safe_finger_upper_limit = finger_upper_limit - finger_limit_margin

        self.gripper_open_pos = float(
            torch.clamp(torch.tensor(0.04, device=self.device), safe_finger_lower_limit, safe_finger_upper_limit).item()
        )
        self.gripper_close_pos = float(
            torch.clamp(torch.tensor(0.0, device=self.device), safe_finger_lower_limit, safe_finger_upper_limit).item()
        )
        self.robot_default_joint_pos[:, self.finger_joint_indices] = self.gripper_open_pos
        self._gripper_span = max(self.gripper_open_pos - self.gripper_close_pos, 1.0e-6)
        self.grasp_frame_offset = torch.tensor(
            (0.0, 0.0, 0.1034),
            device=self.device,
            dtype=self.robot_default_joint_pos.dtype,
        )
        self.finger_contact_offset = torch.tensor(
            (0.0, 0.0, 0.046),
            device=self.device,
            dtype=self.robot_default_joint_pos.dtype,
        )
        self._sample_goal_positions(wp.to_torch(self.robot._ALL_INDICES).to(dtype=torch.long))

    def _setup_scene(self):
        self._register_newton_contact_callback()
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cube = RigidObject(self.cfg.cube)
        # Mirror the stable Franka lift scene so the fixed-base robot is
        # supported by the table instead of contacting the world ground plane.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        table_cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        table_cfg.func(
            "/World/envs/env_.*/Table",
            table_cfg,
            translation=(0.5, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.70711, 0.70711),
        )
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

    def close(self):
        """Cleanup the environment and deregister task-local Newton callbacks."""
        handle = getattr(self, "_newton_model_init_handle", None)
        if handle is not None:
            handle.deregister()
            self._newton_model_init_handle = None
        super().close()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        arm_targets = self.robot_default_joint_pos[:, self.arm_joint_indices] + self.arm_action_scale.unsqueeze(
            0
        ) * self.actions[:, : len(self.arm_joint_indices)]
        self.robot_dof_targets[:, self.arm_joint_indices] = torch.clamp(
            arm_targets,
            self.arm_dof_lower_limits.unsqueeze(0),
            self.arm_dof_upper_limits.unsqueeze(0),
        )

        # Keep the gripper open during the approach phase until the previous
        # step confirmed that the cube is properly enclosed by the fingers.
        allow_close_control = (self.reward_stage > 0) | self.close_ready
        self.actions[:, 7] = torch.where(allow_close_control, self.actions[:, 7], torch.ones_like(self.actions[:, 7]))

        finger_cmd = 0.5 * (self.actions[:, 7] + 1.0)
        finger_target = self.gripper_close_pos + finger_cmd * (self.gripper_open_pos - self.gripper_close_pos)
        self.robot_dof_targets[:, self.finger_joint_indices] = finger_target.unsqueeze(-1).expand(
            -1, len(self.finger_joint_indices)
        )

    def _apply_action(self) -> None:
        # Newton can leak the finger prismatic joints past their USD stops, so
        # clamp them back into range before applying the next target.
        self._enforce_finger_joint_limits()
        self._enforce_arm_joint_velocity_limits()
        self.robot.set_joint_position_target_index(target=self.robot_dof_targets)

    def _resolve_arm_action_scale(self) -> torch.Tensor:
        """Return per-joint arm action scales as a length-7 tensor."""
        arm_action_scale = torch.as_tensor(
            self.cfg.action_scale,
            device=self.device,
            dtype=self.robot_dof_lower_limits.dtype,
        )
        if arm_action_scale.ndim == 0:
            arm_action_scale = arm_action_scale.repeat(len(self.arm_joint_indices))
        else:
            arm_action_scale = arm_action_scale.flatten()
        if arm_action_scale.numel() != len(self.arm_joint_indices):
            raise ValueError(
                "action_scale must be a scalar or provide one value per Franka arm joint "
                f"({len(self.arm_joint_indices)} values)."
            )
        return arm_action_scale

    def _enforce_arm_joint_velocity_limits(self) -> None:
        """Clamp arm joint speeds back into the configured Franka limits."""
        joint_vel = wp.to_torch(self.robot.data.joint_vel)
        arm_joint_vel = joint_vel[:, self.arm_joint_indices]
        clamped_arm_joint_vel = torch.clamp(
            arm_joint_vel,
            -self.arm_dof_velocity_limits.unsqueeze(0),
            self.arm_dof_velocity_limits.unsqueeze(0),
        )
        too_fast = torch.any(torch.abs(clamped_arm_joint_vel - arm_joint_vel) > 1.0e-6, dim=-1)
        if not torch.any(too_fast):
            return

        env_ids = too_fast.nonzero(as_tuple=False).squeeze(-1)
        self.robot.write_joint_velocity_to_sim_index(
            velocity=clamped_arm_joint_vel[env_ids],
            joint_ids=self.arm_joint_indices,
            env_ids=env_ids,
        )

    def _enforce_finger_joint_limits(self) -> None:
        """Clamp finger joints back into their valid range if Newton drifts past the stops."""
        joint_pos = wp.to_torch(self.robot.data.joint_pos)
        finger_joint_pos = joint_pos[:, self.finger_joint_ids]
        clamped_finger_joint_pos = torch.clamp(
            finger_joint_pos,
            self.finger_dof_lower_limits.unsqueeze(0),
            self.finger_dof_upper_limits.unsqueeze(0),
        )
        out_of_bounds = torch.any(torch.abs(clamped_finger_joint_pos - finger_joint_pos) > 1.0e-6, dim=-1)
        if not torch.any(out_of_bounds):
            return

        env_ids = out_of_bounds.nonzero(as_tuple=False).squeeze(-1)
        zero_finger_vel = torch.zeros(
            (env_ids.numel(), len(self.finger_joint_indices)),
            dtype=joint_pos.dtype,
            device=self.device,
        )
        self.robot.write_joint_position_to_sim_index(
            position=clamped_finger_joint_pos[env_ids],
            joint_ids=self.finger_joint_indices,
            env_ids=env_ids,
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=zero_finger_vel,
            joint_ids=self.finger_joint_indices,
            env_ids=env_ids,
        )

    def _get_observations(self) -> dict:
        self._enforce_arm_joint_velocity_limits()
        self._enforce_finger_joint_limits()

        joint_pos = wp.to_torch(self.robot.data.joint_pos)
        joint_vel = wp.to_torch(self.robot.data.joint_vel)
        object_pos = self._get_object_pos()
        grasp_pos = self._get_grasp_pos()
        left_finger_pos, right_finger_pos = self._get_finger_positions()
        finger_joint_pos = joint_pos[:, self.finger_joint_ids].mean(dim=-1)
        (
            finger_midpoint,
            _grasp_pose_gate,
            _close_phase_gate,
            _gripper_close_reward,
            _grasp_reward,
            _secure_grasp_gate,
            enclosure_gate,
            _in_gripper_gate,
            hold_gate,
            finger_open_fraction,
        ) = self._compute_grasp_metrics(
            object_pos=object_pos,
            left_finger_pos=left_finger_pos,
            right_finger_pos=right_finger_pos,
            finger_joint_pos=finger_joint_pos,
        )

        obs = torch.cat(
            [
                joint_pos - self.robot_default_joint_pos,
                joint_vel,
                object_pos - grasp_pos,
                object_pos - left_finger_pos,
                object_pos - right_finger_pos,
                object_pos - finger_midpoint,
                self.previous_actions,
                finger_open_fraction.unsqueeze(-1),
                enclosure_gate.unsqueeze(-1),
                hold_gate.unsqueeze(-1),
            ],
            dim=-1,
        )
        obs = torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        obs = torch.clamp(obs, -100.0, 100.0)
        self.previous_actions[:] = self.actions
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._enforce_arm_joint_velocity_limits()
        self._enforce_finger_joint_limits()
        object_pos = self._get_object_pos()
        left_finger_pos, right_finger_pos = self._get_finger_positions()

        joint_pos = wp.to_torch(self.robot.data.joint_pos)
        joint_vel = wp.to_torch(self.robot.data.joint_vel)
        finger_joint_pos = joint_pos[:, self.finger_joint_ids].mean(dim=-1)
        (
            finger_midpoint,
            grasp_pose_gate,
            _close_phase_gate,
            _gripper_close_reward,
            grasp_reward,
            _secure_grasp_gate,
            enclosure_gate,
            _in_gripper_gate,
            hold_gate,
            finger_open_fraction,
        ) = self._compute_grasp_metrics(
            object_pos=object_pos,
            left_finger_pos=left_finger_pos,
            right_finger_pos=right_finger_pos,
            finger_joint_pos=finger_joint_pos,
        )
        reach_dist = torch.linalg.norm(object_pos - finger_midpoint, dim=-1)
        reaching_reward = 1.0 - torch.tanh(reach_dist / self.cfg.reaching_object_std)
        enclosure_reward = enclosure_gate * grasp_pose_gate * finger_open_fraction
        close_ready = self._compute_close_ready(
            reach_dist=reach_dist,
            grasp_pose_gate=grasp_pose_gate,
            enclosure_gate=enclosure_gate,
            finger_open_fraction=finger_open_fraction,
        )
        self.close_ready[:] = close_ready
        self._update_reward_stage(
            close_ready=close_ready,
            hold_gate=hold_gate,
        )
        lift_progress_reward = self._compute_lift_progress_reward(object_pos=object_pos, grasp_gate=hold_gate)
        lifted_height = torch.clamp(
            object_pos[:, 2] - self.cube_default_root_pose[: object_pos.shape[0], 2].to(device=object_pos.device),
            min=0.0,
        )
        lifting_reward = hold_gate * (lifted_height >= self.cfg.lifted_height).float()
        action_penalty = torch.sum(self.actions.square(), dim=-1)
        joint_vel_penalty = torch.sum(joint_vel.square(), dim=-1)

        rewards = (
            self.cfg.reaching_object_scale * reaching_reward
            + self.cfg.gripper_open_reward_scale * enclosure_reward
            + self.cfg.grasp_reward_scale * grasp_reward
            + self.cfg.lift_progress_reward_scale * lift_progress_reward
            + self.cfg.lifting_object_scale * lifting_reward
            - self.cfg.action_penalty_scale * action_penalty
            - self.cfg.joint_vel_penalty_scale * joint_vel_penalty
        )
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        self.extras["log"] = {
            "reaching_reward": reaching_reward.mean(),
            "grasp_pose_gate": grasp_pose_gate.mean(),
            "enclosure_gate": enclosure_gate.mean(),
            "enclosure_reward": enclosure_reward.mean(),
            "close_ready": close_ready.float().mean(),
            "grasp_reward": grasp_reward.mean(),
            "finger_open_fraction": finger_open_fraction.mean(),
            "hold_gate": hold_gate.mean(),
            "lift_progress_reward": lift_progress_reward.mean(),
            "lifting_reward": lifting_reward.mean(),
            "action_penalty": action_penalty.mean(),
            "joint_vel_penalty": joint_vel_penalty.mean(),
        }
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_height = self._get_object_pos()[:, 2]
        terminated = object_height < self.cfg.object_drop_height
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_tensor = wp.to_torch(self.robot._ALL_INDICES).to(dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids_tensor)

        default_root_pose = wp.to_torch(self.robot.data.default_root_pose)[env_ids_tensor].clone()
        default_root_vel = wp.to_torch(self.robot.data.default_root_vel)[env_ids_tensor].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids_tensor]
        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids_tensor)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids_tensor)

        joint_pos = self.robot_default_joint_pos[env_ids_tensor].clone()
        arm_noise = sample_uniform(
            -self.cfg.reset_arm_noise,
            self.cfg.reset_arm_noise,
            (len(env_ids_tensor), len(self.arm_joint_indices)),
            self.device,
        )
        joint_pos[:, self.arm_joint_indices] = torch.clamp(
            joint_pos[:, self.arm_joint_indices] + arm_noise,
            self.arm_dof_lower_limits.unsqueeze(0),
            self.arm_dof_upper_limits.unsqueeze(0),
        )
        joint_pos[:, self.finger_joint_indices] = self.gripper_open_pos
        joint_vel = torch.zeros_like(joint_pos)
        self.robot_dof_targets[env_ids_tensor] = joint_pos
        self.actions[env_ids_tensor] = 0.0
        self.previous_actions[env_ids_tensor] = 0.0
        self.reward_stage[env_ids_tensor] = 0
        self.close_ready[env_ids_tensor] = False
        self.robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids_tensor)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids_tensor)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids_tensor)

        object_pose = self.cube_default_root_pose[env_ids_tensor].clone()
        object_vel = self.cube_default_root_vel[env_ids_tensor].clone()
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
        self.cube.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids_tensor)
        self.cube.write_root_velocity_to_sim_index(root_velocity=object_vel, env_ids=env_ids_tensor)

        self._sample_goal_positions(env_ids_tensor)

    def _sample_goal_positions(self, env_ids: torch.Tensor) -> None:
        self.goal_pos[env_ids, 0] = sample_uniform(
            self.cfg.goal_pos_x_range[0],
            self.cfg.goal_pos_x_range[1],
            (len(env_ids),),
            self.device,
        )
        self.goal_pos[env_ids, 1] = sample_uniform(
            self.cfg.goal_pos_y_range[0],
            self.cfg.goal_pos_y_range[1],
            (len(env_ids),),
            self.device,
        )
        self.goal_pos[env_ids, 2] = sample_uniform(
            self.cfg.goal_pos_z_range[0],
            self.cfg.goal_pos_z_range[1],
            (len(env_ids),),
            self.device,
        )

    def _get_object_pos(self) -> torch.Tensor:
        """Return the cube root position in environment-local coordinates."""
        return wp.to_torch(self.cube.data.root_pos_w) - self.scene.env_origins

    def _get_grasp_pos(self) -> torch.Tensor:
        """Return the Franka grasp frame in environment-local coordinates."""
        hand_pos = wp.to_torch(self.robot.data.body_pos_w)[:, self.ee_body_idx]
        hand_quat = wp.to_torch(self.robot.data.body_quat_w)[:, self.ee_body_idx]
        grasp_offset = quat_apply(hand_quat, self.grasp_frame_offset.unsqueeze(0).expand(hand_quat.shape[0], -1))
        return hand_pos + grasp_offset - self.scene.env_origins

    def _get_finger_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return left and right fingertip positions in environment-local coordinates."""
        body_pos = wp.to_torch(self.robot.data.body_pos_w)
        body_quat = wp.to_torch(self.robot.data.body_quat_w)
        left_finger_pos = body_pos[:, self.lf_body_idx] + quat_apply(
            body_quat[:, self.lf_body_idx],
            self.finger_contact_offset.unsqueeze(0).expand(body_quat.shape[0], -1),
        )
        right_finger_pos = body_pos[:, self.rf_body_idx] + quat_apply(
            body_quat[:, self.rf_body_idx],
            self.finger_contact_offset.unsqueeze(0).expand(body_quat.shape[0], -1),
        )
        left_finger_pos -= self.scene.env_origins
        right_finger_pos -= self.scene.env_origins
        return left_finger_pos, right_finger_pos

    def _get_finger_open_fraction(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Return the mean normalized Franka finger opening in [0, 1]."""
        finger_joint_pos = joint_pos[:, self.finger_joint_ids].mean(dim=-1)
        return torch.clamp(
            (finger_joint_pos - self.gripper_close_pos) / self._gripper_span,
            min=0.0,
            max=1.0,
        )

    def _compute_grasp_metrics(
        self,
        object_pos: torch.Tensor,
        left_finger_pos: torch.Tensor,
        right_finger_pos: torch.Tensor,
        finger_joint_pos: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute grasp-quality metrics shared by observations and rewards."""
        finger_open_fraction = torch.clamp(
            (finger_joint_pos - self.gripper_close_pos) / self._gripper_span,
            0.0,
            1.0,
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
        grasp_pose_gate = torch.clamp(midpoint_reward * balance_reward * finger_height_reward, 0.0, 1.0)
        gripper_close_reward = 1.0 - finger_open_fraction
        enclosure_gate = self._compute_enclosure_gate(
            object_pos=object_pos,
            left_finger_pos=left_finger_pos,
            right_finger_pos=right_finger_pos,
        )
        close_phase_gate = torch.clamp(grasp_pose_gate * enclosure_gate, 0.0, 1.0)
        grasp_reward = close_phase_gate * gripper_close_reward
        secure_grasp_gate = grasp_reward
        in_gripper_gate = grasp_reward
        hold_gate = grasp_reward

        return (
            finger_midpoint,
            grasp_pose_gate,
            close_phase_gate,
            gripper_close_reward,
            grasp_reward,
            secure_grasp_gate,
            enclosure_gate,
            in_gripper_gate,
            hold_gate,
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
        hold_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the staged reward machine once the cube is actually secured."""
        ready_for_close = (self.reward_stage == 0) & close_ready
        self.reward_stage = torch.where(ready_for_close, torch.ones_like(self.reward_stage), self.reward_stage)

        ready_for_lift = (
            (self.reward_stage == 1)
            & (hold_gate >= self.cfg.lift_stage_secure_grasp_thresh)
        )
        self.reward_stage = torch.where(ready_for_lift, torch.full_like(self.reward_stage, 2), self.reward_stage)
        return ready_for_close.float(), ready_for_lift.float()

    def _compute_lift_progress_reward(self, object_pos: torch.Tensor, grasp_gate: torch.Tensor) -> torch.Tensor:
        """Reward incremental cube lifting once the fingers likely secured the grasp."""
        object_start_height = self.cube_default_root_pose[: object_pos.shape[0], 2].to(device=object_pos.device)
        lifted_height = torch.clamp(object_pos[:, 2] - object_start_height, min=0.0)
        return grasp_gate * torch.tanh(lifted_height / self.cfg.lift_progress_std)

    def _compute_lift_upward_velocity_reward(self, grasp_gate: torch.Tensor) -> torch.Tensor:
        """Reward upward cube motion after the gripper has secured the grasp."""
        upward_velocity = torch.clamp(wp.to_torch(self.cube.data.root_lin_vel_w)[:, 2], min=0.0)
        return grasp_gate * torch.tanh(upward_velocity / self.cfg.lift_upward_velocity_std)

    def _compute_approach_shaping(
        self,
        grasp_pos: torch.Tensor,
        hand_quat: torch.Tensor,
        object_pos: torch.Tensor,
        reach_dist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reward approaching the cube from above with a vertical gripper."""
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

        hand_to_grasp = quat_apply(hand_quat, self.grasp_frame_offset.unsqueeze(0).expand(hand_quat.shape[0], -1))
        grasp_dir = hand_to_grasp / torch.clamp(torch.linalg.norm(hand_to_grasp, dim=-1, keepdim=True), min=1.0e-6)
        top_down_alignment = torch.clamp(-grasp_dir[:, 2], 0.0, 1.0)
        top_down_reward = descend_gate * top_down_alignment.square()
        return approach_reward, top_down_reward, top_down_alignment

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
        """Apply task-local Newton and MuJoCo-solver contact parameters before model finalize."""
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
        if self.cfg.newton_contact.mu is not None and hasattr(shape_cfg, "mu"):
            shape_cfg.mu = self.cfg.newton_contact.mu
        if self.cfg.newton_contact.contact_margin is not None:
            shape_cfg.contact_margin = self.cfg.newton_contact.contact_margin
        self._apply_mujoco_solver_contact_tuning(builder)

    def _apply_mujoco_solver_contact_tuning(self, builder) -> None:
        """Tune MuJoCo-solver contact attributes used by the Newton collision pipeline."""
        from newton import solvers

        solvers.SolverMuJoCo.register_custom_attributes(builder)
        self._set_mujoco_custom_attribute_default(
            builder,
            "mujoco:geom_solimp",
            self.cfg.newton_contact.geom_solimp,
        )
        self._set_mujoco_custom_attribute_default(
            builder,
            "mujoco:solimpfriction",
            self.cfg.newton_contact.solimp_friction,
        )
        self._set_mujoco_custom_attribute_default(
            builder,
            "mujoco:solreffriction",
            self.cfg.newton_contact.solref_friction,
        )
        self._append_support_contact_pairs(builder)
        self._append_grasp_contact_pairs(builder)

    @staticmethod
    def _set_mujoco_custom_attribute_default(builder, key: str, value) -> None:
        """Set a MuJoCo custom-attribute default if a task override is provided."""
        if value is None:
            return
        builder.custom_attributes[key].default = list(value) if isinstance(value, tuple) else value

    @staticmethod
    def _matches_label(label: str, targets: tuple[str, ...]) -> bool:
        """Return whether a prim/body label contains any target token."""
        return any(target in label for target in targets)

    def _append_support_contact_pairs(self, builder) -> None:
        """Apply damped contact overrides for cube support contacts."""
        cube_shapes, _, support_shapes = self._collect_contact_shape_groups(builder)
        pair_keys: set[tuple[int, int, int]] = set()
        for cube_shape in cube_shapes:
            for support_shape in support_shapes:
                self._append_mujoco_contact_pair(
                    builder,
                    shape_a=cube_shape,
                    shape_b=support_shape,
                    condim=self.cfg.newton_contact.support_pair_condim,
                    friction=self.cfg.newton_contact.support_pair_friction,
                    margin=self.cfg.newton_contact.support_pair_margin,
                    solimp=self.cfg.newton_contact.support_pair_solimp,
                    solref=self.cfg.newton_contact.support_pair_solref,
                    pair_keys=pair_keys,
                )

    def _append_grasp_contact_pairs(self, builder) -> None:
        """Apply damped contact overrides for fingertip grasp contacts."""
        cube_shapes, finger_shapes, _ = self._collect_contact_shape_groups(builder)
        pair_keys: set[tuple[int, int, int]] = set()
        for finger_shape in finger_shapes:
            for cube_shape in cube_shapes:
                self._append_mujoco_contact_pair(
                    builder,
                    shape_a=finger_shape,
                    shape_b=cube_shape,
                    condim=self.cfg.newton_contact.grasp_pair_condim,
                    friction=self.cfg.newton_contact.grasp_pair_friction,
                    margin=self.cfg.newton_contact.grasp_pair_margin,
                    solimp=self.cfg.newton_contact.grasp_pair_solimp,
                    solref=self.cfg.newton_contact.grasp_pair_solref,
                    pair_keys=pair_keys,
                )

    def _collect_contact_shape_groups(self, builder) -> tuple[list[int], list[int], list[int]]:
        """Collect shape indices for the cube, fingers, and support surfaces."""
        shape_labels = getattr(builder, "shape_label", None) or getattr(builder, "shape_key", None)
        body_labels = getattr(builder, "body_label", None) or getattr(builder, "body_key", None)

        cube_shapes: list[int] = []
        finger_shapes: list[int] = []
        support_shapes: list[int] = []

        for shape_idx in range(builder.shape_count):
            body_idx = int(builder.shape_body[shape_idx])
            shape_label = str(shape_labels[shape_idx]).lower() if shape_labels is not None else ""
            body_label = (
                str(body_labels[body_idx]).lower()
                if body_labels is not None and 0 <= body_idx < len(body_labels)
                else ""
            )

            if self._matches_label(body_label, ("panda_leftfinger", "panda_rightfinger")):
                finger_shapes.append(shape_idx)
            elif self._matches_label(body_label, ("object",)):
                cube_shapes.append(shape_idx)

            if body_idx == -1 or self._matches_label(shape_label, ("/table", "/ground")) or self._matches_label(
                body_label, ("table", "ground")
            ):
                support_shapes.append(shape_idx)

        return cube_shapes, finger_shapes, support_shapes

    def _append_mujoco_contact_pair(
        self,
        builder,
        shape_a: int,
        shape_b: int,
        condim: int | None,
        friction: tuple[float, float, float, float, float] | None,
        margin: float | None,
        solimp: tuple[float, float, float, float, float] | None,
        solref: tuple[float, float] | None,
        pair_keys: set[tuple[int, int, int]],
    ) -> None:
        """Append a MuJoCo contact-pair override for one shape pair."""
        world_a = int(builder.shape_world[shape_a])
        world_b = int(builder.shape_world[shape_b])
        if world_a == world_b:
            pair_world = world_a
        elif world_a == -1:
            pair_world = world_b
        elif world_b == -1:
            pair_world = world_a
        else:
            return

        geom1, geom2 = sorted((shape_a, shape_b))
        pair_key = (pair_world, geom1, geom2)
        if pair_key in pair_keys:
            return
        pair_keys.add(pair_key)

        attrs = builder.custom_attributes
        attrs["mujoco:pair_world"].values.append(pair_world)
        attrs["mujoco:pair_geom1"].values.append(geom1)
        attrs["mujoco:pair_geom2"].values.append(geom2)
        attrs["mujoco:pair_condim"].values.append(
            condim if condim is not None else attrs["mujoco:pair_condim"].default
        )
        attrs["mujoco:pair_friction"].values.append(
            list(friction) if friction is not None else attrs["mujoco:pair_friction"].default
        )
        attrs["mujoco:pair_gap"].values.append(attrs["mujoco:pair_gap"].default)
        attrs["mujoco:pair_margin"].values.append(
            margin if margin is not None else attrs["mujoco:pair_margin"].default
        )
        attrs["mujoco:pair_solimp"].values.append(
            list(solimp) if solimp is not None else attrs["mujoco:pair_solimp"].default
        )
        attrs["mujoco:pair_solref"].values.append(
            list(solref) if solref is not None else attrs["mujoco:pair_solref"].default
        )
        attrs["mujoco:pair_solreffriction"].values.append(attrs["mujoco:pair_solreffriction"].default)