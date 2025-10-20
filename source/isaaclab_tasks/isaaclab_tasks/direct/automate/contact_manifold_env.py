# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Contact Manifold Sampling Environment

Extended disassembly environment that uses impedance control and trajectory sampling
to construct contact manifolds during peg-in-hole disassembly.
"""

import numpy as np
import os
import torch
from typing import Optional

import isaacsim.core.utils.torch as torch_utils

from isaaclab.controllers import OperationalSpaceController
from isaaclab.utils.math import axis_angle_from_quat

from isaaclab_tasks.direct.automate.disassembly_env import DisassemblyEnv
from isaaclab_tasks.direct.automate.disassembly_env_cfg import DisassemblyEnvCfg
from isaaclab_tasks.direct.automate.contact_manifold_sampler import (
    ContactManifoldSamplingController,
    ContactManifoldDataCollector,
    create_impedance_osc_config,
)


class ContactManifoldEnv(DisassemblyEnv):
    """
    Extended disassembly environment for contact manifold sampling.

    This environment performs systematic exploration of contact configurations during
    disassembly using:
    - Spiral trajectories in XY
    - Oscillatory rotations in roll/pitch/yaw
    - Linear extraction in Z
    - Variable impedance control for compliant contact
    - Force/torque sensing for contact detection
    """

    cfg: DisassemblyEnvCfg

    def __init__(
        self,
        cfg: DisassemblyEnvCfg,
        render_mode: str | None = None,
        # Sampling controller parameters
        spiral_radius_max: float = 0.005,
        spiral_frequency: float = 0.5,
        extraction_velocity: float = 0.002,
        rotation_amplitude: list = None,
        rotation_frequency: list = None,
        # Impedance control parameters
        translational_stiffness: list = None,
        rotational_stiffness: list = None,
        damping_ratio: float = 1.0,
        # Contact detection
        contact_force_threshold: float = 1.0,
        # Data collection
        max_samples_per_env: int = 10000,
        save_directory: str = "./contact_manifolds",
        **kwargs
    ):
        # Initialize base environment
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize sampling controller
        self.sampling_controller = ContactManifoldSamplingController(
            num_envs=self.num_envs,
            device=self.device,
            spiral_radius_max=spiral_radius_max,
            spiral_frequency=spiral_frequency,
            extraction_velocity=extraction_velocity,
            rotation_amplitude=rotation_amplitude,
            rotation_frequency=rotation_frequency,
            contact_force_threshold=contact_force_threshold,
        )

        # Initialize operational space controller for impedance control
        osc_cfg = create_impedance_osc_config(
            translational_stiffness=translational_stiffness,
            rotational_stiffness=rotational_stiffness,
            damping_ratio=damping_ratio,
            variable_impedance=True,
            inertial_compensation=True,
            gravity_compensation=True,
        )
        self.osc_controller = OperationalSpaceController(
            cfg=osc_cfg,
            num_envs=self.num_envs,
            device=self.device
        )

        # Initialize data collector
        self.data_collector = ContactManifoldDataCollector(
            max_samples_per_env=max_samples_per_env,
            device=self.device
        )

        # Storage directory
        self.save_directory = save_directory
        os.makedirs(save_directory, exist_ok=True)

        # Impedance parameters for contact-adaptive control
        if translational_stiffness is None:
            translational_stiffness = [200.0, 200.0, 300.0]
        if rotational_stiffness is None:
            rotational_stiffness = [10.0, 10.0, 10.0]

        self.trans_stiffness_default = torch.tensor(translational_stiffness, device=self.device)
        self.rot_stiffness_default = torch.tensor(rotational_stiffness, device=self.device)
        self.trans_stiffness_contact = self.trans_stiffness_default * 0.3  # Reduce during contact
        self.rot_stiffness_contact = self.rot_stiffness_default * 0.3

        # External wrench tracking (will be computed from joint torques)
        self.external_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        # Gravity torques for OSC
        self.gravity_torques = torch.zeros((self.num_envs, 7), device=self.device)

        # Sampling mode flag
        self.sampling_mode_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _init_tensors(self):
        """Initialize tensors including those for contact manifold sampling."""
        super()._init_tensors()

        # Base pose for sampling (assembled configuration)
        self.sampling_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.sampling_base_quat = torch.zeros((self.num_envs, 4), device=self.device)

    def _reset_idx(self, env_ids):
        """Reset environment and initialize sampling."""
        super()._reset_idx(env_ids)

        # Reset sampling controller
        self.sampling_controller.reset_state(env_ids)

        # Set base pose for sampling (assembled configuration)
        self.sampling_base_pos[env_ids] = self.fixed_pos[env_ids].clone()
        self.sampling_base_quat[env_ids] = self.fixed_quat[env_ids].clone()

        # Activate sampling mode
        self.sampling_mode_active[env_ids] = True

    def _pre_physics_step(self, action):
        """Apply contact manifold sampling control."""
        super()._pre_physics_step(action)

        # Get environments in sampling mode
        sampling_env_ids = torch.where(self.sampling_mode_active)[0]

        if len(sampling_env_ids) > 0:
            # Compute trajectory targets using sampling controller
            target_pos, target_quat = self.sampling_controller.compute_trajectory_targets(
                base_pos=self.sampling_base_pos[sampling_env_ids],
                base_quat=self.sampling_base_quat[sampling_env_ids],
                dt=self.physics_dt,
            )

            # Estimate external wrench from joint torques
            self._estimate_external_wrench()

            # Check for contact
            contact_mask = self.sampling_controller.check_contact(
                self.external_wrench[sampling_env_ids]
            )

            # Adjust impedance based on contact
            stiffness = self._compute_adaptive_stiffness(contact_mask)

            # Create command for operational space controller
            # Format: [pos (3), quat (4), stiffness (6)]
            command = torch.zeros((len(sampling_env_ids), 13), device=self.device)
            command[:, 0:3] = target_pos
            command[:, 3:7] = target_quat  # wxyz format
            command[:, 7:13] = stiffness

            # Set command
            self.osc_controller.set_command(
                command=command,
                current_ee_pose_b=torch.cat([
                    self.fingertip_midpoint_pos[sampling_env_ids],
                    self.fingertip_midpoint_quat[sampling_env_ids]
                ], dim=1)
            )

            # Compute joint torques using operational space controller
            joint_efforts = self.osc_controller.compute(
                jacobian_b=self.fingertip_midpoint_jacobian[sampling_env_ids],
                current_ee_pose_b=torch.cat([
                    self.fingertip_midpoint_pos[sampling_env_ids],
                    self.fingertip_midpoint_quat[sampling_env_ids]
                ], dim=1),
                current_ee_vel_b=torch.cat([
                    self.ee_linvel_fd[sampling_env_ids],
                    self.ee_angvel_fd[sampling_env_ids]
                ], dim=1),
                mass_matrix=self.arm_mass_matrix[sampling_env_ids],
                gravity=self.gravity_torques[sampling_env_ids],
                current_joint_pos=self.joint_pos[sampling_env_ids, 0:7],
                current_joint_vel=self.joint_vel[sampling_env_ids, 0:7],
                nullspace_joint_pos_target=torch.tensor(
                    self.cfg.ctrl.default_dof_pos_tensor,
                    device=self.device
                ).repeat(len(sampling_env_ids), 1)
            )

            # Apply joint efforts
            self.joint_torque[sampling_env_ids, 0:7] = joint_efforts

            # Collect contact manifold data
            self.data_collector.add_sample(
                env_ids=sampling_env_ids,
                positions=self.fingertip_midpoint_pos[sampling_env_ids],
                orientations=self.fingertip_midpoint_quat[sampling_env_ids],
                wrenches=self.external_wrench[sampling_env_ids],
                contact_mask=contact_mask,
                timestamp=self.episode_length_buf[sampling_env_ids[0]].item() * self.physics_dt
            )

    def _estimate_external_wrench(self):
        """
        Estimate external wrench from joint torques and dynamics.

        Uses the relationship: tau_ext = tau_measured - tau_model
        where tau_model includes gravity, Coriolis, and commanded torques.
        """
        # For simplicity, approximate external wrench from joint torques
        # In practice, Franka has built-in torque sensing that can be used directly

        # Map joint torques to task space using Jacobian transpose
        # F_ext = (J^T)^{-1} * tau_ext
        # For better estimate, use: F_ext = (J^T)^{+} * tau_ext

        # Compute external joint torques (simplified)
        # In real robot, this comes from torque sensors
        tau_external = self.joint_torque[:, 0:7].clone()  # Placeholder

        # Map to task space
        jacobian_T = torch.transpose(self.fingertip_midpoint_jacobian, dim0=1, dim1=2)

        # Pseudo-inverse approach
        try:
            jac_T_pinv = torch.linalg.pinv(jacobian_T)
            wrench = (jac_T_pinv @ tau_external.unsqueeze(-1)).squeeze(-1)
            self.external_wrench[:, :] = wrench
        except:
            # If pseudo-inverse fails, use zero wrench
            self.external_wrench[:, :] = 0.0

    def _compute_adaptive_stiffness(self, contact_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive stiffness based on contact state.

        Args:
            contact_mask: Boolean tensor indicating contact for each environment

        Returns:
            stiffness: Stiffness parameters (num_envs, 6)
        """
        num_contact = len(contact_mask)
        stiffness = torch.zeros((num_contact, 6), device=self.device)

        # Default stiffness
        stiffness[:, 0:3] = self.trans_stiffness_default.unsqueeze(0)
        stiffness[:, 3:6] = self.rot_stiffness_default.unsqueeze(0)

        # Reduce stiffness during contact
        if contact_mask.any():
            contact_indices = torch.where(contact_mask)[0]
            stiffness[contact_indices, 0:3] = self.trans_stiffness_contact.unsqueeze(0)
            stiffness[contact_indices, 3:6] = self.rot_stiffness_contact.unsqueeze(0)

        return stiffness

    def save_manifolds(self):
        """Save all collected contact manifolds."""
        base_filepath = os.path.join(
            self.save_directory,
            f"manifold_assembly_{self.cfg_task.assembly_id}"
        )
        self.data_collector.export_all_manifolds(base_filepath)

        print(f"\nContact Manifold Collection Summary:")
        for env_id in range(self.num_envs):
            num_samples = self.data_collector.get_num_samples(env_id)
            if num_samples > 0:
                print(f"  Env {env_id}: {num_samples} contact samples")

    def get_manifold_statistics(self) -> dict:
        """Get statistics about collected manifolds."""
        stats = {
            'total_samples': 0,
            'samples_per_env': [],
            'num_envs_with_data': 0,
        }

        for env_id in range(self.num_envs):
            num_samples = self.data_collector.get_num_samples(env_id)
            if num_samples > 0:
                stats['num_envs_with_data'] += 1
                stats['total_samples'] += num_samples
                stats['samples_per_env'].append(num_samples)

        if stats['samples_per_env']:
            stats['mean_samples'] = np.mean(stats['samples_per_env'])
            stats['std_samples'] = np.std(stats['samples_per_env'])
            stats['min_samples'] = np.min(stats['samples_per_env'])
            stats['max_samples'] = np.max(stats['samples_per_env'])

        return stats
