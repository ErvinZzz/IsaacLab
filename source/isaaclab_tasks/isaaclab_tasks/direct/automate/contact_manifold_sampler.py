# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Contact Manifold Sampling Controller for Disassembly Tasks

This module implements a sampling controller that constructs contact manifolds
during disassembly by executing:
- Spiral trajectories in X, Y (translation)
- Oscillatory trajectories in alpha, beta, gamma (rotation)
- Linearly increasing trajectory in Z (extraction)

The contact manifold represents the geometric relationship between peg and hole
during contact, enabling pose estimation via manifold registration.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

import isaacsim.core.utils.torch as torch_utils
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg


class ContactManifoldSamplingController:
    """
    Motion primitive controller for sampling contact manifolds during disassembly.

    Combines spiral translation, oscillatory rotation, and linear extraction
    to systematically explore contact configurations.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        # Spiral parameters (X, Y)
        spiral_radius_max: float = 0.005,  # 5mm max radius
        spiral_frequency: float = 0.5,  # Hz
        spiral_growth_rate: float = 0.002,  # m/s radial growth
        # Oscillatory parameters (alpha, beta, gamma)
        rotation_amplitude: List[float] = None,  # [roll, pitch, yaw] in radians
        rotation_frequency: List[float] = None,  # Hz for each axis
        # Linear extraction (Z)
        extraction_velocity: float = 0.002,  # 2mm/s
        # Depth-dependent amplitude scaling
        amplitude_scaling_func: Optional[callable] = None,
        # Random perturbations to reduce bias
        position_noise_std: float = 0.0005,  # 0.5mm
        rotation_noise_std: float = 0.01,  # ~0.57 degrees
        # Contact force threshold
        contact_force_threshold: float = 1.0,  # N
    ):
        self.num_envs = num_envs
        self.device = device

        # Spiral parameters
        self.spiral_radius_max = spiral_radius_max
        self.spiral_frequency = spiral_frequency
        self.spiral_growth_rate = spiral_growth_rate

        # Oscillatory rotation parameters
        if rotation_amplitude is None:
            # Default: smaller amplitudes for roll/pitch, larger for yaw
            rotation_amplitude = [0.05, 0.05, 0.1]  # radians
        if rotation_frequency is None:
            rotation_frequency = [0.3, 0.4, 0.2]  # Hz

        self.rotation_amplitude = torch.tensor(rotation_amplitude, device=device)
        self.rotation_frequency = torch.tensor(rotation_frequency, device=device)

        # Linear extraction
        self.extraction_velocity = extraction_velocity

        # Amplitude scaling (reduce amplitude at deeper insertion to prevent damage)
        if amplitude_scaling_func is None:
            # Default: linear decay with depth
            self.amplitude_scaling_func = lambda depth: max(0.3, 1.0 - depth * 10.0)
        else:
            self.amplitude_scaling_func = amplitude_scaling_func

        # Noise parameters
        self.position_noise_std = position_noise_std
        self.rotation_noise_std = rotation_noise_std

        # Contact detection
        self.contact_force_threshold = contact_force_threshold

        # State variables
        self.reset_state()

    def reset_state(self, env_ids: Optional[torch.Tensor] = None):
        """Reset controller state for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Initialize time tracking
        if not hasattr(self, 'time'):
            self.time = torch.zeros(self.num_envs, device=self.device)
            self.extraction_distance = torch.zeros(self.num_envs, device=self.device)
            # Random phase offsets to decorrelate trajectories
            self.phase_offset = torch.rand(self.num_envs, 3, device=self.device) * 2 * np.pi
        else:
            self.time[env_ids] = 0.0
            self.extraction_distance[env_ids] = 0.0
            self.phase_offset[env_ids] = torch.rand(len(env_ids), 3, device=self.device) * 2 * np.pi

    def compute_trajectory_targets(
        self,
        base_pos: torch.Tensor,  # (num_envs, 3)
        base_quat: torch.Tensor,  # (num_envs, 4)
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute target pose using sampling motion primitive.

        Args:
            base_pos: Base position (typically the assembled configuration)
            base_quat: Base orientation
            dt: Time step

        Returns:
            target_pos: Target position (num_envs, 3)
            target_quat: Target orientation (num_envs, 4) in wxyz format
        """
        # Update time
        self.time += dt

        # Get depth-dependent scaling
        depth = self.extraction_distance
        amplitude_scale = torch.tensor(
            [self.amplitude_scaling_func(d.item()) for d in depth],
            device=self.device
        )

        # ===== SPIRAL TRAJECTORY (X, Y) =====
        # r(t) = min(r_max, growth_rate * t)
        radius = torch.minimum(
            torch.ones(self.num_envs, device=self.device) * self.spiral_radius_max,
            self.spiral_growth_rate * self.time
        )

        # theta(t) = 2*pi*f*t
        theta = 2 * np.pi * self.spiral_frequency * self.time

        # Spiral position in XY plane
        spiral_x = radius * torch.cos(theta)
        spiral_y = radius * torch.sin(theta)

        # Add random perturbations
        if self.position_noise_std > 0:
            spiral_x += torch.randn(self.num_envs, device=self.device) * self.position_noise_std
            spiral_y += torch.randn(self.num_envs, device=self.device) * self.position_noise_std

        # Scale by amplitude factor
        spiral_x *= amplitude_scale
        spiral_y *= amplitude_scale

        # ===== LINEAR EXTRACTION (Z) =====
        delta_z = self.extraction_velocity * dt
        self.extraction_distance += delta_z

        # Combine into position delta
        position_delta = torch.stack([spiral_x, spiral_y, self.extraction_distance], dim=1)
        target_pos = base_pos + position_delta

        # ===== OSCILLATORY ROTATION (alpha, beta, gamma) =====
        # Roll, pitch, yaw = alpha, beta, gamma
        # angle(t) = A * sin(2*pi*f*t + phase)

        omega = 2 * np.pi * self.rotation_frequency  # (3,)
        time_expanded = self.time.unsqueeze(1)  # (num_envs, 1)

        # Compute oscillatory angles for each axis
        angles = self.rotation_amplitude.unsqueeze(0) * torch.sin(
            omega.unsqueeze(0) * time_expanded + self.phase_offset
        )  # (num_envs, 3)

        # Add random perturbations
        if self.rotation_noise_std > 0:
            angles += torch.randn(self.num_envs, 3, device=self.device) * self.rotation_noise_std

        # Scale by amplitude factor
        angles *= amplitude_scale.unsqueeze(1)

        # Convert Euler angles to quaternion
        roll, pitch, yaw = angles[:, 0], angles[:, 1], angles[:, 2]
        delta_quat = torch_utils.quat_from_euler_xyz(roll, pitch, yaw)

        # Combine with base orientation
        target_quat = torch_utils.quat_mul(delta_quat, base_quat)

        return target_pos, target_quat

    def check_contact(
        self,
        external_wrench: torch.Tensor,  # (num_envs, 6)
    ) -> torch.Tensor:
        """
        Check if contact force exceeds threshold.

        Args:
            external_wrench: External wrench [fx, fy, fz, tx, ty, tz]

        Returns:
            contact_mask: Boolean tensor indicating contact (num_envs,)
        """
        # Compute force magnitude
        force_magnitude = torch.norm(external_wrench[:, 0:3], dim=1)
        contact_mask = force_magnitude > self.contact_force_threshold

        return contact_mask


def create_impedance_osc_config(
    translational_stiffness: List[float] = None,
    rotational_stiffness: List[float] = None,
    damping_ratio: float = 1.0,
    variable_impedance: bool = True,
    inertial_compensation: bool = True,
    gravity_compensation: bool = True,
) -> OperationalSpaceControllerCfg:
    """
    Create configuration for impedance control using OperationalSpaceController.

    Args:
        translational_stiffness: [kx, ky, kz] in N/m
        rotational_stiffness: [kr, kp, ky] in Nm/rad
        damping_ratio: Damping ratio (1.0 = critical damping)
        variable_impedance: Enable variable impedance mode
        inertial_compensation: Enable inertial dynamics decoupling
        gravity_compensation: Enable gravity compensation

    Returns:
        Configuration for operational space controller with impedance control
    """
    # Default stiffness values for compliant contact
    if translational_stiffness is None:
        translational_stiffness = [200.0, 200.0, 300.0]  # N/m (low stiffness)
    if rotational_stiffness is None:
        rotational_stiffness = [10.0, 10.0, 10.0]  # Nm/rad

    stiffness = translational_stiffness + rotational_stiffness

    cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],  # Absolute pose target
        impedance_mode="variable_kp" if variable_impedance else "fixed",
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],  # Control all 6 DOF
        # Stiffness parameters
        motion_stiffness_task=stiffness,
        motion_damping_ratio_task=damping_ratio,
        # Variable impedance limits
        motion_stiffness_limits_task=(10.0, 1000.0) if variable_impedance else (100.0, 100.0),
        motion_damping_ratio_limits_task=(0.1, 2.0) if variable_impedance else (1.0, 1.0),
        # Dynamics compensation
        inertial_dynamics_decoupling=inertial_compensation,
        partial_inertial_dynamics_decoupling=False,  # Full coupling
        gravity_compensation=gravity_compensation,
        # Null-space control to maintain preferred joint configuration
        nullspace_control="position",
        nullspace_stiffness=10.0,
        nullspace_damping_ratio=1.0,
    )

    return cfg


class ContactManifoldDataCollector:
    """
    Data collector for constructing contact manifolds from disassembly trajectories.

    Stores pose and wrench data at contact points to build a discrete representation
    of the contact manifold M.
    """

    def __init__(
        self,
        max_samples_per_env: int = 10000,
        device: str = "cuda:0",
    ):
        self.max_samples_per_env = max_samples_per_env
        self.device = device

        # Storage for contact manifold samples
        self.reset_data()

    def reset_data(self):
        """Clear all collected data."""
        # Each sample contains: [pos (3), quat (4), wrench (6), timestamp (1)]
        self.samples = {}  # Dict[env_id] -> List[Dict]

    def add_sample(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,  # (num_envs, 3)
        orientations: torch.Tensor,  # (num_envs, 4) quaternions
        wrenches: torch.Tensor,  # (num_envs, 6)
        contact_mask: torch.Tensor,  # (num_envs,) boolean
        timestamp: float,
    ):
        """
        Add samples to the contact manifold for environments in contact.

        Args:
            env_ids: Environment indices
            positions: End-effector positions
            orientations: End-effector orientations (quaternions)
            wrenches: External wrenches
            contact_mask: Boolean mask indicating which envs are in contact
            timestamp: Current simulation time
        """
        # Filter only environments in contact
        contact_env_indices = torch.where(contact_mask)[0]

        for idx in contact_env_indices:
            env_id = env_ids[idx].item()

            if env_id not in self.samples:
                self.samples[env_id] = []

            # Check storage limit
            if len(self.samples[env_id]) >= self.max_samples_per_env:
                continue

            # Create sample dictionary
            sample = {
                'position': positions[idx].cpu().numpy(),
                'orientation': orientations[idx].cpu().numpy(),
                'wrench': wrenches[idx].cpu().numpy(),
                'timestamp': timestamp,
            }

            self.samples[env_id].append(sample)

    def get_manifold_samples(self, env_id: int) -> Optional[Dict]:
        """
        Retrieve all contact manifold samples for a specific environment.

        Returns:
            Dictionary with keys: 'positions', 'orientations', 'wrenches', 'timestamps'
            Each value is a numpy array of shape (num_samples, dim)
        """
        if env_id not in self.samples or len(self.samples[env_id]) == 0:
            return None

        samples = self.samples[env_id]

        return {
            'positions': np.array([s['position'] for s in samples]),
            'orientations': np.array([s['orientation'] for s in samples]),
            'wrenches': np.array([s['wrench'] for s in samples]),
            'timestamps': np.array([s['timestamp'] for s in samples]),
        }

    def get_num_samples(self, env_id: int) -> int:
        """Get number of samples collected for an environment."""
        if env_id not in self.samples:
            return 0
        return len(self.samples[env_id])

    def export_manifold(self, env_id: int, filepath: str):
        """Export contact manifold to file (npz format)."""
        manifold_data = self.get_manifold_samples(env_id)

        if manifold_data is None:
            print(f"No manifold data available for env {env_id}")
            return

        np.savez(
            filepath,
            positions=manifold_data['positions'],
            orientations=manifold_data['orientations'],
            wrenches=manifold_data['wrenches'],
            timestamps=manifold_data['timestamps'],
        )

        print(f"Exported {len(manifold_data['positions'])} samples to {filepath}")

    def export_all_manifolds(self, base_filepath: str):
        """Export all collected manifolds."""
        for env_id in self.samples.keys():
            filepath = f"{base_filepath}_env{env_id}.npz"
            self.export_manifold(env_id, filepath)
