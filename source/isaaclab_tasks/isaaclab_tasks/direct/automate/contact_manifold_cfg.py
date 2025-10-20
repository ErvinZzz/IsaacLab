# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for contact manifold sampling.

This module provides pre-configured parameter sets for different scenarios:
- Conservative: Safe, slow sampling for delicate parts
- Standard: Balanced parameters for typical assemblies
- Aggressive: Fast sampling for robust parts
- Dense: Maximum coverage with fine-grained sampling
"""

from dataclasses import dataclass
from typing import List


@dataclass
class SamplingConfig:
    """Configuration for trajectory sampling parameters."""

    # Spiral parameters
    spiral_radius_max: float
    spiral_frequency: float
    spiral_growth_rate: float

    # Oscillatory rotation
    rotation_amplitude: List[float]  # [roll, pitch, yaw] rad
    rotation_frequency: List[float]  # Hz

    # Linear extraction
    extraction_velocity: float  # m/s

    # Noise for bias reduction
    position_noise_std: float
    rotation_noise_std: float


@dataclass
class ImpedanceConfig:
    """Configuration for impedance control parameters."""

    # Stiffness
    translational_stiffness: List[float]  # [x, y, z] N/m
    rotational_stiffness: List[float]     # [roll, pitch, yaw] Nm/rad

    # Damping
    damping_ratio: float

    # Contact adaptation
    contact_stiffness_scale: float  # Multiplier during contact (0-1)


@dataclass
class ContactDetectionConfig:
    """Configuration for contact detection."""

    force_threshold: float  # N
    torque_threshold: float  # Nm
    use_force_only: bool  # Ignore torque for contact detection


@dataclass
class DataCollectionConfig:
    """Configuration for data collection."""

    max_samples_per_env: int
    save_directory: str
    export_format: str  # 'npz', 'json', 'hdf5'
    save_interval: int  # Save every N samples


# ============================================================================
# Preset Configurations
# ============================================================================

class ConservativeConfig:
    """
    Conservative configuration for safe, compliant sampling.

    Use for:
    - Delicate parts
    - First-time sampling of unknown geometries
    - Small clearances (<0.1mm)
    """

    sampling = SamplingConfig(
        spiral_radius_max=0.003,      # 3mm max
        spiral_frequency=0.3,          # Slow spiral
        spiral_growth_rate=0.001,      # 1mm/s growth
        rotation_amplitude=[0.03, 0.03, 0.05],  # ~1.7°, ~1.7°, ~2.9°
        rotation_frequency=[0.2, 0.25, 0.15],   # Slow oscillations
        extraction_velocity=0.001,     # 1mm/s
        position_noise_std=0.0003,     # 0.3mm
        rotation_noise_std=0.005,      # ~0.29°
    )

    impedance = ImpedanceConfig(
        translational_stiffness=[100.0, 100.0, 150.0],  # Very soft
        rotational_stiffness=[5.0, 5.0, 5.0],           # Very soft
        damping_ratio=1.2,                              # Over-damped
        contact_stiffness_scale=0.2,                    # 20% during contact
    )

    contact = ContactDetectionConfig(
        force_threshold=0.5,      # Sensitive detection
        torque_threshold=0.05,
        use_force_only=True,
    )

    data = DataCollectionConfig(
        max_samples_per_env=20000,
        save_directory="./manifolds_conservative",
        export_format='npz',
        save_interval=1000,
    )


class StandardConfig:
    """
    Standard configuration for typical peg-in-hole assemblies.

    Use for:
    - Standard peg-in-hole tasks
    - Medium clearances (0.1-0.5mm)
    - General purpose sampling
    """

    sampling = SamplingConfig(
        spiral_radius_max=0.005,      # 5mm max
        spiral_frequency=0.5,          # Standard spiral
        spiral_growth_rate=0.002,      # 2mm/s growth
        rotation_amplitude=[0.05, 0.05, 0.1],  # ~2.9°, ~2.9°, ~5.7°
        rotation_frequency=[0.3, 0.4, 0.2],     # Standard oscillations
        extraction_velocity=0.002,     # 2mm/s
        position_noise_std=0.0005,     # 0.5mm
        rotation_noise_std=0.01,       # ~0.57°
    )

    impedance = ImpedanceConfig(
        translational_stiffness=[200.0, 200.0, 300.0],  # Medium compliance
        rotational_stiffness=[10.0, 10.0, 10.0],        # Medium compliance
        damping_ratio=1.0,                              # Critical damping
        contact_stiffness_scale=0.3,                    # 30% during contact
    )

    contact = ContactDetectionConfig(
        force_threshold=1.0,      # Standard detection
        torque_threshold=0.1,
        use_force_only=True,
    )

    data = DataCollectionConfig(
        max_samples_per_env=10000,
        save_directory="./manifolds_standard",
        export_format='npz',
        save_interval=500,
    )


class AggressiveConfig:
    """
    Aggressive configuration for fast sampling.

    Use for:
    - Robust parts (metal, high strength)
    - Large clearances (>0.5mm)
    - Quick manifold sketching
    """

    sampling = SamplingConfig(
        spiral_radius_max=0.008,      # 8mm max
        spiral_frequency=1.0,          # Fast spiral
        spiral_growth_rate=0.005,      # 5mm/s growth
        rotation_amplitude=[0.08, 0.08, 0.15],  # ~4.6°, ~4.6°, ~8.6°
        rotation_frequency=[0.5, 0.6, 0.4],     # Fast oscillations
        extraction_velocity=0.005,     # 5mm/s
        position_noise_std=0.001,      # 1mm
        rotation_noise_std=0.02,       # ~1.15°
    )

    impedance = ImpedanceConfig(
        translational_stiffness=[400.0, 400.0, 600.0],  # Stiff
        rotational_stiffness=[20.0, 20.0, 20.0],        # Stiff
        damping_ratio=0.8,                              # Under-damped
        contact_stiffness_scale=0.5,                    # 50% during contact
    )

    contact = ContactDetectionConfig(
        force_threshold=2.0,      # Less sensitive
        torque_threshold=0.2,
        use_force_only=True,
    )

    data = DataCollectionConfig(
        max_samples_per_env=5000,
        save_directory="./manifolds_aggressive",
        export_format='npz',
        save_interval=250,
    )


class DenseConfig:
    """
    Dense sampling configuration for maximum coverage.

    Use for:
    - Detailed manifold reconstruction
    - Complex geometries (non-circular)
    - Research/analysis purposes
    """

    sampling = SamplingConfig(
        spiral_radius_max=0.006,      # 6mm max
        spiral_frequency=0.8,          # Dense spiral
        spiral_growth_rate=0.0015,     # 1.5mm/s slow growth
        rotation_amplitude=[0.06, 0.06, 0.12],  # ~3.4°, ~3.4°, ~6.9°
        rotation_frequency=[0.4, 0.5, 0.3],     # Multiple frequencies
        extraction_velocity=0.0015,    # 1.5mm/s slow
        position_noise_std=0.0008,     # 0.8mm more exploration
        rotation_noise_std=0.015,      # ~0.86° more exploration
    )

    impedance = ImpedanceConfig(
        translational_stiffness=[150.0, 150.0, 250.0],  # Compliant
        rotational_stiffness=[8.0, 8.0, 8.0],           # Compliant
        damping_ratio=1.1,                              # Slightly over-damped
        contact_stiffness_scale=0.25,                   # 25% during contact
    )

    contact = ContactDetectionConfig(
        force_threshold=0.8,      # Sensitive
        torque_threshold=0.08,
        use_force_only=True,
    )

    data = DataCollectionConfig(
        max_samples_per_env=30000,    # Lots of samples
        save_directory="./manifolds_dense",
        export_format='npz',
        save_interval=1500,
    )


# ============================================================================
# Helper Functions
# ============================================================================

def get_config_by_name(name: str):
    """Get configuration by name."""
    configs = {
        'conservative': ConservativeConfig,
        'standard': StandardConfig,
        'aggressive': AggressiveConfig,
        'dense': DenseConfig,
    }

    if name.lower() not in configs:
        raise ValueError(f"Unknown config: {name}. Choose from {list(configs.keys())}")

    return configs[name.lower()]


def print_config_comparison():
    """Print comparison of all configurations."""
    configs = {
        'Conservative': ConservativeConfig,
        'Standard': StandardConfig,
        'Aggressive': AggressiveConfig,
        'Dense': DenseConfig,
    }

    print("\n" + "=" * 80)
    print("Contact Manifold Sampling Configurations Comparison")
    print("=" * 80)

    for name, cfg in configs.items():
        print(f"\n{name}:")
        print(f"  Extraction: {cfg.sampling.extraction_velocity*1000:.1f} mm/s")
        print(f"  Spiral: {cfg.sampling.spiral_radius_max*1000:.1f} mm @ {cfg.sampling.spiral_frequency:.1f} Hz")
        print(f"  Rotation: {[f'{a*57.3:.1f}°' for a in cfg.sampling.rotation_amplitude]}")
        print(f"  Stiffness: {cfg.impedance.translational_stiffness} N/m")
        print(f"  Force threshold: {cfg.contact.force_threshold:.1f} N")
        print(f"  Max samples: {cfg.data.max_samples_per_env}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_config_comparison()
