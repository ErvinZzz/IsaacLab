# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run contact manifold sampling during disassembly.

This script performs systematic exploration of contact configurations using
spiral trajectories, oscillatory rotations, and compliant impedance control.

Example usage:
    python run_contact_manifold_sampling.py --num_envs 64 --max_iterations 1000
"""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Contact Manifold Sampling for Disassembly")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum simulation iterations")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
parser.add_argument("--assembly_id", type=str, default="00015", help="Assembly ID to sample")
parser.add_argument("--save_dir", type=str, default="./contact_manifolds", help="Directory to save results")

# Sampling parameters
parser.add_argument("--spiral_radius", type=float, default=0.005, help="Max spiral radius (m)")
parser.add_argument("--spiral_freq", type=float, default=0.5, help="Spiral frequency (Hz)")
parser.add_argument("--extraction_vel", type=float, default=0.002, help="Extraction velocity (m/s)")
parser.add_argument("--rot_amplitude", type=float, nargs=3, default=[0.05, 0.05, 0.1],
                    help="Rotation amplitudes [roll, pitch, yaw] (rad)")
parser.add_argument("--rot_freq", type=float, nargs=3, default=[0.3, 0.4, 0.2],
                    help="Rotation frequencies (Hz)")

# Impedance parameters
parser.add_argument("--trans_stiffness", type=float, nargs=3, default=[200.0, 200.0, 300.0],
                    help="Translational stiffness [x, y, z] (N/m)")
parser.add_argument("--rot_stiffness", type=float, nargs=3, default=[10.0, 10.0, 10.0],
                    help="Rotational stiffness [roll, pitch, yaw] (Nm/rad)")
parser.add_argument("--damping_ratio", type=float, default=1.0, help="Damping ratio")

# Contact detection
parser.add_argument("--contact_threshold", type=float, default=1.0, help="Contact force threshold (N)")

args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli={"headless": args.headless})
simulation_app = app_launcher.app

# Import after launching
from isaaclab_tasks.direct.automate.contact_manifold_env import ContactManifoldEnv
from isaaclab_tasks.direct.automate.disassembly_env_cfg import DisassemblyEnvCfg


def main():
    """Run contact manifold sampling."""

    # Create environment configuration
    env_cfg = DisassemblyEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.task_name = "extraction"

    # Update assembly ID
    env_cfg.tasks["extraction"].assembly_id = args.assembly_id

    print("=" * 80)
    print("Contact Manifold Sampling for Disassembly")
    print("=" * 80)
    print(f"Assembly ID: {args.assembly_id}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Device: {args.device}")
    print(f"Spiral radius: {args.spiral_radius} m")
    print(f"Extraction velocity: {args.extraction_vel} m/s")
    print(f"Rotation amplitudes: {args.rot_amplitude} rad")
    print(f"Translational stiffness: {args.trans_stiffness} N/m")
    print(f"Rotational stiffness: {args.rot_stiffness} Nm/rad")
    print(f"Contact threshold: {args.contact_threshold} N")
    print("=" * 80)

    # Create environment
    env = ContactManifoldEnv(
        cfg=env_cfg,
        # Sampling parameters
        spiral_radius_max=args.spiral_radius,
        spiral_frequency=args.spiral_freq,
        extraction_velocity=args.extraction_vel,
        rotation_amplitude=args.rot_amplitude,
        rotation_frequency=args.rot_freq,
        # Impedance parameters
        translational_stiffness=args.trans_stiffness,
        rotational_stiffness=args.rot_stiffness,
        damping_ratio=args.damping_ratio,
        # Contact detection
        contact_force_threshold=args.contact_threshold,
        # Data collection
        save_directory=args.save_dir,
    )

    # Reset environment
    env.reset()

    print("\nStarting contact manifold sampling...")
    print(f"Running for {args.max_iterations} iterations...\n")

    # Run simulation
    for iteration in range(args.max_iterations):
        # Dummy action (controller handles everything)
        action = torch.zeros((args.num_envs, env.action_space.shape[0]), device=args.device)

        # Step environment
        obs, rewards, terminations, truncations, infos = env.step(action)

        # Print progress
        if iteration % 100 == 0:
            stats = env.get_manifold_statistics()
            print(f"Iteration {iteration}/{args.max_iterations}")
            print(f"  Total contact samples: {stats['total_samples']}")
            if stats['total_samples'] > 0:
                print(f"  Mean samples per env: {stats.get('mean_samples', 0):.1f}")
                print(f"  Envs with data: {stats['num_envs_with_data']}/{args.num_envs}")

        # Check if we should stop (e.g., all envs done)
        dones = terminations | truncations
        if dones.all():
            print("\nAll environments completed disassembly!")
            break

    # Save manifold data
    print("\nSaving contact manifolds...")
    env.save_manifolds()

    # Print final statistics
    print("\nFinal Statistics:")
    stats = env.get_manifold_statistics()
    print(f"  Total contact samples: {stats['total_samples']}")
    if stats['total_samples'] > 0:
        print(f"  Mean samples per env: {stats.get('mean_samples', 0):.1f} ± {stats.get('std_samples', 0):.1f}")
        print(f"  Min/Max samples: {stats.get('min_samples', 0)} / {stats.get('max_samples', 0)}")
        print(f"  Envs with data: {stats['num_envs_with_data']}/{args.num_envs}")

    # Close environment
    env.close()

    print("\nContact manifold sampling completed!")
    print(f"Data saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
