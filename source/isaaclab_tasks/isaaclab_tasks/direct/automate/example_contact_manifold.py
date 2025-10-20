# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quick-start example for contact manifold sampling.

This demonstrates how to:
1. Set up the environment with custom parameters
2. Run sampling with different configurations
3. Collect and save manifold data
4. Analyze results

Run with:
    python example_contact_manifold.py
"""

import torch
from isaaclab.app import AppLauncher

# Launch Isaac Sim (headless for faster sampling)
app_launcher = AppLauncher(args_cli={"headless": True})
simulation_app = app_launcher.app

# Import after launching
from isaaclab_tasks.direct.automate.contact_manifold_env import ContactManifoldEnv
from isaaclab_tasks.direct.automate.contact_manifold_cfg import StandardConfig, get_config_by_name
from isaaclab_tasks.direct.automate.disassembly_env_cfg import DisassemblyEnvCfg


def example_basic_sampling():
    """Example 1: Basic sampling with default parameters."""
    print("\n" + "="*80)
    print("Example 1: Basic Contact Manifold Sampling")
    print("="*80)

    # Create environment config
    env_cfg = DisassemblyEnvCfg()
    env_cfg.scene.num_envs = 16  # Small number for quick demo
    env_cfg.task_name = "extraction"

    # Create environment with standard configuration
    cfg = StandardConfig()
    env = ContactManifoldEnv(
        cfg=env_cfg,
        spiral_radius_max=cfg.sampling.spiral_radius_max,
        spiral_frequency=cfg.sampling.spiral_frequency,
        extraction_velocity=cfg.sampling.extraction_velocity,
        rotation_amplitude=cfg.sampling.rotation_amplitude,
        rotation_frequency=cfg.sampling.rotation_frequency,
        translational_stiffness=cfg.impedance.translational_stiffness,
        rotational_stiffness=cfg.impedance.rotational_stiffness,
        damping_ratio=cfg.impedance.damping_ratio,
        contact_force_threshold=cfg.contact.force_threshold,
        save_directory="./example_manifolds_basic",
    )

    # Reset environment
    obs = env.reset()
    print(f"Environment initialized with {env_cfg.scene.num_envs} parallel environments")

    # Run sampling
    num_iterations = 200
    print(f"Running {num_iterations} iterations...")

    for i in range(num_iterations):
        # Zero action (controller handles everything)
        action = torch.zeros((env_cfg.scene.num_envs, env.action_space.shape[0]), device=env.device)

        # Step simulation
        obs, rewards, terminations, truncations, infos = env.step(action)

        # Print progress
        if i % 50 == 0:
            stats = env.get_manifold_statistics()
            print(f"  Iteration {i}: {stats['total_samples']} contact samples collected")

    # Save and analyze
    print("\nSaving manifold data...")
    env.save_manifolds()

    stats = env.get_manifold_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Envs with data: {stats['num_envs_with_data']}/{env_cfg.scene.num_envs}")

    env.close()
    print("Example 1 complete!\n")


def example_custom_parameters():
    """Example 2: Custom sampling parameters for specific requirements."""
    print("\n" + "="*80)
    print("Example 2: Custom Parameters for Tight Clearance")
    print("="*80)

    env_cfg = DisassemblyEnvCfg()
    env_cfg.scene.num_envs = 8

    # Custom parameters for tight clearance (more conservative)
    env = ContactManifoldEnv(
        cfg=env_cfg,
        # Smaller spiral for tight fit
        spiral_radius_max=0.002,  # 2mm instead of 5mm
        spiral_frequency=0.3,
        # Slower extraction
        extraction_velocity=0.0008,  # 0.8mm/s
        # Smaller rotation amplitudes
        rotation_amplitude=[0.02, 0.02, 0.04],  # ~1.1°, ~1.1°, ~2.3°
        rotation_frequency=[0.2, 0.25, 0.15],
        # More compliant
        translational_stiffness=[100.0, 100.0, 150.0],
        rotational_stiffness=[5.0, 5.0, 5.0],
        damping_ratio=1.2,  # Over-damped for stability
        # More sensitive contact detection
        contact_force_threshold=0.3,
        save_directory="./example_manifolds_tight",
    )

    print("Custom parameters configured:")
    print("  Spiral radius: 2mm (tight clearance)")
    print("  Extraction: 0.8 mm/s (slow)")
    print("  Stiffness: Low (compliant)")
    print("  Contact threshold: 0.3 N (sensitive)")

    obs = env.reset()

    # Run shorter demo
    for i in range(100):
        action = torch.zeros((env_cfg.scene.num_envs, env.action_space.shape[0]), device=env.device)
        obs, rewards, terminations, truncations, infos = env.step(action)

    env.save_manifolds()
    print("Example 2 complete!\n")


def example_using_presets():
    """Example 3: Using configuration presets."""
    print("\n" + "="*80)
    print("Example 3: Using Configuration Presets")
    print("="*80)

    # Try different presets
    preset_name = "dense"  # Options: conservative, standard, aggressive, dense

    print(f"Using '{preset_name}' preset configuration")

    env_cfg = DisassemblyEnvCfg()
    env_cfg.scene.num_envs = 4

    # Load preset
    PresetConfig = get_config_by_name(preset_name)
    cfg = PresetConfig()

    print(f"\nPreset parameters:")
    print(f"  Extraction: {cfg.sampling.extraction_velocity*1000:.1f} mm/s")
    print(f"  Spiral: {cfg.sampling.spiral_radius_max*1000:.1f} mm @ {cfg.sampling.spiral_frequency:.1f} Hz")
    print(f"  Stiffness: {cfg.impedance.translational_stiffness} N/m")
    print(f"  Max samples: {cfg.data.max_samples_per_env}")

    env = ContactManifoldEnv(
        cfg=env_cfg,
        spiral_radius_max=cfg.sampling.spiral_radius_max,
        spiral_frequency=cfg.sampling.spiral_frequency,
        extraction_velocity=cfg.sampling.extraction_velocity,
        rotation_amplitude=cfg.sampling.rotation_amplitude,
        rotation_frequency=cfg.sampling.rotation_frequency,
        translational_stiffness=cfg.impedance.translational_stiffness,
        rotational_stiffness=cfg.impedance.rotational_stiffness,
        damping_ratio=cfg.impedance.damping_ratio,
        contact_force_threshold=cfg.contact.force_threshold,
        save_directory=cfg.data.save_directory,
        max_samples_per_env=cfg.data.max_samples_per_env,
    )

    obs = env.reset()

    for i in range(150):
        action = torch.zeros((env_cfg.scene.num_envs, env.action_space.shape[0]), device=env.device)
        obs, rewards, terminations, truncations, infos = env.step(action)

    env.save_manifolds()
    print("Example 3 complete!\n")


def example_multi_stage_sampling():
    """Example 4: Multi-stage sampling with parameter adaptation."""
    print("\n" + "="*80)
    print("Example 4: Multi-Stage Adaptive Sampling")
    print("="*80)

    env_cfg = DisassemblyEnvCfg()
    env_cfg.scene.num_envs = 8

    print("Stage 1: Coarse sampling (fast)")

    # Stage 1: Coarse, fast sampling
    env = ContactManifoldEnv(
        cfg=env_cfg,
        spiral_radius_max=0.008,
        extraction_velocity=0.004,  # Fast
        translational_stiffness=[300.0, 300.0, 400.0],  # Stiff
        contact_force_threshold=2.0,  # Less sensitive
        save_directory="./example_manifolds_stage1",
    )

    obs = env.reset()

    for i in range(50):
        action = torch.zeros((env_cfg.scene.num_envs, env.action_space.shape[0]), device=env.device)
        obs, rewards, terminations, truncations, infos = env.step(action)

    stats1 = env.get_manifold_statistics()
    print(f"  Stage 1 samples: {stats1['total_samples']}")
    env.save_manifolds()
    env.close()

    print("\nStage 2: Fine sampling (slow, compliant)")

    # Stage 2: Fine, compliant sampling
    env = ContactManifoldEnv(
        cfg=env_cfg,
        spiral_radius_max=0.003,  # Smaller
        extraction_velocity=0.001,  # Slower
        translational_stiffness=[100.0, 100.0, 150.0],  # Compliant
        contact_force_threshold=0.5,  # More sensitive
        save_directory="./example_manifolds_stage2",
    )

    obs = env.reset()

    for i in range(100):
        action = torch.zeros((env_cfg.scene.num_envs, env.action_space.shape[0]), device=env.device)
        obs, rewards, terminations, truncations, infos = env.step(action)

    stats2 = env.get_manifold_statistics()
    print(f"  Stage 2 samples: {stats2['total_samples']}")
    env.save_manifolds()
    env.close()

    print(f"\nTotal samples across both stages: {stats1['total_samples'] + stats2['total_samples']}")
    print("Example 4 complete!\n")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Contact Manifold Sampling - Quick Start Examples")
    print("="*80)

    # Run examples
    try:
        example_basic_sampling()
        example_custom_parameters()
        example_using_presets()
        example_multi_stage_sampling()

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        print("\nNext steps:")
        print("1. Visualize collected manifolds:")
        print("   python visualize_contact_manifold.py example_manifolds_basic/manifold_assembly_00015_env0.npz")
        print("\n2. Run full sampling:")
        print("   python run_contact_manifold_sampling.py --num_envs 128 --max_iterations 1000")
        print("\n3. Customize for your specific assembly:")
        print("   - Adjust clearance-dependent parameters")
        print("   - Tune impedance for material properties")
        print("   - Set appropriate contact thresholds")
        print("="*80)

    except Exception as e:
        print(f"\nError during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()
