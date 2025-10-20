# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visualization and analysis tools for contact manifolds.

This script provides utilities to:
- Visualize contact manifolds in 3D
- Analyze contact force distributions
- Export manifold representations for pose estimation
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from pathlib import Path


def load_manifold(filepath: str) -> dict:
    """Load contact manifold from npz file."""
    data = np.load(filepath)
    return {
        'positions': data['positions'],
        'orientations': data['orientations'],
        'wrenches': data['wrenches'],
        'timestamps': data['timestamps'],
    }


def visualize_manifold_3d(manifold: dict, show_forces: bool = True, save_path: str = None):
    """
    Visualize contact manifold in 3D space.

    Args:
        manifold: Dictionary containing manifold data
        show_forces: Whether to show force vectors
        save_path: Optional path to save figure
    """
    positions = manifold['positions']
    wrenches = manifold['wrenches']

    fig = plt.figure(figsize=(15, 5))

    # Plot 1: Position scatter (colored by force magnitude)
    ax1 = fig.add_subplot(131, projection='3d')
    force_mag = np.linalg.norm(wrenches[:, 0:3], axis=1)

    scatter = ax1.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=force_mag,
        cmap='viridis',
        s=20,
        alpha=0.6
    )
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Contact Manifold (colored by force)')
    plt.colorbar(scatter, ax=ax1, label='Force magnitude (N)')

    # Plot 2: XY projection (spiral trajectory)
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(
        positions[:, 0],
        positions[:, 1],
        c=positions[:, 2],  # Color by Z (extraction depth)
        cmap='plasma',
        s=20,
        alpha=0.6
    )
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Projection (colored by Z)')
    ax2.axis('equal')
    plt.colorbar(scatter2, ax=ax2, label='Z (m)')

    # Plot 3: Force distribution over time
    ax3 = fig.add_subplot(133)
    timestamps = manifold['timestamps']
    ax3.plot(timestamps, force_mag, 'b-', alpha=0.7, label='Force magnitude')
    ax3.fill_between(timestamps, 0, force_mag, alpha=0.3)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Force (N)')
    ax3.set_title('Contact Force Evolution')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def visualize_orientation_distribution(manifold: dict, save_path: str = None):
    """
    Visualize orientation distribution of contact configurations.

    Args:
        manifold: Dictionary containing manifold data
        save_path: Optional path to save figure
    """
    orientations = manifold['orientations']  # Quaternions (w, x, y, z)

    # Convert quaternions to Euler angles
    rotations = Rotation.from_quat(orientations)
    euler_angles = rotations.as_euler('xyz', degrees=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot roll, pitch, yaw distributions
    labels = ['Roll', 'Pitch', 'Yaw']
    colors = ['red', 'green', 'blue']

    for i in range(3):
        ax = axes.flat[i]
        ax.hist(euler_angles[:, i], bins=50, color=colors[i], alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{labels[i]} (degrees)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{labels[i]} Distribution')
        ax.grid(True, alpha=0.3)

    # Plot 2D scatter of roll vs pitch
    ax = axes.flat[3]
    scatter = ax.scatter(
        euler_angles[:, 0],  # Roll
        euler_angles[:, 1],  # Pitch
        c=euler_angles[:, 2],  # Yaw
        cmap='twilight',
        s=20,
        alpha=0.6
    )
    ax.set_xlabel('Roll (degrees)')
    ax.set_ylabel('Pitch (degrees)')
    ax.set_title('Roll vs Pitch (colored by Yaw)')
    plt.colorbar(scatter, ax=ax, label='Yaw (degrees)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved orientation visualization to {save_path}")

    plt.show()


def analyze_manifold_statistics(manifold: dict):
    """
    Compute and print statistics about the contact manifold.

    Args:
        manifold: Dictionary containing manifold data
    """
    positions = manifold['positions']
    orientations = manifold['orientations']
    wrenches = manifold['wrenches']

    print("\n" + "=" * 60)
    print("Contact Manifold Statistics")
    print("=" * 60)

    # Position statistics
    print("\nPosition Statistics (m):")
    print(f"  X: mean={positions[:, 0].mean():.6f}, std={positions[:, 0].std():.6f}")
    print(f"     range=[{positions[:, 0].min():.6f}, {positions[:, 0].max():.6f}]")
    print(f"  Y: mean={positions[:, 1].mean():.6f}, std={positions[:, 1].std():.6f}")
    print(f"     range=[{positions[:, 1].min():.6f}, {positions[:, 1].max():.6f}]")
    print(f"  Z: mean={positions[:, 2].mean():.6f}, std={positions[:, 2].std():.6f}")
    print(f"     range=[{positions[:, 2].min():.6f}, {positions[:, 2].max():.6f}]")

    # Orientation statistics
    rotations = Rotation.from_quat(orientations)
    euler_angles = rotations.as_euler('xyz', degrees=True)

    print("\nOrientation Statistics (degrees):")
    print(f"  Roll:  mean={euler_angles[:, 0].mean():.2f}, std={euler_angles[:, 0].std():.2f}")
    print(f"  Pitch: mean={euler_angles[:, 1].mean():.2f}, std={euler_angles[:, 1].std():.2f}")
    print(f"  Yaw:   mean={euler_angles[:, 2].mean():.2f}, std={euler_angles[:, 2].std():.2f}")

    # Force statistics
    forces = wrenches[:, 0:3]
    torques = wrenches[:, 3:6]
    force_mag = np.linalg.norm(forces, axis=1)
    torque_mag = np.linalg.norm(torques, axis=1)

    print("\nForce Statistics (N):")
    print(f"  Magnitude: mean={force_mag.mean():.3f}, std={force_mag.std():.3f}")
    print(f"             range=[{force_mag.min():.3f}, {force_mag.max():.3f}]")
    print(f"  Fx: mean={forces[:, 0].mean():.3f}, std={forces[:, 0].std():.3f}")
    print(f"  Fy: mean={forces[:, 1].mean():.3f}, std={forces[:, 1].std():.3f}")
    print(f"  Fz: mean={forces[:, 2].mean():.3f}, std={forces[:, 2].std():.3f}")

    print("\nTorque Statistics (Nm):")
    print(f"  Magnitude: mean={torque_mag.mean():.3f}, std={torque_mag.std():.3f}")
    print(f"             range=[{torque_mag.min():.3f}, {torque_mag.max():.3f}]")

    # Data coverage
    print("\nData Coverage:")
    print(f"  Total samples: {len(positions)}")
    print(f"  Duration: {manifold['timestamps'][-1] - manifold['timestamps'][0]:.2f} s")
    print(f"  Sampling rate: {len(positions) / (manifold['timestamps'][-1] - manifold['timestamps'][0]):.1f} Hz")

    print("=" * 60 + "\n")


def export_for_registration(manifold: dict, output_path: str):
    """
    Export manifold in format suitable for pose estimation/registration.

    Args:
        manifold: Dictionary containing manifold data
        output_path: Path to save processed manifold
    """
    positions = manifold['positions']
    orientations = manifold['orientations']

    # Create compact representation: [x, y, z, qw, qx, qy, qz]
    manifold_compact = np.hstack([positions, orientations])

    # Save as text file for easy loading
    np.savetxt(
        output_path,
        manifold_compact,
        header='x y z qw qx qy qz',
        fmt='%.8f'
    )

    print(f"Exported manifold for registration to {output_path}")


def main():
    """Main visualization script."""
    parser = argparse.ArgumentParser(description="Visualize contact manifolds")
    parser.add_argument("manifold_file", type=str, help="Path to manifold .npz file")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--export_registration", action="store_true",
                        help="Export manifold for registration")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifold
    print(f"Loading manifold from {args.manifold_file}...")
    manifold = load_manifold(args.manifold_file)

    # Print statistics
    analyze_manifold_statistics(manifold)

    # Generate visualizations
    base_name = Path(args.manifold_file).stem

    print("Generating 3D visualization...")
    visualize_manifold_3d(
        manifold,
        save_path=output_dir / f"{base_name}_3d.png"
    )

    print("Generating orientation visualization...")
    visualize_orientation_distribution(
        manifold,
        save_path=output_dir / f"{base_name}_orientations.png"
    )

    # Export for registration if requested
    if args.export_registration:
        export_path = output_dir / f"{base_name}_registration.txt"
        export_for_registration(manifold, str(export_path))

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
