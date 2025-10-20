# Contact Manifold Sampling for Peg-in-Hole Disassembly

This module implements a contact manifold sampling system for robotic disassembly tasks. It constructs a reference contact manifold by systematically exploring contact configurations during the disassembly process.

## Overview

The contact manifold represents the geometric relationship between the peg and hole during contact. This "fingerprint" in 6D pose space encodes contact characteristics unique to the part geometry.

### Key Features

- **Spiral Trajectories** in X,Y translation for systematic spatial coverage
- **Oscillatory Trajectories** in roll/pitch/yaw for rotational exploration
- **Linear Extraction** in Z for progressive disassembly
- **Variable Impedance Control** using Isaac Lab's OperationalSpaceController
- **Contact Detection** via force/torque sensing
- **Depth-Adaptive Sampling** to prevent part damage
- **Data Collection** and storage in standard formats

## Components

### 1. ContactManifoldSamplingController
Generates motion primitives combining:
- Spiral motion: `r(t) = min(r_max, k*t)`, `θ(t) = 2πf*t`
- Oscillatory rotation: `angle(t) = A*sin(2πf*t + φ)`
- Linear extraction: `z(t) = v*t`

### 2. Impedance Control via OperationalSpaceController
Uses Isaac Lab's built-in OSC with:
- Variable stiffness (contact-adaptive)
- Inertial dynamics decoupling
- Gravity compensation
- Null-space control for redundancy resolution

### 3. ContactManifoldDataCollector
Stores pose-wrench pairs where contact force exceeds threshold:
```
Sample = {position, orientation, wrench, timestamp} where F_ext > ε_f
```

## Installation

The code is already part of the automate module. No additional installation needed beyond Isaac Lab.

## Usage

### Basic Example

```bash
python run_contact_manifold_sampling.py \
    --num_envs 64 \
    --max_iterations 1000 \
    --assembly_id 00015 \
    --save_dir ./contact_manifolds
```

### Advanced Usage with Custom Parameters

```bash
python run_contact_manifold_sampling.py \
    --num_envs 128 \
    --max_iterations 2000 \
    --assembly_id 00015 \
    --spiral_radius 0.008 \
    --spiral_freq 0.3 \
    --extraction_vel 0.001 \
    --rot_amplitude 0.1 0.1 0.15 \
    --rot_freq 0.2 0.3 0.1 \
    --trans_stiffness 150.0 150.0 250.0 \
    --rot_stiffness 8.0 8.0 8.0 \
    --damping_ratio 0.8 \
    --contact_threshold 0.5 \
    --save_dir ./manifolds_custom
```

### Parameter Guide

**Sampling Parameters:**
- `--spiral_radius`: Maximum radius of spiral in XY (m) [default: 0.005]
- `--spiral_freq`: Frequency of spiral rotation (Hz) [default: 0.5]
- `--extraction_vel`: Linear velocity in Z (m/s) [default: 0.002]
- `--rot_amplitude`: Oscillation amplitudes for [roll, pitch, yaw] (rad) [default: 0.05, 0.05, 0.1]
- `--rot_freq`: Oscillation frequencies (Hz) [default: 0.3, 0.4, 0.2]

**Impedance Parameters:**
- `--trans_stiffness`: Translational stiffness [x, y, z] (N/m) [default: 200, 200, 300]
  - Lower values = more compliant (safer for contact)
  - Higher values = stiffer (better tracking)
- `--rot_stiffness`: Rotational stiffness [roll, pitch, yaw] (Nm/rad) [default: 10, 10, 10]
- `--damping_ratio`: Damping ratio (1.0 = critical) [default: 1.0]

**Contact Detection:**
- `--contact_threshold`: Force magnitude threshold for contact (N) [default: 1.0]

## Visualization

After collecting manifold data, visualize it:

```bash
python visualize_contact_manifold.py \
    contact_manifolds/manifold_assembly_00015_env0.npz \
    --output_dir ./visualizations \
    --export_registration
```

This generates:
- 3D visualization of contact points
- Force distribution analysis
- Orientation statistics
- XY projection showing spiral trajectory
- Export for pose estimation/registration

## Integration with Existing Code

The system extends the existing `DisassemblyEnv`:

```python
from contact_manifold_env import ContactManifoldEnv
from disassembly_env_cfg import DisassemblyEnvCfg

# Create config
cfg = DisassemblyEnvCfg()
cfg.scene.num_envs = 64

# Create environment with contact manifold sampling
env = ContactManifoldEnv(
    cfg=cfg,
    spiral_radius_max=0.005,
    translational_stiffness=[200.0, 200.0, 300.0],
    contact_force_threshold=1.0,
)

# Run sampling
env.reset()
for _ in range(1000):
    action = torch.zeros((num_envs, 6), device="cuda:0")
    obs, reward, done, info = env.step(action)

# Save results
env.save_manifolds()
```

## Data Format

Collected manifolds are saved as `.npz` files containing:

```python
{
    'positions': np.array,     # (N, 3) - end-effector positions
    'orientations': np.array,  # (N, 4) - quaternions (w,x,y,z)
    'wrenches': np.array,      # (N, 6) - force/torque [fx,fy,fz,tx,ty,tz]
    'timestamps': np.array,    # (N,) - simulation time
}
```

## Physical Interpretation

The contact manifold M captures:
- **Geometric constraints**: Physical interference between peg/hole
- **Contact modes**: Different contact configurations (edge, face, vertex)
- **Force signatures**: Characteristic force patterns for each geometry
- **Pose-wrench relationship**: Mapping from configuration to contact forces

This enables:
1. **Pose estimation**: Match observed forces to manifold
2. **Contact-rich manipulation**: Plan motions respecting contact constraints
3. **Sim-to-real transfer**: Physics-informed models of contact

## Franka Torque Sensing

The Franka robot has built-in joint torque sensors. To use actual measured torques instead of simulated ones:

1. Access robot torque data in `ContactManifoldEnv._estimate_external_wrench()`
2. For real robot: Use Franka's FCI (Franka Control Interface) external torque estimates
3. For simulation: The current implementation uses joint torque mapping via Jacobian

## Tuning Guidelines

**For compliant contact exploration:**
- Lower stiffness (50-200 N/m translation)
- Higher damping ratio (1.0-1.5)
- Slower extraction velocity (<0.002 m/s)

**For faster sampling:**
- Higher stiffness (200-500 N/m)
- Standard damping (1.0)
- Faster extraction (0.002-0.005 m/s)

**For dense manifold coverage:**
- Smaller spiral radius (0.002-0.003 m)
- Higher spiral frequency (0.5-1.0 Hz)
- More rotation oscillations (increase amplitudes)

## Troubleshooting

**No contact detected:**
- Check contact threshold (reduce if needed)
- Verify force computation in `_estimate_external_wrench()`
- Ensure proper contact sensors activation

**Unstable contact:**
- Reduce stiffness
- Increase damping ratio
- Slow down extraction velocity

**Poor manifold coverage:**
- Increase sampling duration
- Adjust spiral/oscillation parameters
- Add more random perturbations

## References

1. "A unified approach for motion and force control of robot manipulators: The operational space formulation" - Oussama Khatib
2. "Robot Dynamics Lecture Notes" - Marco Hutter (ETH Zurich)
3. Contact-based manipulation papers for manifold-based pose estimation

## Future Enhancements

- [ ] Multi-stage sampling with varying parameters
- [ ] Adaptive sampling based on force gradients
- [ ] Real-time manifold visualization
- [ ] Integration with pose estimation algorithms
- [ ] Learned sampling strategies
- [ ] Support for non-cylindrical geometries
