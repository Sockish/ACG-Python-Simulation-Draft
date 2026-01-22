# Hybrid Simulation - Fluid & Rigid Body Physics

A Python-based physics simulation framework for SPH (Smoothed Particle Hydrodynamics) and MPM (Material Point Method) simulations with rigid body interactions. The project supports both CPU and GPU-accelerated computations using Taichi.

## Features

- **Multiple Solver Options**:
  - SPH (Smoothed Particle Hydrodynamics) for fluid simulation
  - Taichi-accelerated SPH with GPU support for faster computation
  - MPM (Material Point Method) for various materials (water, jelly, snow)
- **Surface Reconstruction**: Convert particle data to meshes using Splashsurf
- **Flexible Configuration**: YAML-based scene configuration
  - customizable asset imports in mesh formats 
  - adjustable simulation parameters (time step, gravity, steps)

- **Rendering Pipeline**: Integration with Blender for high-quality renders

## Prerequisites

- CUDA-capable GPU for Taichi GPU acceleration
- Blender for rendering

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirement.txt
```

## Quick Start

### Basic SPH Simulation (Very Slow. Not Recommended)

Run a standard SPH fluid simulation for 800 steps:
```bash
python scripts/simulate.py --steps 800
```

### GPU-Accelerated Simulation (Recommended)

Use Taichi GPU acceleration for much faster computation:
```bash
python scripts/simulate.py --use-taichi --steps 800
```

### MPM Simulation

Run Material Point Method simulations for different materials:

```bash
python mpm_instances/simulate_mpm4.py
```

## Advanced Workflow

### 1. Configure Your Scene for SPH simulation

Edit `config/scene_config.yaml` to customize:
- Simulation parameters (time step, gravity, step count)
- Fluid properties (density, velocity, particle spacing)
- Rigid/static body meshes
- Import/export directories

MPM solver does not support customized configuration

### 2. Run Simulation

Choose your solver based on needs:
```bash
# SPH with Taichi GPU (fastest for large particle counts)
python scripts/simulate.py --use-taichi --steps 1000

# MPM for elastic/plastic materials
python ./mpm_instances/simulate_mpm1.py
```

### 3. Surface Reconstruction (skip if using mpm solver)

Convert particle data to mesh surfaces (required for smooth rendering):
```bash
python scripts/reconstruct.py --config config/scene_config.yaml --target-fps 60
```


### 4. Render with Blender

Render frames using Blender scenes:
```bash
# Single scene rendering (recommended)
python scripts/render_single.py --scene_file dambreak.blend --input_dir config/outputs_dambreak/show_frames
```

**Available Blender scenes**:
- `dambreak.blend` - Dam break container scene
- `Landscape.blend` - Landscape environment
- `floor.blend` - Simple floor scene

### 5. Create Video

Combine rendered frames into a video using ffmpeg:
```bash
ffmpeg -framerate 60 -i config/outputs_dambreak/renders/%05d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4
```

## Output Directory Structure

```
config/
  └── outputs_<simulation_name>/
      ├── fluid/          # Particle data (.npy files)
      ├── rigid/          # Rigid body states
      ├── static/         # Static mesh data
      ├── frames/         # Reconstructed mesh surfaces
      ├── show_frames/    # Downsampled frames for rendering
      └── renders/        # Final rendered PNG images
```

## Solver Comparison

| Solver | Speed | Best For | GPU Support |
|--------|-------|----------|-------------|
| SPH (CPU) | Slow | Small simulations (<10k particles) | No |
| SPH (Taichi GPU) | Fast | Large fluid simulations | Yes |
| MPM | Medium | Elastic materials, snow, jelly | Yes (via Taichi) |

## Troubleshooting
**Out of memory**: Reduce particle count by increasing `particle_spacing` in scene config

**Slow simulation**: Use `--use-taichi` for GPU acceleration



## Complete Pipeline Example

Here's a complete example from simulation to video:

```bash
# 1. Run simulation (GPU-accelerated)
python scripts/simulate.py --use-taichi --steps 1000

# 2. Reconstruct surfaces
python scripts/reconstruct.py --config config/scene_config.yaml --target-fps 60


# 3. Render with Blender
python scripts/render_single.py --scene_file dambreak.blend --input_dir config/outputs/show_frames

# 4                . Create video
ffmpeg -framerate 60 -i config/outputs/renders/%05d.png -c:v libx264 -pix_fmt yuv420p final_output.mp4
```

## License & Credits

This project uses:
- [Taichi](https://github.com/taichi-dev/taichi) for GPU acceleration
- [Splashsurf](https://github.com/InteractiveComputerGraphics/splashsurf) for surface reconstruction
