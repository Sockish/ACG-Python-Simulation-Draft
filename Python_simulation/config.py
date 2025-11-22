"""Global configuration objects for the SPH project.

Keeping all tunable parameters centralized makes it easier to tweak
simulation behavior, export cadence, and Blender integration without
hunting through multiple files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class SimulationConfig:
    """Container for all physics and export settings."""

    particle_count: int = 5000
    time_step: float = 0.005
    kernel_radius: float = 0.065
    mass: float = 0.024
    rest_density: float = 1000.0
    gas_constant: float = 1000.0
    viscosity: float = 0.001
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    domain_min: Tuple[float, float, float] = (-0.5, -0.5, 0.0)
    domain_max: Tuple[float, float, float] = (0.5, 0.5, 1.0)
    boundary_damping: float = -0.5

    export_every_n_steps: int = 2
    total_steps: int = 1200

    cache_dir: Path = field(default_factory=lambda: Path("output/cache"))
    ply_dir: Path = field(default_factory=lambda: Path("output/ply"))
    mesh_dir: Path = field(default_factory=lambda: Path("output/mesh"))
    render_dir: Path = field(default_factory=lambda: Path("output/render"))

    blender_executable: str = "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe"
    blender_script_particles: Path = Path("blender/render_particles.py")
    blender_script_surface: Path = Path("blender/render_surface.py")

    reconstruction_grid_resolution: int = 64
    reconstruction_radius_scale: float = 2.0

    def ensure_directories(self) -> None:
        for directory in (self.cache_dir, self.ply_dir, self.mesh_dir, self.render_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def to_numpy(self, value: Tuple[float, float, float]) -> np.ndarray:
        return np.array(value, dtype=np.float32)


@dataclass
class ScenePreset:
    """Simple recipes for initial particle blocks and boundaries."""

    fluid_block_min: Tuple[float, float, float]
    fluid_block_max: Tuple[float, float, float]
    obstacle_center: Tuple[float, float, float] | None = None
    obstacle_radius: float = 0.0


PRESETS = {
    "dam_break": ScenePreset(
        fluid_block_min=(-0.3, -0.3, 0.0),
        fluid_block_max=(-0.05, 0.3, 0.6),
    ),
    "drop": ScenePreset(
        fluid_block_min=(-0.1, -0.1, 0.4),
        fluid_block_max=(0.1, 0.1, 0.7),
        obstacle_center=(0.0, 0.0, 0.05),
        obstacle_radius=0.1,
    ),
}

