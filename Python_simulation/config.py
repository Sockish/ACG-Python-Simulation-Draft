"""Global configuration objects for the SPH project.

Keeping all tunable parameters centralized makes it easier to tweak
simulation behavior, export cadence, and Blender integration without
hunting through multiple files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FluidVolumePreset:
    """Definition for spawning a block of fluid particles."""

    name: str
    block_min: Tuple[float, float, float]
    block_max: Tuple[float, float, float]
    jitter: float = 0.002
    particle_count: int | None = None


@dataclass(frozen=True)
class RigidMeshPreset:
    """Definition for spawning a rigid mesh from disk."""

    name: str
    mesh_path: Path
    density: float = 500.0
    scale: float = 1.0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.1)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    linear_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SimulationConfig:
    """Container for all physics and export settings."""

    particle_count: int = 500
    time_step: float = 0.0025
    kernel_radius: float = 0.065
    mass: float | None = None
    rest_density: float = 1000.0
    gas_constant: float = 1000.0
    viscosity: float = 0.001
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    domain_min: Tuple[float, float, float] = (-0.5, -0.5, 0.0)
    domain_max: Tuple[float, float, float] = (0.5, 0.5, 1.0)
    boundary_damping: float = -0.5

    export_every_n_steps: int = 2
    total_steps: int = 15

    ply_dir: Path = field(default_factory=lambda: Path("output/ply"))
    mesh_dir: Path = field(default_factory=lambda: Path("output/mesh"))
    render_dir: Path = field(default_factory=lambda: Path("output/render"))
    rigid_dir: Path = field(default_factory=lambda: Path("output/rigid"))

    blender_executable: str = "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe"
    blender_script_particles: Path = Path("blender/render_particles.py")
    blender_script_surface: Path = Path("blender/render_surface.py")

    reconstruction_grid_resolution: int = 64
    reconstruction_radius_scale: float = 2.0
    extra_fluid_volumes: tuple[FluidVolumePreset, ...] = ()
    extra_rigid_bodies: tuple[RigidMeshPreset, ...] = ()
    include_rigids_in_reconstruction: bool = True

    def ensure_directories(self) -> None:
        for directory in (self.ply_dir, self.mesh_dir, self.render_dir, self.rigid_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def iter_rigid_presets(self, scene: "ScenePreset") -> tuple[RigidMeshPreset, ...]:
        return scene.rigid_bodies + self.extra_rigid_bodies

    def iter_fluid_presets(self, scene: "ScenePreset") -> tuple[FluidVolumePreset, ...]:
        return scene.fluid_volumes + self.extra_fluid_volumes


@dataclass
class ScenePreset:
    """Simple recipes for initial particle blocks and boundaries."""

    fluid_volumes: tuple[FluidVolumePreset, ...]
    obstacle_center: Tuple[float, float, float] | None = None
    obstacle_radius: float = 0.0
    rigid_bodies: tuple[RigidMeshPreset, ...] = ()


PRESETS = {
    "dam_break": ScenePreset(
        fluid_volumes=(
            FluidVolumePreset(
                name="DamColumn",
                block_min=(-0.3, -0.3, 0.0),
                block_max=(-0.05, 0.3, 0.6),
            ),
        ),
    ),
    "drop": ScenePreset(
        fluid_volumes=(
            FluidVolumePreset(
                name="Drop",
                block_min=(-0.1, -0.1, 0.4),
                block_max=(0.1, 0.1, 0.7),
            ),
        ),
        obstacle_center=(0.0, 0.0, 0.05),
        obstacle_radius=0.1,
        rigid_bodies=(
            RigidMeshPreset(
                name="CatchBox",
                mesh_path=Path("assets/rigid/barrier_box.obj"),
                density=400000.0,
                position=(0.0, 0.0, 1.0),
                scale=0.5,
            ),
        ),
    ),
}

