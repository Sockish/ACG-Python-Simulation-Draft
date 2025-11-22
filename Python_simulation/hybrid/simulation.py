"""Hybrid fluid + rigid simulation loop."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from tqdm.auto import tqdm

from boundaries import enforce_box_boundaries
from config import PRESETS, PROJECT_ROOT, ScenePreset, SimulationConfig
from fluid.forces import compute_density_pressure, compute_forces
from fluid.integrator import integrate as integrate_fluid
from fluid.kernels import SmoothingKernels
from fluid.particles import ParticleSystem
from hybrid.coupling import accumulate_fluid_to_rigid_forces, build_coupled_neighbors
from rigid.body import RigidBody
from rigid.world import RigidWorld

Callback = Callable[[int, ParticleSystem], None]


class HybridSimulator:
    def __init__(self, config: SimulationConfig, preset_name: str = "dam_break") -> None:
        self.config = config
        self.scene: ScenePreset = PRESETS[preset_name]
        self.fluid = ParticleSystem(config.particle_count)
        self.kernels = SmoothingKernels(config.kernel_radius)
        self.rigid_world = RigidWorld(gravity=np.asarray(config.gravity, dtype=np.float32))
        self.config.ensure_directories()
        self._seed_scene()
        self._configure_particle_mass()

    def _seed_scene(self) -> None:
        fluid_presets = self.config.iter_fluid_presets(self.scene)
        if not fluid_presets:
            raise ValueError("Scene must define at least one fluid volume preset")
        remaining = self.fluid.count
        explicit = [preset for preset in fluid_presets if preset.particle_count is not None]
        implicit = [preset for preset in fluid_presets if preset.particle_count is None]
        for preset in explicit:
            seeded = self.fluid.seed_block(
                preset.block_min,
                preset.block_max,
                preset.jitter,
                max_particles=preset.particle_count,
            )
            remaining -= seeded
        remaining = max(0, remaining)
        if implicit and remaining > 0:
            base = remaining // len(implicit)
            remainder = remaining - base * len(implicit)
            for idx, preset in enumerate(implicit):
                target = base + (1 if idx < remainder else 0)
                if target <= 0:
                    continue
                seeded = self.fluid.seed_block(
                    preset.block_min,
                    preset.block_max,
                    preset.jitter,
                    max_particles=target,
                )
                remaining = max(0, remaining - seeded)
        for rigid in self.config.iter_rigid_presets(self.scene):
            mesh_path = rigid.mesh_path if rigid.mesh_path.is_absolute() else PROJECT_ROOT / rigid.mesh_path
            body = RigidBody.from_obj(mesh_path, rigid.density, rigid.scale)
            body.position = np.asarray(rigid.position, dtype=np.float32)
            body.orientation = np.asarray(rigid.orientation, dtype=np.float32)
            body.linear_velocity = np.asarray(rigid.linear_velocity, dtype=np.float32)
            self.rigid_world.add_body(body)

    def _configure_particle_mass(self) -> None:
        if self.config.mass is not None:
            return
        active_particles = self.fluid.seeded_count or self.fluid.count
        if active_particles <= 0:
            raise ValueError("No fluid particles were seeded; cannot derive particle mass")
        total_volume = self._estimate_fluid_volume()
        if total_volume <= 0:
            raise ValueError("Fluid volume presets must define positive volume")
        total_mass = self.config.rest_density * total_volume
        self.config.mass = total_mass / active_particles

    def _estimate_fluid_volume(self) -> float:
        volume = 0.0
        for preset in self.config.iter_fluid_presets(self.scene):
            span = np.array(preset.block_max) - np.array(preset.block_min)
            volume += float(np.prod(span))
        return volume

    def add_rigid_body(self, body: RigidBody) -> None:
        self.rigid_world.add_body(body)

    def step(self, dt: float) -> None:
        neighbor_map, ghosts, sources = build_coupled_neighbors(self.fluid, self.rigid_world.bodies, self.config)
        compute_density_pressure(self.fluid, self.config, self.kernels, neighbor_map)
        compute_forces(self.fluid, self.config, self.kernels, neighbor_map)
        accumulate_fluid_to_rigid_forces(self.fluid, ghosts, sources, self.rigid_world.bodies, self.config)
        integrate_fluid(self.fluid, self.config, dt)
        enforce_box_boundaries(
            self.fluid.positions,
            self.fluid.velocities,
            np.asarray(self.config.domain_min, dtype=np.float32),
            np.asarray(self.config.domain_max, dtype=np.float32),
            self.config.boundary_damping,
        )
        self.rigid_world.step(dt)

    def run(self, callback: Optional[Callback] = None) -> None:
        dt = self.config.time_step
        for step_idx in tqdm(range(self.config.total_steps), desc="Hybrid Simulation", unit="step"):
            self.step(dt)
            if callback and step_idx % self.config.export_every_n_steps == 0:
                callback(step_idx, self.fluid)