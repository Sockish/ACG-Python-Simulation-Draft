"""High-level SPH solver orchestration."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from tqdm.auto import tqdm

from boundaries import enforce_box_boundaries, repel_sphere
from config import PRESETS, ScenePreset, SimulationConfig
from forces import compute_density_pressure, compute_forces
from integrator import integrate
from kernels import SmoothingKernels
from particles import ParticleSystem, SpatialHash

Callback = Callable[[int, ParticleSystem], None]


class SPHSolver:
    def __init__(self, config: SimulationConfig, preset_name: str = "dam_break") -> None:
        self.config = config
        self.scene: ScenePreset = PRESETS[preset_name]
        self.particles = ParticleSystem(config.particle_count, config.mass)
        self.kernels = SmoothingKernels(config.kernel_radius)
        self.hash = SpatialHash(config.kernel_radius)
        self._seed_scene()

    def _seed_scene(self) -> None:
        self.particles.seed_block(self.scene.fluid_block_min, self.scene.fluid_block_max)

    def _build_neighbor_map(self) -> list[list[int]]:
        self.hash.build(self.particles.positions)
        neighbor_map: list[list[int]] = [[] for _ in range(self.particles.count)]
        for i in range(self.particles.count):
            pi = self.particles.positions[i]
            local: list[int] = []
            for j in self.hash.iter_neighbors(pi):
                if i == j:
                    continue
                r = np.linalg.norm(pi - self.particles.positions[j])
                if r < self.config.kernel_radius:
                    local.append(j)
            neighbor_map[i] = local
        return neighbor_map

    def step(self, step_idx: int) -> None:
        neighbor_map = self._build_neighbor_map()
        compute_density_pressure(self.particles, self.config, self.kernels, neighbor_map)
        compute_forces(self.particles, self.config, self.kernels, neighbor_map)
        integrate(self.particles, self.config)


    def run(self, callback: Optional[Callback] = None) -> None:
        for step_idx in tqdm(
            range(self.config.total_steps),
            desc="Simulating SPH",
            unit="step",
        ):
            self.step(step_idx)
            if callback and step_idx % self.config.export_every_n_steps == 0:
                callback(step_idx, self.particles)

