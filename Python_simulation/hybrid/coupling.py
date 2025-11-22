"""Fluid/rigid coupling utilities using ghost particles."""
from __future__ import annotations

import numpy as np

from config import SimulationConfig
from fluid.kernels import SmoothingKernels
from fluid.particles import ParticleSystem, SpatialHash
from rigid.body import RigidBody
from hybrid.ghost import sample_ghost_particles


def build_coupled_neighbors(
    fluid: ParticleSystem,
    rigids: list[RigidBody],
    config: SimulationConfig,
) -> tuple[list[list[int]], np.ndarray, list[tuple[int, int]]]:
    """Return neighbor map and ghost particle data."""

    hash_grid = SpatialHash(config.kernel_radius)
    hash_grid.build(fluid.positions)
    neighbor_map: list[list[int]] = [[] for _ in range(fluid.count)]
    for i in range(fluid.count):
        pi = fluid.positions[i]
        local: list[int] = []
        for j in hash_grid.iter_neighbors(pi):
            if i == j:
                continue
            r = np.linalg.norm(pi - fluid.positions[j])
            if r < config.kernel_radius:
                local.append(j)
        neighbor_map[i] = local

    ghost_positions: list[np.ndarray] = []
    ghost_sources: list[tuple[int, int]] = []  # (rigid_idx, layer)
    for rigid_idx, body in enumerate(rigids):
        ghosts = sample_ghost_particles(body, config.kernel_radius * 0.5, layers=2)
        if ghosts.size == 0:
            continue
        ghost_positions.append(ghosts)
        ghost_sources.extend([(rigid_idx, 0) for _ in range(len(ghosts))])
    if ghost_positions:
        ghost_array = np.vstack(ghost_positions)
    else:
        ghost_array = np.empty((0, 3), dtype=np.float32)
    return neighbor_map, ghost_array, ghost_sources


def accumulate_fluid_to_rigid_forces(
    fluid: ParticleSystem,
    ghosts: np.ndarray,
    ghost_sources: list[tuple[int, int]],
    rigids: list[RigidBody],
    config: SimulationConfig,
) -> None:
    if ghosts.size == 0:
        return
    kernels = SmoothingKernels(config.kernel_radius)
    for idx, ghost_pos in enumerate(ghosts):
        rigid_idx, _ = ghost_sources[idx]
        nearest = np.argmin(np.linalg.norm(fluid.positions - ghost_pos, axis=1))
        pressure = fluid.pressures[nearest]
        density = fluid.densities[nearest]
        direction = fluid.positions[nearest] - ghost_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-5:
            continue
        force_mag = -pressure / (density + 1e-6) * kernels.spiky_gradient(direction)
        rigids[rigid_idx].apply_force(force_mag, ghost_pos)
