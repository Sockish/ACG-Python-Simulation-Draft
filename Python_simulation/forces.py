"""Core SPH density, pressure, and force computations."""
from __future__ import annotations

import numpy as np

from config import SimulationConfig
from kernels import SmoothingKernels
from particles import ParticleSystem


def compute_density_pressure(
    particles: ParticleSystem,
    config: SimulationConfig,
    kernels: SmoothingKernels,
    neighbor_map: list[list[int]],
) -> None:
    """Update per-particle density and pressure from current positions."""

    for i in range(particles.count):
        density = config.mass * kernels.poly6(0.0)
        pi = particles.positions[i]
        for j in neighbor_map[i]:
            diff = pi - particles.positions[j]
            r = np.linalg.norm(diff)
            density += config.mass * kernels.poly6(r)
        particles.densities[i] = max(density, config.rest_density * 0.1)
        particles.pressures[i] = config.gas_constant * (particles.densities[i] - config.rest_density)


def compute_forces(
    particles: ParticleSystem,
    config: SimulationConfig,
    kernels: SmoothingKernels,
    neighbor_map: list[list[int]],
) -> None:
    """Accumulation of pressure, viscosity, and gravity forces."""

    gravity = np.asarray(config.gravity, dtype=np.float32)
    for i in range(particles.count):
        pressure_force = np.zeros(3, dtype=np.float32)
        viscosity_force = np.zeros(3, dtype=np.float32)
        pi = particles.positions[i]
        vi = particles.velocities[i]
        for j in neighbor_map[i]:
            pj = particles.positions[j]
            vj = particles.velocities[j]
            rij = pi - pj
            r = np.linalg.norm(rij)
            if r < 1e-5:
                continue
            pressure_term = (
                particles.pressures[i] / (particles.densities[i] ** 2)
                + particles.pressures[j] / (particles.densities[j] ** 2)
            )
            pressure_force += -config.mass * pressure_term * kernels.spiky_gradient(rij)
            visc = kernels.viscosity_laplacian(r)
            viscosity_force += config.viscosity * config.mass * (vj - vi) * visc / particles.densities[j]

        particles.forces[i] = pressure_force + viscosity_force + gravity * particles.densities[i]

