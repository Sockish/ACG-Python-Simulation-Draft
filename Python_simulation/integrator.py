"""Time integration routines."""
from __future__ import annotations

import numpy as np

from config import SimulationConfig
from particles import ParticleSystem


def integrate(particles: ParticleSystem, config: SimulationConfig, dt: float | None = None) -> None:
    """Semi-implicit Euler integration."""

    step = config.time_step if dt is None else dt
    accelerations = particles.forces / particles.densities[:, None]
    particles.velocities += step * accelerations
    particles.positions += step * particles.velocities

