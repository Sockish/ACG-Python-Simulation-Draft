"""Boundary handling utilities."""
from __future__ import annotations

import numpy as np


def enforce_box_boundaries(
    positions: np.ndarray,
    velocities: np.ndarray,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    damping: float,
) -> None:
    """Keep particles inside an axis-aligned box with simple reflection."""

    for axis in range(3):
        mask_low = positions[:, axis] < domain_min[axis]
        mask_high = positions[:, axis] > domain_max[axis]
        if np.any(mask_low):
            positions[mask_low, axis] = domain_min[axis]
            velocities[mask_low, axis] *= damping
        if np.any(mask_high):
            positions[mask_high, axis] = domain_max[axis]
            velocities[mask_high, axis] *= damping


def repel_sphere(
    positions: np.ndarray,
    velocities: np.ndarray,
    center: np.ndarray,
    radius: float,
    stiffness: float = 200.0,
    damping: float = 0.5,
) -> None:
    """Basic sphere obstacle using spring forces."""

    offsets = positions - center
    distances = np.linalg.norm(offsets, axis=1)
    mask = distances < radius
    if not np.any(mask):
        return
    penetration = radius - distances[mask]
    normals = offsets[mask] / np.maximum(distances[mask][:, None], 1e-6)
    forces = stiffness * penetration[:, None] * normals
    velocities[mask] += forces
    velocities[mask] *= damping

