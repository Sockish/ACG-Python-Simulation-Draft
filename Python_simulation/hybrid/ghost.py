"""Ghost particle generation for rigid bodies."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from rigid.body import RigidBody


def sample_ghost_particles(body: RigidBody, spacing: float, layers: int = 1) -> np.ndarray:
    """Generate ghost particles by sampling the rigid surface and offsetting along normals."""

    verts = body.world_vertices()
    faces = body.faces
    ghosts: List[np.ndarray] = []
    for face in faces:
        v0, v1, v2 = verts[face]
        normal = np.cross(v1 - v0, v2 - v0)
        if np.linalg.norm(normal) < 1e-6:
            continue
        normal = normal / np.linalg.norm(normal)
        area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2
        samples = max(1, int(area / (spacing ** 2)))
        for _ in range(samples):
            a, b = np.random.random(2)
            if a + b > 1:
                a = 1 - a
                b = 1 - b
            point = v0 + a * (v1 - v0) + b * (v2 - v0)
            for layer in range(1, layers + 1):
                ghosts.append(point + normal * spacing * layer)
    if not ghosts:
        return np.empty((0, 3), dtype=np.float32)
    return np.vstack(ghosts).astype(np.float32)
