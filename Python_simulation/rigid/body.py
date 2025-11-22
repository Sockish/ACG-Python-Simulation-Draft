"""Rigid body definition and basic utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class RigidBody:
    """Represents a rigid mesh with pose and mass properties."""

    name: str
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3)
    mass: float
    inertia_tensor: np.ndarray  # 3x3 in body space

    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    force_accumulator: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    torque_accumulator: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    restitution: float = 0.2
    friction: float = 0.5

    def apply_force(self, force: np.ndarray, point: Optional[np.ndarray] = None) -> None:
        """Accumulate a force (optionally at a world-space point)."""

        self.force_accumulator += force
        if point is not None:
            r = point - self.position
            self.torque_accumulator += np.cross(r, force)

    def clear_accumulators(self) -> None:
        self.force_accumulator.fill(0.0)
        self.torque_accumulator.fill(0.0)

    def world_vertices(self) -> np.ndarray:
        """Return vertices transformed by the current pose."""

        rot = quaternion_to_matrix(self.orientation)
        return (rot @ self.vertices.T).T + self.position

    @staticmethod
    def from_obj(path: Path, density: float, scale: float = 1.0) -> "RigidBody":
        """Load a simple OBJ mesh and compute approximate mass/inertia."""

        verts: list[list[float]] = []
        faces: list[list[int]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("v "):
                    _, x, y, z = line.strip().split()
                    verts.append([float(x) * scale, float(y) * scale, float(z) * scale])
                elif line.startswith("f "):
                    indices = [int(part.split("/")[0]) - 1 for part in line.strip().split()[1:4]]
                    faces.append(indices)
        vertices = np.asarray(verts, dtype=np.float32)
        faces_arr = np.asarray(faces, dtype=np.int32)
        volume = compute_mesh_volume(vertices, faces_arr)
        mass = density * volume
        inertia = approximate_inertia(vertices, faces_arr, mass)
        return RigidBody(
            name=path.stem,
            vertices=vertices,
            faces=faces_arr,
            mass=mass,
            inertia_tensor=inertia,
        )


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def compute_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    volume = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face]
        volume += np.dot(v0, np.cross(v1, v2))
    return abs(volume) / 6.0


def approximate_inertia(vertices: np.ndarray, faces: np.ndarray, mass: float) -> np.ndarray:
    """Very rough inertia via bounding box; replace with exact integration if needed."""

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    extents = maxs - mins
    width, height, depth = extents
    inertia = np.diag([
        (1 / 12) * mass * (height ** 2 + depth ** 2),
        (1 / 12) * mass * (width ** 2 + depth ** 2),
        (1 / 12) * mass * (width ** 2 + height ** 2),
    ]).astype(np.float32)
    return inertia
