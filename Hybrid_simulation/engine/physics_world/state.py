"""Dataclasses describing the evolving physics state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .math_utils import Vec3, quaternion_to_matrix, transform_point


Quaternion = Tuple[float, float, float, float]


@dataclass
class FluidState:
    positions: List[Vec3]  # meters (m)
    velocities: List[Vec3]  # meters per second (m/s)
    densities: List[float]  # kilograms per cubic meter (kg/m^3)
    pressures: List[float]  # Pascals (N/m^2)
    particle_mass: float  # kilograms (kg)
    smoothing_length: float  # meters (m)
    rest_density: float  # kilograms per cubic meter (kg/m^3)
    bounds_min: Vec3  # meters (m)
    bounds_max: Vec3  # meters (m)

    def particle_count(self) -> int:
        return len(self.positions)


@dataclass
class RigidBodyState:
    name: str
    mesh_path: Path
    mass: float  # kilograms (kg)
    inertia: Tuple[float, float, float]  # kg·m^2
    position: Vec3  # world-space center of mass position in meters (m)
    orientation: Quaternion  # unit quaternion
    linear_velocity: Vec3  # meters per second (m/s)
    angular_velocity: Vec3  # radians per second (rad/s)
    bounding_radius: float  # meters (m)
    local_bounds_min: Vec3  # meters (m)
    local_bounds_max: Vec3  # meters (m)
    center_of_mass: Vec3  # local-space center of mass offset (m) - usually (0,0,0) after centering
    centered_vertices: List[Vec3]  # vertices relative to center of mass (local space)

    @property
    def inverse_mass(self) -> float:
        return 0.0 if self.mass <= 0.0 else 1.0 / self.mass
    
    def get_world_vertices(self) -> List[Vec3]:
        """Transform centered vertices to world space using current position and orientation.
        
        Returns:
            List of vertex positions in world coordinates.
        """
        rotation_matrix = quaternion_to_matrix(self.orientation)
        return [
            transform_point(v, rotation_matrix, self.position)
            for v in self.centered_vertices
        ]


@dataclass
class StaticBodyState:
    name: str
    mesh_path: Path
    position: Vec3  # world-space reference position in meters (m) - always (0,0,0) for absolute coordinates
    orientation: Quaternion  # unit quaternion - always (0,0,0,1) for no rotation
    local_bounds_min: Vec3  # meters (m) - bounds from OBJ file
    local_bounds_max: Vec3  # meters (m) - bounds from OBJ file
    vertices: List[Vec3]  # All vertices in world space
    faces: List[Tuple[int, int, int]]  # Triangle indices (v0, v1, v2)

    @property
    def world_bounds(self) -> tuple[Vec3, Vec3]:
        # Static bodies use absolute coordinates from OBJ, no transformation needed
        return self.local_bounds_min, self.local_bounds_max
    
    def get_triangle(self, face_idx: int) -> Tuple[Vec3, Vec3, Vec3]:
        """Get world-space vertices of a triangle by face index."""
        i0, i1, i2 = self.faces[face_idx]
        return self.vertices[i0], self.vertices[i1], self.vertices[i2]
    
    @property
    def triangle_count(self) -> int:
        """Get total number of triangles in the mesh."""
        return len(self.faces)


@dataclass
class WorldSnapshot:
    step_index: int
    time: float
    fluids: FluidState | None
    rigids: List[RigidBodyState]
    statics: List[StaticBodyState]
