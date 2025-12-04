"""Dataclasses describing the evolving physics state."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    triangles: List[Tuple[int, int, int]]
    ghost_local_positions: List[Vec3] = field(default_factory=list)
    ghost_local_normals: List[Vec3] = field(default_factory=list)
    ghost_pseudo_masses: List[float] = field(default_factory=list)

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

    def get_world_ghost_particles(self) -> List[Tuple[Vec3, Vec3]]:
        """Transform stored ghost particles (position + normal) to world space."""

        if not self.ghost_local_positions:
            return []
        rotation_matrix = quaternion_to_matrix(self.orientation)
        ghosts: List[Tuple[Vec3, Vec3]] = []
        for pos, normal in zip(self.ghost_local_positions, self.ghost_local_normals):
            world_pos = transform_point(pos, rotation_matrix, self.position)
            world_normal = transform_point(normal, rotation_matrix, (0.0, 0.0, 0.0))
            ghosts.append((world_pos, world_normal))
        return ghosts


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
    ghost_positions: List[Vec3] = field(default_factory=list)
    ghost_normals: List[Vec3] = field(default_factory=list)
    ghost_pseudo_masses: List[float] = field(default_factory=list)

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

    def get_world_ghost_particles(self) -> List[Tuple[Vec3, Vec3]]:
        """Return pre-sampled ghost particles with normals."""

        return list(zip(self.ghost_positions, self.ghost_normals))


@dataclass
class WorldSnapshot:
    step_index: int
    time: float
    fluids: FluidState | None
    rigids: List[RigidBodyState]
    statics: List[StaticBodyState]
