"""Dataclasses describing the evolving physics state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .math_utils import Vec3


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
    position: Vec3  # meters (m)
    orientation: Quaternion  # unit quaternion
    linear_velocity: Vec3  # meters per second (m/s)
    angular_velocity: Vec3  # radians per second (rad/s)
    bounding_radius: float  # meters (m)
    local_bounds_min: Vec3  # meters (m)
    local_bounds_max: Vec3  # meters (m)

    @property
    def inverse_mass(self) -> float:
        return 0.0 if self.mass <= 0.0 else 1.0 / self.mass


@dataclass
class StaticBodyState:
    name: str
    mesh_path: Path
    position: Vec3  # meters (m)
    orientation: Quaternion  # unit quaternion
    local_bounds_min: Vec3  # meters (m)
    local_bounds_max: Vec3  # meters (m)

    @property
    def world_bounds(self) -> tuple[Vec3, Vec3]:
        offset = self.position
        min_world = (
            self.local_bounds_min[0] + offset[0],
            self.local_bounds_min[1] + offset[1],
            self.local_bounds_min[2] + offset[2],
        )
        max_world = (
            self.local_bounds_max[0] + offset[0],
            self.local_bounds_max[1] + offset[1],
            self.local_bounds_max[2] + offset[2],
        )
        return min_world, max_world


@dataclass
class WorldSnapshot:
    step_index: int
    time: float
    fluids: FluidState
    rigids: List[RigidBodyState]
    statics: List[StaticBodyState]
