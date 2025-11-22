"""Rigid body dynamics integrator with simple gravity and damping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ....configuration import RigidBodyConfig
from ....mesh_utils import bounding_radius, mesh_bounds
from ...math_utils import Vec3, add, clamp, mul
from ...state import RigidBodyState


def _vec(values) -> Vec3:
    return float(values[0]), float(values[1]), float(values[2])


@dataclass
class RigidBodySolver:
    rigid_configs: List[RigidBodyConfig]
    gravity: Vec3  # m/s^2
    bounds_min: Vec3  # m
    bounds_max: Vec3  # m
    linear_damping: float = 0.01  # dimensionless per-second damping

    def initialize(self) -> List[RigidBodyState]:
        states: List[RigidBodyState] = []
        for cfg in self.rigid_configs:
            local_min, local_max = mesh_bounds(cfg.mesh_path)
            radius = bounding_radius(local_min, local_max)
            states.append(
                RigidBodyState(
                    name=cfg.name,
                    mesh_path=cfg.mesh_path,
                    mass=cfg.mass,
                    inertia=tuple(cfg.inertia),
                    position=_vec(cfg.initial_position),
                    orientation=tuple(cfg.initial_orientation),
                    linear_velocity=_vec(cfg.initial_linear_velocity),
                    angular_velocity=_vec(cfg.initial_angular_velocity),
                    bounding_radius=radius,
                    local_bounds_min=local_min,
                    local_bounds_max=local_max,
                )
            )
        return states

    def step(self, states: List[RigidBodyState], dt: float) -> None:
        for state in states:
            acceleration = self.gravity if state.mass > 0 else (0.0, 0.0, 0.0)
            velocity = add(state.linear_velocity, mul(acceleration, dt))
            velocity = mul(velocity, 1.0 - self.linear_damping * dt)
            position = add(state.position, mul(velocity, dt))
            position, velocity = self._enforce_bounds(state, position, velocity)
            state.position = position
            state.linear_velocity = velocity

    def _enforce_bounds(self, state: RigidBodyState, position: Vec3, velocity: Vec3) -> tuple[Vec3, Vec3]:
        px, py, pz = position
        vx, vy, vz = velocity
        radius = state.bounding_radius

        def bounce(axis_value, axis_velocity, min_val, max_val):
            clamped = clamp(axis_value, min_val + radius, max_val - radius)
            if clamped != axis_value:
                axis_velocity *= -0.4
            return clamped, axis_velocity

        px, vx = bounce(px, vx, self.bounds_min[0], self.bounds_max[0])
        py, vy = bounce(py, vy, self.bounds_min[1], self.bounds_max[1])
        pz, vz = bounce(pz, vz, self.bounds_min[2], self.bounds_max[2])
        return (px, py, pz), (vx, vy, vz)
