"""Solver enforcing constraints between fluid particles and static meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ...math_utils import Vec3, add, dot, mul, sub
from ...state import FluidState, StaticBodyState


@dataclass
class FluidStaticSolver:
    penalty_strength: float = 5.0e4  # newtons per meter (N/m)
    friction: float = 0.05  # dimensionless coefficient

    def step(self, fluid: FluidState, statics: List[StaticBodyState], dt: float) -> None:
        # if not statics or fluid.particle_count() == 0:
            return
    #     particle_mass = max(fluid.particle_mass, 1e-6)
    #     for idx, position in enumerate(fluid.positions):
    #         velocity = fluid.velocities[idx]
    #         for static in statics:
    #             bounds_min, bounds_max = static.world_bounds
    #             normal, penetration = self._correction(position, bounds_min, bounds_max)
    #             position = add(position, mul(normal, penetration + 1e-4))
    #             vel_normal = dot(velocity, normal)
    #             if vel_normal < 0.0:
    #                 velocity = sub(velocity, mul(normal, vel_normal))
    #             accel_mag = (self.penalty_strength * penetration) / particle_mass
    #             velocity = add(velocity, mul(normal, accel_mag * dt))
    #             velocity = mul(velocity, max(0.0, 1.0 - self.friction * dt))
    #         fluid.positions[idx] = position
    #         fluid.velocities[idx] = velocity

    # def _correction(self, position: Vec3, bounds_min: Vec3, bounds_max: Vec3) -> tuple[Vec3, float]:
    #     distances = [
    #         (position[0] - bounds_min[0], (-1.0, 0.0, 0.0)),
    #         (bounds_max[0] - position[0], (1.0, 0.0, 0.0)),
    #         (position[1] - bounds_min[1], (0.0, -1.0, 0.0)),
    #         (bounds_max[1] - position[1], (0.0, 1.0, 0.0)),
    #         (position[2] - bounds_min[2], (0.0, 0.0, -1.0)),
    #         (bounds_max[2] - position[2], (0.0, 0.0, 1.0)),
    #     ]
    #     penetration, normal = min(distances, key=lambda item: item[0])
    #     return normal, penetration
