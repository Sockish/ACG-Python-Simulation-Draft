"""Solver coordinating fluid-rigid two-way interaction."""

from __future__ import annotations

from dataclasses import dataclass

from ...math_utils import Vec3, add, length, mul, normalize, sub
from ...state import FluidState, RigidBodyState


@dataclass
class FluidRigidCouplingSolver:
    coupling_strength: float = 1.0e4  # penalty stiffness (N/m)
    drag_coefficient: float = 0.2  # dimensionless drag factor

    def step(self, fluid: FluidState, rigids: list[RigidBodyState], dt: float) -> None:
        if fluid.particle_count() == 0 or not rigids:
            return
        for rigid in rigids:
            self._couple_with_body(fluid, rigid, dt)

    def _couple_with_body(self, fluid: FluidState, rigid: RigidBodyState, dt: float) -> None:
        interaction_radius = rigid.bounding_radius + fluid.smoothing_length * 0.5
        if interaction_radius <= 0.0:
            return
        for idx, pos in enumerate(fluid.positions):
            delta = sub(pos, rigid.position)
            dist = length(delta)
            if dist >= interaction_radius or dist < 1e-6:
                continue
            normal = normalize(delta)
            penetration = interaction_radius - dist
            force_mag = self.coupling_strength * penetration
            self._apply_impulses(fluid, rigid, idx, normal, force_mag, dt)

    def _apply_impulses(
        self,
        fluid: FluidState,
        rigid: RigidBodyState,
        particle_index: int,
        normal: Vec3,
        force_mag: float,
        dt: float,
    ) -> None:
        particle_mass = max(fluid.particle_mass, 1e-6)
        rigid_mass = max(rigid.mass, 1e-6)
        accel_fluid = (force_mag / particle_mass) * dt
        accel_rigid = (force_mag / rigid_mass) * dt
        fluid.velocities[particle_index] = add(fluid.velocities[particle_index], mul(normal, accel_fluid))
        rigid.linear_velocity = sub(rigid.linear_velocity, mul(normal, accel_rigid))

        rel_vel = sub(fluid.velocities[particle_index], rigid.linear_velocity)
        drag = self.drag_coefficient * dt
        fluid.velocities[particle_index] = sub(fluid.velocities[particle_index], mul(rel_vel, drag))
        rigid.linear_velocity = add(rigid.linear_velocity, mul(rel_vel, drag * particle_mass / rigid_mass))
