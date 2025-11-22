"""Rigid body world that manages integration and collisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from rigid.body import RigidBody


@dataclass
class RigidWorld:
    bodies: List[RigidBody] = field(default_factory=list)
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=np.float32))
    restitution: float = 0.2
    friction: float = 0.5

    def add_body(self, body: RigidBody) -> None:
        self.bodies.append(body)

    def step(self, dt: float) -> None:
        for body in self.bodies:
            body.apply_force(body.mass * self.gravity)
            inv_mass = 1.0 / body.mass
            linear_acc = body.force_accumulator * inv_mass
            body.linear_velocity += dt * linear_acc
            body.position += dt * body.linear_velocity
            # Skipping angular integration details for brevity
            body.clear_accumulators()
        self.resolve_collisions()

    def resolve_collisions(self) -> None:
        # Placeholder: implement rigid-rigid collision response
        for i, body_a in enumerate(self.bodies):
            for body_b in self.bodies[i + 1 :]:
                penetration, normal = self._check_overlap(body_a, body_b)
                if penetration > 0:
                    self._apply_impulse(body_a, body_b, normal)

    def _check_overlap(self, body_a: RigidBody, body_b: RigidBody) -> tuple[float, np.ndarray]:
        # Placeholder: bounding-sphere test
        pos_a, pos_b = body_a.position, body_b.position
        radius_a = np.linalg.norm(body_a.vertices.max(axis=0) - body_a.vertices.min(axis=0)) / 2
        radius_b = np.linalg.norm(body_b.vertices.max(axis=0) - body_b.vertices.min(axis=0)) / 2
        delta = pos_b - pos_a
        dist = np.linalg.norm(delta)
        overlap = radius_a + radius_b - dist
        normal = delta / dist if dist > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return overlap, normal

    def _apply_impulse(self, body_a: RigidBody, body_b: RigidBody, normal: np.ndarray) -> None:
        rel_vel = body_b.linear_velocity - body_a.linear_velocity
        vel_along_normal = np.dot(rel_vel, normal)
        if vel_along_normal > 0:
            return
        impulse_mag = -(1 + self.restitution) * vel_along_normal
        impulse_mag /= 1 / body_a.mass + 1 / body_b.mass
        impulse = impulse_mag * normal
        body_a.linear_velocity -= impulse / body_a.mass
        body_b.linear_velocity += impulse / body_b.mass
