"""Solver handling rigid body collisions against immovable meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ...math_utils import add, closest_point_on_aabb, dot, length, mul, normalize, sub
from ...state import RigidBodyState, StaticBodyState


@dataclass
class RigidStaticSolver:
    restitution: float = 0.4  # dimensionless coefficient

    def step(self, rigids: List[RigidBodyState], statics: List[StaticBodyState], dt: float) -> None:
        del dt
        if not rigids or not statics:
            return
        for rigid in rigids:
            for static in statics:
                bounds_min, bounds_max = static.world_bounds
                closest = closest_point_on_aabb(rigid.position, bounds_min, bounds_max)
                delta = sub(rigid.position, closest)
                dist = length(delta)
                if dist >= rigid.bounding_radius or rigid.bounding_radius <= 0.0:
                    continue
                normal = normalize(delta) if dist > 1e-6 else (0.0, 1.0, 0.0)
                penetration = rigid.bounding_radius - dist
                rigid.position = add(rigid.position, mul(normal, penetration + 1e-4))
                vel_normal = dot(rigid.linear_velocity, normal)
                rigid.linear_velocity = sub(
                    rigid.linear_velocity,
                    mul(normal, (1.0 + self.restitution) * vel_normal),
                )
