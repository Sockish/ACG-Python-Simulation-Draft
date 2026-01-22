"""Rigid body dynamics integrator with simple gravity and damping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ....configuration import RigidBodyConfig
from ....mesh_utils import mesh_bounds, load_obj_mesh, compute_center_of_mass, center_mesh_vertices, triangulate_faces
from ...math_utils import Vec3, add, mul, integrate_quaternion
from ...state import RigidBodyState


def _vec(values) -> Vec3:
    return float(values[0]), float(values[1]), float(values[2])


@dataclass
class RigidBodySolver:
    rigid_configs: List[RigidBodyConfig]
    gravity: Vec3  # m/s^2
    linear_damping: float = 0.8  # dimensionless per-second damping
    angular_damping: float = 0.8  # dimensionless per-second angular damping (increased for realistic rolling decay)

    def initialize(self) -> List[RigidBodyState]:
        states: List[RigidBodyState] = []
        for cfg in self.rigid_configs:
            # Load mesh and compute geometric center
            mesh = load_obj_mesh(cfg.mesh_path)
            center = compute_center_of_mass(mesh.vertices)
            
            # Center the mesh vertices around the center of mass
            centered_verts = center_mesh_vertices(mesh.vertices, center)
            
            # Compute bounds in centered coordinates
            centered_mesh = type(mesh)(vertices=centered_verts, faces=mesh.faces)
            local_min, local_max = centered_mesh.bounds()
            triangles = triangulate_faces(mesh.faces)
            
            print(f"Initialized rigid body '{cfg.name}'")
            print(f"  Local center of mass: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
            print(f"  World position (CoM): ({cfg.initial_position[0]:.3f}, {cfg.initial_position[1]:.3f}, {cfg.initial_position[2]:.3f})")
            
            states.append(
                RigidBodyState(
                    name=cfg.name,
                    mesh_path=cfg.mesh_path,
                    mass=cfg.mass,
                    inertia=tuple(cfg.inertia),
                    position=_vec(cfg.initial_position),  # This is now the world-space CoM
                    orientation=tuple(cfg.initial_orientation),
                    linear_velocity=_vec(cfg.initial_linear_velocity),
                    angular_velocity=_vec(cfg.initial_angular_velocity),
                    bounding_radius=0.0,  # No longer used for collision detection
                    local_bounds_min=local_min,
                    local_bounds_max=local_max,
                    center_of_mass=center,
                    centered_vertices=centered_verts,
                    triangles=triangles,
                )
            )
        return states

    def step(self, states: List[RigidBodyState], dt: float) -> None:
        """Update rigid body states using semi-implicit Euler integration."""
        for state in states:
            # Linear dynamics
            if state.mass > 0:
                # Apply gravity
                linear_acceleration = self.gravity
                # Update velocity: v = v + a*dt
                velocity = add(state.linear_velocity, mul(linear_acceleration, dt))
                # Apply linear damping: v = v * (1 - damping*dt)
                velocity = mul(velocity, 1.0 - self.linear_damping * dt)
            else:
                # Static/kinematic object (infinite mass)
                velocity = state.linear_velocity
            
            # Update position: p = p + v*dt
            position = add(state.position, mul(velocity, dt))
            
            # Angular dynamics
            # Apply angular damping: ω = ω * (1 - damping*dt)
            angular_velocity = mul(state.angular_velocity, 1.0 - self.angular_damping * dt)
            
            # Update orientation using quaternion integration
            # q(t+dt) = q(t) + dt * 0.5 * [ω] * q(t)
            orientation = integrate_quaternion(state.orientation, angular_velocity, dt)
            
            # Update state
            state.position = position
            state.linear_velocity = velocity
            state.angular_velocity = angular_velocity
            state.orientation = orientation
