"""Symplectic Euler time integration."""
from __future__ import annotations

from typing import List, Tuple

Vec3 = Tuple[float, float, float]

def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def mul(v: Vec3, s: float) -> Vec3:
    return (v[0]*s, v[1]*s, v[2]*s)

def integrate_symplectic(
    positions: List[Vec3],
    velocities: List[Vec3],
    forces: List[Vec3],
    mass: float,
    dt: float,
    force_damp: float,
    gravity: Vec3 = (0.0, 0.0, -9.81),
    extra_forces: List[Vec3] | None = None,
) -> Tuple[List[Vec3], List[Vec3]]:
    """
    Integrate using Symplectic Euler.
    v(t+dt) = v(t) + dt * a(t)
    x(t+dt) = x(t) + dt * v(t+dt)
    """
    n = len(positions)
    new_positions = [(0.0, 0.0, 0.0)] * n
    new_velocities = [(0.0, 0.0, 0.0)] * n
    
    inv_mass = 1.0 / mass if mass > 0 else 0.0
    
    for i in range(n):
        # a(t) = forces / m + gravity
        total_force = forces[i]

        if i < 5:
            print(f"Particle {i}: Base Force = {forces[i]}")
        if extra_forces is not None:
            total_force = add(total_force, extra_forces[i])

        
        a = add(mul(total_force, inv_mass * force_damp), gravity)
        
        # v(t+dt) = v(t) + dt * a
        v_new = add(velocities[i], mul(a, dt))
        new_velocities[i] = v_new
        
        # x(t+dt) = x(t) + dt * v(t+dt)
        x_new = add(positions[i], mul(v_new, dt * force_damp))
        new_positions[i] = x_new
        
        if i < 5:  # Print first 5 particles for debugging
            print(f"Particle {i}: Pos {positions[i]} -> {x_new}, Vel {velocities[i]} -> {v_new}, Acc {a}")
        
    return new_positions, new_velocities
