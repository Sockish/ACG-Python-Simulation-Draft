"""Surface tension helper for SPH particles."""
from typing import List, Tuple

from .kernels import spiky_grad

Vec3 = Tuple[float, float, float]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def length_sq(v: Vec3) -> float:
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]


def compute_surface_tension_forces(
    positions: List[Vec3],
    mass: float,
    h: float,
    surface_tension_kappa: float,
    neighbors: List[List[int]],
) -> List[Vec3]:
    """Compute surface tension force term per particle.

    Implements the DVA capillary term: a = -kappa / m_a * \sum_b m_b W(x_a - x_b).
    Here W is taken as the gradient of the spiky kernel to provide direction.
    """
    n = len(positions)
    if mass <= 0.0:
        return [(0.0, 0.0, 0.0)] * n

    factor = -surface_tension_kappa
    forces = [(0.0, 0.0, 0.0)] * n

    for i in range(n):
        f_i = (0.0, 0.0, 0.0)
        p_i = positions[i]
        for j in neighbors[i]:
            r = sub(p_i, positions[j])
            grad = mul(spiky_grad(r, h), length_sq(r)) 
            f_i = add(f_i, mul(grad, factor * mass))
        forces[i] = f_i
        if i < 5:  # Print first 5 forces for debugging
            print(f"surface_tension_i[{i}] = {f_i}")

    return forces