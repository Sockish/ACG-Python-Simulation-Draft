"""SPH kernel functions (poly6, spiky gradient, viscosity laplacian).

All kernels assume a support radius h > 0. Functions operate on scalar r (distance) or
on the displacement vector for gradients.
"""
from math import pi
from typing import Tuple

Vec3 = Tuple[float, float, float]


def length(vec: Vec3) -> float:
    return (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]) ** 0.5


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul_scalar(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def poly6(r: float, h: float) -> float:
    """Poly6 kernel value for distance r and support h."""
    if r < 0 or r >= h:
        return 0.0
    coef = 315.0 / (64.0 * pi * (h ** 9))
    return coef * (h * h - r * r) ** 3


def spiky_grad(r_vec: Vec3, h: float) -> Vec3:
    """Gradient of the spiky kernel (returns vector)."""
    r = length(r_vec)
    if r <= 0.0 or r >= h:
        return (0.0, 0.0, 0.0)
    coef = -45.0 / (pi * (h ** 6))
    factor = coef * (h - r) ** 2 / r
    return                 mul_scalar(r_vec, factor)


def visc_laplacian(r: float, h: float) -> float:
    """Laplacian of viscosity kernel (scalar)."""
    if r < 0 or r >= h:
        return 0.0
    coef = 45.0 / (pi * (h ** 6))
    return coef * (h - r)
