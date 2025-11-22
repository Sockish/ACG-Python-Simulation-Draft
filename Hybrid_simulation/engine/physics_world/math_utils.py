"""Lightweight vector math helpers used throughout the physics world."""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Tuple

Vec3 = Tuple[float, float, float]


def vec3(values: Iterable[float]) -> Vec3:
    x, y, z = values
    return float(x), float(y), float(z)


def add(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def mul(a: Vec3, scalar: float) -> Vec3:
    return a[0] * scalar, a[1] * scalar, a[2] * scalar


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def length(a: Vec3) -> float:
    return sqrt(max(dot(a, a), 0.0))


def normalize(a: Vec3) -> Vec3:
    l = length(a)
    if l <= 1e-8:
        return 0.0, 0.0, 0.0
    inv = 1.0 / l
    return mul(a, inv)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def clamp_vec(a: Vec3, minimum: Vec3, maximum: Vec3) -> Vec3:
    return (
        clamp(a[0], minimum[0], maximum[0]),
        clamp(a[1], minimum[1], maximum[1]),
        clamp(a[2], minimum[2], maximum[2]),
    )


def closest_point_on_aabb(point: Vec3, bounds_min: Vec3, bounds_max: Vec3) -> Vec3:
    return clamp_vec(point, bounds_min, bounds_max)


def quaternion_to_matrix(quat: Vec3 | Tuple[float, float, float, float]) -> Tuple[Vec3, Vec3, Vec3]:
    if len(quat) == 3:
        x, y, z = quat
        w = 1.0
    else:
        x, y, z, w = quat
    norm = sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-8:
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = (1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy))
    row1 = (2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx))
    row2 = (2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy))
    return row0, row1, row2


def transform_point(point: Vec3, matrix: Tuple[Vec3, Vec3, Vec3], translation: Vec3) -> Vec3:
    x = point[0] * matrix[0][0] + point[1] * matrix[0][1] + point[2] * matrix[0][2]
    y = point[0] * matrix[1][0] + point[1] * matrix[1][1] + point[2] * matrix[1][2]
    z = point[0] * matrix[2][0] + point[1] * matrix[2][1] + point[2] * matrix[2][2]
    return x + translation[0], y + translation[1], z + translation[2]
