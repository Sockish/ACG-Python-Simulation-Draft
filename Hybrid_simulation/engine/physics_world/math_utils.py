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


def cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product of two 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def quaternion_multiply(q1: Tuple[float, float, float, float], q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Multiply two quaternions (q1 * q2)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quaternion_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Normalize a quaternion to unit length."""
    x, y, z, w = q
    norm = sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    inv = 1.0 / norm
    return (x * inv, y * inv, z * inv, w * inv)


def integrate_quaternion(q: Tuple[float, float, float, float], omega: Vec3, dt: float) -> Tuple[float, float, float, float]:
    """Integrate quaternion orientation given angular velocity.
    
    Uses the formula: q(t+dt) = q(t) + 0.5 * dt * omega_quat * q(t)
    where omega_quat = (omega_x, omega_y, omega_z, 0)
    """
    x, y, z, w = q
    wx, wy, wz = omega
    
    # Compute quaternion derivative: dq/dt = 0.5 * omega_quat * q
    dx = 0.5 * (wx * w + wy * z - wz * y)
    dy = 0.5 * (wy * w + wz * x - wx * z)
    dz = 0.5 * (wz * w + wx * y - wy * x)
    dw = 0.5 * (-wx * x - wy * y - wz * z)
    
    # Euler integration
    new_q = (x + dx * dt, y + dy * dt, z + dz * dt, w + dw * dt)
    
    # Normalize to prevent drift
    return quaternion_normalize(new_q)


def rotate_vector_by_quaternion(v: Vec3, q: Tuple[float, float, float, float]) -> Vec3:
    """Rotate a vector by a quaternion.
    
    Uses the formula: v' = q * v * q^(-1)
    For unit quaternions, q^(-1) = q* (conjugate)
    """
    x, y, z, w = q
    vx, vy, vz = v
    
    # q * v (treating v as quaternion with w=0)
    tx = w * vx + y * vz - z * vy
    ty = w * vy + z * vx - x * vz
    tz = w * vz + x * vy - y * vx
    tw = -x * vx - y * vy - z * vz
    
    # (q * v) * q^(-1) where q^(-1) = (-x, -y, -z, w) for unit quaternion
    return (
        tw * (-x) + tx * w + ty * (-z) - tz * (-y),
        tw * (-y) + ty * w + tz * (-x) - tx * (-z),
        tw * (-z) + tz * w + tx * (-y) - ty * (-x),
    )


def point_in_aabb(point: Vec3, bounds_min: Vec3, bounds_max: Vec3) -> bool:
    """Check if a point is inside an AABB."""
    return (
        bounds_min[0] <= point[0] <= bounds_max[0] and
        bounds_min[1] <= point[1] <= bounds_max[1] and
        bounds_min[2] <= point[2] <= bounds_max[2]
    )


def distance_to_aabb(point: Vec3, bounds_min: Vec3, bounds_max: Vec3) -> float:
    """Calculate signed distance from point to AABB surface.
    
    Negative if inside, positive if outside.
    """
    # Compute closest point on AABB
    closest = closest_point_on_aabb(point, bounds_min, bounds_max)
    dist = length(sub(point, closest))
    
    # Check if inside
    if point_in_aabb(point, bounds_min, bounds_max):
        return -dist
    return dist


def aabb_normal_at_point(point: Vec3, bounds_min: Vec3, bounds_max: Vec3, epsilon: float = 1e-4) -> Vec3:
    """Get the outward normal of an AABB at a point on its surface.
    
    Returns the axis-aligned normal direction.
    """
    # Find which face the point is closest to
    distances = [
        abs(point[0] - bounds_min[0]),  # -X face
        abs(point[0] - bounds_max[0]),  # +X face
        abs(point[1] - bounds_min[1]),  # -Y face
        abs(point[1] - bounds_max[1]),  # +Y face
        abs(point[2] - bounds_min[2]),  # -Z face
        abs(point[2] - bounds_max[2]),  # +Z face
    ]
    
    min_dist = min(distances)
    min_idx = distances.index(min_dist)
    
    # Return normal for the closest face
    normals = [
        (-1.0, 0.0, 0.0),  # -X
        (1.0, 0.0, 0.0),   # +X
        (0.0, -1.0, 0.0),  # -Y
        (0.0, 1.0, 0.0),   # +Y
        (0.0, 0.0, -1.0),  # -Z
        (0.0, 0.0, 1.0),   # +Z
    ]
    
    return normals[min_idx]


def aabb_intersects_aabb(min_a: Vec3, max_a: Vec3, min_b: Vec3, max_b: Vec3) -> bool:
    """Check if two AABBs intersect."""
    return (
        min_a[0] <= max_b[0] and max_a[0] >= min_b[0] and
        min_a[1] <= max_b[1] and max_a[1] >= min_b[1] and
        min_a[2] <= max_b[2] and max_a[2] >= min_b[2]
    )
