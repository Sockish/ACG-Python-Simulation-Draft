"""Triangle mesh collision detection for rigid-static interactions."""

from __future__ import annotations

from typing import List, Tuple, Optional

from ...math_utils import Vec3, add, cross, dot, length, mul, normalize, sub


def closest_point_on_triangle(
    point: Vec3, 
    v0: Vec3, 
    v1: Vec3, 
    v2: Vec3
) -> Tuple[Vec3, Vec3]:
    """Find closest point on triangle to given point.
    
    Returns:
        Tuple of (closest_point, barycentric_coords)
    
    Based on Real-Time Collision Detection by Christer Ericson.
    """
    # Check if point is in vertex region outside v0
    edge0 = sub(v1, v0)
    edge1 = sub(v2, v0)
    v0_to_p = sub(point, v0)
    
    d00 = dot(edge0, edge0)
    d01 = dot(edge0, edge1)
    d11 = dot(edge1, edge1)
    d20 = dot(v0_to_p, edge0)
    d21 = dot(v0_to_p, edge1)
    
    denom = d00 * d11 - d01 * d01
    
    # Compute barycentric coordinates
    if abs(denom) < 1e-10:
        # Degenerate triangle, return v0
        return v0, (1.0, 0.0, 0.0)
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    # Clamp to triangle
    if u < 0:
        u = 0.0
    if v < 0:
        v = 0.0
    if w < 0:
        w = 0.0
    
    total = u + v + w
    if total > 1e-10:
        u /= total
        v /= total
        w /= total
    
    # Compute closest point
    closest = (
        u * v0[0] + v * v1[0] + w * v2[0],
        u * v0[1] + v * v1[1] + w * v2[1],
        u * v0[2] + v * v1[2] + w * v2[2],
    )
    
    return closest, (u, v, w)


def point_to_triangle_distance(
    point: Vec3,
    v0: Vec3,
    v1: Vec3, 
    v2: Vec3
) -> Tuple[float, Vec3]:
    """Calculate signed distance from point to triangle.
    
    Returns:
        Tuple of (signed_distance, closest_point)
        Negative if point is below triangle (opposite normal direction)
    """
    # Get triangle normal
    edge0 = sub(v1, v0)
    edge1 = sub(v2, v0)
    normal = normalize(cross(edge0, edge1))
    
    # Find closest point on triangle
    closest, _ = closest_point_on_triangle(point, v0, v1, v2)
    
    # Calculate vector from closest point to query point
    to_point = sub(point, closest)
    distance = length(to_point)
    
    # Determine sign based on which side of triangle
    if distance > 1e-10:
        direction = mul(to_point, 1.0 / distance)
        if dot(direction, normal) < 0:
            distance = -distance
    
    return distance, closest


def triangle_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3:
    """Calculate triangle normal (counter-clockwise winding)."""
    edge0 = sub(v1, v0)
    edge1 = sub(v2, v0)
    return normalize(cross(edge0, edge1))


def point_in_triangle_projection(
    point: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    normal: Vec3
) -> bool:
    """Check if point's projection onto triangle plane is inside triangle.
    
    Uses barycentric coordinate test.
    """
    # Project point onto triangle plane
    v0_to_p = sub(point, v0)
    plane_distance = dot(v0_to_p, normal)
    projected = sub(point, mul(normal, plane_distance))
    
    # Barycentric coordinate test
    edge0 = sub(v1, v0)
    edge1 = sub(v2, v0)
    edge2 = sub(v2, v1)
    
    to_p0 = sub(projected, v0)
    to_p1 = sub(projected, v1)
    
    cross0 = cross(edge0, to_p0)
    cross1 = cross(edge1, to_p0)
    cross2 = cross(edge2, to_p1)
    
    # All cross products should point same direction as normal
    d0 = dot(cross0, normal)
    d1 = dot(cross1, normal)
    d2 = dot(cross2, normal)
    
    return d0 >= 0 and d1 >= 0 and d2 >= 0


def vertex_triangle_collision(
    vertex: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    threshold: float = 0.0
) -> Optional[Tuple[Vec3, Vec3, float]]:
    """Detect collision between vertex and triangle.
    
    Args:
        vertex: The vertex position
        v0, v1, v2: Triangle vertices
        threshold: Distance threshold for contact (0 = touching)
    
    Returns:
        If collision detected: (contact_point, normal, penetration)
        If no collision: None
    """
    signed_distance, closest = point_to_triangle_distance(vertex, v0, v1, v2)
    
    # Check if penetrating or within threshold
    if signed_distance < threshold:
        #print(f"Vertex at {vertex} is within threshold of triangle "f"({v0}, {v1}, {v2}): signed_distance={signed_distance}")
        penetration = -signed_distance
        normal = triangle_normal(v0, v1, v2)
        return closest, normal, penetration
    
    return None
