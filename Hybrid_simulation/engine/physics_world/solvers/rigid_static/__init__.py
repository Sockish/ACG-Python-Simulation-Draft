"""Rigid-static collision solver package."""

from .solver import RigidStaticSolver
from .mesh_collision import (
    closest_point_on_triangle,
    point_to_triangle_distance,
    triangle_normal,
    vertex_triangle_collision,
)
from .spatial_hash import SpatialHashGrid
from .triangle_detector import TriangleMeshCollisionDetector

__all__ = [
    "RigidStaticSolver",
    "closest_point_on_triangle",
    "point_to_triangle_distance",
    "triangle_normal",
    "vertex_triangle_collision",
    "SpatialHashGrid",
    "TriangleMeshCollisionDetector",
]
