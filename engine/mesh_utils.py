"""Helpers for loading OBJ meshes and computing simple bounds."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
UVCoord = Tuple[float, ...]


@dataclass
class OBJMesh:
    vertices: List[Vec3]
    faces: List[Tuple[int, ...]]
    uvs: List[UVCoord] = field(default_factory=list)
    uv_faces: List[Tuple[Optional[int], ...]] = field(default_factory=list)
    normals: List[Vec3] = field(default_factory=list)
    normal_faces: List[Tuple[Optional[int], ...]] = field(default_factory=list)

    def bounds(self) -> Tuple[Vec3, Vec3]:
        if not self.vertices:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


_mesh_cache: Dict[Path, OBJMesh] = {}


def load_obj_mesh(path: Path) -> OBJMesh:

    path = path.expanduser().resolve()


    if path in _mesh_cache:
        return _mesh_cache[path]


    vertices: List[Vec3] = []
    faces: List[Tuple[int, ...]] = []
    uvs: List[UVCoord] = []
    uv_faces: List[Tuple[Optional[int], ...]] = []
    normals: List[Vec3] = []
    normal_faces: List[Tuple[Optional[int], ...]] = []

    if path.exists():
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith("vt "):
                    parts = line.split()
                    coords = tuple(float(value) for value in parts[1:])
                    if coords:
                        uvs.append(coords)
                elif line.startswith("vn "):
                    parts = line.split()
                    if len(parts) >= 4:
                        normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    v_indices: List[int] = []
                    uv_indices: List[Optional[int]] = []
                    normal_indices: List[Optional[int]] = []
                    for token in parts:
                        tokens = token.split("/")
                        v_idx = int(tokens[0]) - 1 if tokens and tokens[0] else None
                        vt_idx = int(tokens[1]) - 1 if len(tokens) > 1 and tokens[1] else None
                        vn_idx = int(tokens[2]) - 1 if len(tokens) > 2 and tokens[2] else None
                        if v_idx is None:
                            continue
                        v_indices.append(v_idx)
                        uv_indices.append(vt_idx)
                        normal_indices.append(vn_idx)
                    if len(v_indices) >= 3:
                        faces.append(tuple(v_indices))
                        uv_faces.append(tuple(uv_indices))
                        normal_faces.append(tuple(normal_indices))
    else:
        raise FileNotFoundError(f"OBJ mesh file not found: {path}")

    mesh = OBJMesh(
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        uv_faces=uv_faces,
        normals=normals,
        normal_faces=normal_faces,
    )
    _mesh_cache[path] = mesh
    return mesh


def mesh_bounds(path: Path) -> Tuple[Vec3, Vec3]:
    mesh = load_obj_mesh(path)
    return mesh.bounds()

def bounding_radius(bounds_min: Vec3, bounds_max: Vec3) -> float:
    dx = bounds_max[0] - bounds_min[0]
    dy = bounds_max[1] - bounds_min[1]
    dz = bounds_max[2] - bounds_min[2]
    return 0.5 * sqrt(dx * dx + dy * dy + dz * dz)


def compute_center_of_mass(vertices: List[Vec3]) -> Vec3:
    """Compute the geometric center (centroid) of mesh vertices.
    
    Assumes uniform density throughout the mesh.
    For complex shapes, this is an approximation.
    """
    if not vertices:
        return (0.0, 0.0, 0.0)
    
    sum_x = sum(v[0] for v in vertices)
    sum_y = sum(v[1] for v in vertices)
    sum_z = sum(v[2] for v in vertices)
    count = len(vertices)
    
    return (sum_x / count, sum_y / count, sum_z / count)


def center_mesh_vertices(vertices: List[Vec3], center: Vec3) -> List[Vec3]:
    """Translate all vertices so that the given center becomes the origin.
    
    Returns vertices in center-relative coordinates.
    """
    return [
        (v[0] - center[0], v[1] - center[1], v[2] - center[2])
        for v in vertices
    ]


def triangulate_faces(faces: List[Tuple[int, ...]]) -> List[Tuple[int, int, int]]:
    """Convert polygon faces into triangle indices using fan triangulation."""

    triangles: List[Tuple[int, int, int]] = []
    for face in faces:
        if len(face) == 3:
            triangles.append((face[0], face[1], face[2]))
        elif len(face) == 4:
            triangles.append((face[0], face[1], face[2]))
            triangles.append((face[0], face[2], face[3]))
        elif len(face) > 4:
            for i in range(1, len(face) - 1):
                triangles.append((face[0], face[i], face[i + 1]))
    return triangles

