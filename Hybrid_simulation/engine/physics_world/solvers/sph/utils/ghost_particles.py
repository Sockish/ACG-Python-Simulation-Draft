"""Surface sampling utilities for ghost particle generation."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple, Dict
from tqdm import tqdm

from ....math_utils import Vec3, add, cross, length, mul, normalize, sub
from .kernels import poly6


def _triangle_area_and_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Tuple[float, Vec3]:
    edge0 = sub(v1, v0)
    edge1 = sub(v2, v0)
    area_vec = cross(edge0, edge1)
    area = 0.5 * length(area_vec)
    normal = normalize(area_vec)
    return area, normal


def _barycentric_point(v0: Vec3, v1: Vec3, v2: Vec3, b0: float, b1: float, b2: float) -> Vec3:
    return (
        v0[0] * b2 + v1[0] * b0 + v2[0] * b1,
        v0[1] * b2 + v1[1] * b0 + v2[1] * b1,
        v0[2] * b2 + v1[2] * b0 + v2[2] * b1,
    )


def sample_mesh_surface(
    vertices: Sequence[Vec3],
    triangles: Iterable[Tuple[int, int, int]],
    smoothing_length: float,
    layer_offsets: Tuple[float, float] | None = None,
) -> List[Tuple[Vec3, Vec3]]:
    """Sample ghost particles across a mesh surface.

    Args:
        vertices: Mesh vertices in either local or world space.
        triangles: Triangles referencing the ``vertices`` array (counter-clockwise winding).
        smoothing_length: Fluid smoothing length ``h`` for spacing heuristics.
        layer_offsets: Optional pair of signed offsets applied along the triangle normal
            for dual-layer sampling. Defaults to ``(+0.3h, -0.2h)``.
    Returns:
        List of ``(position, normal)`` tuples. Normals are normalized and aligned with the
        underlying triangle winding.
    """

    if smoothing_length <= 0.0:
        return []

    sample_spacing = max(1e-5, 0.3 * smoothing_length)
    if layer_offsets is None:
        layer_offsets = (0.1 * smoothing_length, -0.1 * smoothing_length)

    samples: List[Tuple[Vec3, Vec3]] = []

    for tri in triangles:
        i0, i1, i2 = tri
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        area, normal = _triangle_area_and_normal(v0, v1, v2)
        if area <= 1e-10 or length(normal) <= 1e-8:
            continue

        edge_lengths = (
            length(sub(v1, v0)),
            length(sub(v2, v1)),
            length(sub(v0, v2)),
        )
        max_edge = max(edge_lengths)
        divisions = max(1, int(math.ceil(max_edge / sample_spacing)))
        step = 1.0 / divisions

        for i in range(divisions + 1):
            for j in range(divisions + 1 - i):
                b0 = i * step
                b1 = j * step
                b2 = 1.0 - b0 - b1
                if b2 < -1e-6:
                    continue
                base_point = _barycentric_point(v0, v1, v2, b0, b1, b2)
                for offset in layer_offsets:
                    samples.append((add(base_point, mul(normal, offset)), normal))

        # Always include the centroid as a stable sample
        centroid = (
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0,
        )
        for offset in layer_offsets:
            samples.append((add(centroid, mul(normal, offset)), normal))

    return samples


def compute_local_pseudo_masses(
    positions: Sequence[Vec3],
    smoothing_length: float,
    rest_density: float,
) -> List[float]:
    """Compute pseudo masses Ψ using a spatial hash for neighbor queries."""

    count = len(positions)
    if smoothing_length <= 0.0 or rest_density <= 0.0 or count == 0:
        return [0.0] * count

    cell_size = smoothing_length
    grid: Dict[Tuple[int, int, int], List[int]] = {}

    def _cell_index(pos: Vec3) -> Tuple[int, int, int]:
        return (
            int(pos[0] // cell_size),
            int(pos[1] // cell_size),
            int(pos[2] // cell_size),
        )

    for idx, pos in enumerate(positions):
        grid.setdefault(_cell_index(pos), []).append(idx)

    masses: List[float] = [0.0] * count
    for i, pos_i in tqdm(enumerate(positions), total=len(positions), desc="Computing masses"):
        ci, cj, ck = _cell_index(pos_i)
        total_w = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    cell = (ci + di, cj + dj, ck + dk)
                    if cell not in grid:
                        continue
                    for neighbor_idx in grid[cell]:
                        pos_j = positions[neighbor_idx]
                        r = length(sub(pos_i, pos_j))
                        total_w += poly6(r, smoothing_length)
        masses[i] =  rest_density / total_w if total_w > 1e-9 else 0.0

    return masses
