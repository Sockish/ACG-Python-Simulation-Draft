"""Surface sampling utilities for ghost particle generation."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple, Dict
from tqdm import tqdm
import numpy as np

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

    # Convert to numpy array for vectorized operations
    pos_array = np.array(positions, dtype=np.float32)
    cell_size = smoothing_length
    
    # Compute cell indices for all positions at once
    cell_indices = np.floor(pos_array / cell_size).astype(np.int32)
    
    # Build spatial hash grid
    grid: Dict[Tuple[int, int, int], np.ndarray] = {}
    for idx in range(count):
        cell = tuple(cell_indices[idx])
        if cell not in grid:
            grid[cell] = []
        grid[cell].append(idx)
    
    # Convert lists to numpy arrays for faster indexing
    for cell in grid:
        grid[cell] = np.array(grid[cell], dtype=np.int32)
    
    # Precompute poly6 kernel coefficient
    h9 = smoothing_length ** 9
    h2 = smoothing_length * smoothing_length
    poly6_coef = 315.0 / (64.0 * np.pi * h9)
    
    masses = np.zeros(count, dtype=np.float32)
    
    # Process in batches for better cache performance
    batch_size = 1024
    num_batches = (count + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Computing masses"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, count)
        batch_cells = cell_indices[start_idx:end_idx]
        
        for local_i, i in enumerate(range(start_idx, end_idx)):
            ci, cj, ck = batch_cells[local_i]
            pos_i = pos_array[i]
            total_w = 0.0
            
            # Check 27 neighboring cells
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        cell = (ci + di, cj + dj, ck + dk)
                        if cell not in grid:
                            continue
                        
                        neighbor_indices = grid[cell]
                        if len(neighbor_indices) == 0:
                            continue
                        
                        # Vectorized distance computation for all neighbors in this cell
                        neighbor_positions = pos_array[neighbor_indices]
                        diff = neighbor_positions - pos_i
                        r_squared = np.sum(diff * diff, axis=1)
                        
                        # Vectorized poly6 kernel computation
                        # Only compute for r < h (i.e., r^2 < h^2)
                        valid_mask = r_squared < h2
                        if np.any(valid_mask):
                            h2_minus_r2 = h2 - r_squared[valid_mask]
                            kernel_values = poly6_coef * (h2_minus_r2 ** 3)
                            total_w += np.sum(kernel_values)
            
            masses[i] = rest_density / total_w if total_w > 1e-9 else 0.0

    return masses.tolist()
