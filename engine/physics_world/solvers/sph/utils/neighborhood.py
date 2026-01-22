"""Spatial hashing neighborhood search for SPH particles and boundaries."""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ....state import RigidBodyState, StaticBodyState
from .ghost_particles import compute_local_pseudo_masses

Vec3 = Tuple[float, float, float]
CellIndex = Tuple[int, int, int]


@dataclass
class BoundarySample:
    """Discrete sample point representing a rigid or static feature."""

    kind: Literal["rigid", "static"]
    body_index: int
    feature_index: int
    position: Vec3
    normal: Vec3
    pseudo_mass: float = 0.0


def cell_index(pos: Vec3, cell_size: float) -> CellIndex:
    return (
        int(pos[0] // cell_size),
        int(pos[1] // cell_size),
        int(pos[2] // cell_size),
    )


def build_hash(positions: List[Vec3], cell_size: float) -> Dict[CellIndex, List[int]]:
    """Build a spatial hash mapping cell indices to lists of particle indices."""
    grid = {}
    for i, p in enumerate(positions):
        idx = cell_index(p, cell_size)
        grid.setdefault(idx, []).append(i)
    return grid


def neighborhood_indices(i: int, positions: List[Vec3], grid: Dict[CellIndex, List[int]], cell_size: float) -> List[int]:
    """Return neighbor indices for particle i by searching adjacent cells.

    Checks the 3x3x3 neighborhood (works for 2D data too where z is zero).
    """
    p = positions[i]
    ci, cj, ck = cell_index(p, cell_size)
    neighbors = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                cell = (ci + di, cj + dj, ck + dk)
                if cell in grid:
                    neighbors.extend(grid[cell])
    # remove self
    return [n for n in neighbors if n != i]


# Optional numba acceleration for neighborhood_indices using numpy arrays
try:
    import numba as nb
except ImportError:  # pragma: no cover
    nb = None


def neighborhood_indices_numpy(positions: np.ndarray, cell_size: float):
    """Build spatial hash and neighbor lists in numpy; returns (flat, offsets).

    This avoids Python loops when numba is available. Fallback to Python if numba missing.
    """
    if nb is None:
        # Fallback to Python lists
        grid = build_hash([tuple(p) for p in positions], cell_size)
        neighbors = [neighborhood_indices(i, [tuple(p) for p in positions], grid, cell_size) for i in range(len(positions))]
        from .neighbor_flatten import flatten_neighbors

        return flatten_neighbors(neighbors)

    return _neighborhood_indices_numba(positions, cell_size)


if nb:
    @nb.njit(fastmath=True)
    def _hash_index(px: float, py: float, pz: float, cell_size: float):
        return int(px // cell_size), int(py // cell_size), int(pz // cell_size)


    @nb.njit(fastmath=True)
    def _build_grid_indices(positions: np.ndarray, cell_size: float):
        n = positions.shape[0]
        keys = np.empty((n, 3), dtype=np.int32)
        for i in range(n):
            keys[i, 0], keys[i, 1], keys[i, 2] = _hash_index(positions[i, 0], positions[i, 1], positions[i, 2], cell_size)
        return keys


    @nb.njit(fastmath=True)
    def _neighbor_pairs(keys: np.ndarray):
        # Very simple O(n^2) grouping by keys for compatibility; for large n consider better hashing
        n = keys.shape[0]
        counts = np.zeros(n, dtype=np.int32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if keys[i, 0] == keys[j, 0] and keys[i, 1] == keys[j, 1] and keys[i, 2] == keys[j, 2]:
                    counts[i] += 1
        offsets = np.zeros(n + 1, dtype=np.int32)
        for i in range(n):
            offsets[i + 1] = offsets[i] + counts[i]
        flat = np.empty(offsets[-1], dtype=np.int32)
        for i in range(n):
            write = offsets[i]
            for j in range(n):
                if i == j:
                    continue
                if keys[i, 0] == keys[j, 0] and keys[i, 1] == keys[j, 1] and keys[i, 2] == keys[j, 2]:
                    flat[write] = j
                    write += 1
        return flat, offsets


    @nb.njit(parallel=True, fastmath=True)
    def _expand_neighbors_same_cell(keys: np.ndarray, cell_size: float):
        # 3x3x3 search using same-cell grouping as approximation; still O(n^2) but in numba
        n = keys.shape[0]
        counts = np.zeros(n, dtype=np.int32)
        for i in nb.prange(n):
            ci0, ci1, ci2 = keys[i, 0], keys[i, 1], keys[i, 2]
            cnt = 0
            for j in range(n):
                if i == j:
                    continue
                dj0, dj1, dj2 = keys[j, 0], keys[j, 1], keys[j, 2]
                if abs(ci0 - dj0) <= 1 and abs(ci1 - dj1) <= 1 and abs(ci2 - dj2) <= 1:
                    cnt += 1
            counts[i] = cnt

        offsets = np.zeros(n + 1, dtype=np.int32)
        for i in range(n):
            offsets[i + 1] = offsets[i] + counts[i]
        flat = np.empty(offsets[-1], dtype=np.int32)

        for i in nb.prange(n):
            ci0, ci1, ci2 = keys[i, 0], keys[i, 1], keys[i, 2]
            write = offsets[i]
            for j in range(n):
                if i == j:
                    continue
                dj0, dj1, dj2 = keys[j, 0], keys[j, 1], keys[j, 2]
                if abs(ci0 - dj0) <= 1 and abs(ci1 - dj1) <= 1 and abs(ci2 - dj2) <= 1:
                    flat[write] = j
                    write += 1
        return flat, offsets


    def _neighborhood_indices_numba(positions: np.ndarray, cell_size: float):
        keys = _build_grid_indices(positions, cell_size)
        return _expand_neighbors_same_cell(keys, cell_size)


def _collect_boundary_samples(
    bodies: Optional[List[Union[RigidBodyState, StaticBodyState]]],
    kind: Literal["rigid", "static"],
    smoothing_length: float,
    rest_density: Optional[float],
) -> List[BoundarySample]:
    samples: List[BoundarySample] = []
    if not bodies:
        return samples

    for body_idx, body in enumerate(bodies):
        if kind == "rigid":
            ghosts = body.get_world_ghost_particles()
            if not ghosts:
                vertices = body.get_world_vertices()  # type: ignore[attr-defined]
                ghosts = [(vertex, (0.0, 0.0, 0.0)) for vertex in vertices]
        else:
            ghosts = body.get_world_ghost_particles()
            if not ghosts:
                ghosts = [(vertex, (0.0, 0.0, 0.0)) for vertex in body.vertices]

        mass_cache: List[float] = getattr(body, "ghost_pseudo_masses", [])
        if rest_density and rest_density > 0.0 and not mass_cache:
            local_positions = [pos for pos, _ in ghosts]
            mass_cache[:] = compute_local_pseudo_masses(local_positions, smoothing_length, rest_density)
        elif len(mass_cache) < len(ghosts):
            mass_cache.extend([0.0] * (len(ghosts) - len(mass_cache)))

        for sample_idx, (position, normal) in enumerate(ghosts):
            pseudo_mass = mass_cache[sample_idx] if sample_idx < len(mass_cache) else 0.0
            samples.append(BoundarySample(kind, body_idx, sample_idx, position, normal, pseudo_mass=pseudo_mass))
    return samples


def _grid_from_samples(samples: List[BoundarySample], cell_size: float) -> Dict[CellIndex, List[BoundarySample]]:
    grid: Dict[CellIndex, List[BoundarySample]] = {}
    for sample in samples:
        idx = cell_index(sample.position, cell_size)
        grid.setdefault(idx, []).append(sample)
    return grid


def build_boundary_grids(
    rigids: Optional[List[RigidBodyState]],
    statics: Optional[List[StaticBodyState]],
    cell_size: float,
    rest_density: Optional[float] = None,
) -> tuple[Dict[CellIndex, List[BoundarySample]], Dict[CellIndex, List[BoundarySample]]]:
    """Create spatial hashes for rigid and static body samples."""

    rigid_samples = _collect_boundary_samples(rigids, "rigid", cell_size, rest_density)
    static_samples = _collect_boundary_samples(statics, "static", cell_size, rest_density)

    rigid_grid = _grid_from_samples(rigid_samples, cell_size)
    static_grid = _grid_from_samples(static_samples, cell_size)

    return rigid_grid, static_grid


def boundary_neighbors(
    position: Vec3,
    boundary_grid: Dict[CellIndex, List[BoundarySample]],
    cell_size: float,
) -> List[BoundarySample]:
    """Gather nearby boundary samples for the provided position."""

    if not boundary_grid:
        return []

    ci, cj, ck = cell_index(position, cell_size)
    neighbors: List[BoundarySample] = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                cell = (ci + di, cj + dj, ck + dk)
                if cell in boundary_grid:
                    neighbors.extend(boundary_grid[cell])
    return neighbors
