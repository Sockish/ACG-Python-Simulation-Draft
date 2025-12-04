"""Spatial hashing neighborhood search for SPH particles and boundaries."""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

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
