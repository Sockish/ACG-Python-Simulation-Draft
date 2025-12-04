"""Spatial hashing neighborhood search for SPH particles."""
from typing import Dict, List, Tuple

Vec3 = Tuple[float, float, float]


def cell_index(pos: Vec3, cell_size: float) -> Tuple[int, int, int]:
    return (int(pos[0] // cell_size), int(pos[1] // cell_size), int(pos[2] // cell_size))


def build_hash(positions: List[Vec3], cell_size: float) -> Dict[Tuple[int, int, int], List[int]]:
    """Build a spatial hash mapping cell indices to lists of particle indices."""
    grid = {}
    for i, p in enumerate(positions):
        idx = cell_index(p, cell_size)
        grid.setdefault(idx, []).append(i)
    return grid


def neighborhood_indices(i: int, positions: List[Vec3], grid: Dict[Tuple[int, int, int], List[int]], cell_size: float) -> List[int]:
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
