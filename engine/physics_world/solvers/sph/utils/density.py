"""SPH density computation with optional numba acceleration."""
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np

from .kernels import poly6

if TYPE_CHECKING:
    from .neighborhood import BoundarySample

try:
    import numba as nb
except ImportError:  # pragma: no cover
    nb = None

Vec3 = Tuple[float, float, float]


def length_sq(v: Vec3) -> float:
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def compute_density(
    positions: List[Vec3],
    mass: float,
    h: float,
    neighbors: List[List[int]],
    boundary_neighbors: Optional[List[List["BoundarySample"]]] = None,
) -> List[float]:
    """Pure-Python density computation."""
    n = len(positions)
    densities = [0.0] * n

    for i in range(n):
        rho = 0.0
        p_i = positions[i]

        for j in neighbors[i]:
            r_vec = sub(p_i, positions[j])
            r = (length_sq(r_vec)) ** 0.5
            rho += mass * poly6(r, h)

        if boundary_neighbors:
            for sample in boundary_neighbors[i]:
                r_vec = sub(p_i, sample.position)
                r = (length_sq(r_vec)) ** 0.5
                rho += sample.pseudo_mass * poly6(r, h)

        rho += mass * poly6(0.0, h)
        densities[i] = rho

    return densities


if nb:
    @nb.njit(parallel=True, fastmath=True)
    def compute_density_numba(
        positions: np.ndarray,
        mass: float,
        h: float,
        neighbors_flat: np.ndarray,
        neighbors_offsets: np.ndarray,
        b_pos_flat: np.ndarray,
        b_mass_flat: np.ndarray,
        b_offsets: np.ndarray,
    ) -> np.ndarray:
        n = positions.shape[0]
        out = np.zeros(n, dtype=np.float32)
        h9 = h ** 9
        poly6_coef = 315.0 / (64.0 * np.pi * h9)
        for i in nb.prange(n):
            rho = 0.0
            px, py, pz = positions[i, 0], positions[i, 1], positions[i, 2]

            start = neighbors_offsets[i]
            end = neighbors_offsets[i + 1]
            for k in range(start, end):
                j = neighbors_flat[k]
                dx = px - positions[j, 0]
                dy = py - positions[j, 1]
                dz = pz - positions[j, 2]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < h * h:
                    rho += mass * poly6_coef * (h * h - r2) ** 3

            b_start = b_offsets[i]
            b_end = b_offsets[i + 1]
            for k in range(b_start, b_end):
                dx = px - b_pos_flat[k, 0]
                dy = py - b_pos_flat[k, 1]
                dz = pz - b_pos_flat[k, 2]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < h * h:
                    rho += b_mass_flat[k] * poly6_coef * (h * h - r2) ** 3

            rho += mass * poly6_coef * (h * h) ** 3
            out[i] = rho
        return out
else:
    compute_density_numba = None
