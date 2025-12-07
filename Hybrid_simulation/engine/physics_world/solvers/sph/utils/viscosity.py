"""SPH viscosity (XSPH) with optional numba acceleration."""
from typing import List, Tuple

import numpy as np

from .kernels import poly6

try:
    import numba as nb
except ImportError:  # pragma: no cover
    nb = None

Vec3 = Tuple[float, float, float]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def length(v: Vec3) -> float:
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5


def compute_xsph_velocities(
    positions: List[Vec3],
    velocities: List[Vec3],
    densities: List[float],
    mass: float,
    h: float,
    alpha: float,
    neighbors: List[List[int]],
) -> List[Vec3]:
    """Pure-Python XSPH smoothing."""
    n = len(positions)
    v_hat = [(0.0, 0.0, 0.0)] * n

    for i in range(n):
        v_i = velocities[i]
        sum_term = (0.0, 0.0, 0.0)

        for j in neighbors[i]:
            rho_j = densities[j]
            if rho_j < 1e-6:
                continue

            r_vec = sub(positions[i], positions[j])
            r = length(r_vec)
            w_ij = poly6(r, h)

            v_diff = sub(velocities[j], v_i)
            factor = (mass / rho_j) * w_ij
            term = mul(v_diff, factor)

            sum_term = add(sum_term, term)

        v_hat[i] = add(v_i, mul(sum_term, alpha))

    return v_hat


if nb:
    @nb.njit(parallel=True, fastmath=True)
    def compute_xsph_velocities_numba(
        positions: np.ndarray,
        velocities: np.ndarray,
        densities: np.ndarray,
        mass: float,
        h: float,
        alpha: float,
        neighbors_flat: np.ndarray,
        neighbors_offsets: np.ndarray,
    ) -> np.ndarray:
        n = positions.shape[0]
        out = np.zeros((n, 3), dtype=np.float32)
        h9 = h ** 9
        poly6_coef = 315.0 / (64.0 * np.pi * h9)
        for i in nb.prange(n):
            vx_i = velocities[i, 0]
            vy_i = velocities[i, 1]
            vz_i = velocities[i, 2]
            sx = 0.0
            sy = 0.0
            sz = 0.0
            px = positions[i, 0]
            py = positions[i, 1]
            pz = positions[i, 2]

            start = neighbors_offsets[i]
            end = neighbors_offsets[i + 1]
            for k in range(start, end):
                j = neighbors_flat[k]
                rho_j = densities[j]
                if rho_j < 1e-6:
                    continue
                dx = px - positions[j, 0]
                dy = py - positions[j, 1]
                dz = pz - positions[j, 2]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < h * h:
                    w_ij = poly6_coef * (h * h - r2) ** 3
                    factor = (mass / rho_j) * w_ij
                    sx += (velocities[j, 0] - vx_i) * factor
                    sy += (velocities[j, 1] - vy_i) * factor
                    sz += (velocities[j, 2] - vz_i) * factor

            out[i, 0] = vx_i + alpha * sx
            out[i, 1] = vy_i + alpha * sy
            out[i, 2] = vz_i + alpha * sz
        return out
else:
    compute_xsph_velocities_numba = None
