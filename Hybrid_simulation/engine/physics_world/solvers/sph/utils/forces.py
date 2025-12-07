"""SPH pressure forces with optional numba acceleration."""
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np

from .kernels import spiky_grad

if TYPE_CHECKING:
    from .neighborhood import BoundarySample

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


def compute_pressure_forces(
    positions: List[Vec3],
    densities: List[float],
    pressures: List[float],
    mass: float,
    h: float,
    neighbors: List[List[int]],
    boundary_neighbors: Optional[List[List["BoundarySample"]]] = None,
) -> List[Vec3]:
    """Pure-Python pressure force computation."""
    n = len(positions)
    forces = [(0.0, 0.0, 0.0)] * n

    for i in range(n):
        f_i = (0.0, 0.0, 0.0)
        p_i = positions[i]
        rho_i = densities[i]
        pres_i = pressures[i]

        if rho_i < 1e-6:
            continue

        term_i = pres_i / (rho_i * rho_i)

        for j in neighbors[i]:
            rho_j = densities[j]
            if rho_j < 1e-6:
                continue

            pres_j = pressures[j]
            term_j = pres_j / (rho_j * rho_j)

            r_vec = sub(p_i, positions[j])
            grad_w = spiky_grad(r_vec, h)

            scalar = -mass * mass * (term_i + term_j)
            f_pair = mul(grad_w, scalar)
            f_i = add(f_i, f_pair)

        if boundary_neighbors:
            for sample in boundary_neighbors[i]:
                r_vec = sub(p_i, sample.position)
                grad_w = spiky_grad(r_vec, h)
                scalar = -mass * sample.pseudo_mass * term_i
                f_boundary = mul(grad_w, scalar)
                f_i = add(f_i, f_boundary)

        forces[i] = f_i

    return forces


if nb:
    @nb.njit(parallel=True, fastmath=True)
    def compute_pressure_forces_numba(
        positions: np.ndarray,
        densities: np.ndarray,
        pressures: np.ndarray,
        mass: float,
        h: float,
        neighbors_flat: np.ndarray,
        neighbors_offsets: np.ndarray,
        b_pos_flat: np.ndarray,
        b_mass_flat: np.ndarray,
        b_offsets: np.ndarray,
    ) -> np.ndarray:
        n = positions.shape[0]
        out = np.zeros((n, 3), dtype=np.float32)
        coef = -45.0 / (np.pi * (h ** 6))
        for i in nb.prange(n):
            rho_i = densities[i]
            if rho_i < 1e-6:
                continue
            term_i = pressures[i] / (rho_i * rho_i)
            px, py, pz = positions[i, 0], positions[i, 1], positions[i, 2]
            fx = 0.0
            fy = 0.0
            fz = 0.0

            start = neighbors_offsets[i]
            end = neighbors_offsets[i + 1]
            for k in range(start, end):
                j = neighbors_flat[k]
                rho_j = densities[j]
                if rho_j < 1e-6:
                    continue
                term_j = pressures[j] / (rho_j * rho_j)
                rx = px - positions[j, 0]
                ry = py - positions[j, 1]
                rz = pz - positions[j, 2]
                r2 = rx * rx + ry * ry + rz * rz
                if r2 > 0.0:
                    r = r2 ** 0.5
                    if r < h:
                        factor = coef * (h - r) * (h - r) / r
                        scalar = -mass * mass * (term_i + term_j)
                        fx += rx * factor * scalar
                        fy += ry * factor * scalar
                        fz += rz * factor * scalar

            b_start = b_offsets[i]
            b_end = b_offsets[i + 1]
            for k in range(b_start, b_end):
                rx = px - b_pos_flat[k, 0]
                ry = py - b_pos_flat[k, 1]
                rz = pz - b_pos_flat[k, 2]
                r2 = rx * rx + ry * ry + rz * rz
                if r2 > 0.0:
                    r = r2 ** 0.5
                    if r < h:
                        factor = coef * (h - r) * (h - r) / r
                        scalar = -mass * b_mass_flat[k] * term_i
                        fx += rx * factor * scalar
                        fy += ry * factor * scalar
                        fz += rz * factor * scalar

            out[i, 0] = fx
            out[i, 1] = fy
            out[i, 2] = fz
        return out
else:
    compute_pressure_forces_numba = None
