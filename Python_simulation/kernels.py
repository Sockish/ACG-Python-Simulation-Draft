"""Smoothing kernels used by the SPH solver."""
from __future__ import annotations

import numpy as np


class SmoothingKernels:
    """Encapsulates standard SPH kernels so they can be reused efficiently."""

    def __init__(self, h: float) -> None:
        self.h = h
        self.h2 = h * h
        self.poly6_const = 315.0 / (64.0 * np.pi * h ** 9)
        self.spiky_const = -45.0 / (np.pi * h ** 6)
        self.visc_const = 45.0 / (np.pi * h ** 6)

    def poly6(self, r: float) -> float:
        if r >= self.h:
            return 0.0
        diff = self.h2 - r * r
        return self.poly6_const * diff * diff * diff

    def spiky_gradient(self, r_vec: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(r_vec)
        if r == 0.0 or r >= self.h:
            return np.zeros(3, dtype=np.float32)
        scale = self.spiky_const * (self.h - r) ** 2 / r
        return scale * r_vec

    def viscosity_laplacian(self, r: float) -> float:
        if r >= self.h:
            return 0.0
        return self.visc_const * (self.h - r)

