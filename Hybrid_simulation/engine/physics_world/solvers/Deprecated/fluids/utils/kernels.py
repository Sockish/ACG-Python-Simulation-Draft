"""Common SPH kernel functions (poly6, spiky, viscosity)."""

from __future__ import annotations

from math import pi


class SmoothingKernels:
    def __init__(self, h: float) -> None:
        self.h = h
        self.h2 = h * h
        self.h3 = self.h2 * h
        self.h6 = self.h3 * self.h3
        self.poly6_const = 315.0 / (64.0 * pi * self.h6)
        self.spiky_const = -45.0 / (pi * self.h6)
        self.visc_const = 45.0 / (pi * self.h6)

    def poly6(self, r2: float) -> float:
        if r2 >= self.h2:
            return 0.0
        term = self.h2 - r2
        return self.poly6_const * term * term * term

    def spiky_grad(self, r: float) -> float:
        if r <= 0.0 or r >= self.h:
            return 0.0
        term = self.h - r
        return self.spiky_const * term * term

    def viscosity_laplacian(self, r: float) -> float:
        if r >= self.h:
            return 0.0
        return self.visc_const * (self.h - r)
