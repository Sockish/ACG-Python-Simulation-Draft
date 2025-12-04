"""Standard SPH solver implementation using simple cubic kernel hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ....configuration import LiquidBoxConfig
from ...math_utils import Vec3, add, clamp, dot, length, mul, normalize, sub
from ...state import FluidState
from .utils.kernels import SmoothingKernels
from .utils.spatial_hash import SpatialHashGrid


def _seq_to_vec3(values) -> Vec3:
    return float(values[0]), float(values[1]), float(values[2])


@dataclass
class FluidSolver:
    liquid_box: LiquidBoxConfig
    gravity: Vec3  # m/s^2
    stiffness: float = 200  # bulk modulus approximation for water (Pa)
    viscosity: float = 0.001  # dynamic viscosity of water (Pa·s)
    damping: float = 0.05  # dimensionless per-second damping

    def __post_init__(self) -> None:
        self.bounds_min = _seq_to_vec3(self.liquid_box.min_corner)
        self.bounds_max = _seq_to_vec3(self.liquid_box.max_corner)
        self.smoothing_length = float(self.liquid_box.smoothing_length)
        spacing = float(self.liquid_box.particle_spacing)
        self.particle_mass = self.liquid_box.rest_density * spacing ** 3
        self.kernels = SmoothingKernels(self.smoothing_length)
        self.grid = SpatialHashGrid(self.smoothing_length)

    def initialize(self) -> FluidState:
        positions: List[Vec3] = []
        velocities: List[Vec3] = []
        densities: List[float] = []
        pressures: List[float] = []

        spacing = float(self.liquid_box.particle_spacing)
        min_corner = self.bounds_min
        max_corner = self.bounds_max

        x = min_corner[0] + spacing * 0.5
        while x < max_corner[0] - spacing * 0.5 + 1e-6:
            y = min_corner[1] + spacing * 0.5
            while y < max_corner[1] - spacing * 0.5 + 1e-6:
                z = min_corner[2] + spacing * 0.5
                while z < max_corner[2] - spacing * 0.5 + 1e-6:
                    positions.append((x, y, z))
                    velocities.append((0.0, 0.0, 0.0))
                    densities.append(self.liquid_box.rest_density)
                    print(f"Adding particle at position: ({x}, {y}, {z})")
                    print(f"Density: {self.liquid_box.rest_density}")
                    pressures.append(0.0)
                    z += spacing
                y += spacing
            x += spacing

        return FluidState(
            positions=positions,
            velocities=velocities,
            densities=densities,
            pressures=pressures,
            particle_mass=self.particle_mass,
            smoothing_length=self.smoothing_length,
            rest_density=self.liquid_box.rest_density,
            bounds_min=self.bounds_min,
            bounds_max=self.bounds_max,
        )

    def step(self, state: FluidState, dt: float) -> None:
        if state.particle_count() == 0:
            return

        self.grid.build(state.positions)
        self._update_densities(state)
        self._update_pressures(state)
        forces = self._compute_forces(state)
        self._integrate(state, forces, dt)
