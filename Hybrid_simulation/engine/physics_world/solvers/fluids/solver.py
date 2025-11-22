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

    def _update_densities(self, state: FluidState) -> None:
        for i, pos_i in enumerate(state.positions):
            density = 0.0
            for j in self.grid.neighbors(pos_i):
                rij = sub(pos_i, state.positions[j])
                density += state.particle_mass * self.kernels.poly6(dot(rij, rij))
            if i==0:
                print (f"Particle {i} computed density: {density}")
            state.densities[i] = max(density, state.rest_density * 0.5)

    def _update_pressures(self, state: FluidState) -> None:
        for i, rho in enumerate(state.densities):
            state.pressures[i] = self.stiffness * (rho - state.rest_density)

    def _compute_forces(self, state: FluidState) -> List[Vec3]:
        forces: List[Vec3] = [(0.0, 0.0, 0.0) for _ in state.positions]
        h = state.smoothing_length
        for i, pos_i in enumerate(state.positions):
            pressure_force = (0.0, 0.0, 0.0)
            viscosity_force = (0.0, 0.0, 0.0)
            vel_i = state.velocities[i]
            for j in self.grid.neighbors(pos_i):
                if i == j:
                    continue
                pos_j = state.positions[j]
                rij = sub(pos_i, pos_j)
                dist = length(rij)
                if dist < 1e-6 or dist >= h:
                    continue
                normalized = normalize(rij)
                mass_term = state.particle_mass / state.densities[j]
                grad = self.kernels.spiky_grad(dist)
                pressure_term = (state.pressures[i] + state.pressures[j]) * 0.5
                pressure_force = add(
                    pressure_force,
                    mul(normalized, -mass_term * pressure_term * grad),
                )
                visc = self.kernels.viscosity_laplacian(dist)
                vel_diff = sub(state.velocities[j], vel_i)
                viscosity_force = add(
                    viscosity_force,
                    mul(vel_diff, self.viscosity * mass_term * visc),
                )
            gravity_force = mul(self.gravity, state.densities[i])
            forces[i] = add(add(pressure_force, viscosity_force), gravity_force)
        return forces

    def _integrate(self, state: FluidState, forces: List[Vec3], dt: float) -> None:
        for i in range(state.particle_count()):
            inv_density = 1.0 / max(state.densities[i], 1e-6)
            if i==0:
                print(f"Density of particle {i}: {state.densities[i]}, Inverse density: {inv_density}")
            acceleration = mul(forces[i], inv_density)
            velocity = add(state.velocities[i], mul(acceleration, dt))
            if i==0:
                print(f"Particle {i} acceleration: {acceleration}, velocity before damping: {velocity}")
            velocity = mul(velocity, 1.0 - self.damping * dt)
            position = add(state.positions[i], mul(velocity, dt))
            position, velocity = self._enforce_container(position, velocity, state)
            state.velocities[i] = velocity
            state.positions[i] = position

    def _enforce_container(self, position: Vec3, velocity: Vec3, state: FluidState) -> tuple[Vec3, Vec3]:
        px, py, pz = position
        vx, vy, vz = velocity
        min_b = state.bounds_min
        max_b = state.bounds_max
        margin = state.smoothing_length * 0.5

        def bounce(axis_value, axis_velocity, min_val, max_val):
            clamped = clamp(axis_value, min_val + margin, max_val - margin)
            if clamped != axis_value:
                axis_velocity *= -0.5
            return clamped, axis_velocity

        px, vx = bounce(px, vx, min_b[0], max_b[0])
        py, vy = bounce(py, vy, min_b[1], max_b[1])
        pz, vz = bounce(pz, vz, min_b[2], max_b[2])
        return (px, py, pz), (vx, vy, vz)
