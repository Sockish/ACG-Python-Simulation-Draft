"""WCSPH solver implementation (Becker & Teschner style).

This module provides a simple, self-contained WCSPH implementation using spatial hashing
for neighborhood search and standard SPH kernels. The implementation is intentionally
minimal and focuses on density, pressure, pressure force, XSPH viscosity and symplectic
Euler integration as described.

Usage:
    from engine.physics_world.solvers.sph.solver import SphSolver
    solver = SphSolver(liquid_box_config, gravity)
    state = solver.initialize()
    solver.step(state, dt)

Note: This module avoids importing other (possibly faulty) project files and uses
plain Python lists/tuples for Vec3 arithmetic.
"""
from random import Random
from typing import List, Optional, Tuple

from ....configuration import LiquidBoxConfig
from ...state import FluidState, RigidBodyState, StaticBodyState
from .utils.neighborhood import (
    BoundarySample,
    build_boundary_grids,
    build_hash,
    boundary_neighbors,
    neighborhood_indices,
)
from .utils.density import compute_density
from .utils.eos import compute_pressure
from .utils.forces import compute_pressure_forces
from .utils.surface_tension import compute_surface_tension_forces
from .utils.viscosity import compute_xsph_velocities
from .utils.integrator import integrate_symplectic

Vec3 = Tuple[float, float, float]


def _seq_to_vec3(values) -> Vec3:
    return float(values[0]), float(values[1]), float(values[2])


class WCSphSolver:
    def __init__(
        self,
        liquid_box: LiquidBoxConfig,
        gravity: Vec3,
        kappa: float = 3000.0,
        gamma: float = 5.0,
        viscosity_alpha: float = 0.02,
        surface_tension_kappa: float = 0.00073,
        eps: float = 1e-6,
        noise_amplitude: float = 0.003,
        noise_seed: Optional[int] = None,
    ):
        self.liquid_box = liquid_box
        self.gravity = gravity
        self.kappa = kappa
        self.gamma = gamma
        self.viscosity_alpha = viscosity_alpha
        self.eps = eps
        self.noise_amplitude = noise_amplitude
        self._noise_rng = Random(noise_seed)
        self.surface_tension_kappa = surface_tension_kappa
        self.initial_velocity = _seq_to_vec3(liquid_box.initial_velocity)
        
        # Initialize derived properties from config
        self.bounds_min = _seq_to_vec3(self.liquid_box.min_corner)
        self.bounds_max = _seq_to_vec3(self.liquid_box.max_corner)
        self.smoothing_length = float(self.liquid_box.smoothing_length)
        spacing = float(self.liquid_box.particle_spacing)
        self.particle_mass = self.liquid_box.rest_density * spacing ** 3
        self._last_boundary_neighbors: dict[str, List[List[BoundarySample]]] | None = None

    def initialize(self) -> FluidState:
        """Initialize the fluid state with particles in a grid."""
        positions: List[Vec3] = []
        velocities: List[Vec3] = []
        densities: List[float] = []
        pressures: List[float] = []

        spacing = float(self.liquid_box.particle_spacing)
        min_corner = self.bounds_min
        max_corner = self.bounds_max

        # Simple grid initialization
        x = min_corner[0] + spacing * 0.5
        while x < max_corner[0] - spacing * 0.5 + 1e-6:
            y = min_corner[1] + spacing * 0.5
            while y < max_corner[1] - spacing * 0.5 + 1e-6:
                z = min_corner[2] + spacing * 0.5
                while z < max_corner[2] - spacing * 0.5 + 1e-6:
                    positions.append(
                        (
                            x + self._noise_rng.uniform(-self.noise_amplitude, self.noise_amplitude),
                            y + self._noise_rng.uniform(-self.noise_amplitude, self.noise_amplitude),
                            z + self._noise_rng.uniform(-self.noise_amplitude, self.noise_amplitude),
                        )
                    )
                    velocities.append(self.initial_velocity)
                    densities.append(self.liquid_box.rest_density)
                    pressures.append(0.0)
                    z += spacing
                y += spacing
            x += spacing
            
        print(f"Initialized {len(positions)} fluid particles.")

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

    def step(
        self,
        fluid: FluidState,
        force_damp: float,
        dt: float,
        rigids: Optional[List[RigidBodyState]] = None,
        statics: Optional[List[StaticBodyState]] = None,
    ):
        """Advance the fluid state by one timestep dt."""
        n = len(fluid.positions)
        if n == 0:
            return

        h = fluid.smoothing_length
        cell_size = h

        # Build spatial hashes for fluid particles and boundary geometry
        fluid_grid = build_hash(fluid.positions, cell_size)
        rigid_grid, static_grid = build_boundary_grids(
            rigids,
            statics,
            cell_size,
            rest_density=fluid.rest_density,
        )

        print(
            f"[BoundaryGrid] h={cell_size:.4f}, rigids={len(rigid_grid)} cells, statics={len(static_grid)} cells"
        )

        # 1) Neighborhood lists (fluid-fluid and fluid-boundary)
        neighbors = [neighborhood_indices(i, fluid.positions, fluid_grid, cell_size) for i in range(n)]
        rigid_neighbors = [
            boundary_neighbors(fluid.positions[i], rigid_grid, cell_size)
            for i in range(n)
        ]
        static_neighbors = [
            boundary_neighbors(fluid.positions[i], static_grid, cell_size)
            for i in range(n)
        ]
        boundary_neighbor_lists = [
            rigid_neighbors[i] + static_neighbors[i]
            for i in range(n)
        ]

        avg_rigid = sum(len(lst) for lst in rigid_neighbors) / max(1, n)
        avg_static = sum(len(lst) for lst in static_neighbors) / max(1, n)
        print(
            f"[BoundaryNeighbors] avg_rigid={avg_rigid:.2f}, avg_static={avg_static:.2f}"
        )

        # 2) Compute densities
        rho = compute_density(
            fluid.positions,
            fluid.particle_mass,
            h,
            neighbors,
            boundary_neighbor_lists,
        )
        rho = [max(r, self.eps) for r in rho]
        fluid.densities[:] = rho

        # 3) Equation of state -> pressures
        pressures = compute_pressure(
            rho,
            fluid.rest_density,
            self.kappa,
            self.gamma
        )
        fluid.pressures[:] = pressures

        # 4) Pressure forces
        forces = compute_pressure_forces(
            fluid.positions,
            rho,
            pressures,
            fluid.particle_mass,
            h,
            neighbors,
            boundary_neighbor_lists,
        )

        # 4.5) Surface tension
        surface_forces = compute_surface_tension_forces(
            fluid.positions,
            fluid.particle_mass,
            h,
            self.surface_tension_kappa,
            neighbors,
        )

        # 5) Viscosity (XSPH) smoothing of velocity
        if self.viscosity_alpha > 0.0:
            v_hat = compute_xsph_velocities(
                fluid.positions,
                fluid.velocities,
                rho,
                fluid.particle_mass,
                h,
                self.viscosity_alpha,
                neighbors
            )
            fluid.velocities[:] = v_hat

        # 6) Integrate (symplectic Euler)
        # v(t+dt) = v(t) + dt * a(t)
        # x(t+dt) = x(t) + dt * v(t+dt)
        # We use the current (possibly smoothed) velocities for integration.
        new_pos, new_vel = integrate_symplectic(
            fluid.positions,
            fluid.velocities,
            forces,
            fluid.particle_mass,
            dt,
            force_damp,
            self.gravity,
            extra_forces=surface_forces,
        )

        # Hook for fluid-solid interaction forces (user-implemented)
        self._apply_boundary_interactions(
            fluid,
            rho,
            rigid_neighbors,
            static_neighbors,
            dt,
        )

        # Write back to fluid state
        for i in range(n):
            fluid.velocities[i] = new_vel[i]
            fluid.positions[i] = new_pos[i]

    def _apply_boundary_interactions(
        self,
        fluid: FluidState,
        densities: List[float],
        rigid_neighbors: List[List[BoundarySample]],
        static_neighbors: List[List[BoundarySample]],
        dt: float,
    ) -> None:
        """Placeholder hook for fluid-solid coupling.

        Store neighbor information for future use so that user-defined coupling
        forces can be implemented without touching the core integration loop.
        """

        _ = (fluid, densities, dt)
        self._last_boundary_neighbors = {
            "rigid": rigid_neighbors,
            "static": static_neighbors,
        }

