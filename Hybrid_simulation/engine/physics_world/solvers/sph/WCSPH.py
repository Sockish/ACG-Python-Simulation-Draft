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

import numpy as np

from ....configuration import LiquidBoxConfig
from ...state import FluidState, RigidBodyState, StaticBodyState
from ...math_utils import add, cross, dot, mul, sub
from .utils.neighborhood import (
    BoundarySample,
    build_boundary_grids,
    build_hash,
    boundary_neighbors,
    neighborhood_indices,
    neighborhood_indices_numpy,
)
from .utils.kernels import spiky_grad
from .utils.density import compute_density, compute_density_numba
from .utils.eos import compute_pressure
from .utils.forces import compute_pressure_forces, compute_pressure_forces_numba
from .utils.surface_tension import compute_surface_tension_forces
from .utils.viscosity import compute_xsph_velocities, compute_xsph_velocities_numba
from .utils.integrator import integrate_symplectic
from .utils.neighbor_flatten import flatten_neighbors, flatten_boundary_neighbors

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
        viscosity_alpha: float = 0.25,
        boundary_friction_sigma: float = 4.0,
        surface_tension_kappa: float = 0.15,
        eps: float = 1e-6,
        noise_amplitude: float = 0.003,
        noise_seed: Optional[int] = None,
    ):
        self.liquid_box = liquid_box
        self.gravity = gravity
        self.kappa = kappa
        self.gamma = gamma
        self.viscosity_alpha = viscosity_alpha
        self.boundary_friction_sigma = boundary_friction_sigma
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
        # Speed of sound used by the Müller et al. style friction (eq. 14 derivative of Tait EOS)
        self.speed_of_sound = (self.kappa * self.gamma / max(self.liquid_box.rest_density, 1e-6)) ** 0.5
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
        # Flatten neighbors for numba path
        neigh_flat, neigh_offsets = flatten_neighbors(neighbors)
        b_pos_flat, b_mass_flat, b_offsets = flatten_boundary_neighbors(boundary_neighbor_lists)

        use_numba = compute_density_numba is not None

        if use_numba:
            pos_np = np.asarray(fluid.positions, dtype=np.float32)
            rho_np = compute_density_numba(
                pos_np,
                fluid.particle_mass,
                h,
                neigh_flat,
                neigh_offsets,
                b_pos_flat,
                b_mass_flat,
                b_offsets,
            )
            rho = rho_np.tolist()
        else:
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
        if use_numba and compute_pressure_forces_numba is not None:
            pos_np = np.asarray(fluid.positions, dtype=np.float32)
            rho_np = np.asarray(rho, dtype=np.float32)
            pres_np = np.asarray(pressures, dtype=np.float32)
            forces_np = compute_pressure_forces_numba(
                pos_np,
                rho_np,
                pres_np,
                fluid.particle_mass,
                h,
                neigh_flat,
                neigh_offsets,
                b_pos_flat,
                b_mass_flat,
                b_offsets,
            )
            forces = [
                (float(fx), float(fy), float(fz))
                for fx, fy, fz in forces_np.tolist()
            ]
        else:
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
            if use_numba and compute_xsph_velocities_numba is not None:
                pos_np = np.asarray(fluid.positions, dtype=np.float32)
                vel_np = np.asarray(fluid.velocities, dtype=np.float32)
                rho_np = np.asarray(rho, dtype=np.float32)
                v_hat_np = compute_xsph_velocities_numba(
                    pos_np,
                    vel_np,
                    rho_np,
                    fluid.particle_mass,
                    h,
                    self.viscosity_alpha,
                    neigh_flat,
                    neigh_offsets,
                )
                fluid.velocities[:] = [tuple(map(float, v)) for v in v_hat_np.tolist()]
            else:
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

        # Hook for fluid-solid interaction forces (now includes friction)
        self._apply_boundary_interactions(
            fluid,
            rho,
            rigid_neighbors,
            static_neighbors,
            rigids,
            statics,
            new_vel,
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
        rigids: Optional[List[RigidBodyState]],
        statics: Optional[List[StaticBodyState]],
        new_velocities: List[Vec3],
        dt: float,
    ) -> None:
        """Apply boundary-fluid friction following Müller et al. (2004) style model.

        Implements eq. (11)–(15) from the provided reference using ghost particles.
        The reaction force is accumulated on dynamic rigid bodies (if any).
        """

        if not rigid_neighbors and not static_neighbors:
            return

        h = fluid.smoothing_length
        eps = 0.01  # epsilon term in eq. (11) to avoid singularities

        rigid_impulses: dict[int, Tuple[Vec3, Vec3]] = {}

        for i, (p_i, v_i) in enumerate(zip(fluid.positions, new_velocities)):
            rho_i = max(densities[i], self.eps)

            # ν = σ h c_s / (2 ρ_fi)
            nu = (
                self.boundary_friction_sigma
                * h
                * self.speed_of_sound
                / (2.0 * rho_i)
            )

            total_force = (0.0, 0.0, 0.0)

            all_neighbors = []
            if rigid_neighbors:
                all_neighbors.extend(rigid_neighbors[i])
            if static_neighbors:
                all_neighbors.extend(static_neighbors[i])

            for sample in all_neighbors:
                # Boundary velocity: static = 0, rigid = v + ω × r
                if sample.kind == "rigid" and rigids:
                    rigid_state = rigids[sample.body_index]
                    rel = sub(sample.position, rigid_state.position)
                    v_b = add(
                        rigid_state.linear_velocity,
                        cross(rigid_state.angular_velocity, rel),
                    )
                else:
                    v_b = (0.0, 0.0, 0.0)

                v_ij = sub(v_i, v_b)
                x_ij = sub(p_i, sample.position)
                x_sq = max(dot(x_ij, x_ij), self.eps)

                vij_xij = dot(v_ij, x_ij)
                limiter = min(vij_xij, 0.0)

                pi_ij = -nu * limiter / (x_sq + eps * h * h)

                grad_w = spiky_grad(x_ij, h)
                scalar = -fluid.particle_mass * sample.pseudo_mass * pi_ij
                f_v = mul(grad_w, scalar)
                total_force = add(total_force, f_v)

                # Action-reaction on dynamic rigid body
                if sample.kind == "rigid" and rigids:
                    reaction = mul(f_v, -1.0)
                    rel = sub(sample.position, rigids[sample.body_index].position)
                    torque = cross(rel, reaction)
                    prev_f, prev_t = rigid_impulses.get(sample.body_index, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
                    rigid_impulses[sample.body_index] = (
                        add(prev_f, reaction),
                        add(prev_t, torque),
                    )

            if total_force != (0.0, 0.0, 0.0):
                accel = (
                    total_force[0] / fluid.particle_mass,
                    total_force[1] / fluid.particle_mass,
                    total_force[2] / fluid.particle_mass,
                )
                new_velocities[i] = add(new_velocities[i], mul(accel, dt))

        # Apply accumulated impulses to rigids (linear + angular)
        if rigids and rigid_impulses:
            for idx, (force, torque) in rigid_impulses.items():
                rigid = rigids[idx]
                if rigid.mass <= 0:
                    continue
                inv_m = 1.0 / rigid.mass
                delta_v = mul(force, dt * inv_m)
                rigid.linear_velocity = add(rigid.linear_velocity, delta_v)

                inertia_avg = max(sum(rigid.inertia) / 3.0, 1e-6)
                inv_inertia = 1.0 / inertia_avg
                delta_w = mul(torque, dt * inv_inertia)
                rigid.angular_velocity = add(rigid.angular_velocity, delta_w)

        self._last_boundary_neighbors = {
            "rigid": rigid_neighbors,
            "static": static_neighbors,
        }

