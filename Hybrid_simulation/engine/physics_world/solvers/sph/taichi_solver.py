"""Taichi-accelerated WCSPH solver.

This is a GPU-accelerated version of the Weakly Compressible SPH solver using Taichi.
It provides significant performance improvements over the pure Python/NumPy/Numba version,
especially for large particle counts (50k+).

Usage:
    import taichi as ti
    ti.init(arch=ti.gpu)  # or ti.cuda, ti.vulkan, ti.cpu
    
    solver = TaichiWCSPHSolver(max_particles=100000, ...)
    solver.initialize_particles(positions, velocities)
    solver.step(dt=0.001)
    positions_np = solver.get_positions()
"""
import taichi as ti
import numpy as np
from typing import Tuple, Optional
from random import Random

from .taichi_kernels import poly6_kernel, spiky_grad_kernel, poly6_gradient, poly6_laplacian
from .taichi_spatial_hash import SpatialHashGrid
from .taichi_boundary import BoundaryParticles


@ti.data_oriented
class TaichiWCSPHSolver:
    """GPU-accelerated Weakly Compressible SPH solver using Taichi."""
    
    def __init__(
        self,
        max_particles: int,
        domain_size: Tuple[float, float, float],
        smoothing_length: float,
        particle_mass: float,
        rest_density: float,
        stiffness: float = 3000.0,
        gamma: float = 7.0,
        viscosity_alpha: float = 0.25,
        surface_tension_kappa: float = 0.15,
        boundary_friction_sigma: float = 4.0,
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81),
        grid_resolution: int = 64,
        max_boundary_particles: int = 5000000,
        eps: float = 1e-6,
    ):
        """Initialize Taichi SPH solver.
        
        Args:
            max_particles: Maximum number of particles (fixed allocation)
            domain_size: (width, height, depth) of simulation domain
            smoothing_length: SPH smoothing length h
            particle_mass: Mass of each particle
            rest_density: Reference density ρ₀
            stiffness: Pressure stiffness κ (higher = more incompressible)
            gamma: Tait equation exponent γ
            viscosity_alpha: XSPH viscosity coefficient
            surface_tension_kappa: Surface tension coefficient
            boundary_friction_sigma: Boundary friction coefficient σ
            gravity: Gravity vector
            grid_resolution: Number of grid cells per dimension (adaptive based on domain)
            max_boundary_particles: Maximum boundary ghost particles
            eps: Small epsilon to avoid division by zero
        """
        self.max_particles = max_particles
        self.h = smoothing_length
        self.mass = particle_mass
        self.rho0 = rest_density
        self.kappa = stiffness
        self.gamma = gamma
        self.visc_alpha = viscosity_alpha
        self.surf_tension_kappa = surface_tension_kappa
        self.boundary_friction_sigma = boundary_friction_sigma
        self.eps = eps
        
        # Speed of sound for boundary friction
        self.speed_of_sound = (stiffness * gamma / max(rest_density, 1e-6)) ** 0.5
        
        # Compute grid dimensions based on domain and smoothing length
        cell_size = smoothing_length
        grid_x = max(8, int(domain_size[0] / cell_size) + 1)
        grid_y = max(8, int(domain_size[1] / cell_size) + 1)
        grid_z = max(8, int(domain_size[2] / cell_size) + 1)
        grid_size = (grid_x, grid_y, grid_z)
        
        # Particle data fields
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.densities = ti.field(dtype=ti.f32, shape=max_particles)
        self.pressures = ti.field(dtype=ti.f32, shape=max_particles)
        
        # Additional force fields
        self.surface_forces = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        
        # Simulation parameters (stored as fields for kernel access)
        self.n_particles = ti.field(dtype=ti.i32, shape=())
        self.gravity_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravity_field[None] = ti.math.vec3(*gravity)
        
        # Domain bounds
        self.domain_min = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.domain_max = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.domain_min[None] = ti.math.vec3(0, 0, 0)
        self.domain_max[None] = ti.math.vec3(*domain_size)
        
        # Spatial hash for neighbor search
        self.spatial_hash = SpatialHashGrid(grid_size, max_particles_per_cell=64)
        self.cell_size = cell_size
        
        # Boundary particles (ghost particles from rigid/static bodies)
        self.boundary_particles = BoundaryParticles(max_boundary_particles)
        
        # Separate spatial hash for boundary particles (ghost particles)
        # Boundary cells may be more crowded, so use higher capacity
        # Increased to 400 to handle dense mesh sampling (observed max ~314)
        self.boundary_hash = SpatialHashGrid(grid_size, max_particles_per_cell=1500)
        
        print(f"[TaichiSPH] Initialized with:")
        print(f"  - Max particles: {max_particles}")
        print(f"  - Max boundary particles: {max_boundary_particles}")
        print(f"  - Grid size: {grid_size}")
        print(f"  - Cell size (smoothing length h): {cell_size:.4f}")
        print(f"  - Domain: {domain_size}")
        print(f"  - Surface tension: {surface_tension_kappa}")
        print(f"  - Boundary friction: {boundary_friction_sigma}")
        print(f"  - Boundary hash max per cell: (for dense ghost particles)")
    
    def initialize_particles(self, positions_np: np.ndarray, velocities_np: np.ndarray):
        """Load initial particle data from numpy arrays.
        
        Args:
            positions_np: (N, 3) array of positions
            velocities_np: (N, 3) array of velocities
        """
        n = len(positions_np)
        assert n <= self.max_particles, f"Too many particles: {n} > {self.max_particles}"
        
        self.n_particles[None] = n
        self.positions.from_numpy(positions_np.astype(np.float32))
        self.velocities.from_numpy(velocities_np.astype(np.float32))
        
        print(f"[TaichiSPH] Loaded {n} particles")
    
    @ti.kernel
    def compute_density(self):
        """Compute density for all particles using SPH summation.
        
        Includes both fluid-fluid and fluid-boundary contributions.
        Boundary contributions are automatically skipped if n_boundary == 0.
        """
        n = self.n_particles[None]
        n_boundary = self.boundary_particles.n_particles[None]
        h = self.h
        mass = self.mass
        
        for i in range(n):
            rho = 0.0
            pos_i = self.positions[i]
            
            # Get cell index for particle i
            cell_i = ti.cast((pos_i - self.domain_min[None]) / self.cell_size, ti.i32)
            
            # Search 27 neighboring cells (3x3x3) for fluid-fluid contribution
            for offset_x in ti.static(range(-1, 2)):
                for offset_y in ti.static(range(-1, 2)):
                    for offset_z in ti.static(range(-1, 2)):
                        cell = cell_i + ti.math.ivec3(offset_x, offset_y, offset_z)
                        
                        if self.spatial_hash.is_valid_cell(cell):
                            n_neighbors = self.spatial_hash.get_cell_particle_count(cell)
                            
                            for k in range(n_neighbors):
                                j = self.spatial_hash.get_cell_particle(cell, k)
                                r_vec = pos_i - self.positions[j]
                                r = r_vec.norm()
                                
                                if r < h:
                                    rho += mass * poly6_kernel(r, h)
            
            # Add boundary contributions (automatically skipped if n_boundary == 0)
            if n_boundary > 0:
                for offset_x in ti.static(range(-1, 2)):
                    for offset_y in ti.static(range(-1, 2)):
                        for offset_z in ti.static(range(-1, 2)):
                            cell = cell_i + ti.math.ivec3(offset_x, offset_y, offset_z)
                            
                            if self.boundary_hash.is_valid_cell(cell):
                                n_boundary_neighbors = self.boundary_hash.get_cell_particle_count(cell)
                                
                                for k in range(n_boundary_neighbors):
                                    j = self.boundary_hash.get_cell_particle(cell, k)
                                    r_vec = pos_i - self.boundary_particles.positions[j]
                                    r = r_vec.norm()
                                    
                                    if r < h:
                                        rho += self.boundary_particles.pseudo_masses[j] * poly6_kernel(r, h)
            
            # Add self-contribution (CRITICAL: same as WCSPH.py line 46)
            # This ensures particle always has minimum density even if no neighbors
            rho += mass * poly6_kernel(0.0, h)  
            
            # Clamp to avoid numerical issues
            self.densities[i] = ti.max(rho, self.rho0 * 0.01)
    
    @ti.kernel
    def compute_pressure(self):
        """Compute pressure from density using Tait equation of state.
        
        Formula (same as WCSPH.py): p = (κ * ρ₀ / γ) * ((ρ/ρ₀)^γ - 1)
        """
        n = self.n_particles[None]
        
        for i in range(n):
            rho = self.densities[i]
            # Tait EOS: p = (κ * ρ₀ / γ) * ((ρ/ρ₀)^γ - 1) ？？？？？ in wcsph
            # This formula ensures p=0 when ρ=ρ₀
            ratio = rho / self.rho0
            self.pressures[i] = self.kappa * (ti.pow(ratio, self.gamma) - 1.0)
            # ratio = rho / self.rho0
            # factor = (self.kappa * self.rho0) / self.gamma
            # self.pressures[i] = factor * (ti.pow(ratio, self.gamma) - 1.0)ratio = rho / self.rho0
            # factor = (self.kappa * self.rho0) / self.gamma
            # self.pressures[i] = factor * (ti.pow(ratio, self.gamma) - 1.0)
    
    @ti.kernel
    def compute_forces(self):
        """Compute pressure, viscosity, and boundary pressure forces.
        
        Includes both fluid-fluid and fluid-boundary interactions.
        Boundary forces are automatically skipped if n_boundary == 0.
        """
        n = self.n_particles[None]
        n_boundary = self.boundary_particles.n_particles[None]
        h = self.h
        mass = self.mass
        
        for i in range(n):
            force = ti.math.vec3(0.0, 0.0, 0.0)
            pos_i = self.positions[i]
            vel_i = self.velocities[i]
            rho_i = self.densities[i]
            p_i = self.pressures[i]
            
            if rho_i < 1e-6:
                continue
            
            term_i = p_i / (rho_i * rho_i)
            cell_i = ti.cast((pos_i - self.domain_min[None]) / self.cell_size, ti.i32)
            
            # Fluid-fluid forces
            for offset_x in ti.static(range(-1, 2)):
                for offset_y in ti.static(range(-1, 2)):
                    for offset_z in ti.static(range(-1, 2)):
                        cell = cell_i + ti.math.ivec3(offset_x, offset_y, offset_z)
                        
                        if self.spatial_hash.is_valid_cell(cell):
                            n_neighbors = self.spatial_hash.get_cell_particle_count(cell)
                            
                            for k in range(n_neighbors):
                                j = self.spatial_hash.get_cell_particle(cell, k)
                                
                                if i != j:
                                    pos_j = self.positions[j]
                                    rho_j = self.densities[j]
                                    
                                    if rho_j < 1e-6:
                                        continue
                                    
                                    r_vec = pos_i - pos_j
                                    r = r_vec.norm()
                                    
                                    if r < h and r > 1e-6:
                                        # Pressure force
                                        p_j = self.pressures[j]
                                        term_j = p_j / (rho_j * rho_j)
                                        grad_w = spiky_grad_kernel(r_vec, h)
                                        f_pressure = -mass * mass * (term_i + term_j) * grad_w
                                        force += f_pressure
                                        
                                        # XSPH viscosity (velocity smoothing)
                                        vel_j = self.velocities[j]
                                        v_ij = vel_j - vel_i
                                        w = poly6_kernel(r, h)
                                        f_visc = mass * mass * (v_ij / rho_j) * w * self.visc_alpha
                                        force += f_visc
            
            # Boundary pressure forces (automatically skipped if n_boundary == 0)
            if n_boundary > 0:
                for offset_x in ti.static(range(-1, 2)):
                    for offset_y in ti.static(range(-1, 2)):
                        for offset_z in ti.static(range(-1, 2)):
                            cell = cell_i + ti.math.ivec3(offset_x, offset_y, offset_z)
                            
                            if self.boundary_hash.is_valid_cell(cell):
                                n_boundary_neighbors = self.boundary_hash.get_cell_particle_count(cell)
                                
                                for k in range(n_boundary_neighbors):
                                    j = self.boundary_hash.get_cell_particle(cell, k)
                                    r_vec = pos_i - self.boundary_particles.positions[j]
                                    r = r_vec.norm()
                                    
                                    if r < h and r > 1e-6:
                                        grad_w = spiky_grad_kernel(r_vec, h)
                                        b_mass = self.boundary_particles.pseudo_masses[j]
                                        # Boundary pressure force: F = -m_i * Ψ_j * (p_i/ρ_i²) * ∇W
                                        f_boundary = -mass * b_mass * term_i * grad_w
                                        force += f_boundary
            
            self.forces[i] = force
    
    @ti.kernel
    def compute_surface_tension(self):
        """Compute surface tension forces using color field method."""
        n = self.n_particles[None]
        h = self.h
        mass = self.mass
        kappa = self.surf_tension_kappa
        
        for i in range(n):
            force = ti.math.vec3(0.0, 0.0, 0.0)
            pos_i = self.positions[i]
            cell_i = ti.cast((pos_i - self.domain_min[None]) / self.cell_size, ti.i32)
            
            # Compute color field gradient and laplacian
            grad_c = ti.math.vec3(0.0, 0.0, 0.0)
            lapl_c = 0.0
            
            for offset_x in ti.static(range(-1, 2)):
                for offset_y in ti.static(range(-1, 2)):
                    for offset_z in ti.static(range(-1, 2)):
                        cell = cell_i + ti.math.ivec3(offset_x, offset_y, offset_z)
                        
                        if self.spatial_hash.is_valid_cell(cell):
                            n_neighbors = self.spatial_hash.get_cell_particle_count(cell)
                            
                            for k in range(n_neighbors):
                                j = self.spatial_hash.get_cell_particle(cell, k)
                                
                                if i != j:
                                    r_vec = pos_i - self.positions[j]
                                    r = r_vec.norm()
                                    
                                    if r < h and r > 1e-6:
                                        rho_j = self.densities[j]
                                        if rho_j > 1e-6:
                                            grad_c += (mass / rho_j) * poly6_gradient(r_vec, h)
                                            lapl_c += (mass / rho_j) * poly6_laplacian(r, h)
            
            # Surface normal and curvature
            grad_c_norm = grad_c.norm()
            if grad_c_norm > 1e-6:
                normal = grad_c / grad_c_norm
                force = -kappa * lapl_c * normal
            
            self.surface_forces[i] = force
    
    @ti.kernel
    def apply_boundary_friction(self, dt: ti.f32):
        """Apply boundary friction to velocities after integration using spatial hash.
        
        This is called AFTER integrate() to modify velocities, exactly matching
        WCSPH.py's _apply_boundary_interactions() which modifies new_velocities.
        
        Implements Müller et al. (2004) friction model:
        π_ij = -ν × min(v_ij·x_ij, 0) / (|x_ij|² + ε×h²)
        where ν = σ × h × c_s / (2ρ_i)
        
        Then applies: Δv = (F/m) × dt
        """
        n = self.n_particles[None]
        h = self.h
        mass = self.mass
        eps_term = 0.01  # ε term to avoid singularities (same as WCSPH.py)
        
        for i in range(n):
            total_force = ti.math.vec3(0.0, 0.0, 0.0)
            pos_i = self.positions[i]
            vel_i = self.velocities[i]
            rho_i = ti.max(self.densities[i], self.eps)
            
            # ν = σ h c_s / (2 ρ_i) - exact formula from WCSPH.py
            nu = self.boundary_friction_sigma * h * self.speed_of_sound / (2.0 * rho_i)
            
            # Get cell for this particle
            cell_i = ti.cast((pos_i - self.domain_min[None]) / self.cell_size, ti.i32)
            
            # Search neighboring cells for boundary particles
            for offset_x in ti.static(range(-1, 2)):
                for offset_y in ti.static(range(-1, 2)):
                    for offset_z in ti.static(range(-1, 2)):
                        cell = cell_i + ti.math.ivec3(offset_x, offset_y, offset_z)
                        
                        if self.boundary_hash.is_valid_cell(cell):
                            n_boundary_neighbors = self.boundary_hash.get_cell_particle_count(cell)
                            
                            for k in range(n_boundary_neighbors):
                                j = self.boundary_hash.get_cell_particle(cell, k)
                                r_vec = pos_i - self.boundary_particles.positions[j]  # x_ij
                                r_sq = r_vec.norm_sqr()
                                
                                if r_sq < h * h and r_sq > 1e-12:  # Within smoothing radius
                                    # Boundary velocity (from boundary_particles field)
                                    v_b = self.boundary_particles.velocities[j]
                                    v_ij = vel_i - v_b
                                    
                                    # Friction formula: only apply when approaching boundary
                                    vij_dot_xij = v_ij.dot(r_vec)
                                    limiter = ti.min(vij_dot_xij, 0.0)  # Only compress, not separate
                                    
                                    # π_ij = -ν × limiter / (|x_ij|² + ε×h²)
                                    pi_ij = -nu * limiter / (r_sq + eps_term * h * h)
                                    
                                    # F = -m × Ψ × π_ij × ∇W  (exact formula from WCSPH.py)
                                    grad_w = spiky_grad_kernel(r_vec, h)
                                    b_mass = self.boundary_particles.pseudo_masses[j]
                                    total_force += -mass * b_mass * pi_ij * grad_w
            
            # Apply friction force to velocity: Δv = (F/m) × dt
            if total_force.norm_sqr() > 1e-12:
                accel = total_force / mass
                self.velocities[i] += accel * dt
    
    @ti.kernel
    def integrate(self, dt: ti.f32, force_damp: ti.f32):
        """Symplectic Euler time integration (same as WCSPH.py).
        
        Note: Boundary friction is NOT included here. It's applied separately
        after integration via apply_boundary_friction(), matching WCSPH.py's
        _apply_boundary_interactions() behavior.
        """
        n = self.n_particles[None]
        gravity = self.gravity_field[None]
        mass = self.mass
        
        for i in range(n):
            # Total acceleration: a = (F_pressure + F_surface)/m + g
            # Note: boundary_forces NOT included here - applied separately after integration
            total_force = (self.forces[i] + self.surface_forces[i]) * force_damp
            acc = total_force / mass + gravity
            
            # Update velocity: v(t+dt) = v(t) + a*dt
            self.velocities[i] += acc * dt
            
            # Update position: x(t+dt) = x(t) + v(t+dt)*dt
            self.positions[i] += self.velocities[i] * dt
            
            # NOTE: No box boundary collision here!
            # Boundaries are handled purely through ghost particles (boundary_particles).
            # The domain_min/domain_max are only used for spatial hashing, not as physical walls.
    
    def step(self, dt: float, force_damp: float = 1.0):
        """Execute one simulation timestep.
        
        Args:
            dt: Time step size
            force_damp: Force damping factor
        """
        n = self.n_particles[None]
        
        if n == 0:
            return
        
        has_boundaries = self.boundary_particles.n_particles[None] > 0
        
        # 1. Build spatial hash for fluid particles
        self.spatial_hash.build(
            self.positions, 
            n,
            self.cell_size,
            self.domain_min[None]
        )
        
        # 1b. Build spatial hash for boundary particles (ghost particles)
        if has_boundaries:
            n_boundary = self.boundary_particles.n_particles[None]
            if not hasattr(self, '_boundary_hash_built_once'):
                print(f"[TaichiSPH] Building boundary hash for {n_boundary} particles (first time, may compile kernels)...")
                print(f"  - cell_size: {self.cell_size:.4f}")
                print(f"  - domain_min: {self.domain_min[None]}")
                self._boundary_hash_built_once = True
            
            self.boundary_hash.build(
                self.boundary_particles.positions,
                n_boundary,
                self.cell_size,
                self.domain_min[None]
            )
            
            # Check stats only on first build
            if not hasattr(self, '_boundary_stats_printed'):
                stats = self.boundary_hash.get_stats()
                print(f"  - Boundary hash built: occupancy={stats['occupancy_rate']:.2%}, max_per_cell={stats['max_particles_in_cell']}")
                if stats['overflow_warning']:
                    print(f"  [WARNING] Boundary hash overflow detected! Some cells exceeded capacity 400")
                    print(f"           Consider increasing max_particles_per_cell or refining mesh sampling")
                self._boundary_stats_printed = True
        
        # 2. Compute densities (unified function handles both fluid and boundary)
        if not hasattr(self, '_first_density_done'):
            boundary_msg = "with" if has_boundaries else "without"
            print(f"[TaichiSPH] Computing densities (first time, {boundary_msg} boundary interactions)...")
            print(f"  This may take a few minutes on first run...")
            if has_boundaries:
                n_boundary = self.boundary_particles.n_particles[None]
                print(f"  [TaichiSPH] n_boundary = {n_boundary}")
            self._first_density_done = True
        
        self.compute_density()
        
        if not hasattr(self, '_density_compiled'):
            print(f"[TaichiSPH] Density kernel compiled successfully")
            # Check if densities actually increased due to boundaries
            if has_boundaries:
                densities_np = self.get_densities()
                print(f"  [TaichiSPH] Density after boundary: min={densities_np.min():.2f}, max={densities_np.max():.2f}, avg={densities_np.mean():.2f}")
            self._density_compiled = True
        
        # 3. Compute pressures
        self.compute_pressure()
        
        # 4. Compute forces (unified function handles both fluid and boundary)
        self.compute_forces()
        
        # 5. Surface tension
        if self.surf_tension_kappa > 0:
            self.compute_surface_tension()
        
        # 6. Time integration (positions and velocities updated)
        # Note: This is done BEFORE boundary friction, matching WCSPH.py's flow
        self.integrate(dt, force_damp)
        
        # 7. Apply boundary friction to velocities (after integration)
        # This matches WCSPH.py's _apply_boundary_interactions() which modifies
        # velocities after integrate_symplectic()， here we don't apply friction to test
        if has_boundaries and self.boundary_friction_sigma > 0:
            self.apply_boundary_friction(dt)
        
        # Print confirmation after first complete step
        if not hasattr(self, '_first_step_complete'):
            print(f"[TaichiSPH] First simulation step completed successfully!")
            self._first_step_complete = True
    
    def get_positions(self) -> np.ndarray:
        """Get current particle positions as numpy array."""
        n = self.n_particles[None]
        return self.positions.to_numpy()[:n]
    
    def get_velocities(self) -> np.ndarray:
        """Get current particle velocities as numpy array."""
        n = self.n_particles[None]
        return self.velocities.to_numpy()[:n]
    
    def get_densities(self) -> np.ndarray:
        """Get current particle densities as numpy array."""
        n = self.n_particles[None]
        return self.densities.to_numpy()[:n]
    
    def get_grid_stats(self) -> dict:
        """Get spatial hash grid statistics."""
        return self.spatial_hash.get_stats()
    
    def get_boundary_stats(self) -> dict:
        """Get statistics about boundary particles."""
        n_boundary = self.boundary_particles.n_particles[None]
        if n_boundary == 0:
            return {
                'n_boundary': 0,
                'avg_pseudo_mass': 0,
                'boundary_hash_stats': {}
            }
        
        masses = self.boundary_particles.pseudo_masses.to_numpy()[:n_boundary]
        positions = self.boundary_particles.positions.to_numpy()[:n_boundary]
        
        return {
            'n_boundary': n_boundary,
            'avg_pseudo_mass': float(np.mean(masses)),
            'min_pseudo_mass': float(np.min(masses)),
            'max_pseudo_mass': float(np.max(masses)),
            'position_range': {
                'min': positions.min(axis=0).tolist(),
                'max': positions.max(axis=0).tolist()
            },
            'boundary_hash_stats': self.boundary_hash.get_stats()
        }
