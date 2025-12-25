"""
MPM solver - main Material Point Method simulation engine.
"""
import numpy as np
import taichi as ti
from typing import Optional

from .mpm_state import MPMState
from .mpm_grid import MPMGrid
from .mpm_materials import compute_stress, MATERIAL_WATER
from .mpm_kernels import compute_grid_influence
from .mpm_boundary import MPMBoundary


@ti.data_oriented
class MPMSolver:
    """Main MPM solver using Taichi for GPU acceleration."""
    
    def __init__(self, 
                 max_particles: int = 500000,
                 grid_resolution: int = 128,
                 domain_min: float = -2.0,
                 domain_max: float = 2.0,
                 dt: float = 1e-4,
                 gravity: tuple = (0.0, 0.0, -9.81),
                 bulk_modulus: float = 500.0,
                 youngs_modulus: float = 0.0,
                 boundary_mode: str = 'bounce'):
        """
        Initialize MPM solver.
        
        Args:
            max_particles: Maximum number of particles
            grid_resolution: Grid resolution (e.g., 64 for 64^3 grid)
            domain_min: Minimum coordinate of simulation domain
            domain_max: Maximum coordinate of simulation domain
            dt: Time step size
            gravity: Gravity vector (m/s^2)
            bulk_modulus: Bulk modulus for water (similar to kappa in SPH)
            youngs_modulus: Young's modulus (0 for ideal fluid, >0 for elasticity)
            boundary_mode: Boundary condition mode ('sticky', 'slip', 'separate', 'bounce')
        """
        # Simulation parameters
        self.dt = dt
        self.gravity = ti.Vector(gravity)
        self.bulk_modulus = bulk_modulus
        
        # Material parameters - allow override from config
        self.E = youngs_modulus  # 0 for ideal fluid, >0 for elasticity
        self.nu = 0.2    # Poisson's ratio (only used if E>0)
        self.mu = self.E / (2.0 * (1.0 + self.nu)) if self.E > 0 else 0.0
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu)) if self.E > 0 else 0.0
        
        # Initialize components
        self.state = MPMState(max_particles)
        self.grid = MPMGrid(grid_resolution, domain_min, domain_max)
        self.boundary = MPMBoundary(grid_resolution, boundary_mode)
        
        # Store parameters as Taichi fields for kernel access
        self.dt_field = ti.field(dtype=ti.f32, shape=())
        self.dt_field[None] = dt
        
        self.gravity_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravity_field[None] = self.gravity
        
        self.material_params = ti.field(dtype=ti.f32, shape=4)
        self.material_params[0] = bulk_modulus
        self.material_params[1] = self.mu
        self.material_params[2] = self.lam
        self.material_params[3] = 0.0  # Reserved
        
        # Debug tracking
        self.step_count = 0
        self.debug_interval = 200  # Output debug info every 50 steps
        self.leaked_count = ti.field(dtype=ti.i32, shape=())  # Count particles below z=-1
    
    def initialize_box(self, box_min: np.ndarray, box_max: np.ndarray, 
                       spacing: float, density: float, 
                       initial_velocity: Optional[np.ndarray] = None):
        """
        Initialize water particles within a spherical volume derived from box bounds.
        """
        if initial_velocity is None:
            initial_velocity = np.array([0.0, 0.0, 0.0])
        
        # 1. Geometry Calculation
        box_size = box_max - box_min
        sphere_center = (box_min + box_max) / 2.0
        # BUG FIX: Use the maximum dimension to define radius to ensure a uniform sphere.
        # To prevent clipping, the sampling box must be a cube covering 2 * radius.
        sphere_radius = 0.5 * np.max(box_size)
        
        # 2. Particle Count Estimation
        sphere_volume = (4.0 / 3.0) * np.pi * (sphere_radius ** 3)
        dx = self.grid.dx
        # Standard MPM practice: 8 particles per cell (2x2x2)
        particle_volume = (dx * 0.5) ** 3
        n_particles_target = int(sphere_volume / particle_volume)
        
        print(f"[MPMSolver] Initializing SPHERE: center={sphere_center}, radius={sphere_radius:.3f}m")
        print(f"[MPMSolver] Target particle count: {n_particles_target}")
    
        # 3. Vectorized Rejection Sampling
        # We oversample because the volume of a sphere is ~52.4% of its bounding cube.
        oversample_factor = 2.2 
        n_to_sample = int(n_particles_target * oversample_factor)
        
        # CRITICAL FIX: Define a sampling cube that fully contains the sphere
        sample_min = sphere_center - sphere_radius
        sample_max = sphere_center + sphere_radius
        
        # Generate all candidate points at once for high performance
        candidate_pos = sample_min + np.random.rand(n_to_sample, 3) * (2 * sphere_radius)
        
        # Calculate squared distance to center (avoiding sqrt for speed)
        dists_sq = np.sum((candidate_pos - sphere_center)**2, axis=1)
        
        # Filter points that reside within the sphere radius
        inside_mask = dists_sq <= (sphere_radius ** 2)
        positions = candidate_pos[inside_mask]
        
        # Truncate to the exact target count to maintain consistency
        if len(positions) > n_particles_target:
            positions = positions[:n_particles_target]
        
        positions = positions.astype(np.float32)
        n_particles = len(positions)
    
        # 4. State Assignment
        # Set spatial coordinates
        self.state.set_positions(positions)
        
        # Set uniform initial velocity
        velocities = np.tile(initial_velocity, (n_particles, 1)).astype(np.float32)
        self.state.set_velocities(velocities)
        
        # Calculate physical mass based on material density
        particle_mass = density * particle_volume
        
        # Initialize solver-specific particle properties (MATERIAL_WATER = 0)
        self.state.initialize_particles(n_particles, particle_mass, particle_volume, MATERIAL_WATER)
        
        print(f"[MPMSolver] Successfully generated {n_particles} particles.")
        print(f"[MPMSolver] Particle mass: {particle_mass:.6e} kg, volume: {particle_volume:.6e} m^3")
        
        self.material_type = 'water'
    
    def load_static_bodies(self, static_bodies: list):
        """
        Load static mesh obstacles for collision detection.
        
        Args:
            static_bodies: List of StaticBodyState objects
        """
        if not static_bodies:
            print("[MPMSolver] No static bodies to load")
            return
        
        print(f"[MPMSolver] Loading {len(static_bodies)} static bodies...")
        
        for body in static_bodies:
            print(f"[MPMSolver] Loading static body: {body.name}")
            print(f"  Vertices: {len(body.vertices)}, Faces: {len(body.faces)}")
            
            # Convert to numpy arrays
            vertices = np.array(body.vertices, dtype=np.float32)
            triangles = np.array(body.faces, dtype=np.int32)
            
            # Load into boundary handler (voxelize onto grid)
            self.boundary.load_static_mesh(
                vertices=vertices,
                triangles=triangles,
                domain_min=self.grid.domain_min,
                domain_max=self.grid.domain_max
            )
        
        print(f"[MPMSolver] Static bodies loaded successfully")
    
    @ti.kernel
    def particle_to_grid(self):
        """P2G: Transfer particle data to grid (momentum and mass)."""
        # Clear grid
        for I in ti.grouped(self.grid.grid_v):
            self.grid.grid_v[I] = ti.Vector.zero(ti.f32, 3)
            self.grid.grid_m[I] = 0.0
        
        # Particle to grid transfer
        n = self.state.n_particles[None]
        inv_dx = self.grid.grid_params[3]
        
        for p in range(n):
            # Particle position in grid coordinates
            Xp = self.grid.world_to_grid(self.state.x[p])
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)
            
            # Quadratic B-spline weights
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            # CRITICAL: Update F BEFORE computing stress (like mpm_3d_sim.py)
            self.state.F[p] = (ti.Matrix.identity(ti.f32, 3) + self.dt_field[None] * self.state.C[p]) @ self.state.F[p]
            
            # For WATER: Reset F to avoid shear accumulation
            mat = self.state.material[p]
            if mat == 0:  # MATERIAL_WATER
                J = self.state.F[p].determinant()
                self.state.F[p] = ti.Matrix.identity(ti.f32, 3) * ti.pow(J, 1.0 / 3.0)
            
            # Update J
            self.state.J[p] = self.state.F[p].determinant()
            
            # Compute stress using mpm_3d_sim.py formula
            stress = compute_stress(mat, self.state.F[p], self.state.J[p],
                                   self.material_params[0], self.material_params[1], 
                                   self.material_params[2])
            
            # Affine momentum from APIC
            affine = stress * (-4.0 * inv_dx * inv_dx * self.dt_field[None]) * self.state.volume[p]
            affine += self.state.mass[p] * self.state.C[p]
            
            # Scatter to grid
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (ti.cast(offset, ti.f32) - fx) * self.grid.grid_params[2]
                weight = w[i].x * w[j].y * w[k].z
                grid_pos = base + offset
                
                if self.grid.is_valid_grid_pos(grid_pos):
                    # Momentum transfer
                    momentum = weight * (self.state.mass[p] * self.state.v[p] + affine @ dpos)
                    self.grid.grid_v[grid_pos] += momentum
                    self.grid.grid_m[grid_pos] += weight * self.state.mass[p]
    
    @ti.kernel
    def grid_operations(self):
        """Update grid velocities (convert momentum to velocity, add gravity)."""
        dt = self.dt_field[None]
        gravity = self.gravity_field[None]
        
        for I in ti.grouped(self.grid.grid_v):
            if self.grid.grid_m[I] > 0:
                # Convert momentum to velocity
                self.grid.grid_v[I] /= self.grid.grid_m[I]
                
                # Add gravity
                self.grid.grid_v[I] += dt * gravity
    
    @ti.kernel
    def apply_domain_boundaries(self):
        """Apply boundary conditions at domain edges (after collision)."""
        # Domain boundary conditions (sticky walls at simulation box edges)
        bound = 3  # Boundary thickness
        for I in ti.grouped(self.grid.grid_v):
            if self.grid.grid_m[I] > 0:
                if self.grid.is_boundary_node(I, bound):
                    self.grid.grid_v[I] = ti.Vector.zero(ti.f32, 3)
    
    def apply_static_collisions(self):
        """Apply collisions with static meshes (called after grid_operations)."""
        if self.boundary.has_static_meshes:
            self.boundary.apply_static_collision(self.grid.grid_v, self.grid.grid_m)
    
    @ti.kernel
    def grid_to_particle(self):
        """G2P: Transfer grid velocities back to particles and update positions."""
        n = self.state.n_particles[None]
        dt = self.dt_field[None]
        inv_dx = self.grid.grid_params[3]
        
        for p in range(n):
            # Particle position in grid coordinates
            Xp = self.grid.world_to_grid(self.state.x[p])
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)
            
            # Quadratic weights
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            # Gather from grid
            new_v = ti.Vector.zero(ti.f32, 3)
            new_C = ti.Matrix.zero(ti.f32, 3, 3)
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (ti.cast(offset, ti.f32) - fx) * self.grid.grid_params[2]
                weight = w[i].x * w[j].y * w[k].z
                grid_pos = base + offset
                
                if self.grid.is_valid_grid_pos(grid_pos):
                    g_v = self.grid.grid_v[grid_pos]
                    new_v += weight * g_v
                    new_C += 4.0 * inv_dx * weight * g_v.outer_product(dpos)
            
            # Update particle velocity and affine matrix
            self.state.v[p] = new_v
            self.state.C[p] = new_C
            
            # Update particle position
            self.state.x[p] += dt * new_v
            
            # NOTE: F is updated in P2G phase, NOT here!
    
    @ti.kernel
    def count_leaked_particles(self) -> ti.i32:
        """Count particles with z < -0 (leaked through floor)."""
        count = 0
        for p in range(self.state.n_particles[None]):
            if self.state.x[p].z < -0.0:
                count += 1
        return count
    
    @ti.kernel
    def get_particle_bounds(self, min_pos: ti.template(), max_pos: ti.template()):
        """Calculate bounding box of all particles."""
        n = self.state.n_particles[None]
        if n > 0:
            # Initialize with first particle
            min_pos[None] = self.state.x[0]
            max_pos[None] = self.state.x[0]
            
            # Find min/max
            for p in range(1, n):
                for d in ti.static(range(3)):
                    ti.atomic_min(min_pos[None][d], self.state.x[p][d])
                    ti.atomic_max(max_pos[None][d], self.state.x[p][d])
    
    def step(self):
        """Advance simulation by one time step."""
        self.particle_to_grid()
        self.grid_operations()  # Convert momentum->velocity, add gravity
        self.apply_static_collisions()  # Apply static mesh collisions ONLY
        # DISABLED: Domain boundaries removed to allow free expansion
        # self.apply_domain_boundaries()
        self.grid_to_particle()
        
        # Debug output every N steps
        self.step_count += 1
        if self.step_count % self.debug_interval == 0:
            n_particles = self.state.n_particles[None]
            leaked = self.count_leaked_particles()
            leak_pct = (leaked / n_particles * 100.0) if n_particles > 0 else 0.0
            
            # Calculate particle bounds
            min_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
            max_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.get_particle_bounds(min_pos, max_pos)
            
            # Grid info
            dx = self.grid.dx
            
            print(f"[MPM Step {self.step_count}] Particles: {n_particles}, Leaked: {leaked} ({leak_pct:.2f}%)")
            print(f"  Grid: dx={dx:.4f}m ({self.grid.n_grid}³ cells), Domain: [{self.grid.domain_min:.1f}, {self.grid.domain_max:.1f}]")
            if n_particles > 0:
                pmin = min_pos[None]
                pmax = max_pos[None]
                print(f"  Particles range: X[{pmin[0]:.3f}, {pmax[0]:.3f}], Y[{pmin[1]:.3f}, {pmax[1]:.3f}], Z[{pmin[2]:.3f}, {pmax[2]:.3f}]")
    
    def get_state(self) -> MPMState:
        """Get current particle state."""
        return self.state
