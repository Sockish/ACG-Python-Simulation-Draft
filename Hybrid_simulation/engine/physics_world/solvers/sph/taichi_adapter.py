"""Adapter to make TaichiWCSPHSolver compatible with existing FluidState interface.

This wrapper allows using the Taichi-accelerated solver as a drop-in replacement
for the original WCSPH solver without changing existing simulation code.
"""
import taichi as ti
import numpy as np
from typing import List, Optional, Tuple

from ...state import FluidState, RigidBodyState, StaticBodyState
from ....configuration import LiquidBoxConfig
from .taichi_solver import TaichiWCSPHSolver

Vec3 = Tuple[float, float, float]


class TaichiSolverAdapter:
    """Adapter that wraps TaichiWCSPHSolver to work with FluidState."""
    
    def __init__(
        self,
        liquid_box: LiquidBoxConfig,
        gravity: Vec3,
        kappa: float = 3000.0,
        gamma: float = 5.0,
        viscosity_alpha: float = 0.25,
        boundary_friction_sigma: float = 4.0,
        surface_tension_kappa: float = 0.15,
        max_particles: Optional[int] = None,        max_boundary_particles: int = 300000,        use_gpu: bool = True,
        noise_amplitude: float = 0.003,
        noise_seed: Optional[int] = None,
    ):
        """Initialize Taichi solver with configuration.
        
        Args:
            liquid_box: Liquid box configuration
            gravity: Gravity vector
            kappa: Pressure stiffness
            gamma: Tait equation exponent
            viscosity_alpha: XSPH viscosity coefficient
            boundary_friction_sigma: Boundary friction coefficient
            surface_tension_kappa: Surface tension coefficient
            max_particles: Maximum particle count (auto-estimated if None)
            max_boundary_particles: Maximum ghost/boundary particles (default 300000)
            use_gpu: Try to use GPU backend (falls back to CPU if unavailable)
            noise_amplitude: Initial position noise amplitude
            noise_seed: Random seed for noise
        """
        self.liquid_box = liquid_box
        self.gravity = gravity
        self.smoothing_length = float(liquid_box.smoothing_length)  # Add attribute for compatibility
        self.noise_amplitude = noise_amplitude
        self._noise_rng = np.random.RandomState(noise_seed)
        
        # Taichi should already be initialized by calling script (simulate.py)
        # Skip re-initialization to avoid runtime errors
        
        # Estimate max particles if not provided
        if max_particles is None:
            spacing = float(liquid_box.particle_spacing)
            box_vol = 1.0
            for dim in ['x', 'y', 'z']:
                box_vol *= (liquid_box.max_corner[['x', 'y', 'z'].index(dim)] - 
                           liquid_box.min_corner[['x', 'y', 'z'].index(dim)])
            estimated = int(box_vol / (spacing ** 3))
            max_particles = int(estimated * 1.5)  # 50% buffer
            print(f"[TaichiAdapter] Estimated max particles: {max_particles}")
        
        # Calculate domain size
        min_corner = liquid_box.min_corner
        max_corner = liquid_box.max_corner
        domain_size = (
            max_corner[0] - min_corner[0],
            max_corner[1] - min_corner[1],
            max_corner[2] - min_corner[2],
        )
        
        # Create Taichi solver
        spacing = float(liquid_box.particle_spacing)
        smoothing_length = float(liquid_box.smoothing_length)
        particle_mass = liquid_box.rest_density * spacing ** 3
        
        self.taichi_solver = TaichiWCSPHSolver(
            max_particles=max_particles,
            domain_size=(10.0, 10.0, 10.0),  # Fixed domain for spatial hash: [-5, 5] in each dimension
            smoothing_length=smoothing_length,
            particle_mass=particle_mass,
            rest_density=liquid_box.rest_density,
            stiffness=kappa,
            gamma=gamma,
            viscosity_alpha=viscosity_alpha,
            surface_tension_kappa=surface_tension_kappa,
            boundary_friction_sigma=boundary_friction_sigma,
            gravity=gravity,
            max_boundary_particles=max_boundary_particles,
        )
        
        # Set domain bounds for spatial hashing (covers entire scene including boundaries)
        self.taichi_solver.domain_min[None] = ti.math.vec3(-5.0, -5.0, -5.0)
        self.taichi_solver.domain_max[None] = ti.math.vec3(5.0, 5.0, 5.0)
        
        print(f"[TaichiAdapter] Spatial hash domain: [-5, -5, -5] to [5, 5, 5]")
        print(f"[TaichiAdapter] Liquid box: {list(min_corner)} to {list(max_corner)}")
        
        self._initialized = False
    
    def initialize(self) -> FluidState:
        """Initialize fluid state with particles in a grid."""
        positions: List[Vec3] = []
        velocities: List[Vec3] = []
        
        spacing = float(self.liquid_box.particle_spacing)
        min_corner = tuple(self.liquid_box.min_corner)
        max_corner = tuple(self.liquid_box.max_corner)
        initial_velocity = tuple(self.liquid_box.initial_velocity)
        
        # Simple grid initialization with noise
        x = min_corner[0] + spacing * 0.5
        while x < max_corner[0] - spacing * 0.5 + 1e-6:
            y = min_corner[1] + spacing * 0.5
            while y < max_corner[1] - spacing * 0.5 + 1e-6:
                z = min_corner[2] + spacing * 0.5
                while z < max_corner[2] - spacing * 0.5 + 1e-6:
                    # Add noise to avoid artificial regularity
                    noise_x = self._noise_rng.uniform(-self.noise_amplitude, self.noise_amplitude)
                    noise_y = self._noise_rng.uniform(-self.noise_amplitude, self.noise_amplitude)
                    noise_z = self._noise_rng.uniform(-self.noise_amplitude, self.noise_amplitude)
                    positions.append((x + noise_x, y + noise_y, z + noise_z))
                    velocities.append(initial_velocity)
                    z += spacing
                y += spacing
            x += spacing
        
        print(f"[TaichiAdapter] Initialized {len(positions)} fluid particles.")
        
        # Convert to numpy and load into Taichi solver
        positions_np = np.array(positions, dtype=np.float32)
        velocities_np = np.array(velocities, dtype=np.float32)
        self.taichi_solver.initialize_particles(positions_np, velocities_np)
        self._initialized = True
        
        # Create FluidState for compatibility
        particle_mass = self.liquid_box.rest_density * spacing ** 3
        fluid_state = FluidState(
            positions=positions,
            velocities=velocities,
            densities=[self.liquid_box.rest_density] * len(positions),
            pressures=[0.0] * len(positions),
            particle_mass=particle_mass,
            smoothing_length=float(self.liquid_box.smoothing_length),
            rest_density=self.liquid_box.rest_density,
            bounds_min=min_corner,
            bounds_max=max_corner,
        )
        
        return fluid_state
    
    def step(
        self,
        fluid: FluidState,
        force_damp: float,
        dt: float,
        rigids: Optional[List[RigidBodyState]] = None,
        statics: Optional[List[StaticBodyState]] = None,
    ):
        """Execute one simulation step using Taichi solver.
        
        Args:
            fluid: FluidState to update (modified in-place)
            force_damp: Force damping factor
            dt: Time step size
            rigids: Rigid body states
            statics: Static body states
        """
        if not self._initialized:
            raise RuntimeError("Solver not initialized. Call initialize() first.")
        
        # Load/update boundary particles if present
        if (rigids or statics) and not hasattr(self, '_boundaries_loaded'):
            print(f"[TaichiAdapter] Loading boundary particles...")
            self.taichi_solver.boundary_particles.clear()
            
            # Load static boundaries
            if statics:
                for static in statics:
                    if hasattr(static, 'ghost_positions') and static.ghost_positions:
                        n_ghost = len(static.ghost_positions)
                        print(f"  [TaichiAdapter] Loading {n_ghost} ghost particles from static '{static.name}'")
                        self.taichi_solver.boundary_particles.load_static_boundaries(
                            static.ghost_positions,
                            static.ghost_normals,
                            static.ghost_pseudo_masses
                        )
            
            # Load rigid boundaries
            if rigids:
                for idx, rigid in enumerate(rigids):
                    if hasattr(rigid, 'ghost_local_positions') and rigid.ghost_local_positions:
                        # Transform local to world space
                        world_positions = rigid.get_world_ghost_particles()
                        if world_positions:
                            positions = [pos for pos, _ in world_positions]
                            normals = [normal for _, normal in world_positions]
                            n_ghost = len(positions)
                            print(f"  [TaichiAdapter] Loading {n_ghost} ghost particles from rigid '{rigid.name}'")
                            self.taichi_solver.boundary_particles.load_rigid_boundaries(
                                idx,
                                positions,
                                normals,
                                rigid.ghost_pseudo_masses
                            )
            
            total_boundary = self.taichi_solver.boundary_particles.n_particles[None]
            print(f"[TaichiAdapter] Total boundary particles loaded: {total_boundary}")
            self._boundaries_loaded = True
        
        # Update rigid body ghost particle positions each step
        if rigids:
            for idx, rigid in enumerate(rigids):
                if hasattr(rigid, 'ghost_local_positions') and rigid.ghost_local_positions:
                    # Update world positions based on rigid body transform
                    world_positions = rigid.get_world_ghost_particles()
                    if world_positions:
                        # TODO: Implement efficient GPU update
                        # For now, we assume boundaries don't move much
                        pass
        
        # Run Taichi simulation step with force damping
        self.taichi_solver.step(dt, force_damp)
        
        # Debug output every 10 steps
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1
        
        if self._step_count % 10 == 0:
            densities_np = self.taichi_solver.get_densities()
            pressures_np = self.taichi_solver.pressures.to_numpy()[:len(densities_np)]
            print(f"[TaichiAdapter Step {self._step_count}]")
            print(f"  - Density range: [{densities_np.min():.2f}, {densities_np.max():.2f}], avg={densities_np.mean():.2f}")
            print(f"  - Pressure range: [{pressures_np.min():.2f}, {pressures_np.max():.2f}], avg={pressures_np.mean():.2f}")
        
        # Copy results back to FluidState
        positions_np = self.taichi_solver.get_positions()
        velocities_np = self.taichi_solver.get_velocities()
        densities_np = self.taichi_solver.get_densities()
        pressures_np = self.taichi_solver.pressures.to_numpy()[:len(positions_np)]
        
        # Update FluidState
        n = len(positions_np)
        fluid.positions[:] = [tuple(p) for p in positions_np]
        fluid.velocities[:] = [tuple(v) for v in velocities_np]
        fluid.densities[:] = densities_np.tolist()
        fluid.pressures[:] = pressures_np.tolist()
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics from spatial hash grid."""
        return self.taichi_solver.get_grid_stats()
