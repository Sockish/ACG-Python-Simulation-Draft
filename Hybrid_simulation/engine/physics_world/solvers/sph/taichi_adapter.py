"""Adapter to make TaichiWCSPHSolver compatible with existing FluidState interface.

This wrapper allows using the Taichi-accelerated solver as a drop-in replacement
for the original WCSPH solver without changing existing simulation code.
"""
import taichi as ti
import numpy as np
from typing import Dict, List, Optional, Tuple

from ...state import FluidState, RigidBodyState, StaticBodyState
from ....configuration import LiquidBoxConfig
from .taichi_solver import TaichiWCSPHSolver
from ...math_utils import add, cross, sub

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
        max_particles: Optional[int] = None,
        max_boundary_particles: int = 50000000,
        use_gpu: bool = True,
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
            max_boundary_particles: Maximum ghost/boundary particles (default 500000)
            use_gpu: Try to use GPU backend (falls back to CPU if unavailable)
            noise_amplitude: Initial position noise amplitude
            noise_seed: Random seed for noise
        """
        self.liquid_box = liquid_box
        self.gravity = gravity
        self.smoothing_length = float(liquid_box.smoothing_length)
        self.noise_amplitude = noise_amplitude
        self._noise_rng = np.random.RandomState(noise_seed)
        self.boundary_friction_sigma = boundary_friction_sigma
        self.rest_density = float(liquid_box.rest_density)
        self.eps = 1e-6
        spacing = float(liquid_box.particle_spacing)
        self.particle_mass = self.rest_density * spacing ** 3 if spacing > 0 else 0.0
        self.speed_of_sound = (kappa * gamma / max(self.rest_density, 1e-6)) ** 0.5
        self._static_boundaries_loaded = False
        self._rigid_ranges: List[Dict[str, int]] = []
        
        # Taichi should already be initialized by calling script (simulate.py)
        # Skip re-initialization to avoid runtime errors
        
        # Estimate max particles if not provided
        if max_particles is None:
            min_corner_np = np.array(liquid_box.min_corner, dtype=np.float32)
            max_corner_np = np.array(liquid_box.max_corner, dtype=np.float32)
            extent = np.maximum(max_corner_np - min_corner_np, 1e-6)
            box_vol = float(extent.prod())
            estimated = int(box_vol / max(spacing ** 3, 1e-12))
            max_particles = int(estimated * 1.5)  # 50% buffer
            print(f"[TaichiAdapter] Estimated max particles: {max_particles}")
        
        min_corner = np.array(liquid_box.min_corner, dtype=np.float32)
        max_corner = np.array(liquid_box.max_corner, dtype=np.float32)
        margin = max(2.0 * self.smoothing_length, spacing)
        domain_min = min_corner - margin
        domain_max = max_corner + margin
        domain_size = tuple((domain_max - domain_min).tolist())

        # Create Taichi solver
        self.taichi_solver = TaichiWCSPHSolver(
            max_particles=max_particles,
            domain_size=domain_size,
            smoothing_length=self.smoothing_length,
            particle_mass=self.particle_mass,
            rest_density=self.rest_density,
            stiffness=kappa,
            gamma=gamma,
            viscosity_alpha=viscosity_alpha,
            surface_tension_kappa=surface_tension_kappa,
            boundary_friction_sigma=boundary_friction_sigma,
            gravity=gravity,
            max_boundary_particles=max_boundary_particles,
        )
        self.taichi_solver.domain_min[None] = ti.math.vec3(*domain_min.tolist())
        self.taichi_solver.domain_max[None] = ti.math.vec3(*domain_max.tolist())
        
        print(f"[TaichiAdapter] Spatial hash domain: {domain_min.tolist()} to {domain_max.tolist()}")
        print(f"[TaichiAdapter] Liquid box: {min_corner.tolist()} to {max_corner.tolist()}")
        
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
        fluid_state = FluidState(
            positions=positions,
            velocities=velocities,
            densities=[self.liquid_box.rest_density] * len(positions),
            pressures=[0.0] * len(positions),
            particle_mass=self.particle_mass,
            smoothing_length=float(self.liquid_box.smoothing_length),
            rest_density=self.liquid_box.rest_density,
            bounds_min=min_corner,
            bounds_max=max_corner,
        )
        
        return fluid_state
    
    def _reload_boundary_particles(
        self,
        rigids: Optional[List[RigidBodyState]],
        statics: Optional[List[StaticBodyState]],
    ) -> bool:
        """Upload or refresh boundary particle data inside Taichi buffers."""
        has_boundaries = False
        
        if statics and not self._static_boundaries_loaded:
            for static in statics:
                if getattr(static, "ghost_positions", None):
                    n_ghost = len(static.ghost_positions)
                    if n_ghost == 0:
                        continue
                    positions = [tuple(map(float, p)) for p in static.ghost_positions]
                    normals = [tuple(map(float, n)) for n in static.ghost_normals]
                    pseudo_masses = list(static.ghost_pseudo_masses)
                    if len(pseudo_masses) < len(positions):
                        pseudo_masses.extend([0.0] * (len(positions) - len(pseudo_masses)))
                    self.taichi_solver.boundary_particles.load_static_boundaries(
                        positions,
                        normals,
                        pseudo_masses,
                    )
                    has_boundaries = True
            if has_boundaries:
                self._static_boundaries_loaded = True
        
        has_boundaries = has_boundaries or self._static_boundaries_loaded
        
        if rigids:
            for idx, rigid in enumerate(rigids):
                if not getattr(rigid, "ghost_local_positions", None):
                    continue
                world_samples = rigid.get_world_ghost_particles()
                if not world_samples:
                    continue
                
                positions = [tuple(map(float, pos)) for pos, _ in world_samples]
                normals = [tuple(map(float, normal)) for _, normal in world_samples]
                pseudo_masses = list(rigid.ghost_pseudo_masses)
                if len(pseudo_masses) < len(positions):
                    pseudo_masses.extend([0.0] * (len(positions) - len(pseudo_masses)))
                velocities = []
                for pos in positions:
                    rel = sub(pos, rigid.position)
                    v_b = add(rigid.linear_velocity, cross(rigid.angular_velocity, rel))
                    velocities.append(v_b)
                
                record = next((entry for entry in self._rigid_ranges if entry["body_index"] == idx), None)
                if record is None:
                    start_idx = self.taichi_solver.boundary_particles.load_rigid_boundaries(
                        idx,
                        positions,
                        normals,
                        pseudo_masses,
                        velocities,
                    )
                    self._rigid_ranges.append(
                        {"body_index": idx, "start": start_idx, "count": len(positions)}
                    )
                else:
                    if record["count"] != len(positions):
                        # Rigid sampling changed significantly – rebuild all boundary buffers
                        self.taichi_solver.boundary_particles.clear()
                        self._static_boundaries_loaded = False
                        self._rigid_ranges.clear()
                        return self._reload_boundary_particles(rigids, statics)
                    self.taichi_solver.boundary_particles.update_rigid_range(
                        record["start"],
                        positions,
                        normals,
                        velocities,
                    )
                has_boundaries = True
        
        total_boundary = self.taichi_solver.boundary_particles.n_particles[None]
        if has_boundaries and total_boundary > 0 and not hasattr(self, "_boundary_stats_printed"):
            masses_np = self.taichi_solver.boundary_particles.pseudo_masses.to_numpy()[:total_boundary]
            positions_np = self.taichi_solver.boundary_particles.positions.to_numpy()[:total_boundary]
            print(f"[TaichiAdapter] Boundary ghost particles: {total_boundary}")
            print(
                f"  [TaichiAdapter] Pseudo mass stats: "
                f"min={masses_np.min():.6f}, max={masses_np.max():.6f}, avg={masses_np.mean():.6f}"
            )
            print(
                f"  [TaichiAdapter] Position range: "
                f"x=[{positions_np[:,0].min():.2f}, {positions_np[:,0].max():.2f}], "
                f"y=[{positions_np[:,1].min():.2f}, {positions_np[:,1].max():.2f}], "
                f"z=[{positions_np[:,2].min():.2f}, {positions_np[:,2].max():.2f}]"
            )
            self._boundary_stats_printed = True
        
        return has_boundaries
    
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
        
        has_boundaries = self._reload_boundary_particles(rigids, statics)
        
        self.taichi_solver.step(dt, force_damp)
        
        # Debug output every 10 steps
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1
        
        if self._step_count % 100 == 0:
            densities_np = self.taichi_solver.get_densities()
            pressures_np = self.taichi_solver.pressures.to_numpy()[:len(densities_np)]
            positions_np = self.taichi_solver.get_positions()
            velocities_np = self.taichi_solver.get_velocities()
            print(f"[TaichiAdapter Step {self._step_count}]")
            print(f"  - Density range: [{densities_np.min():.2f}, {densities_np.max():.2f}], avg={densities_np.mean():.2f}")
            print(f"  - Pressure range: [{pressures_np.min():.2f}, {pressures_np.max():.2f}], avg={pressures_np.mean():.2f}")
            print(f"  - Velocity magnitude range: [{np.linalg.norm(velocities_np, axis=1).min():.2f}, "
                  f"{np.linalg.norm(velocities_np, axis=1).max():.2f}], avg={np.linalg.norm(velocities_np, axis=1).mean():.2f}")
            print(f"  - Fluid position range: z=[{positions_np[:, 2].min():.3f}, {positions_np[:, 2].max():.3f}]")
            
            # Check for particles falling through floor
            if positions_np[:, 2].min() < -1.0:
                below_count = np.sum(positions_np[:, 2] < -1.0)
                proportion = below_count / len(positions_np) * 100
                print(f"  [WARNING] {below_count}/{len(positions_np)} particles below z=-1.0 (floor)! Proportion: {proportion:.2f}%")
        
        # Copy results back to FluidState
        positions_np = self.taichi_solver.get_positions()
        densities_np = self.taichi_solver.get_densities()
        pressures_np = self.taichi_solver.pressures.to_numpy()[:len(positions_np)]
        velocities_np = self.taichi_solver.get_velocities()
        
        # Update FluidState directly from Taichi fields
        fluid.positions[:] = [tuple(map(float, p)) for p in positions_np]
        fluid.velocities[:] = [tuple(map(float, v)) for v in velocities_np]
        fluid.densities[:] = densities_np.tolist()
        fluid.pressures[:] = pressures_np.tolist()
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics from spatial hash grid."""
        return self.taichi_solver.get_grid_stats()
