"""Physics world core that orchestrates individual solvers."""

from __future__ import annotations

from dataclasses import dataclass

try:  # tqdm is optional but useful for long sampling runs
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is missing
    tqdm = None

from .solvers.sph.WCSPH import WCSphSolver
from .solvers.sph.utils.ghost_particles import sample_mesh_surface, compute_local_pseudo_masses
from .solvers.sph.taichi_ghost_sampler import TaichiGhostSampler

from ..configuration import SceneConfig
from ..mesh_utils import load_obj_mesh, triangulate_faces
from .math_utils import Vec3, quaternion_to_matrix, transform_point
from .state import FluidState, RigidBodyState, StaticBodyState, WorldSnapshot
from .solvers import (
    RigidBodySolver,
    RigidStaticSolver,
    RigidRigidSolver,
)


@dataclass
class PhysicsWorld:
    config: SceneConfig
    fluid_solver: WCSphSolver | None
    mpm_solver: any | None  # MPM solver (imported dynamically)
    rigid_solver: RigidBodySolver
    rigid_rigid_solver: RigidRigidSolver
    #fluid_rigid_solver: FluidRigidCouplingSolver
    #fluid_static_solver: FluidStaticSolver
    rigid_static_solver: RigidStaticSolver
    fluid_state: FluidState | None
    rigid_states: list[RigidBodyState]
    static_states: list[StaticBodyState]
    current_time: float = 0.0
    current_step: int = 0

    @classmethod
    def from_config(cls, config: SceneConfig, use_taichi: bool = False, use_mpm: bool = False) -> "PhysicsWorld":
        gravity = _vec(config.simulation.gravity)
        
        # ==== STEP 1: Initialize rigid and static bodies FIRST (before fluid) ====
        
        # Initialize rigid body solver and states
        rigid_solver = RigidBodySolver(config.rigid_bodies, gravity)
        rigid_rigid_solver = RigidRigidSolver()
        rigid_static_solver = RigidStaticSolver()
        rigid_states = rigid_solver.initialize()
        
        # Initialize static bodies with proper transformations
        static_states = []
        for body in config.static_bodies:
            mesh = load_obj_mesh(body.mesh_path)
            triangles = triangulate_faces(mesh.faces)
            
            # Apply initial_position and initial_orientation transformation
            initial_pos = _vec(body.initial_position)
            initial_ori = tuple(body.initial_orientation)
            rotation_matrix = quaternion_to_matrix(initial_ori)
            
            # Transform all vertices from local OBJ space to world space
            transformed_vertices = [
                transform_point(v, rotation_matrix, initial_pos)
                for v in mesh.vertices
            ]
            
            # Compute transformed bounds
            if transformed_vertices:
                xs = [v[0] for v in transformed_vertices]
                ys = [v[1] for v in transformed_vertices]
                zs = [v[2] for v in transformed_vertices]
                world_min = (min(xs), min(ys), min(zs))
                world_max = (max(xs), max(ys), max(zs))
            else:
                world_min, world_max = mesh.bounds()
            
            print(f"[StaticInit] {body.name}: pos={initial_pos}, ori={initial_ori}")
            print(f"  Original bounds: {mesh.bounds()}")
            print(f"  Transformed bounds: {world_min} to {world_max}")
            
            static_states.append(
                StaticBodyState(
                    name=body.name,
                    mesh_path=body.mesh_path,
                    position=initial_pos,  # Store the applied transformation
                    orientation=initial_ori,  # Store the applied transformation
                    local_bounds_min=world_min,  # Now these are world-space bounds
                    local_bounds_max=world_max,  # Now these are world-space bounds
                    vertices=transformed_vertices,  # Already in world space
                    faces=triangles,
                )
            )
        
        # ==== STEP 2: Initialize fluid solver AFTER rigid/static bodies are ready ====
        mpm_solver = None
        
        print(f"[PhysicsWorld] Checking MPM initialization: use_mpm={use_mpm}, has_mpm_box={config.mpm_box is not None}")
        if config.mpm_box:
            print(f"[PhysicsWorld] MPM box config: {config.mpm_box}")
        
        if use_mpm and config.mpm_box:
            # Initialize MPM solver
            from .solvers.mpm import MPMSolver
            import numpy as np
            print("[PhysicsWorld] Using MPM solver")
            
            mpm_solver = MPMSolver(
                max_particles=1000000,
                grid_resolution=config.mpm_box.grid_resolution,
                domain_min=config.mpm_box.domain_min,
                domain_max=config.mpm_box.domain_max,
                dt=config.simulation.time_step,
                gravity=gravity,
                bulk_modulus=config.mpm_box.bulk_modulus,
                youngs_modulus=config.mpm_box.youngs_modulus,
                poisson_ratio=config.mpm_box.poisson_ratio,
                material_type=config.mpm_box.material_type,
                boundary_mode=config.mpm_box.boundary_mode,
            )
            
            # Initialize MPM particles
            mpm_solver.initialize_box(
                box_min=np.array(config.mpm_box.min_corner),
                box_max=np.array(config.mpm_box.max_corner),
                spacing=config.mpm_box.particle_spacing,
                density=config.mpm_box.density,
                initial_velocity=np.array(config.mpm_box.initial_velocity),
            )
            
            # Load static bodies into MPM for collision detection
            print(f"[PhysicsWorld] Loading {len(static_states)} static bodies into MPM...")
            mpm_solver.load_static_bodies(static_states)
            
            fluid_solver = None
            fluid_state = None
            
        elif config.liquid_box:
            if use_taichi:
                # Use Taichi-accelerated solver
                from .solvers.sph.taichi_adapter import TaichiSolverAdapter
                print("[PhysicsWorld] Using Taichi SPH solver")
                fluid_solver = TaichiSolverAdapter(
                    liquid_box=config.liquid_box,
                    gravity=gravity,
                )
            else:
                # Use original Numba solver
                fluid_solver = WCSphSolver(
                    liquid_box=config.liquid_box,
                    gravity=gravity)
            fluid_state = fluid_solver.initialize()
        else:
            fluid_solver = None
            fluid_state = None

        # Ghost particle sampling (only for SPH, not MPM)
        if not use_mpm:
            smoothing_length = fluid_solver.smoothing_length if fluid_solver else None
            rest_density = fluid_solver.liquid_box.rest_density if fluid_solver else None
            if smoothing_length and rest_density:
                # Use GPU sampler for Taichi, CPU for WCSPH
                if use_taichi:
                    print(f"[GhostInit] Using Taichi GPU sampler | h={smoothing_length:.4f}, rho0={rest_density:.2f}")
                    import sys
                    sys.stdout.flush()
                    sampler = TaichiGhostSampler()
                    sample_func = sampler.sample_mesh_surface
                    mass_func = sampler.compute_local_pseudo_masses
                else:
                    print(f"[GhostInit] Using NumPy CPU sampler | h={smoothing_length:.4f}, rho0={rest_density:.2f}")
                    import sys
                    sys.stdout.flush()
                    sample_func = sample_mesh_surface
                    mass_func = compute_local_pseudo_masses
                
                print(f"[GhostInit] Starting rigid body sampling ({len(rigid_states)} bodies)...")
                sys.stdout.flush()
                for rigid in rigid_states:                
                    print(f"  [GhostInit][Rigid:{rigid.name}] Sampling {len(rigid.triangles)} triangles...")
                    sys.stdout.flush()
                    samples = sample_func(rigid.centered_vertices, rigid.triangles, smoothing_length)
                    rigid.ghost_local_positions = [pos for pos, _ in samples]
                    rigid.ghost_local_normals = [normal for _, normal in samples]
                    print(f"  [GhostInit][Rigid:{rigid.name}] Computing masses for {len(rigid.ghost_local_positions)} samples...")
                    sys.stdout.flush()
                    rigid.ghost_pseudo_masses = mass_func(
                        rigid.ghost_local_positions,
                        smoothing_length,
                        rest_density,
                    )
                    print(f"  [GhostInit][Rigid:{rigid.name}] Complete: {len(rigid.ghost_local_positions)} samples")
                    sys.stdout.flush()
            
            print(f"[GhostInit] Starting static body sampling ({len(static_states)} bodies)...")
            sys.stdout.flush()
            static_iterator = (
                tqdm(static_states, desc="[GhostInit][Static] Sampling", unit="mesh")
                if tqdm
                else static_states
            )
            for static in static_iterator:
                    print(f"  [GhostInit][Static:{static.name}] Sampling {len(static.faces)} triangles...")
                    sys.stdout.flush()
                    samples = sample_func(static.vertices, static.faces, smoothing_length)
                    static.ghost_positions = [pos for pos, _ in samples]
                    static.ghost_normals = [normal for _, normal in samples]
                    print(f"  [GhostInit][Static:{static.name}] Computing masses for {len(static.ghost_positions)} samples...")
                    sys.stdout.flush()
                    static.ghost_pseudo_masses = mass_func(
                        static.ghost_positions,
                        smoothing_length,
                        rest_density,
                    )
                    print(f"  [GhostInit][Static:{static.name}] Complete: {len(static.ghost_positions)} samples")
                    sys.stdout.flush()
        return cls(
            config=config,
            fluid_solver=fluid_solver,
            mpm_solver=mpm_solver,
            rigid_solver=rigid_solver,
            rigid_rigid_solver=rigid_rigid_solver,
           # fluid_rigid_solver=fluid_rigid_solver,
            #fluid_static_solver=fluid_static_solver,
            rigid_static_solver=rigid_static_solver,
            fluid_state=fluid_state,
            rigid_states=rigid_states,
            static_states=static_states,
        )

    def step(self, liquid_force_damp: float = 0.2, dt: float | None = None) -> WorldSnapshot:
        """Advance simulation by one time step."""
        if dt is None:
            dt = self.config.simulation.time_step
        
        # Run MPM simulation if mpm_solver exists
        if self.mpm_solver:
            self.mpm_solver.step()
        
        # Only run SPH fluid simulation if fluid_solver exists
        elif self.fluid_solver and self.fluid_state:
            self.fluid_solver.step(
                self.fluid_state,
                liquid_force_damp,
                dt,
                self.rigid_states,
                self.static_states,
            )
            # self.fluid_static_solver.step(self.fluid_state, self.static_states, dt)
            # self.fluid_rigid_solver.step(self.fluid_state, self.rigid_states, dt)
        
        # run rigid body simulation, inside will couple no rigids situations
        self.rigid_solver.step(self.rigid_states, dt)
        
        # Rigid-rigid collision (between multiple rigid bodies)
        if len(self.rigid_states) > 1:
            self.rigid_rigid_solver.step(self.rigid_states, dt)
        
        # Rigid-static collision
        self.rigid_static_solver.step(self.rigid_states, self.static_states, dt)

        self.current_time += dt
        
        # Create fluid state for snapshot (either SPH or MPM)
        fluid_state_for_snapshot = self.fluid_state
        mpm_particles_for_snapshot = None
        
        if self.mpm_solver:
            # OPTIMIZATION: Only export MPM particles every N steps to save time
            # Adjust export_interval based on your needs (1=every frame, 10=every 10 frames)
            export_interval = 1  # Change to 5 or 10 for faster simulation
            
            if self.current_step % export_interval == 0:
                # Get MPM particle count
                n_particles = self.mpm_solver.state.n_particles[None]
                
                if n_particles > 0:
                    # FAST: Use to_numpy() for batch export (10-100x faster!)
                    import numpy as np
                    from .state import MPMState
                    
                    # Batch export all fields at once
                    positions_np = self.mpm_solver.state.x.to_numpy()[:n_particles]  # (N, 3)
                    velocities_np = self.mpm_solver.state.v.to_numpy()[:n_particles]  # (N, 3)
                    masses_np = self.mpm_solver.state.mass.to_numpy()[:n_particles]  # (N,)
                    volumes_np = self.mpm_solver.state.volume.to_numpy()[:n_particles]  # (N,)
                    
                    # Create MPMState with numpy arrays (no conversion needed!)
                    mpm_particles_for_snapshot = MPMState(
                        positions=positions_np,
                        velocities=velocities_np,
                        masses=masses_np,
                        volumes=volumes_np,
                        particle_count=n_particles,
                        material_type='water',
                    )
        
        snapshot = WorldSnapshot(
            step_index=self.current_step,
            time=self.current_time,
            fluids=fluid_state_for_snapshot,
            mpm_particles=mpm_particles_for_snapshot,
            rigids=list(self.rigid_states),
            statics=list(self.static_states),
        )
        #print(f"[DEBUG] WorldSnapshot created: step={self.current_step}, has_fluids={snapshot.fluids is not None}, has_mpm={snapshot.mpm_particles is not None}")
        self.current_step += 1
        return snapshot


def _vec(values) -> Vec3:
    return float(values[0]), float(values[1]), float(values[2])
