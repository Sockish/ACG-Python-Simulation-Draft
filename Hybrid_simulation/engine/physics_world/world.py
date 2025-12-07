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
    def from_config(cls, config: SceneConfig, use_taichi: bool = False) -> "PhysicsWorld":
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
        if config.liquid_box:
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
            rigid_solver=rigid_solver,
            rigid_rigid_solver=rigid_rigid_solver,
           # fluid_rigid_solver=fluid_rigid_solver,
            #fluid_static_solver=fluid_static_solver,
            rigid_static_solver=rigid_static_solver,
            fluid_state=fluid_state,
            rigid_states=rigid_states,
            static_states=static_states,
        )

    def step(self, liquid_force_damp: float, dt: float) -> WorldSnapshot:
        # Only run fluid simulation if fluid_solver exists
        if self.fluid_solver and self.fluid_state:
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
        snapshot = WorldSnapshot(
            step_index=self.current_step,
            time=self.current_time,
            fluids=self.fluid_state,
            rigids=list(self.rigid_states),
            statics=list(self.static_states),
        )
        self.current_step += 1
        return snapshot


def _vec(values) -> Vec3:
    return float(values[0]), float(values[1]), float(values[2])
