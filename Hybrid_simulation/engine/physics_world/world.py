"""Physics world core that orchestrates individual solvers."""

from __future__ import annotations

from dataclasses import dataclass

try:  # tqdm is optional but useful for long sampling runs
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is missing
    tqdm = None

from .solvers.sph.WCSPH import WCSphSolver
from .solvers.sph.utils.ghost_particles import sample_mesh_surface, compute_local_pseudo_masses

from ..configuration import SceneConfig
from ..mesh_utils import load_obj_mesh, triangulate_faces
from .math_utils import Vec3
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
    fluid_rigid_solver: FluidRigidCouplingSolver
    fluid_static_solver: FluidStaticSolver
    rigid_static_solver: RigidStaticSolver
    fluid_state: FluidState | None
    rigid_states: list[RigidBodyState]
    static_states: list[StaticBodyState]
    current_time: float = 0.0
    current_step: int = 0

    @classmethod
    def from_config(cls, config: SceneConfig) -> "PhysicsWorld":
        gravity = _vec(config.simulation.gravity)
        
        # Initialize fluid solver only if liquid_box is defined
        if config.liquid_box:
            fluid_solver = WCSphSolver(
                liquid_box=config.liquid_box,
                gravity=gravity)
            fluid_state = fluid_solver.initialize()
        else:
            fluid_solver = None
            fluid_state = None
        # fluid_rigid_solver = FluidRigidCouplingSolver()
        # fluid_static_solver = FluidStaticSolver() 


        rigid_solver = RigidBodySolver(config.rigid_bodies, gravity)
        rigid_rigid_solver = RigidRigidSolver()
        fluid_rigid_solver = FluidRigidCouplingSolver()
        fluid_static_solver = FluidStaticSolver()
        rigid_static_solver = RigidStaticSolver()

        rigid_states = rigid_solver.initialize()
        static_states = []
        for body in config.static_bodies:
            mesh = load_obj_mesh(body.mesh_path)
            local_min, local_max = mesh.bounds()
            triangles = triangulate_faces(mesh.faces)
            
            static_states.append(
                StaticBodyState(
                    name=body.name,
                    mesh_path=body.mesh_path,
                    position=_vec(body.initial_position),  # Should be (0,0,0)
                    orientation=tuple(body.initial_orientation),  # Should be (0,0,0,1)
                    local_bounds_min=local_min,
                    local_bounds_max=local_max,
                    vertices=mesh.vertices,
                    faces=triangles,
                )
            )

        smoothing_length = fluid_solver.smoothing_length if fluid_solver else None
        rest_density = fluid_solver.liquid_box.rest_density if fluid_solver else None
        if smoothing_length and rest_density:
            print(
                f"[GhostInit] h={smoothing_length:.4f}, rho0={rest_density:.2f}"
            )
            for rigid in rigid_states:
                samples = sample_mesh_surface(rigid.centered_vertices, rigid.triangles, smoothing_length)
                rigid.ghost_local_positions = [pos for pos, _ in samples]
                rigid.ghost_local_normals = [normal for _, normal in samples]
                rigid.ghost_pseudo_masses = compute_local_pseudo_masses(
                    rigid.ghost_local_positions,
                    smoothing_length,
                    rest_density,
                )
                print(
                    f"  [GhostInit][Rigid:{rigid.name}] samples={len(rigid.ghost_local_positions)}"
                )
            static_iterator = (
                tqdm(static_states, desc="[GhostInit][Static] Sampling", unit="mesh")
                if tqdm
                else static_states
            )
            for static in static_iterator:
                samples = sample_mesh_surface(static.vertices, static.faces, smoothing_length)
                static.ghost_positions = [pos for pos, _ in samples]
                static.ghost_normals = [normal for _, normal in samples]
                static.ghost_pseudo_masses = compute_local_pseudo_masses(
                    static.ghost_positions,
                    smoothing_length,
                    rest_density,
                )
                print(
                    f"  [GhostInit][Static:{static.name}] samples={len(static.ghost_positions)}"
                )

        return cls(
            config=config,
            fluid_solver=fluid_solver,
            rigid_solver=rigid_solver,
            rigid_rigid_solver=rigid_rigid_solver,
            fluid_rigid_solver=fluid_rigid_solver,
            fluid_static_solver=fluid_static_solver,
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
