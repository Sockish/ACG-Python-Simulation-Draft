"""Physics world core that orchestrates individual solvers."""

from __future__ import annotations

from dataclasses import dataclass

from ..configuration import SceneConfig
from ..mesh_utils import mesh_bounds
from .math_utils import Vec3
from .state import FluidState, RigidBodyState, StaticBodyState, WorldSnapshot
from .solvers import (
    FluidRigidCouplingSolver,
    FluidSolver,
    FluidStaticSolver,
    RigidBodySolver,
    RigidStaticSolver,
    RigidRigidSolver,
)


@dataclass
class PhysicsWorld:
    config: SceneConfig
    fluid_solver: FluidSolver | None
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
            fluid_solver = FluidSolver(config.liquid_box, gravity)
            fluid_state = fluid_solver.initialize()
        else:
            fluid_solver = None
            fluid_state = None
        
        rigid_solver = RigidBodySolver(config.rigid_bodies, gravity)
        rigid_rigid_solver = RigidRigidSolver()
        fluid_rigid_solver = FluidRigidCouplingSolver()
        fluid_static_solver = FluidStaticSolver()
        rigid_static_solver = RigidStaticSolver()

        rigid_states = rigid_solver.initialize()
        static_states = []
        for body in config.static_bodies:
            # Load mesh to get vertices and faces
            from ..mesh_utils import load_obj_mesh
            mesh = load_obj_mesh(body.mesh_path)
            local_min, local_max = mesh.bounds()
            
            # Convert faces to triangles (handle quads and n-gons)
            triangles = []
            for face in mesh.faces:
                if len(face) == 3:
                    # Already a triangle
                    triangles.append(face)
                elif len(face) == 4:
                    # Quad: split into two triangles
                    triangles.append((face[0], face[1], face[2]))
                    triangles.append((face[0], face[2], face[3]))
                elif len(face) > 4:
                    # N-gon: fan triangulation from first vertex
                    for i in range(1, len(face) - 1):
                        triangles.append((face[0], face[i], face[i + 1]))
            
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

    def step(self, dt: float) -> WorldSnapshot:
        # Only run fluid simulation if fluid_solver exists
        if self.fluid_solver and self.fluid_state:
            self.fluid_solver.step(self.fluid_state, dt)
            self.fluid_static_solver.step(self.fluid_state, self.static_states, dt)
            self.fluid_rigid_solver.step(self.fluid_state, self.rigid_states, dt)
        
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
