"""Collection of specialized solvers used by the physics world."""

from .fluids.solver import FluidSolver
from .rigid.solver import RigidBodySolver
from .fluid_rigid.solver import FluidRigidCouplingSolver
from .fluid_static.solver import FluidStaticSolver
from .rigid_static.solver import RigidStaticSolver
from .rigid_rigid.solver import RigidRigidSolver

__all__ = [
    "FluidSolver",
    "RigidBodySolver",
    "FluidRigidCouplingSolver",
    "FluidStaticSolver",
    "RigidStaticSolver",
    "RigidRigidSolver",
]
