"""Collection of specialized solvers used by the physics world."""


from .rigid.solver import RigidBodySolver
from .sph.solver import SphSolver
from .rigid_static.solver import RigidStaticSolver
from .rigid_rigid.solver import RigidRigidSolver

__all__ = [
    "SphSolver",
    "RigidBodySolver",
    "RigidStaticSolver",
    "RigidRigidSolver",
]
