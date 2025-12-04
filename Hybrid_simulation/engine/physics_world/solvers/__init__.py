"""Collection of specialized solvers used by the physics world."""


from .rigid.solver import RigidBodySolver
from .sph.WCSPH import WCSphSolver
from .rigid_static.solver import RigidStaticSolver

__all__ = [
    "SphSolver",
    "RigidBodySolver",
    "RigidStaticSolver",
]
