"""SPH density computation."""
from typing import List, Tuple, Optional, TYPE_CHECKING
from .kernels import poly6

if TYPE_CHECKING:
    from .neighborhood import BoundarySample

Vec3 = Tuple[float, float, float]

def length_sq(v: Vec3) -> float:
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]

def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def compute_density(
    positions: List[Vec3],
    mass: float,
    h: float,
    neighbors: List[List[int]],
    boundary_neighbors: Optional[List[List["BoundarySample"]]] = None,
) -> List[float]:
    """Compute density rho_i = sum_j m_j W_ij."""
    n = len(positions)
    densities = [0.0] * n
    
    # Precompute h squared for fast distance check if needed, 
    # but poly6 handles check.
    
    for i in range(n):
        rho = 0.0
        p_i = positions[i]
        
        # Fluid neighbors
        # rho_i = sum_j m_j W_ij
        # Self-contribution is included if i is in neighbors[i] or handled separately.
        # Usually neighbors list includes self or we add it manually.
        # The formula sum_j includes j=i.
        # My neighborhood search excludes self. I must add self contribution.
        # W(0) is valid.
        
        
        
        for j in neighbors[i]:
            r_vec = sub(p_i, positions[j])
            r = (length_sq(r_vec)) ** 0.5
            rho += mass * poly6(r, h)

        if boundary_neighbors:
            for sample in boundary_neighbors[i]:
                r_vec = sub(p_i, sample.position)
                r = (length_sq(r_vec)) ** 0.5
                rho += sample.pseudo_mass * poly6(r, h)


        # Add self contribution
        rho += mass * poly6(0.0, h)            
        densities[i] = rho

    return densities
