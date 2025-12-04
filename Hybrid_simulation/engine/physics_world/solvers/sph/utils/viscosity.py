"""SPH viscosity (XSPH)."""
from typing import List, Tuple
from .kernels import poly6

Vec3 = Tuple[float, float, float]

def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def mul(v: Vec3, s: float) -> Vec3:
    return (v[0]*s, v[1]*s, v[2]*s)

def length(v: Vec3) -> float:
    return (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5

def compute_xsph_velocities(
    positions: List[Vec3],
    velocities: List[Vec3],
    densities: List[float],
    mass: float,
    h: float,
    alpha: float,
    neighbors: List[List[int]]
) -> List[Vec3]:
    """Compute smoothed velocities v^i = v_i + alpha * sum_j (m_j/rho_j) * (v_j - v_i) * W_ij."""
    n = len(positions)
    # We will return the updated velocities directly, effectively replacing v_i with v^i
    v_hat = [(0.0, 0.0, 0.0)] * n
    
    for i in range(n):
        v_i = velocities[i]
        sum_term = (0.0, 0.0, 0.0)
        
        for j in neighbors[i]:
            rho_j = densities[j]
            if rho_j < 1e-6:
                continue
                
            # W_ij
            r_vec = sub(positions[i], positions[j])
            r = length(r_vec)
            w_ij = poly6(r, h)
            
            # (v_j - v_i)
            v_diff = sub(velocities[j], v_i)
            
            # term = (m_j / rho_j) * (v_j - v_i) * W_ij
            factor = (mass / rho_j) * w_ij
            term = mul(v_diff, factor)
            
            sum_term = add(sum_term, term)
            
        # v^i = v_i + alpha * sum_term
        v_hat[i] = add(v_i, mul(sum_term, alpha))
        
    return v_hat
