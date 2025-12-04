"""SPH pressure forces."""
from typing import List, Tuple
from .kernels import spiky_grad

Vec3 = Tuple[float, float, float]

def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def mul(v: Vec3, s: float) -> Vec3:
    return (v[0]*s, v[1]*s, v[2]*s)

def compute_pressure_forces(
    positions: List[Vec3],
    densities: List[float],
    pressures: List[float],
    mass: float,
    h: float,
    neighbors: List[List[int]]
) -> List[Vec3]:
    """Compute pressure force f_i = -m_i sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad W_ij."""
    n = len(positions)
    forces = [(0.0, 0.0, 0.0)] * n
    
    for i in range(n):
        f_i = (0.0, 0.0, 0.0)
        p_i = positions[i]
        rho_i = densities[i]
        pres_i = pressures[i]
        
        # Avoid division by zero
        if rho_i < 1e-6:
            continue
            
        # term_i = p_i / rho_i^2
        term_i = pres_i / (rho_i * rho_i)
        
        for j in neighbors[i]:
            rho_j = densities[j]
            if rho_j < 1e-6:
                continue
                
            pres_j = pressures[j]
            # term_j = p_j / rho_j^2
            term_j = pres_j / (rho_j * rho_j)
            
            # grad W_ij = grad W(r_i - r_j)
            r_vec = sub(p_i, positions[j])
            grad_w = spiky_grad(r_vec, h)
            
            # Formula: f_i = -m_i * sum_j [ m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad W_ij ]
            # scalar = -m_i * m_j * (term_i + term_j)
            scalar = -mass * mass * (term_i + term_j)
            
            f_pair = mul(grad_w, scalar)
            f_i = add(f_i, f_pair)
            
        forces[i] = f_i
        
    return forces
