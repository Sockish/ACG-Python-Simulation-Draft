"""Equation of State for SPH."""
from typing import List

def compute_pressure(
    densities: List[float],
    rest_density: float,
    kappa: float,
    gamma: float
) -> List[float]:
    """Compute pressure p_i = (kappa * rho0 / gamma) * ((rho_i/rho0)^gamma - 1)."""
    pressures = []
    rho0 = rest_density
    # rho0_gamma = rho0 ** gamma # Not needed for the new formula
    
    factor = (kappa * rho0) / gamma

    for rho in densities:
        # Formula: p_i = (kappa * rho0 / gamma) * ((rho_i/rho0)^gamma - 1)
        
        p = factor * ((rho / rho0) ** gamma - 1.0)
        pressures.append(p)
        
    return pressures
