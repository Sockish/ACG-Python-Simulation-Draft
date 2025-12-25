"""
MPM state management - stores particle positions, velocities, etc.
"""
import numpy as np
import taichi as ti


@ti.data_oriented
class MPMState:
    """Manages MPM particle state (positions, velocities, deformation gradients, etc.)"""
    
    def __init__(self, max_particles: int):
        """
        Initialize MPM state.
        
        Args:
            max_particles: Maximum number of MPM particles
        """
        self.max_particles = max_particles
        
        # Particle data (using Taichi fields for GPU acceleration)
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)      # positions
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)      # velocities
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)   # APIC affine matrix
        self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)   # deformation gradient
        self.J = ti.field(dtype=ti.f32, shape=max_particles)                # determinant of F
        
        # Particle properties
        self.mass = ti.field(dtype=ti.f32, shape=max_particles)
        self.volume = ti.field(dtype=ti.f32, shape=max_particles)
        self.material = ti.field(dtype=ti.i32, shape=max_particles)         # material type
        
        # Active particle count
        self.n_particles = ti.field(dtype=ti.i32, shape=())
        
    def get_positions(self) -> np.ndarray:
        """Get particle positions as numpy array."""
        n = self.n_particles[None]
        return self.x.to_numpy()[:n]
    
    def get_velocities(self) -> np.ndarray:
        """Get particle velocities as numpy array."""
        n = self.n_particles[None]
        return self.v.to_numpy()[:n]
    
    def set_positions(self, positions: np.ndarray):
        """Set particle positions from numpy array."""
        n = len(positions)
        if n > self.max_particles:
            raise ValueError(f"Too many particles: {n} > {self.max_particles}")
        # Ensure positions array is the right shape
        if len(positions.shape) == 1:
            positions = positions.reshape(-1, 3)
        self.x.from_numpy(positions.astype(np.float32))
        self.n_particles[None] = n
        print(f"[MPMState] set_positions: {n} particles, n_particles field set to {self.n_particles[None]}")
    
    def set_velocities(self, velocities: np.ndarray):
        """Set particle velocities from numpy array."""
        n = len(velocities)
        if n > self.max_particles:
            raise ValueError(f"Too many particles: {n} > {self.max_particles}")
        self.v.from_numpy(velocities.astype(np.float32))
    
    @ti.kernel
    def initialize_particles(self, n: ti.i32, p_mass: ti.f32, p_vol: ti.f32, mat_type: ti.i32):
        """
        Initialize particle properties.
        
        Args:
            n: Number of particles
            p_mass: Particle mass
            p_vol: Particle volume
            mat_type: Material type (0=water, 1=jelly, 2=snow)
        """
        for i in range(n):
            self.mass[i] = p_mass
            self.volume[i] = p_vol
            self.material[i] = mat_type
            self.F[i] = ti.Matrix.identity(ti.f32, 3)  # Initial deformation = identity
            self.J[i] = 1.0
            self.C[i] = ti.Matrix.zero(ti.f32, 3, 3)
    
    def clear(self):
        """Clear all particles."""
        self.n_particles[None] = 0
