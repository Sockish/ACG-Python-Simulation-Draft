"""Taichi-based boundary particle management for ghost particles.

Handles rigid and static body boundary samples in GPU-compatible format.
"""
import taichi as ti
import numpy as np
from typing import List, Tuple, Literal

from .taichi_ghost_sampler import TaichiGhostSampler

Vec3 = Tuple[float, float, float]


@ti.data_oriented
class BoundaryParticles:
    """GPU-accelerated boundary particle storage for SPH interactions."""
    
    def __init__(self, max_boundary_particles: int):
        """Initialize boundary particle storage.
        
        Args:
            max_boundary_particles: Maximum number of boundary samples to store
        """
        self.max_particles = max_boundary_particles
        self.sampler = TaichiGhostSampler()
        
        # Boundary particle data
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=max_boundary_particles)
        self.normals = ti.Vector.field(3, dtype=ti.f32, shape=max_boundary_particles)
        self.pseudo_masses = ti.field(dtype=ti.f32, shape=max_boundary_particles)
        
        # Type: 0=static, 1=rigid
        self.types = ti.field(dtype=ti.i32, shape=max_boundary_particles)
        
        # For rigid bodies: body index and local position/velocity
        self.body_indices = ti.field(dtype=ti.i32, shape=max_boundary_particles)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=max_boundary_particles)
        
        # Actual count
        self.n_particles = ti.field(dtype=ti.i32, shape=())
        self.n_particles[None] = 0
    
    def load_static_boundaries(
        self,
        positions: List[Vec3],
        normals: List[Vec3],
        pseudo_masses: List[float]
    ) -> int:
        """Load static boundary particles.
        
        Returns:
            Starting index of loaded particles
        """
        n_new = len(positions)
        start_idx = self.n_particles[None]
        
        if start_idx + n_new > self.max_particles:
            raise ValueError(f"Too many boundary particles: {start_idx + n_new} > {self.max_particles}")
        
        # Convert to numpy
        pos_np = np.array(positions, dtype=np.float32)
        if pos_np.ndim == 1:
            pos_np = pos_np.reshape(-1, 3)
        norm_np = np.array(normals, dtype=np.float32)
        if norm_np.ndim == 1:
            norm_np = norm_np.reshape(-1, 3)
        mass_np = np.array(pseudo_masses, dtype=np.float32)
        types_np = np.zeros(n_new, dtype=np.int32)  # 0 = static
        
        # Copy to GPU - get full array, modify slice, write back
        end_idx = start_idx + n_new
        positions_full = self.positions.to_numpy()
        positions_full[start_idx:end_idx] = pos_np
        self.positions.from_numpy(positions_full)
        
        normals_full = self.normals.to_numpy()
        normals_full[start_idx:end_idx] = norm_np
        self.normals.from_numpy(normals_full)
        
        masses_full = self.pseudo_masses.to_numpy()
        masses_full[start_idx:end_idx] = mass_np
        self.pseudo_masses.from_numpy(masses_full)
        
        types_full = self.types.to_numpy()
        types_full[start_idx:end_idx] = types_np
        self.types.from_numpy(types_full)
        
        self.n_particles[None] = start_idx + n_new
        return start_idx
    
    def load_rigid_boundaries(
        self,
        body_index: int,
        positions: List[Vec3],
        normals: List[Vec3],
        pseudo_masses: List[float]
    ) -> int:
        """Load rigid body boundary particles.
        
        Args:
            body_index: Index of the rigid body
            positions: World-space positions of ghost particles
            normals: Normals of ghost particles
            pseudo_masses: Pseudo masses
            
        Returns:
            Starting index of loaded particles
        """
        n_new = len(positions)
        start_idx = self.n_particles[None]
        
        if start_idx + n_new > self.max_particles:
            raise ValueError(f"Too many boundary particles: {start_idx + n_new} > {self.max_particles}")
        
        # Convert to numpy - ensure correct shape
        pos_np = np.array(positions, dtype=np.float32)
        if pos_np.ndim == 1:
            pos_np = pos_np.reshape(-1, 3)
        norm_np = np.array(normals, dtype=np.float32)
        if norm_np.ndim == 1:
            norm_np = norm_np.reshape(-1, 3)
        mass_np = np.array(pseudo_masses, dtype=np.float32)
        types_np = np.ones(n_new, dtype=np.int32)  # 1 = rigid
        body_np = np.full(n_new, body_index, dtype=np.int32)
        
        # Copy to GPU - get full array, modify slice, write back
        end_idx = start_idx + n_new
        positions_full = self.positions.to_numpy()
        positions_full[start_idx:end_idx] = pos_np
        self.positions.from_numpy(positions_full)
        
        normals_full = self.normals.to_numpy()
        normals_full[start_idx:end_idx] = norm_np
        self.normals.from_numpy(normals_full)
        
        masses_full = self.pseudo_masses.to_numpy()
        masses_full[start_idx:end_idx] = mass_np
        self.pseudo_masses.from_numpy(masses_full)
        
        types_full = self.types.to_numpy()
        types_full[start_idx:end_idx] = types_np
        self.types.from_numpy(types_full)
        
        body_full = self.body_indices.to_numpy()
        body_full[start_idx:end_idx] = body_np
        self.body_indices.from_numpy(body_full)
        
        self.n_particles[None] = start_idx + n_new
        return start_idx
    
    @ti.kernel
    def update_rigid_positions(
        self,
        start_idx: ti.i32,
        n_particles: ti.i32,
        local_positions: ti.template(),
        rigid_position: ti.math.vec3,
        rigid_rotation: ti.math.mat3
    ):
        """Update world positions of rigid body ghost particles.
        
        Args:
            start_idx: Starting index in boundary arrays
            n_particles: Number of particles to update
            local_positions: Local positions field
            rigid_position: Rigid body center position
            rigid_rotation: Rigid body rotation matrix
        """
        for i in range(n_particles):
            idx = start_idx + i
            local_pos = local_positions[i]
            world_pos = rigid_rotation @ local_pos + rigid_position
            self.positions[idx] = world_pos
    
    @ti.kernel
    def update_rigid_velocities(
        self,
        start_idx: ti.i32,
        n_particles: ti.i32,
        rigid_linear_vel: ti.math.vec3,
        rigid_angular_vel: ti.math.vec3,
        rigid_position: ti.math.vec3
    ):
        """Update velocities of rigid body ghost particles.
        
        v = v_linear + ω × r
        """
        for i in range(n_particles):
            idx = start_idx + i
            r = self.positions[idx] - rigid_position
            velocity = rigid_linear_vel + rigid_angular_vel.cross(r)
            self.velocities[idx] = velocity
    
    def clear(self):
        """Clear all boundary particles."""
        self.n_particles[None] = 0
    
    def sample_and_load_mesh(
        self,
        vertices: List[Vec3],
        triangles: List[Tuple[int, int, int]],
        smoothing_length: float,
        rest_density: float,
        body_type: Literal['static', 'rigid'] = 'static',
        body_index: int = 0,
        layer_offsets: Tuple[float, float] | None = None,
    ) -> int:
        """Sample mesh surface and load as boundary particles using GPU acceleration.
        
        Args:
            vertices: Mesh vertices
            triangles: Triangle indices
            smoothing_length: SPH smoothing length
            rest_density: SPH rest density
            body_type: 'static' or 'rigid'
            body_index: Rigid body index (only used if body_type='rigid')
            layer_offsets: Optional layer offsets for dual-layer sampling
            
        Returns:
            Starting index of loaded particles
        """
        # GPU-accelerated mesh sampling
        samples = self.sampler.sample_mesh_surface(
            vertices, triangles, smoothing_length, layer_offsets
        )
        
        if not samples:
            return self.n_particles[None]
        
        # Extract positions and normals
        positions = [s[0] for s in samples]
        normals = [s[1] for s in samples]
        
        # GPU-accelerated pseudo mass computation
        pseudo_masses = self.sampler.compute_local_pseudo_masses(
            positions, smoothing_length, rest_density
        )
        
        # Load into boundary particles
        if body_type == 'static':
            return self.load_static_boundaries(positions, normals, pseudo_masses)
        else:
            return self.load_rigid_boundaries(body_index, positions, normals, pseudo_masses)
