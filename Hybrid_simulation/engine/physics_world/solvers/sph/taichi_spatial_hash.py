"""Taichi-based spatial hash grid for neighbor search.

This module implements a fixed-size spatial hash grid optimized for GPU execution.
The grid uses a simple 3D array structure with pre-allocated capacity.
"""
import taichi as ti
import numpy as np


@ti.data_oriented
class SpatialHashGrid:
    """GPU-accelerated spatial hash grid for particle neighbor queries.
    
    Uses a fixed-size 3D grid where each cell stores up to `max_particles_per_cell` 
    particle indices. This avoids dynamic memory allocation on the GPU.
    """
    
    def __init__(self, grid_size: tuple, max_particles_per_cell: int = 100):
        """Initialize spatial hash grid.
        
        Args:
            grid_size: (nx, ny, nz) dimensions of the grid
            max_particles_per_cell: Maximum particles that can be stored in one cell
        """
        self.grid_size = grid_size
        self.max_per_cell = max_particles_per_cell
        
        # Number of particles in each cell
        self.cell_count = ti.field(dtype=ti.i32, shape=grid_size)
        
        # Particle indices stored in each cell [grid_x, grid_y, grid_z, particle_slot]
        self.cell_particles = ti.field(dtype=ti.i32, 
                                       shape=(*grid_size, max_particles_per_cell))
    
    @ti.kernel
    def clear(self):
        """Clear all cell counts (prepares for rebuild)."""
        for I in ti.grouped(self.cell_count):
            self.cell_count[I] = 0
    
    @ti.kernel
    def build(self, 
              positions: ti.template(), 
              n_particles: ti.i32,
              cell_size: ti.f32,
              world_min: ti.math.vec3):
        """Build spatial hash from particle positions.
        
        Args:
            positions: Particle position field
            n_particles: Number of active particles
            cell_size: Size of each grid cell (typically smoothing_length h)
            world_min: Minimum corner of the simulation domain
        """
        # Clear existing data
        for I in ti.grouped(self.cell_count):
            self.cell_count[I] = 0
        
        # Insert particles into grid
        for i in range(n_particles):
            pos = positions[i]
            # Compute cell index
            cell_idx = ti.cast((pos - world_min) / cell_size, ti.i32)
            
            # Clamp to grid bounds
            cell_idx = ti.math.clamp(cell_idx, 0, ti.math.ivec3(*self.grid_size) - 1)
            
            # Atomic add to get slot in this cell
            slot = ti.atomic_add(self.cell_count[cell_idx], 1)
            
            # Store particle index if there's space
            if slot < self.max_per_cell:
                self.cell_particles[cell_idx, slot] = i
    
    @ti.func
    def is_valid_cell(self, cell: ti.math.ivec3) -> bool:
        """Check if cell indices are within grid bounds."""
        return (cell[0] >= 0 and cell[0] < self.grid_size[0] and
                cell[1] >= 0 and cell[1] < self.grid_size[1] and
                cell[2] >= 0 and cell[2] < self.grid_size[2])
    
    @ti.func
    def get_cell_particle_count(self, cell: ti.math.ivec3) -> ti.i32:
        """Get number of particles in a cell."""
        count = 0
        if self.is_valid_cell(cell):
            count = self.cell_count[cell]
        return count
    
    @ti.func
    def get_cell_particle(self, cell: ti.math.ivec3, slot: ti.i32) -> ti.i32:
        """Get particle index from cell at given slot."""
        return self.cell_particles[cell, slot]
    
    def get_stats(self) -> dict:
        """Get statistics about the grid (for debugging/tuning)."""
        cell_counts = self.cell_count.to_numpy()
        occupied = np.sum(cell_counts > 0)
        total_cells = np.prod(self.grid_size)
        max_count = np.max(cell_counts)
        avg_count = np.mean(cell_counts[cell_counts > 0]) if occupied > 0 else 0
        
        return {
            'total_cells': total_cells,
            'occupied_cells': occupied,
            'occupancy_rate': occupied / total_cells,
            'max_particles_in_cell': max_count,
            'avg_particles_per_occupied_cell': avg_count,
            'overflow_warning': max_count >= self.max_per_cell
        }
