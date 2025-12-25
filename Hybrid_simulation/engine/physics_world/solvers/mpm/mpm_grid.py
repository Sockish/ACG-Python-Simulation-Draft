"""
MPM grid operations - background Eulerian grid for momentum transfer.
"""
import taichi as ti


@ti.data_oriented
class MPMGrid:
    """Background Eulerian grid for MPM simulation."""
    
    def __init__(self, resolution: int, domain_min: float, domain_max: float):
        """
        Initialize MPM grid.
        
        Args:
            resolution: Grid resolution (e.g., 64 means 64x64x64 grid)
            domain_min: Minimum coordinate of simulation domain
            domain_max: Maximum coordinate of simulation domain
        """
        self.n_grid = resolution
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.dx = (domain_max - domain_min) / resolution  # Grid spacing
        self.inv_dx = 1.0 / self.dx
        
        # OPTIMIZATION: Use sparse grid for 10-100x speedup!
        # Sparse grids only allocate memory for active blocks
        USE_SPARSE_GRID = True  # Change to False for dense grid
        
        if USE_SPARSE_GRID:
            # Sparse grid: only allocate 8x8x8 blocks where particles exist
            # Memory usage: ~100x less, Speed: ~10-50x faster!
            self.grid_v = ti.Vector.field(3, dtype=ti.f32)
            self.grid_m = ti.field(dtype=ti.f32)
            
            # Create sparse grid with block size 8 (3D indices!)
            block = ti.root.pointer(ti.ijk, (resolution // 8, resolution // 8, resolution // 8))
            block.dense(ti.ijk, (8, 8, 8)).place(self.grid_v, self.grid_m)
            
            print(f"[MPMGrid] Using SPARSE grid: {resolution}³ (blocks allocated on-demand)")
        else:
            # Dense grid: allocate full memory
            self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(resolution, resolution, resolution))
            self.grid_m = ti.field(dtype=ti.f32, shape=(resolution, resolution, resolution))
            print(f"[MPMGrid] Using DENSE grid: {resolution}³")
        
        # Store grid parameters as Taichi fields for kernel access
        self.grid_params = ti.field(dtype=ti.f32, shape=4)
        self.grid_params[0] = domain_min
        self.grid_params[1] = domain_max
        self.grid_params[2] = self.dx
        self.grid_params[3] = self.inv_dx
    
    @ti.kernel
    def clear_grid(self):
        """Clear grid momentum and mass."""
        for I in ti.grouped(self.grid_v):
            self.grid_v[I] = ti.Vector.zero(ti.f32, 3)
            self.grid_m[I] = 0.0
    
    @ti.func
    def world_to_grid(self, x: ti.template()) -> ti.template():
        """
        Convert world position to grid coordinates.
        
        Args:
            x: World position (3D vector)
        
        Returns:
            Grid coordinates (floating point)
        """
        domain_min = self.grid_params[0]
        inv_dx = self.grid_params[3]
        return (x - domain_min) * inv_dx
    
    @ti.func
    def is_valid_grid_pos(self, grid_pos: ti.template()) -> ti.i32:
        """
        Check if grid position is within valid range.
        
        Args:
            grid_pos: Grid position (integer indices)
        
        Returns:
            1 if valid, 0 otherwise
        """
        valid = 1
        for d in ti.static(range(3)):
            if grid_pos[d] < 0 or grid_pos[d] >= self.n_grid:
                valid = 0
        return valid
    
    @ti.func
    def is_boundary_node(self, grid_pos: ti.template(), boundary_thickness: ti.i32) -> ti.i32:
        """
        Check if grid node is on domain boundary.
        
        Args:
            grid_pos: Grid position (integer indices)
            boundary_thickness: Thickness of boundary layer (in grid cells)
        
        Returns:
            1 if boundary node, 0 otherwise
        """
        is_boundary = 0
        for d in ti.static(range(3)):
            if grid_pos[d] < boundary_thickness or grid_pos[d] >= self.n_grid - boundary_thickness:
                is_boundary = 1
        return is_boundary
