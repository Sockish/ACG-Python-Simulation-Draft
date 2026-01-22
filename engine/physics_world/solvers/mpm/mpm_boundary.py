"""
MPM boundary handling - static rigid body collisions.
"""
import numpy as np
import taichi as ti
from typing import List, Tuple


@ti.data_oriented
class MPMBoundary:
    """Handles boundary conditions and collisions with static meshes."""
    
    def __init__(self, grid_resolution: int, boundary_mode: str = 'sticky'):
        """
        Initialize MPM boundary handler.
        
        Args:
            grid_resolution: Grid resolution
            boundary_mode: 'sticky', 'slip', or 'separate'
        """
        self.n_grid = grid_resolution
        self.boundary_mode = boundary_mode
        
        # Boundary mode constants
        self.MODE_STICKY = 0    # Zero velocity at boundary
        self.MODE_SLIP = 1      # Zero normal velocity, preserve tangential
        self.MODE_SEPARATE = 2  # Only prevent penetration
        self.MODE_BOUNCE = 3    # Elastic/semi-elastic collision with restitution
        
        # Convert string mode to constant
        mode_map = {'sticky': 0, 'slip': 1, 'separate': 2, 'bounce': 3}
        self.mode_value = ti.field(dtype=ti.i32, shape=())
        self.mode_value[None] = mode_map.get(boundary_mode, 0)
        
        # Coefficient of restitution for bounce mode (0=no bounce, 1=perfect elastic)
        self.restitution = ti.field(dtype=ti.f32, shape=())
        self.restitution[None] = 0.8  # Default: 50% bounce
        
        # Static mesh collision: voxelized occupancy grid
        # 1 = inside static mesh (solid), 0 = empty space
        self.occupancy = ti.field(dtype=ti.i32, shape=(grid_resolution, grid_resolution, grid_resolution))
        
        # Normal field for slip boundary (computed from occupancy gradient)
        self.normal = ti.Vector.field(3, dtype=ti.f32, shape=(grid_resolution, grid_resolution, grid_resolution))
        
        # Whether static meshes have been loaded (as Taichi field for kernel access)
        self.has_static_meshes_field = ti.field(dtype=ti.i32, shape=())
        self.has_static_meshes_field[None] = 0
        self.has_static_meshes = False
    
    def load_static_mesh(self, vertices: np.ndarray, triangles: np.ndarray, 
                         domain_min: float, domain_max: float):
        """
        Load a static mesh and voxelize it onto the grid using proper triangle rasterization.
        
        Args:
            vertices: Vertex positions (Nx3 array)
            triangles: Triangle indices (Mx3 array)
            domain_min: Minimum coordinate of simulation domain
            domain_max: Maximum coordinate of simulation domain
        """
        print(f"[MPMBoundary] Voxelizing static mesh: {len(vertices)} vertices, {len(triangles)} triangles")
        
        dx = (domain_max - domain_min) / self.n_grid
        
        # Initialize occupancy grid (0 = empty/fluid space, 1 = solid wall)
        occupancy_np = np.zeros((self.n_grid, self.n_grid, self.n_grid), dtype=np.int32)
        
        # STEP 1: Proper triangle rasterization (NOT just bounding boxes!)
        # For each triangle, use conservative voxelization
        print(f"[MPMBoundary] Rasterizing {len(triangles)} triangles...")
        
        for tri_idx in range(len(triangles)):
            if tri_idx % 1000 == 0:
                print(f"  Progress: {tri_idx}/{len(triangles)}", end='\r')
            
            i0, i1, i2 = triangles[tri_idx]
            v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
            
            # Convert to grid coordinates
            v0_grid = (v0 - domain_min) / dx
            v1_grid = (v1 - domain_min) / dx
            v2_grid = (v2 - domain_min) / dx
            
            # Triangle bounding box in grid coordinates
            min_idx = np.maximum(0, np.floor(np.min([v0_grid, v1_grid, v2_grid], axis=0)).astype(int))
            max_idx = np.minimum(self.n_grid - 1, np.ceil(np.max([v0_grid, v1_grid, v2_grid], axis=0)).astype(int))
            
            # Conservative rasterization: check each cell in bounding box
            for i in range(min_idx[0], max_idx[0] + 1):
                for j in range(min_idx[1], max_idx[1] + 1):
                    for k in range(min_idx[2], max_idx[2] + 1):
                        # Cell center in grid coordinates
                        cell_center = np.array([i + 0.5, j + 0.5, k + 0.5])
                        
                        # Check if cell center is close to triangle (distance < sqrt(3)*dx)
                        # This is conservative - marks cells that triangle passes through
                        dist = self._point_to_triangle_distance(cell_center, v0_grid, v1_grid, v2_grid)
                        if dist < 1.5:  # Conservative threshold (1.5 cells)
                            occupancy_np[i, j, k] = 1
        
        print(f"\n[MPMBoundary] Initial rasterization: {np.sum(occupancy_np)} voxels")
        
        # STEP 2: Fill interior using flood-fill to create solid volume
        print(f"[MPMBoundary] Filling interior...")
        from scipy.ndimage import binary_fill_holes
        filled = binary_fill_holes(occupancy_np)
        
        # STEP 3: Extract shell by eroding and XOR
        print(f"[MPMBoundary] Extracting surface shell...")
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(filled, iterations=1)
        shell = (filled.astype(bool) ^ eroded).astype(np.int32)
        
        occupancy_np = shell  # Only keep the shell!
        
        # Transfer to Taichi field
        self.occupancy.from_numpy(occupancy_np)
        
        # Compute normals from occupancy gradient
        self._compute_normals()
        
        self.has_static_meshes = True        
        self.has_static_meshes_field[None] = 1        
        occupied_cells = np.sum(occupancy_np)
        print(f"[MPMBoundary] Voxelization complete: {occupied_cells} shell cells (hollow interior)")
    
    def _point_to_triangle_distance(self, p: np.ndarray, v0: np.ndarray, 
                                     v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute distance from point p to triangle (v0, v1, v2).
        Returns distance in grid cell units.
        
        Reference: Real-Time Collision Detection by Christer Ericson
        """
        # Edge vectors
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2
        
        # Normal (not normalized - we only need direction)
        normal = np.cross(e0, -e2)
        normal_len = np.linalg.norm(normal)
        
        if normal_len < 1e-8:
            # Degenerate triangle - use distance to closest vertex
            d0 = np.linalg.norm(p - v0)
            d1 = np.linalg.norm(p - v1)
            d2 = np.linalg.norm(p - v2)
            return min(d0, d1, d2)
        
        normal = normal / normal_len
        
        # Project point onto triangle plane
        v0_to_p = p - v0
        dist_to_plane = abs(np.dot(v0_to_p, normal))
        
        # Check if projection is inside triangle using barycentric coordinates
        p_proj = p - dist_to_plane * normal * np.sign(np.dot(v0_to_p, normal))
        
        # Barycentric coordinates
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = p_proj - v0
        
        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p, v0v1)
        d21 = np.dot(v0p, v0v2)
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-8:
            # Degenerate triangle
            return dist_to_plane
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        # If inside triangle, return distance to plane
        if u >= 0 and v >= 0 and w >= 0:
            return dist_to_plane
        
        # Outside triangle - find distance to closest edge or vertex
        min_dist = float('inf')
        
        # Distance to edges
        edges = [(v0, v1), (v1, v2), (v2, v0)]
        for edge_start, edge_end in edges:
            edge_vec = edge_end - edge_start
            edge_len_sq = np.dot(edge_vec, edge_vec)
            
            if edge_len_sq < 1e-8:
                # Degenerate edge
                dist = np.linalg.norm(p - edge_start)
            else:
                # Project point onto edge
                t = np.dot(p - edge_start, edge_vec) / edge_len_sq
                t = np.clip(t, 0.0, 1.0)
                closest = edge_start + t * edge_vec
                dist = np.linalg.norm(p - closest)
            
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _point_in_triangle_prism(self, p: np.ndarray, v0: np.ndarray, v1: np.ndarray, 
                                  v2: np.ndarray, thickness: float) -> bool:
        """
        Conservative check: is point p inside the triangular prism formed by triangle + thickness.
        Uses barycentric coordinates for 2D check (ignoring one dimension).
        """
        # Project to XY plane and check if inside triangle
        # Compute barycentric coordinates
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = p - v0
        
        # Use XY projection (ignore Z for now, just check 2D containment)
        d00 = np.dot(v0v1[:2], v0v1[:2])
        d01 = np.dot(v0v1[:2], v0v2[:2])
        d11 = np.dot(v0v2[:2], v0v2[:2])
        d20 = np.dot(v0p[:2], v0v1[:2])
        d21 = np.dot(v0p[:2], v0v2[:2])
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-8:
            return False
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        # Check if inside triangle (with some tolerance)
        tolerance = 0.5  # Allow cells near triangle edges
        if u >= -tolerance and v >= -tolerance and w >= -tolerance:
            # Check Z distance to triangle plane
            z_on_triangle = v0[2] + v * v0v1[2] + w * v0v2[2]
            if abs(p[2] - z_on_triangle) < thickness * 2.0:
                return True
        
        return False
    
    @ti.kernel
    def _compute_normals(self):
        """Compute surface normals from occupancy gradient (points INTO fluid space)."""
        for I in ti.grouped(self.occupancy):
            # Compute gradient at every cell (will be zero in interior)
            grad = ti.Vector([0.0, 0.0, 0.0])
            for d in ti.static(range(3)):
                ip = I[d] + 1 if I[d] < self.n_grid - 1 else I[d]
                im = I[d] - 1 if I[d] > 0 else I[d]
                
                idx_p = [I[0], I[1], I[2]]
                idx_m = [I[0], I[1], I[2]]
                idx_p[d] = ip
                idx_m[d] = im
                
                # Gradient points from low to high: solid(1) -> fluid(0)
                # So we compute: fluid - solid = 0 - 1 = -1 (points into solid)
                # We want normal pointing INTO FLUID, so reverse:
                grad[d] = float(self.occupancy[idx_p[0], idx_p[1], idx_p[2]]) - \
                          float(self.occupancy[idx_m[0], idx_m[1], idx_m[2]])
            
            # Normalize
            norm = grad.norm()
            if norm > 1e-6:
                self.normal[I] = grad / norm
    
    @ti.kernel
    def apply_static_collision(self, grid_v: ti.template(), grid_m: ti.template()):
        """
        Apply collision with static meshes on grid velocities.
        
        CRITICAL: Grid nodes and voxel centers are offset!
        - occupancy[i,j,k] represents cell center at world position (i+0.5, j+0.5, k+0.5)*dx
        - grid_v[i,j,k] represents grid node at world position (i, j, k)*dx
        
        Solution: Check if grid node is NEAR any solid cell (including neighbors)
        """
        mode = self.mode_value[None]
        restitution = self.restitution[None]
        
        for I in ti.grouped(grid_v):
            if grid_m[I] > 1e-12:  # Only process nodes with mass
                # Check if this grid node is near a solid surface
                # Strategy: check the 8 cells that this grid node belongs to
                has_solid_neighbor = 0
                normal_sum = ti.Vector([0.0, 0.0, 0.0])
                
                # Check the 8 surrounding cells (grid node is at corner of cells)
                for di in ti.static([-1, 0]):
                    for dj in ti.static([-1, 0]):
                        for dk in ti.static([-1, 0]):
                            ni = I[0] + di
                            nj = I[1] + dj
                            nk = I[2] + dk
                            
                            # Check bounds
                            if 0 <= ni < self.n_grid and 0 <= nj < self.n_grid and 0 <= nk < self.n_grid:
                                if self.occupancy[ni, nj, nk] == 1:
                                    has_solid_neighbor = 1
                                    # Accumulate normal (will normalize later)
                                    normal_sum += self.normal[ni, nj, nk]
                
                # Apply collision if near solid
                if has_solid_neighbor == 1:
                    n = normal_sum
                    n_norm = n.norm()
                    
                    if n_norm > 1e-6:
                        n = n / n_norm  # Normalize
                        v = grid_v[I]
                        v_n = v.dot(n)  # Normal component
                        
                        # Only handle particles moving INTO solid (v_n < 0)
                        if v_n < 0:
                            if mode == 0:  # STICKY
                                grid_v[I] = ti.Vector.zero(ti.f32, 3)
                                
                            elif mode == 1:  # SLIP
                                # Remove normal component, keep tangential
                                v_t = v - v_n * n
                                grid_v[I] = v_t
                                
                            elif mode == 2:  # SEPARATE
                                # Only prevent penetration
                                v_t = v - v_n * n
                                grid_v[I] = v_t
                                
                            else:  # mode == 3: BOUNCE
                                # Reflect with restitution
                                v_t = v - v_n * n  # Tangential component
                                v_n_reflected = -restitution * v_n  # Reflected normal
                                grid_v[I] = v_t + v_n_reflected * n

