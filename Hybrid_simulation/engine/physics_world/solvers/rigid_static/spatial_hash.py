"""Spatial hash grid for accelerating triangle collision queries."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from ...math_utils import Vec3


class SpatialHashGrid:
    """3D spatial hash grid for fast triangle lookup.
    
    Divides space into uniform cells and stores triangle indices in each cell.
    This dramatically reduces collision detection complexity from O(n*m) to O(n*k)
    where k is the average number of triangles per cell.
    """
    
    def __init__(self, triangles: List[Tuple[Vec3, Vec3, Vec3]], cell_size: float = 0.5):
        """Initialize spatial hash grid.
        
        Args:
            triangles: List of triangle tuples (v0, v1, v2)
            cell_size: Size of each grid cell in meters (smaller = more precision, more memory)
        """
        self.cell_size = cell_size
        self.inv_cell_size = 1.0 / cell_size
        self.grid: Dict[Tuple[int, int, int], List[int]] = {}
        
        # Build grid by inserting each triangle
        for tri_idx, (v0, v1, v2) in enumerate(triangles):
            self._insert_triangle(tri_idx, v0, v1, v2)
    
    def _insert_triangle(self, tri_idx: int, v0: Vec3, v1: Vec3, v2: Vec3) -> None:
        """Insert a triangle into all cells it overlaps."""
        # Compute triangle AABB
        min_x = min(v0[0], v1[0], v2[0])
        min_y = min(v0[1], v1[1], v2[1])
        min_z = min(v0[2], v1[2], v2[2])
        max_x = max(v0[0], v1[0], v2[0])
        max_y = max(v0[1], v1[1], v2[1])
        max_z = max(v0[2], v1[2], v2[2])
        
        # Find cell range
        min_cell = self._point_to_cell((min_x, min_y, min_z))
        max_cell = self._point_to_cell((max_x, max_y, max_z))
        
        # Insert into all overlapping cells
        for cx in range(min_cell[0], max_cell[0] + 1):
            for cy in range(min_cell[1], max_cell[1] + 1):
                for cz in range(min_cell[2], max_cell[2] + 1):
                    cell = (cx, cy, cz)
                    if cell not in self.grid:
                        self.grid[cell] = []
                    self.grid[cell].append(tri_idx)
    
    def query_point(self, point: Vec3) -> List[int]:
        """Query triangles in the cell containing the point.
        
        Args:
            point: Query position in world space
        
        Returns:
            List of triangle indices that might collide with this point
        """
        cell = self._point_to_cell(point)
        return self.grid.get(cell, [])
    
    def query_sphere(self, center: Vec3, radius: float) -> Set[int]:
        """Query triangles in cells overlapping a sphere.
        
        Args:
            center: Sphere center
            radius: Sphere radius
        
        Returns:
            Set of unique triangle indices
        """
        # Find cell range for sphere AABB
        min_point = (center[0] - radius, center[1] - radius, center[2] - radius)
        max_point = (center[0] + radius, center[1] + radius, center[2] + radius)
        
        min_cell = self._point_to_cell(min_point)
        max_cell = self._point_to_cell(max_point)
        
        # Collect unique triangles from all cells
        triangles: Set[int] = set()
        for cx in range(min_cell[0], max_cell[0] + 1):
            for cy in range(min_cell[1], max_cell[1] + 1):
                for cz in range(min_cell[2], max_cell[2] + 1):
                    cell = (cx, cy, cz)
                    if cell in self.grid:
                        triangles.update(self.grid[cell])
        
        return triangles
    
    def _point_to_cell(self, point: Vec3) -> Tuple[int, int, int]:
        """Convert world position to cell coordinates."""
        return (
            int(point[0] * self.inv_cell_size),
            int(point[1] * self.inv_cell_size),
            int(point[2] * self.inv_cell_size),
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the grid for debugging."""
        if not self.grid:
            return {
                'total_cells': 0,
                'total_triangles': 0,
                'avg_triangles_per_cell': 0,
                'max_triangles_per_cell': 0,
            }
        
        cell_counts = [len(tris) for tris in self.grid.values()]
        return {
            'total_cells': len(self.grid),
            'total_triangles': sum(cell_counts),
            'avg_triangles_per_cell': sum(cell_counts) // len(cell_counts) if cell_counts else 0,
            'max_triangles_per_cell': max(cell_counts) if cell_counts else 0,
        }
