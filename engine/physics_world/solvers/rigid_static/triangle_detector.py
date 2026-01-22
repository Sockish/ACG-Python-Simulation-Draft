"""Triangle-based collision detection for rigid-static interactions."""

from __future__ import annotations

from typing import List, Optional

from ...math_utils import Vec3, cross, distance_to_aabb, normalize, sub
from ...state import RigidBodyState, StaticBodyState
from .mesh_collision import vertex_triangle_collision
from .spatial_hash import SpatialHashGrid


class TriangleMeshCollisionDetector:
    """Handles collision detection between rigid body vertices and static triangle meshes.
    
    Uses spatial hashing for acceleration when meshes are large.
    """
    
    def __init__(self, use_spatial_hash: bool = True, cell_size: float = 0.5):
        """Initialize collision detector.
        
        Args:
            use_spatial_hash: Enable spatial hash acceleration for large meshes
            cell_size: Cell size for spatial hash grid (meters)
        """
        self.use_spatial_hash = use_spatial_hash
        self.cell_size = cell_size
        self._spatial_grids: dict[str, SpatialHashGrid] = {}
    
    def build_acceleration_structure(self, static: StaticBodyState) -> None:
        """Build spatial hash grid for a static mesh (called once during initialization).
        
        Args:
            static: Static body to build structure for
        """
        if not self.use_spatial_hash:
            return
        
        # Skip if already built
        if static.name in self._spatial_grids:
            return
        
        # Build list of triangles
        triangles = [static.get_triangle(i) for i in range(static.triangle_count)]
        
        # Create spatial hash grid
        self._spatial_grids[static.name] = SpatialHashGrid(triangles, self.cell_size)
        
        # Print statistics
        stats = self._spatial_grids[static.name].get_stats()
        print(f"Built spatial hash for '{static.name}': "
              f"{stats['total_cells']} cells, "
              f"avg {stats['avg_triangles_per_cell']} tris/cell, "
              f"max {stats['max_triangles_per_cell']} tris/cell")
    
    def detect_vertex_contacts(
        self,
        vertex: Vec3,
        local_vert: Vec3,
        static: StaticBodyState,
        contact_threshold: float = 0.0
    ) -> Optional[tuple[Vec3, Vec3, float]]:
        """Detect collision between a single vertex and static mesh.
        
        Args:
            vertex: Vertex position in world space
            local_vert: Vertex position in rigid body local space
            static: Static mesh to check collision against
            contact_threshold: Distance threshold for contact detection
        
        Returns:
            If collision: (contact_point, normal, penetration)
            If no collision: None
        """
        # Broadphase: Quick AABB rejection
        bounds_min, bounds_max = static.world_bounds
        signed_distance_aabb = distance_to_aabb(vertex, bounds_min, bounds_max)
        
        # If vertex is far from AABB, no collision possible
        if signed_distance_aabb > -0.0001:  # 0.1cm threshold
            return None
        
        # Get candidate triangles
        if self.use_spatial_hash and static.name in self._spatial_grids:
            # Use spatial hash to get nearby triangles only
            #print(f"Querying spatial hash for vertex at {vertex}")
            candidate_indices = self._spatial_grids[static.name].query_point(vertex)
            print(f"Spatial hash returned {len(candidate_indices)} candidate triangles for vertex at {vertex}")
        else:
            #print("No spatial hash available, checking all triangles")
            # Check all triangles (fallback for small meshes)
            candidate_indices = list(range(static.triangle_count))
            #print(f"Total triangles to check: {len(candidate_indices)}")
        
        # Narrowphase: Check collision with candidate triangles
        closest_contact = None
        min_penetration = float('inf')
        
        for tri_idx in candidate_indices:
            v0, v1, v2 = static.get_triangle(tri_idx)
            
            # Check vertex-triangle collision
            result = vertex_triangle_collision(vertex, v0, v1, v2, threshold=contact_threshold)
            #if tri_idx == 0:
                #print(f"Checking triangle {tri_idx}: result = {result}, triangle vertices = {v0}, {v1}, {v2}, normal = {normalize(cross(sub(v1, v0), sub(v2, v0)))}")
            
            if result is not None:
                contact_pos, normal, penetration = result
                
                # Keep the contact with maximum penetration (closest surface)
                if penetration < min_penetration:
                    min_penetration = penetration
                    closest_contact = (contact_pos, normal, penetration)
                #print (f"Vertex-triangle collision detected with triangle {tri_idx}: "
				#							 f"contact_pos={contact_pos}, normal={normal}, penetration={penetration}")
        
        return closest_contact
    
    def get_statistics(self) -> dict[str, any]:
        """Get statistics about acceleration structures."""
        return {
            'spatial_hash_enabled': self.use_spatial_hash,
            'meshes_cached': len(self._spatial_grids),
            'grids': {
                name: grid.get_stats()
                for name, grid in self._spatial_grids.items()
            }
        }
