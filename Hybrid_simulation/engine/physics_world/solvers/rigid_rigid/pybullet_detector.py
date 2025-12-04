"""PyBullet-based collision detection for rigid-rigid body pairs.

Uses PyBullet's collision detection for arbitrary closed triangle meshes.
For rigid-rigid collision, we use convex decomposition (V-HACD) to allow
accurate collision between concave shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pybullet as p
import pybullet_data

from ...math_utils import Vec3
from ...state import RigidBodyState


@dataclass
class RigidRigidContact:
    """Contact information between two rigid bodies."""
    position_on_a: Vec3  # Contact point on body A (world space)
    position_on_b: Vec3  # Contact point on body B (world space)
    normal: Vec3  # Contact normal (from B toward A)
    penetration: float  # Penetration depth (positive = penetrating)
    body_a_name: str  # Name of body A
    body_b_name: str  # Name of body B


@dataclass
class PyBulletRigidCollisionDetector:
    """Collision detector for rigid-rigid pairs using PyBullet.
    
    For concave meshes, PyBullet requires either:
    1. Convex hull (loses concave details)
    2. Convex decomposition via V-HACD (accurate but slower setup)
    
    This detector uses V-HACD for accurate concave mesh collision.
    """
    
    use_vhacd: bool = True  # Use V-HACD convex decomposition for concave meshes
    vhacd_resolution: int = 100000  # V-HACD resolution (higher = more accurate)
    vhacd_max_hulls: int = 64  # Maximum convex hulls per mesh
    margin: float = 0.001  # Collision margin (meters)
    
    # Internal state
    _physics_client: int = field(default=-1, init=False, repr=False)
    _body_ids: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _collision_shapes: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    
    def initialize(self) -> None:
        """Initialize PyBullet physics client in DIRECT mode."""
        if self._initialized:
            return
        
        self._physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0, physicsClientId=self._physics_client)
        
        self._initialized = True
        print("[PyBullet Rigid-Rigid] Collision detector initialized (DIRECT mode)")
    
    def shutdown(self) -> None:
        """Clean up PyBullet resources."""
        if self._initialized and self._physics_client >= 0:
            p.disconnect(physicsClientId=self._physics_client)
            self._physics_client = -1
            self._initialized = False
            self._body_ids.clear()
            self._collision_shapes.clear()
            print("[PyBullet Rigid-Rigid] Collision detector shut down")
    
    def _create_collision_shape(self, rigid: RigidBodyState) -> int:
        """Create collision shape for a rigid body.
        
        Uses V-HACD convex decomposition for accurate concave mesh collision.
        Falls back to simple convex hull if V-HACD fails.
        """
        mesh_path = str(rigid.mesh_path.resolve())
        
        if self.use_vhacd:
            try:
                # Use V-HACD for convex decomposition
                # This allows accurate collision between concave meshes
                name_out = f"vhacd_{rigid.name}.obj"
                name_log = f"vhacd_{rigid.name}.log"
                
                p.vhacd(
                    mesh_path,
                    name_out,
                    name_log,
                    resolution=self.vhacd_resolution,
                    maxNumVerticesPerCH=64,
                    maxConvexHulls=self.vhacd_max_hulls,
                    physicsClientId=self._physics_client
                )
                
                # Load the decomposed mesh as compound collision shape
                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    fileName=name_out,
                    physicsClientId=self._physics_client
                )
                print(f"[PyBullet] Created V-HACD decomposition for '{rigid.name}'")
                return collision_shape
                
            except Exception as e:
                print(f"[PyBullet] V-HACD failed for '{rigid.name}': {e}, using convex hull")
        
        # Fallback: simple convex hull
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            physicsClientId=self._physics_client
        )
        print(f"[PyBullet] Created convex hull for '{rigid.name}'")
        return collision_shape
    
    def register_rigid_body(self, rigid: RigidBodyState) -> int:
        """Register a rigid body for collision detection.
        
        Returns the PyBullet body ID.
        """
        if not self._initialized:
            self.initialize()
        
        if rigid.name in self._body_ids:
            return self._body_ids[rigid.name]
        
        # Create or reuse collision shape
        if rigid.name not in self._collision_shapes:
            self._collision_shapes[rigid.name] = self._create_collision_shape(rigid)
        
        collision_shape = self._collision_shapes[rigid.name]
        
        # Create physics body
        body_id = p.createMultiBody(
            baseMass=rigid.mass,
            baseCollisionShapeIndex=collision_shape,
            basePosition=rigid.position,
            baseOrientation=rigid.orientation,
            physicsClientId=self._physics_client
        )
        
        self._body_ids[rigid.name] = body_id
        print(f"[PyBullet Rigid-Rigid] Registered body '{rigid.name}' as ID {body_id}")
        return body_id
    
    def sync_body(self, rigid: RigidBodyState) -> None:
        """Synchronize PyBullet body pose with simulation state."""
        if rigid.name not in self._body_ids:
            self.register_rigid_body(rigid)
            return
        
        body_id = self._body_ids[rigid.name]
        p.resetBasePositionAndOrientation(
            body_id,
            rigid.position,
            rigid.orientation,
            physicsClientId=self._physics_client
        )
    
    def detect_contacts(
        self,
        rigid_a: RigidBodyState,
        rigid_b: RigidBodyState,
        max_contacts: int = 8
    ) -> List[RigidRigidContact]:
        """Detect contacts between two rigid bodies.
        
        Returns list of contact points with positions, normals, and penetration.
        """
        if not self._initialized:
            self.initialize()
        
        # Ensure both bodies are registered
        body_a_id = self.register_rigid_body(rigid_a)
        body_b_id = self.register_rigid_body(rigid_b)
        
        # Sync poses
        self.sync_body(rigid_a)
        self.sync_body(rigid_b)
        
        # Perform collision detection
        p.performCollisionDetection(physicsClientId=self._physics_client)
        
        # Get contact points
        contact_points = p.getContactPoints(
            bodyA=body_a_id,
            bodyB=body_b_id,
            physicsClientId=self._physics_client
        )
        
        contacts: List[RigidRigidContact] = []
        
        for cp in contact_points:
            # PyBullet contact structure:
            # [5] positionOnA, [6] positionOnB, [7] normalOnB, [8] distance
            pos_on_a = cp[5]
            pos_on_b = cp[6]
            normal = cp[7]  # Points from B toward A
            distance = cp[8]  # Negative = penetrating
            
            penetration = -distance
            
            if penetration > 0:
                contacts.append(RigidRigidContact(
                    position_on_a=pos_on_a,
                    position_on_b=pos_on_b,
                    normal=normal,
                    penetration=penetration,
                    body_a_name=rigid_a.name,
                    body_b_name=rigid_b.name,
                ))
        
        # Sort by penetration and limit
        contacts.sort(key=lambda c: c.penetration, reverse=True)
        return contacts[:max_contacts]
    
    def detect_all_contacts(
        self,
        rigids: List[RigidBodyState],
        max_contacts_per_pair: int = 4
    ) -> Dict[Tuple[str, str], List[RigidRigidContact]]:
        """Detect contacts for all rigid body pairs.
        
        Returns dictionary mapping (name_a, name_b) to list of contacts.
        """
        if not self._initialized:
            self.initialize()
        
        # Register and sync all bodies
        for rigid in rigids:
            self.register_rigid_body(rigid)
            self.sync_body(rigid)
        
        # Single collision detection pass
        p.performCollisionDetection(physicsClientId=self._physics_client)
        
        # Check all pairs
        all_contacts: Dict[Tuple[str, str], List[RigidRigidContact]] = {}
        
        n = len(rigids)
        for i in range(n):
            for j in range(i + 1, n):
                rigid_a = rigids[i]
                rigid_b = rigids[j]
                
                body_a_id = self._body_ids[rigid_a.name]
                body_b_id = self._body_ids[rigid_b.name]
                
                contact_points = p.getContactPoints(
                    bodyA=body_a_id,
                    bodyB=body_b_id,
                    physicsClientId=self._physics_client
                )
                
                contacts = []
                for cp in contact_points:
                    penetration = -cp[8]
                    if penetration > 0:
                        contacts.append(RigidRigidContact(
                            position_on_a=cp[5],
                            position_on_b=cp[6],
                            normal=cp[7],
                            penetration=penetration,
                            body_a_name=rigid_a.name,
                            body_b_name=rigid_b.name,
                        ))
                
                if contacts:
                    contacts.sort(key=lambda c: c.penetration, reverse=True)
                    all_contacts[(rigid_a.name, rigid_b.name)] = contacts[:max_contacts_per_pair]
        
        return all_contacts
