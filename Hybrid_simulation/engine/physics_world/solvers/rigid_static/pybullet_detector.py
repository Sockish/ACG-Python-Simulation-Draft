"""PyBullet-based collision detection for rigid-static body pairs.

Uses PyBullet's robust collision detection to find contact points,
penetration depths, and contact normals. The collision response
(impulse resolution) is still handled by our custom solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pybullet as p
import pybullet_data

from ...math_utils import Vec3
from ...state import RigidBodyState, StaticBodyState


@dataclass
class PyBulletContact:
    """Contact information returned by PyBullet."""
    position_on_rigid: Vec3  # Contact point on rigid body (world space)
    position_on_static: Vec3  # Contact point on static body (world space)
    normal: Vec3  # Contact normal (from static toward rigid)
    penetration: float  # Penetration depth (positive = penetrating)


@dataclass
class PyBulletCollisionDetector:
    """Collision detector using PyBullet for accurate contact detection.
    
    This detector creates a PyBullet physics simulation in DIRECT mode
    (no GUI) purely for collision queries. Bodies are synchronized with
    our simulation state before each query.
    """
    
    margin: float = 0.001  # Collision margin for shapes (meters)
    
    # Internal state
    _physics_client: int = field(default=-1, init=False, repr=False)
    _rigid_body_ids: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _static_body_ids: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    
    def initialize(self) -> None:
        """Initialize PyBullet physics client in DIRECT mode (headless)."""
        if self._initialized:
            return
        
        # Create physics client (DIRECT = no GUI, fast)
        self._physics_client = p.connect(p.DIRECT)
        
        # Set up PyBullet data path for built-in shapes
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Disable gravity (we only use PyBullet for collision detection)
        p.setGravity(0, 0, 0, physicsClientId=self._physics_client)
        
        self._initialized = True
        print("[PyBullet] Collision detector initialized (DIRECT mode)")
    
    def shutdown(self) -> None:
        """Clean up PyBullet resources."""
        if self._initialized and self._physics_client >= 0:
            p.disconnect(physicsClientId=self._physics_client)
            self._physics_client = -1
            self._initialized = False
            self._rigid_body_ids.clear()
            self._static_body_ids.clear()
            print("[PyBullet] Collision detector shut down")
    
    def register_rigid_body(self, rigid: RigidBodyState) -> int:
        """Create a PyBullet collision body for a rigid body mesh.
        
        IMPORTANT: Rigid bodies must use CONVEX hull collision shapes.
        PyBullet does NOT support concave-concave collision detection.
        Only convex-convex or convex-concave pairs can collide.
        
        Returns the PyBullet body ID.
        """
        if not self._initialized:
            self.initialize()
        
        if rigid.name in self._rigid_body_ids:
            return self._rigid_body_ids[rigid.name]
        
        # Load mesh as convex hull collision shape
        mesh_path = str(rigid.mesh_path.resolve())
        
        # CRITICAL: Dynamic/rigid bodies MUST use convex hull (default behavior)
        # PyBullet cannot detect collisions between two concave meshes!
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            # NO flags = convex hull (required for collision detection to work)
            physicsClientId=self._physics_client
        )
        
        # Create body at origin (we'll update position/orientation before queries)
        body_id = p.createMultiBody(
            baseMass=rigid.mass,
            baseCollisionShapeIndex=collision_shape,
            basePosition=rigid.position,
            baseOrientation=rigid.orientation,  # PyBullet uses (x,y,z,w) quaternion
            physicsClientId=self._physics_client
        )
        
        self._rigid_body_ids[rigid.name] = body_id
        print(f"[PyBullet] Registered rigid body '{rigid.name}' as CONVEX body {body_id}")
        return body_id
    
    def register_static_body(self, static: StaticBodyState) -> int:
        """Create a PyBullet collision body for a static mesh.
        
        Static bodies CAN use concave triangle mesh for accurate collision
        geometry, as long as they only collide with convex rigid bodies.
        
        Returns the PyBullet body ID.
        """
        if not self._initialized:
            self.initialize()
        
        if static.name in self._static_body_ids:
            return self._static_body_ids[static.name]
        
        mesh_path = str(static.mesh_path.resolve())
        
        # Static bodies can use concave triangle mesh (accurate geometry)
        # This works because rigid bodies use convex hulls
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,  # Exact triangle mesh for static
            physicsClientId=self._physics_client
        )
        
        # Static body: mass = 0
        body_id = p.createMultiBody(
            baseMass=0,  # Static body
            baseCollisionShapeIndex=collision_shape,
            basePosition=static.position,
            baseOrientation=static.orientation,
            physicsClientId=self._physics_client
        )
        
        self._static_body_ids[static.name] = body_id
        print(f"[PyBullet] Registered static body '{static.name}' as CONCAVE body {body_id}")
        return body_id
    
    def sync_rigid_body(self, rigid: RigidBodyState) -> None:
        """Synchronize PyBullet body pose with our simulation state."""
        if rigid.name not in self._rigid_body_ids:
            self.register_rigid_body(rigid)
        
        body_id = self._rigid_body_ids[rigid.name]
        
        # PyBullet quaternion format: (x, y, z, w)
        # Our format is also (x, y, z, w)
        p.resetBasePositionAndOrientation(
            body_id,
            rigid.position,
            rigid.orientation,
            physicsClientId=self._physics_client
        )
    
    def detect_contacts(
        self,
        rigid: RigidBodyState,
        static: StaticBodyState,
        max_contacts: int = 8
    ) -> List[PyBulletContact]:
        """Detect contacts between a rigid body and a static body.
        
        Args:
            rigid: The rigid body state
            static: The static body state
            max_contacts: Maximum number of contacts to return
            
        Returns:
            List of PyBulletContact objects describing contact points
        """
        if not self._initialized:
            self.initialize()
        
        # Ensure bodies are registered
        rigid_id = self.register_rigid_body(rigid)
        static_id = self.register_static_body(static)
        
        # Sync rigid body pose
        self.sync_rigid_body(rigid)
        
        # Perform collision detection
        # stepSimulation is needed to update collision pairs
        p.performCollisionDetection(physicsClientId=self._physics_client)
        
        # Get contact points
        contact_points = p.getContactPoints(
            bodyA=rigid_id,
            bodyB=static_id,
            physicsClientId=self._physics_client
        )
        
        contacts: List[PyBulletContact] = []
        
        for cp in contact_points:
            # PyBullet contact point structure:
            # [0] contactFlag
            # [1] bodyUniqueIdA
            # [2] bodyUniqueIdB
            # [3] linkIndexA
            # [4] linkIndexB
            # [5] positionOnA (x, y, z)
            # [6] positionOnB (x, y, z)
            # [7] contactNormalOnB (x, y, z) - points from B toward A
            # [8] contactDistance (negative = penetration)
            # [9] normalForce
            
            pos_on_rigid = cp[5]  # Position on body A (rigid)
            pos_on_static = cp[6]  # Position on body B (static)
            normal_on_b = cp[7]  # Normal points from static toward rigid
            contact_distance = cp[8]  # Negative = penetrating
            
            # Convert to our contact format
            # Penetration is positive when overlapping
            penetration = -contact_distance
            
            if penetration > 0:  # Only report actual penetrating contacts
                contacts.append(PyBulletContact(
                    position_on_rigid=pos_on_rigid,
                    position_on_static=pos_on_static,
                    normal=normal_on_b,  # Already points from static to rigid
                    penetration=penetration
                ))
        
        # Sort by penetration depth and limit
        contacts.sort(key=lambda c: c.penetration, reverse=True)
        return contacts[:max_contacts]
    
    def detect_all_contacts(
        self,
        rigids: List[RigidBodyState],
        statics: List[StaticBodyState],
        max_contacts_per_pair: int = 4
    ) -> Dict[Tuple[str, str], List[PyBulletContact]]:
        """Detect contacts for all rigid-static pairs.
        
        Returns:
            Dictionary mapping (rigid_name, static_name) to list of contacts
        """
        if not self._initialized:
            self.initialize()
        
        # Register all bodies
        for rigid in rigids:
            self.register_rigid_body(rigid)
        for static in statics:
            self.register_static_body(static)
        
        # Sync all rigid body poses
        for rigid in rigids:
            self.sync_rigid_body(rigid)
        
        # Single collision detection pass
        p.performCollisionDetection(physicsClientId=self._physics_client)
        
        # Collect contacts
        all_contacts: Dict[Tuple[str, str], List[PyBulletContact]] = {}
        
        for rigid in rigids:
            for static in statics:
                rigid_id = self._rigid_body_ids[rigid.name]
                static_id = self._static_body_ids[static.name]
                
                contact_points = p.getContactPoints(
                    bodyA=rigid_id,
                    bodyB=static_id,
                    physicsClientId=self._physics_client
                )
                
                contacts = []
                for cp in contact_points:
                    penetration = -cp[8]
                    if penetration > 0:
                        contacts.append(PyBulletContact(
                            position_on_rigid=cp[5],
                            position_on_static=cp[6],
                            normal=cp[7],
                            penetration=penetration
                        ))
                
                if contacts:
                    contacts.sort(key=lambda c: c.penetration, reverse=True)
                    all_contacts[(rigid.name, static.name)] = contacts[:max_contacts_per_pair]
        
        return all_contacts
