"""Solver handling rigid body collisions against static meshes using vertex-level collision detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ....mesh_utils import load_obj_mesh, OBJMesh
from ...math_utils import (
    Vec3, add, aabb_intersects_aabb, aabb_normal_at_point, closest_point_on_aabb,
    cross, distance_to_aabb, dot, length, mul, normalize, point_in_aabb, 
    quaternion_to_matrix, sub, transform_point
)
from ...state import RigidBodyState, StaticBodyState


@dataclass
class ContactPoint:
    """Represents a single contact point between rigid and static body."""
    position: Vec3  # World space contact position
    normal: Vec3  # Contact normal (from static to rigid)
    penetration: float  # Penetration depth
    rigid_local: Vec3  # Contact point in rigid body local space


@dataclass
class RigidStaticSolver:
    restitution: float = 0.4  # Coefficient of restitution
    friction: float = 0.3  # Coefficient of friction
    use_broadphase: bool = True  # Use AABB broadphase culling
    max_contacts_per_pair: int = 4  # Maximum contact points to process per collision pair

    def step(self, rigids: List[RigidBodyState], statics: List[StaticBodyState], dt: float) -> None:
        """Detect and resolve collisions between rigid and static bodies."""
        del dt
        if not rigids or not statics:
            return
        
        for rigid in rigids:
            if rigid.mass <= 0:  # Skip static/kinematic bodies
                continue
            
            # Get rotation matrix for transforming local vertices to world space
            rotation_matrix = quaternion_to_matrix(rigid.orientation)
            
            # Get world-space vertices (centered vertices transformed to world)
            world_vertices = rigid.get_world_vertices()
            
            # Compute world-space AABB for rigid body
            if not world_vertices:
                continue
            
            rigid_aabb_min = (
                min(v[0] for v in world_vertices),
                min(v[1] for v in world_vertices),
                min(v[2] for v in world_vertices),
            )
            rigid_aabb_max = (
                max(v[0] for v in world_vertices),
                max(v[1] for v in world_vertices),
                max(v[2] for v in world_vertices),
            )
            
            # Check collision with each static body
            for static in statics:
                # Broadphase: AABB intersection test
                if self.use_broadphase:
                    static_min, static_max = static.world_bounds
                    if not aabb_intersects_aabb(rigid_aabb_min, rigid_aabb_max, static_min, static_max):
                        continue
                
                # Fine-phase: Check which vertices penetrate the static AABB
                contacts = self._detect_contacts(world_vertices, rigid.centered_vertices, 
                                                 rigid.position, rotation_matrix, static)
                
                if not contacts:
                    continue
                
                # Resolve collisions using the detected contact points
                self._resolve_contacts(rigid, contacts)
    
    def _detect_contacts(self, world_vertices: List[Vec3], local_vertices: List[Vec3],
                        rigid_position: Vec3, rotation_matrix: Tuple[Vec3, Vec3, Vec3],
                        static: StaticBodyState) -> List[ContactPoint]:
        """Detect contact points between rigid body vertices and static AABB."""
        contacts: List[ContactPoint] = []
        bounds_min, bounds_max = static.world_bounds
        
        for world_vert, local_vert in zip(world_vertices, local_vertices):
            # Check if vertex is inside or very close to static AABB
            signed_distance = distance_to_aabb(world_vert, bounds_min, bounds_max)
            
            # Only process penetrating or touching vertices
            if signed_distance < 0.01:  # Small threshold for contact
                # Get contact point (closest point on AABB)
                contact_pos = closest_point_on_aabb(world_vert, bounds_min, bounds_max)
                
                # Get surface normal at contact point
                normal = aabb_normal_at_point(contact_pos, bounds_min, bounds_max)
                
                # Calculate penetration depth (negative distance = penetration)
                penetration = max(0.0, -signed_distance)
                
                contacts.append(ContactPoint(
                    position=contact_pos,
                    normal=normal,
                    penetration=penetration,
                    rigid_local=local_vert
                ))
        
        # Limit number of contacts to avoid over-processing
        if len(contacts) > self.max_contacts_per_pair:
            # Sort by penetration depth and keep the deepest contacts
            contacts.sort(key=lambda c: c.penetration, reverse=True)
            contacts = contacts[:self.max_contacts_per_pair]
        
        return contacts
    
    def _resolve_contacts(self, rigid: RigidBodyState, contacts: List[ContactPoint]) -> None:
        """Resolve collision using impulse-based method for all contact points."""
        if not contacts:
            return
        
        # Average penetration for position correction
        avg_penetration = sum(c.penetration for c in contacts) / len(contacts)
        avg_normal = (
            sum(c.normal[0] for c in contacts) / len(contacts),
            sum(c.normal[1] for c in contacts) / len(contacts),
            sum(c.normal[2] for c in contacts) / len(contacts),
        )
        avg_normal = normalize(avg_normal)
        
        # Position correction: push rigid body out
        if avg_penetration > 1e-6:
            correction = mul(avg_normal, avg_penetration + 1e-4)
            rigid.position = add(rigid.position, correction)
        
        # Compute average inertia
        I_avg = sum(rigid.inertia) / 3.0
        inv_I = 1.0 / max(I_avg, 1e-6)
        inv_mass = 1.0 / rigid.mass
        
        # Process each contact point for impulse
        for contact in contacts:
            # Vector from rigid center to contact point
            r = sub(contact.position, rigid.position)
            
            # Velocity at contact point: v = v_linear + ω × r
            v_angular = cross(rigid.angular_velocity, r)
            v_contact = add(rigid.linear_velocity, v_angular)
            
            # Relative velocity along normal
            v_n = dot(v_contact, contact.normal)
            
            # Only resolve if moving into surface
            if v_n >= 0:
                continue
            
            # Calculate normal impulse
            r_cross_n = cross(r, contact.normal)
            angular_factor = inv_I * dot(r_cross_n, r_cross_n)
            denominator = inv_mass + angular_factor
            j_n = -(1.0 + self.restitution) * v_n / max(denominator, 1e-6)
            
            # Apply normal impulse
            impulse_n = mul(contact.normal, j_n)
            rigid.linear_velocity = add(rigid.linear_velocity, mul(impulse_n, inv_mass))
            
            # Angular impulse from normal force
            torque_n = cross(r, impulse_n)
            angular_impulse_n = mul(torque_n, inv_I)
            rigid.angular_velocity = add(rigid.angular_velocity, angular_impulse_n)
            
            # Tangential friction
            v_contact_updated = add(rigid.linear_velocity, cross(rigid.angular_velocity, r))
            v_t = sub(v_contact_updated, mul(contact.normal, dot(v_contact_updated, contact.normal)))
            v_t_mag = length(v_t)
            
            if v_t_mag > 1e-6:
                tangent = mul(v_t, 1.0 / v_t_mag)
                
                # Friction impulse magnitude
                r_cross_t = cross(r, tangent)
                angular_factor_t = inv_I * dot(r_cross_t, r_cross_t)
                denominator_t = inv_mass + angular_factor_t
                j_t = -v_t_mag / max(denominator_t, 1e-6)
                
                # Apply Coulomb friction limit
                j_t_max = self.friction * abs(j_n)
                j_t = max(-j_t_max, min(j_t, j_t_max))
                
                # Apply friction impulse
                impulse_t = mul(tangent, j_t)
                rigid.linear_velocity = add(rigid.linear_velocity, mul(impulse_t, inv_mass))
                
                # Angular impulse from friction
                torque_t = cross(r, impulse_t)
                angular_impulse_t = mul(torque_t, inv_I)
                rigid.angular_velocity = add(rigid.angular_velocity, angular_impulse_t)
