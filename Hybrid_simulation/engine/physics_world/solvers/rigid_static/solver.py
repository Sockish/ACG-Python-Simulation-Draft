"""Solver handling rigid body collisions against static meshes using vertex-level collision detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ...math_utils import (
    Vec3, add, aabb_intersects_aabb, aabb_normal_at_point, closest_point_on_aabb,
    cross, distance_to_aabb, dot, length, mul, normalize, point_in_aabb, 
    quaternion_to_matrix, sub, inverse_transform_point
)
from ...state import RigidBodyState, StaticBodyState
from .triangle_detector import TriangleMeshCollisionDetector
from .pybullet_detector import PyBulletCollisionDetector


@dataclass
class ContactPoint:
    """Represents a single contact point between rigid and static body."""
    position: Vec3  # World space contact position
    normal: Vec3  # Contact normal (from static to rigid)
    penetration: float  # Penetration depth
    rigid_local: Vec3  # Contact point in rigid body local space


@dataclass
class RigidStaticSolver:
    restitution: float = 0.9  # Coefficient of restitution
    friction: float = 1.0  # Coefficient of friction
    rolling_resistance_coefficient: float = 0.15  # Rolling resistance coefficient (increased for faster decay)
    use_broadphase: bool = True  # Use AABB broadphase culling
    max_contacts_per_pair: int = 4  # Maximum contact points to process per collision pair
    max_position_iterations: int = 10  # Maximum iterations to eliminate penetration
    baumgarte_factor: float = 0.8  # Position correction strength (aggressive)
    penetration_tolerance: float = 0.0001  # Stop when penetration below this (0.1mm)
    velocity_threshold: float = 0.1  # Velocity below which to apply resting contact (m/s)
    max_penetration_correction: float = 0.5  # Maximum position correction per iteration (m)
    resting_angular_threshold: float = 0.01  # Angular velocity threshold for damping (rad/s)
    use_triangle_collision: bool = True  # Use precise triangle mesh collision (vs AABB approximation)
    use_spatial_hash: bool = True  # Use spatial hash acceleration for triangle collision
    spatial_hash_cell_size: float = 0.3  # Cell size for spatial hash grid (meters)
    use_pybullet_collision: bool = True  # Use PyBullet for collision detection (most accurate)
    
    # Internal collision detector (initialized on first use)
    _collision_detector: TriangleMeshCollisionDetector = field(default=None, init=False, repr=False)
    _pybullet_detector: PyBulletCollisionDetector = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    
    def _ensure_initialized(self, rigids: List[RigidBodyState], statics: List[StaticBodyState]) -> None:
        """Initialize collision detector and build acceleration structures."""
        if self._initialized:
            return
        
        if self.use_pybullet_collision:
            # Initialize PyBullet detector
            self._pybullet_detector = PyBulletCollisionDetector()
            self._pybullet_detector.initialize()
            
            # Register all bodies
            for rigid in rigids:
                self._pybullet_detector.register_rigid_body(rigid)
            for static in statics:
                self._pybullet_detector.register_static_body(static)
                
            print(f"[RigidStaticSolver] Using PyBullet collision detection")
        elif self.use_triangle_collision:
            self._collision_detector = TriangleMeshCollisionDetector(
                use_spatial_hash=self.use_spatial_hash,
                cell_size=self.spatial_hash_cell_size
            )
            
            # Build acceleration structures for all static meshes
            for static in statics:
                self._collision_detector.build_acceleration_structure(static)
            
            print(f"[RigidStaticSolver] Using triangle mesh collision detection")
        else:
            print(f"[RigidStaticSolver] Using AABB collision detection")
        
        self._initialized = True

    def step(self, rigids: List[RigidBodyState], statics: List[StaticBodyState], dt: float) -> None:
        """Detect and resolve collisions between rigid and static bodies."""
        if not rigids or not statics:
            return
        
        # Initialize collision detector on first call
        self._ensure_initialized(rigids, statics)
        
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
                # Broadphase: AABB intersection test (skip for PyBullet - it does its own broadphase)
                if self.use_broadphase and not self.use_pybullet_collision:
                    static_min, static_max = static.world_bounds
                    if not aabb_intersects_aabb(rigid_aabb_min, rigid_aabb_max, static_min, static_max):
                        continue
                
                # Fine-phase: Detect contacts using selected method
                if self.use_pybullet_collision:
                    contacts = self._detect_contacts_pybullet(rigid, static, rotation_matrix)
                else:
                    contacts = self._detect_contacts(world_vertices, rigid.centered_vertices, 
                                                     rigid.position, rotation_matrix, static)
                
                if not contacts:
                    continue
                
                print(f"Initial collision: {len(contacts)} contacts, point: {contacts[0].position} all penetrations: {[c.penetration for c in contacts]}")
                
                # Iteratively resolve position penetration until eliminated
                iteration = 0
                max_penetration = max(c.penetration for c in contacts)
                
                while max_penetration > self.penetration_tolerance and iteration < self.max_position_iterations:
                    # Resolve collisions using the detected contact points
                    self._resolve_contacts(rigid, contacts, dt)
                    print(f"  Iteration {iteration}: max penetration = {max_penetration:.6f}m")
                    
                    # Re-compute rotation matrix and world vertices after position correction
                    rotation_matrix = quaternion_to_matrix(rigid.orientation)
                    world_vertices = rigid.get_world_vertices()
                    
                    # Re-detect contacts using the same method
                    if self.use_pybullet_collision:
                        contacts = self._detect_contacts_pybullet(rigid, static, rotation_matrix)
                    else:
                        contacts = self._detect_contacts(world_vertices, rigid.centered_vertices,
                                                        rigid.position, rotation_matrix, static)
                    
                    if not contacts:
                        print(f"  No contacts after iteration {iteration}, breaking")
                        break
                    
                    max_penetration = max(c.penetration for c in contacts)
                    iteration += 1
                print(f"Resolved collision after {iteration} iterations, remaining max penetration: {max_penetration:.6f}m")
                
                # Apply rolling resistance when in contact with ground
                self._apply_rolling_resistance(rigid, contacts, dt)
                print(f"Resolved collision after {iteration} iterations, remaining max penetration: {max_penetration:.6f}m")
                # self._apply_sliding_resistance(rigid, contacts, dt) ## This term is very small!
                
    
    def _detect_contacts(self, world_vertices: List[Vec3], local_vertices: List[Vec3],
                        rigid_position: Vec3, rotation_matrix: Tuple[Vec3, Vec3, Vec3],
                        static: StaticBodyState) -> List[ContactPoint]:
        """Detect contact points between rigid body vertices and static mesh.
        
        Supports both triangle mesh collision (precise) and AABB collision (fast approximation).
        """
        contacts: List[ContactPoint] = []
        bounds_min, bounds_max = static.world_bounds
        
        contact_threshold = 0.0000  # Strict penetration threshold
        
        for world_vert, local_vert in zip(world_vertices, local_vertices):
            contact_result = None
            
            if self.use_triangle_collision and self._collision_detector is not None:
                # Use precise triangle mesh collision
                contact_result = self._collision_detector.detect_vertex_contacts(
                    world_vert, local_vert, static, contact_threshold
                )
                
                if contact_result is not None and contact_result[2] < 0.02:  # 6cm max penetration for valid contact
                    contact_pos, normal, penetration = contact_result
                    contacts.append(ContactPoint(
                        position=contact_pos,
                        normal=normal,
                        penetration=penetration,
                        rigid_local=local_vert
                    ))
            else:
                # Fallback to AABB collision (original behavior)
                signed_distance = distance_to_aabb(world_vert, bounds_min, bounds_max)
                
                if signed_distance < contact_threshold:
                    contact_pos = closest_point_on_aabb(world_vert, bounds_min, bounds_max)
                    normal = aabb_normal_at_point(contact_pos, bounds_min, bounds_max)
                    penetration = -signed_distance
                    
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
    
    def _detect_contacts_pybullet(
        self,
        rigid: RigidBodyState,
        static: StaticBodyState,
        rotation_matrix: Tuple[Vec3, Vec3, Vec3]
    ) -> List[ContactPoint]:
        """Detect contact points using PyBullet collision detection.
        
        PyBullet provides robust, accurate collision detection for complex meshes.
        The detected contacts are converted to our ContactPoint format for resolution.
        """
        if self._pybullet_detector is None:
            return []
        
        # Get contacts from PyBullet
        pybullet_contacts = self._pybullet_detector.detect_contacts(
            rigid, static, max_contacts=self.max_contacts_per_pair
        )
        
        contacts: List[ContactPoint] = []
        
        for pb_contact in pybullet_contacts:
            # Convert PyBullet contact to our ContactPoint format
            # Use position on rigid body as contact position
            contact_pos = pb_contact.position_on_rigid
            
            # Compute local position by inverse transforming
            local_pos = inverse_transform_point(contact_pos, rotation_matrix, rigid.position)
            
            contacts.append(ContactPoint(
                position=contact_pos,
                normal=pb_contact.normal,
                penetration=pb_contact.penetration,
                rigid_local=local_pos
            ))
        
        return contacts

    def _resolve_contacts(self, rigid: RigidBodyState, contacts: List[ContactPoint], dt: float) -> None:
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
        
        # Position correction: aggressively eliminate all penetration
        # Use high baumgarte factor to quickly resolve penetration
        if avg_penetration > self.penetration_tolerance:
            # Apply aggressive correction clamped to maximum
            correction_amount = min(avg_penetration * self.baumgarte_factor, 
                                   self.max_penetration_correction)
            correction = mul(avg_normal, correction_amount)
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
            
            # Calculate normal impulse
            r_cross_n = cross(r, contact.normal)
            angular_factor = inv_I * dot(r_cross_n, r_cross_n)
            denominator = inv_mass + angular_factor
            
            effective_restitution = self.restitution
            j_n = -(1.0 + effective_restitution) * v_n / max(denominator, 1e-6)
            
            # Clamp to ensure impulse is at least slightly repulsive
            # Allow very small negative for numerical stability, but mainly push outward （since here may v_n be slightly negative, meaning it's going out now but haven't done yet!）
            j_n = max(j_n, 0.0)
            
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
                print(f"Applying friction impulse: {impulse_t}, j_t: {j_t:.6f}, old linear velocity: {rigid.linear_velocity}")
                rigid.linear_velocity = add(rigid.linear_velocity, mul(impulse_t, inv_mass))
                print(f"Applied friction impulse: {impulse_t}, j_t: {j_t:.6f}, new linear velocity: {rigid.linear_velocity}")

                # Angular impulse from friction
                torque_t = cross(r, impulse_t)
                angular_impulse_t = mul(torque_t, inv_I)
                rigid.angular_velocity = add(rigid.angular_velocity, angular_impulse_t)
    
    def _apply_rolling_resistance(self, rigid: RigidBodyState, contacts: List[ContactPoint], dt: float) -> None:
        """Apply rolling resistance torque to simulate energy loss during rolling.
        
        Rolling resistance opposes the angular velocity and is proportional to the normal force.
        This simulates deformation, surface irregularities, and material hysteresis.
        """
        if not contacts:
            return
        
        # Calculate average normal force magnitude from contacts, may used to do mg's projection
        # Approximate normal force from weight component
        avg_normal = (
            sum(c.normal[0] for c in contacts) / len(contacts),
            sum(c.normal[1] for c in contacts) / len(contacts),
            sum(c.normal[2] for c in contacts) / len(contacts),
        )
        
        # Normal force magnitude (approximate from gravity)
        # F_n ≈ m * g (assuming object is resting on surface)
        normal_force = rigid.mass * 9.81 * dot(avg_normal, (0.0, 0.0, 1.0))  # Approximate normal force
        
        # Rolling resistance torque magnitude: τ_r = C_rr * F_n * r
        # where C_rr is rolling resistance coefficient, r is contact radius
        # For simplicity, use average distance from CoM to contact points
        avg_contact_distance = 0.0
        for contact in contacts:
            r = sub(contact.position, rigid.position)
            avg_contact_distance += length(r)
        avg_contact_distance /= len(contacts)
        
        # Rolling resistance torque magnitude
        rolling_torque_magnitude = self.rolling_resistance_coefficient * normal_force * avg_contact_distance
        #print(f"Rolling resistance torque magnitude: {rolling_torque_magnitude:.4f} N·m, normal force: {normal_force:.4f} N")
        
        # Direction: opposite to angular velocity
        angular_speed = length(rigid.angular_velocity)
        if angular_speed > 1e-6:
            # Torque opposes rotation
            torque_direction = mul(rigid.angular_velocity, -1.0 / angular_speed)
            rolling_torque = mul(torque_direction, rolling_torque_magnitude)
            
            # Compute angular deceleration: α = τ / I
            I_avg = sum(rigid.inertia) / 3.0
            angular_deceleration = mul(rolling_torque, 1.0 / max(I_avg, 1e-6))
            
            # Update angular velocity: ω = ω + α * dt
            new_angular_velocity = add(rigid.angular_velocity, mul(angular_deceleration, dt))
            
            # Check if we would reverse direction (overdamping)
            new_speed = length(new_angular_velocity)
            dot_product = dot(rigid.angular_velocity, new_angular_velocity)
            
            if dot_product < 0 or new_speed < self.resting_angular_threshold:
                # Would reverse or too slow - stop rotation completely
                rigid.angular_velocity = (0.0, 0.0, 0.0)
                print(f"Rolling resistance stopped rotation: old speed={angular_speed:.4f} rad/s → 0")
            else:
                rigid.angular_velocity = new_angular_velocity
                new_speed_actual = length(rigid.angular_velocity)
                reduction_percent = (angular_speed - new_speed_actual) / angular_speed * 100
                print(f"Rolling resistance: torque_mag={rolling_torque_magnitude:.4f} N·m, speed: {angular_speed:.4f} → {new_speed_actual:.4f} rad/s ({reduction_percent:.1f}% reduction)")

    def _apply_sliding_resistance(self, rigid: RigidBodyState, contacts: List[ContactPoint], dt: float) -> None:
        """Apply sliding resistance (kinetic friction) to reduce linear velocity when in contact."""
        if not contacts:
            return
        
        # Average normal from contacts
        avg_normal = (
            sum(c.normal[0] for c in contacts) / len(contacts),
            sum(c.normal[1] for c in contacts) / len(contacts),
            sum(c.normal[2] for c in contacts) / len(contacts),
        )
        avg_normal = normalize(avg_normal)
        
        # compute the tangential friction direction according to the contact points' velocity's projection on the contact plane, v = v_linear + ω × r and then subtract normal component
        v_contact_avg = tuple(sum(add(rigid.linear_velocity, cross(rigid.angular_velocity, sub(contact.position, rigid.position)))[i] for contact in contacts) / len(contacts) for i in range(3))
        v_t_avg = sub(v_contact_avg, mul(avg_normal, dot(v_contact_avg, avg_normal)))
        v_t_mag = length(v_t_avg)
        print(f"Average tangential velocity at contacts: {v_t_avg}, magnitude: {v_t_mag:.4f} m/s")
        if v_t_mag < 1e-6:
            return  # No significant tangential velocity
        tangent = mul(v_t_avg, 1.0 / v_t_mag)
        # Friction acceleration magnitude: a_f = μ_k * g * cos(θ)
        friction_acceleration_mag = self.friction * 9.81 * dot(avg_normal, (0.0, 0.0, 1.0))  # Approximate normal force component
        friction_acceleration = mul(tangent, -friction_acceleration_mag)
        # Update linear velocity: v = v + a_f * dt
        new_linear_velocity = add(rigid.linear_velocity, mul(friction_acceleration, dt))
        # Check if we would reverse direction (overdamping)
        dot_product = dot(rigid.linear_velocity, new_linear_velocity)
        if dot_product < 0:
            # Would reverse - stop linear motion in that direction
            rigid.linear_velocity = sub(rigid.linear_velocity, mul(tangent, dot(rigid.linear_velocity, tangent)))
            print(f"Sliding resistance stopped linear motion along tangent: old velocity={rigid.linear_velocity} → {rigid.linear_velocity}")
        else:
            rigid.linear_velocity = new_linear_velocity
            new_speed_actual = length(rigid.linear_velocity)
            old_speed = length(add(rigid.linear_velocity, mul(friction_acceleration, -dt)))
            reduction_percent = (old_speed - new_speed_actual) / old_speed * 100 if old_speed > 1e-6 else 0.0
            print(f"Sliding resistance: acc_mag={friction_acceleration_mag:.4f} m/s², speed: {old_speed:.4f} → {new_speed_actual:.4f} m/s ({reduction_percent:.1f}% reduction)")
