"""Solver handling rigid-rigid body collisions using PyBullet collision detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ...math_utils import (
    Vec3, add, sub, mul, dot, cross, length, normalize,
    quaternion_to_matrix, inverse_transform_point
)
from ...state import RigidBodyState
from .pybullet_detector import PyBulletRigidCollisionDetector, RigidRigidContact


@dataclass
class RigidContactPoint:
    """Internal contact point representation for collision resolution."""
    position: Vec3  # World space contact position
    normal: Vec3  # Contact normal (from body B to body A)
    penetration: float  # Penetration depth
    local_a: Vec3  # Contact point in body A local space
    local_b: Vec3  # Contact point in body B local space


@dataclass
class RigidRigidSolver:
    """Solver for rigid-rigid body collisions.
    
    Uses PyBullet for collision detection and custom impulse-based
    response for collision resolution.
    """
    
    restitution: float = 0.5  # Coefficient of restitution (bounciness)
    friction: float = 0.3  # Coefficient of friction
    max_contacts_per_pair: int = 4  # Maximum contacts per collision pair
    max_position_iterations: int = 5  # Iterations to resolve penetration
    baumgarte_factor: float = 0.5  # Position correction strength
    penetration_tolerance: float = 0.001  # Acceptable penetration (1mm)
    velocity_threshold: float = 0.5  # Resting contact velocity threshold
    
    # V-HACD settings for concave mesh decomposition
    use_vhacd: bool = False  # Enable V-HACD (slower but more accurate for concave)
    vhacd_resolution: int = 50000
    vhacd_max_hulls: int = 32
    
    # Internal state
    _detector: PyBulletRigidCollisionDetector = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    
    def _ensure_initialized(self, rigids: List[RigidBodyState]) -> None:
        """Initialize collision detector on first use."""
        if self._initialized:
            return
        
        self._detector = PyBulletRigidCollisionDetector(
            use_vhacd=self.use_vhacd,
            vhacd_resolution=self.vhacd_resolution,
            vhacd_max_hulls=self.vhacd_max_hulls,
        )
        self._detector.initialize()
        
        # Register all rigid bodies
        for rigid in rigids:
            self._detector.register_rigid_body(rigid)
        
        print(f"[RigidRigidSolver] Initialized with {len(rigids)} rigid bodies")
        self._initialized = True
    
    def step(self, rigids: List[RigidBodyState], dt: float) -> None:
        """Detect and resolve collisions between all rigid body pairs."""
        if len(rigids) < 2:
            return
        
        # Initialize detector
        self._ensure_initialized(rigids)
        
        # Detect all contacts
        all_contacts = self._detector.detect_all_contacts(rigids, self.max_contacts_per_pair)
        
        if not all_contacts:
            return
        
        # Build name -> state map
        rigid_map = {r.name: r for r in rigids}
        
        # Process each colliding pair
        for (name_a, name_b), contacts in all_contacts.items():
            rigid_a = rigid_map[name_a]
            rigid_b = rigid_map[name_b]
            
            # Convert to internal contact format
            internal_contacts = self._convert_contacts(contacts, rigid_a, rigid_b)
            
            if not internal_contacts:
                continue
            
            # Resolve collision
            self._resolve_contacts(rigid_a, rigid_b, internal_contacts, dt)
    
    def _convert_contacts(
        self,
        contacts: List[RigidRigidContact],
        rigid_a: RigidBodyState,
        rigid_b: RigidBodyState
    ) -> List[RigidContactPoint]:
        """Convert PyBullet contacts to internal format with local coordinates."""
        rot_a = quaternion_to_matrix(rigid_a.orientation)
        rot_b = quaternion_to_matrix(rigid_b.orientation)
        
        internal = []
        for c in contacts:
            # Use midpoint as contact position
            position = (
                (c.position_on_a[0] + c.position_on_b[0]) * 0.5,
                (c.position_on_a[1] + c.position_on_b[1]) * 0.5,
                (c.position_on_a[2] + c.position_on_b[2]) * 0.5,
            )
            
            # Compute local positions
            local_a = inverse_transform_point(position, rot_a, rigid_a.position)
            local_b = inverse_transform_point(position, rot_b, rigid_b.position)
            
            internal.append(RigidContactPoint(
                position=position,
                normal=c.normal,  # Points from B toward A
                penetration=c.penetration,
                local_a=local_a,
                local_b=local_b,
            ))
        
        return internal
    
    def _resolve_contacts(
        self,
        rigid_a: RigidBodyState,
        rigid_b: RigidBodyState,
        contacts: List[RigidContactPoint],
        dt: float
    ) -> None:
        """Resolve collision using impulse-based method."""
        if not contacts:
            return
        
        # Check if either body is static (infinite mass)
        inv_mass_a = rigid_a.inverse_mass
        inv_mass_b = rigid_b.inverse_mass
        
        if inv_mass_a == 0 and inv_mass_b == 0:
            return  # Both static, no response needed
        
        # Compute inverse inertia tensors (diagonal approximation)
        inv_inertia_a = (
            1.0 / rigid_a.inertia[0] if rigid_a.inertia[0] > 0 else 0.0,
            1.0 / rigid_a.inertia[1] if rigid_a.inertia[1] > 0 else 0.0,
            1.0 / rigid_a.inertia[2] if rigid_a.inertia[2] > 0 else 0.0,
        )
        inv_inertia_b = (
            1.0 / rigid_b.inertia[0] if rigid_b.inertia[0] > 0 else 0.0,
            1.0 / rigid_b.inertia[1] if rigid_b.inertia[1] > 0 else 0.0,
            1.0 / rigid_b.inertia[2] if rigid_b.inertia[2] > 0 else 0.0,
        )
        
        for contact in contacts:
            # Vector from CoM to contact point
            r_a = sub(contact.position, rigid_a.position)
            r_b = sub(contact.position, rigid_b.position)
            
            # Compute relative velocity at contact point
            # v_rel = (v_a + ω_a × r_a) - (v_b + ω_b × r_b)
            vel_a = add(rigid_a.linear_velocity, cross(rigid_a.angular_velocity, r_a))
            vel_b = add(rigid_b.linear_velocity, cross(rigid_b.angular_velocity, r_b))
            v_rel = sub(vel_a, vel_b)
            
            # Normal component of relative velocity
            v_n = dot(v_rel, contact.normal)
            
            # If separating, skip this contact
            if v_n > 0:
                continue
            
            # Compute effective mass for normal impulse
            # K = 1/m_a + 1/m_b + (I_a^-1 * (r_a × n)) × r_a · n + (I_b^-1 * (r_b × n)) × r_b · n
            r_a_cross_n = cross(r_a, contact.normal)
            r_b_cross_n = cross(r_b, contact.normal)
            
            # Apply inverse inertia (diagonal)
            angular_a = (
                r_a_cross_n[0] * inv_inertia_a[0],
                r_a_cross_n[1] * inv_inertia_a[1],
                r_a_cross_n[2] * inv_inertia_a[2],
            )
            angular_b = (
                r_b_cross_n[0] * inv_inertia_b[0],
                r_b_cross_n[1] * inv_inertia_b[1],
                r_b_cross_n[2] * inv_inertia_b[2],
            )
            
            angular_effect_a = dot(cross(angular_a, r_a), contact.normal)
            angular_effect_b = dot(cross(angular_b, r_b), contact.normal)
            
            effective_mass = inv_mass_a + inv_mass_b + angular_effect_a + angular_effect_b
            
            if effective_mass <= 1e-10:
                continue
            
            # Compute normal impulse magnitude
            # j_n = -(1 + e) * v_n / K
            restitution = self.restitution if abs(v_n) > self.velocity_threshold else 0.0
            j_n = -(1.0 + restitution) * v_n / effective_mass
            
            # Add Baumgarte stabilization for penetration correction
            if contact.penetration > self.penetration_tolerance:
                bias = self.baumgarte_factor * contact.penetration / dt
                j_n += bias / effective_mass
            
            # Clamp impulse to be non-negative (no pulling)
            j_n = max(0.0, j_n)
            
            # Apply normal impulse
            impulse_n = mul(contact.normal, j_n)
            
            if inv_mass_a > 0:
                rigid_a.linear_velocity = add(rigid_a.linear_velocity, mul(impulse_n, inv_mass_a))
                delta_omega_a = (
                    angular_a[0] * j_n,
                    angular_a[1] * j_n,
                    angular_a[2] * j_n,
                )
                rigid_a.angular_velocity = add(rigid_a.angular_velocity, delta_omega_a)
            
            if inv_mass_b > 0:
                rigid_b.linear_velocity = sub(rigid_b.linear_velocity, mul(impulse_n, inv_mass_b))
                delta_omega_b = (
                    angular_b[0] * j_n,
                    angular_b[1] * j_n,
                    angular_b[2] * j_n,
                )
                rigid_b.angular_velocity = sub(rigid_b.angular_velocity, delta_omega_b)
            
            # Friction impulse
            if self.friction > 0 and j_n > 0:
                # Recompute relative velocity after normal impulse
                vel_a = add(rigid_a.linear_velocity, cross(rigid_a.angular_velocity, r_a))
                vel_b = add(rigid_b.linear_velocity, cross(rigid_b.angular_velocity, r_b))
                v_rel = sub(vel_a, vel_b)
                
                # Tangent velocity
                v_t_vec = sub(v_rel, mul(contact.normal, dot(v_rel, contact.normal)))
                v_t_mag = length(v_t_vec)
                
                if v_t_mag > 1e-6:
                    tangent = mul(v_t_vec, 1.0 / v_t_mag)
                    
                    # Friction impulse magnitude (clamped by Coulomb friction)
                    j_t = min(self.friction * j_n, v_t_mag * (inv_mass_a + inv_mass_b))
                    
                    impulse_t = mul(tangent, -j_t)
                    
                    if inv_mass_a > 0:
                        rigid_a.linear_velocity = add(rigid_a.linear_velocity, mul(impulse_t, inv_mass_a))
                        r_a_cross_t = cross(r_a, tangent)
                        delta_omega_a = (
                            -r_a_cross_t[0] * inv_inertia_a[0] * j_t,
                            -r_a_cross_t[1] * inv_inertia_a[1] * j_t,
                            -r_a_cross_t[2] * inv_inertia_a[2] * j_t,
                        )
                        rigid_a.angular_velocity = add(rigid_a.angular_velocity, delta_omega_a)
                    
                    if inv_mass_b > 0:
                        rigid_b.linear_velocity = sub(rigid_b.linear_velocity, mul(impulse_t, inv_mass_b))
                        r_b_cross_t = cross(r_b, tangent)
                        delta_omega_b = (
                            r_b_cross_t[0] * inv_inertia_b[0] * j_t,
                            r_b_cross_t[1] * inv_inertia_b[1] * j_t,
                            r_b_cross_t[2] * inv_inertia_b[2] * j_t,
                        )
                        rigid_b.angular_velocity = add(rigid_b.angular_velocity, delta_omega_b)
        
        # Position correction (direct displacement to resolve penetration)
        max_penetration = max(c.penetration for c in contacts)
        if max_penetration > self.penetration_tolerance:
            deepest = max(contacts, key=lambda c: c.penetration)
            correction = max(0, deepest.penetration - self.penetration_tolerance) * 0.5
            
            total_inv_mass = inv_mass_a + inv_mass_b
            if total_inv_mass > 0:
                correction_a = correction * (inv_mass_a / total_inv_mass)
                correction_b = correction * (inv_mass_b / total_inv_mass)
                
                if inv_mass_a > 0:
                    rigid_a.position = add(rigid_a.position, mul(deepest.normal, correction_a))
                if inv_mass_b > 0:
                    rigid_b.position = sub(rigid_b.position, mul(deepest.normal, correction_b))
    
    def shutdown(self) -> None:
        """Clean up resources."""
        if self._detector:
            self._detector.shutdown()
            self._detector = None
        self._initialized = False
