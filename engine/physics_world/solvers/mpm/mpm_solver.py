"""
MPM solver - main Material Point Method simulation engine.
"""
import numpy as np
import taichi as ti
from typing import Optional
export_file=""
ti.init(arch=ti.gpu)

# from .mpm_state import MPMState
# from .mpm_grid import MPMGrid
# from .mpm_materials import compute_stress, MATERIAL_WATER
# from .mpm_kernels import compute_grid_influence
# from .mpm_boundary import MPMBoundary

################# NOW JUST DEBUG MODELS #################
# enumerations for materials
WATER = 0
JELLY = 1
SNOW = 2

dim = 3
n_grid = 128
dx = 1.0 / n_grid #!!!!!!!!!!!!!!

inv_dx = float(n_grid)
p_vol, p_rho = (dx * 0.5)**3, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters
neighbour = (3, ) * dim

# boundary condition modes
STICKY = 0      # Current default: velocity clamped to zero
ELASTIC = 1     # Bouncy: velocity reflected with restitution
FRICTION = 2    # Frictional: tangential velocity reduced

@ti.data_oriented
class MPMSolver:
    """Main MPM solver using Taichi for GPU acceleration."""
    
    def __init__(self, 
                 max_particles: int = 500000,
                 grid_resolution: int = 128,
                 domain_min: float = -1.0,
                 domain_max: float = 1.0,
                 dt: float = 2e-4,
                 gravity: tuple = (0.0, 0.0, -9.81),
                 bulk_modulus: float = 500.0,
                 youngs_modulus: float = 0.0,
                 poisson_ratio: float = 0.2,
                 materials: str = 'water',
                 boundary_mode: str = 'bounce'):
        """
        Initialize MPM solver.
        
        Args:
            max_particles: Maximum number of particles
            grid_resolution: Grid resolution (e.g., 64 for 64^3 grid)
            domain_min: Minimum coordinate of simulation domain
            domain_max: Maximum coordinate of simulation domain
            dt: Time step size
            gravity: Gravity vector (m/s^2)
            bulk_modulus: Bulk modulus for water (similar to kappa in SPH)
            youngs_modulus: Young's modulus (0 for ideal fluid, >0 for elasticity)
            poisson_ratio: Poisson's ratio (0.0-0.5)
            material_type: Material type ('water', 'jelly', 'snow')
            boundary_mode: Boundary condition mode ('sticky', 'slip', 'separate', 'bounce')
        """
        
        print(f"[MPMSolver] Initializing with max_particles={max_particles}, grid_resolution={grid_resolution}³")
        print(f"[MPMSolver] Domain: [{domain_min}, {domain_max}], dt={dt}s, gravity={gravity}")
        print(f"[MPMSolver] Material params: E={youngs_modulus}, nu={poisson_ratio}, bulk={bulk_modulus}")
        print(f"[MPMSolver] Boundary mode: {boundary_mode}")
        
        # Simulation parameters
        self.dt = dt
        self.dx = dx
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.inv_dx = inv_dx
        self.gravity = -9.81
        self.bulk_modulus = bulk_modulus
        
        # Material parameters - use defaults for debugging
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0

        # for simulation
        self.x= ti.Vector.field(3, dtype=ti.f32, shape=max_particles)     
        self.v= ti.Vector.field(3, dtype=ti.f32, shape=max_particles)      #
        self.C= ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)   
        self.F= ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)  
        self.J= ti.field(dtype=ti.f32, shape=max_particles)
        self.materials= ti.field(dtype=ti.i32, shape=max_particles)
        self.grid_v= ti.Vector.field(3, dtype=ti.f32, shape=(grid_resolution, grid_resolution, grid_resolution))
        self.grid_m= ti.field(dtype=ti.f32, shape=(grid_resolution, grid_resolution, grid_resolution))
        # Active particle count
        self.is_used= ti.field(dtype=ti.i32, shape=max_particles)  # should be a boolean field
        self.n_particles = ti.field(dtype=ti.i32, shape=())

        self.boundary_mode = STICKY        
        
        # Debug tracking
        self.step_count = 0
        self.debug_interval = 200  # Output debug info every 50 steps
        self.leaked_count = ti.field(dtype=ti.i32, shape=())  # Count particles below z=-1        
        # Debug: Track average J for elastic materials
        self.avg_J = ti.field(dtype=ti.f32, shape=())
        self.max_J = ti.field(dtype=ti.f32, shape=())
        self.min_J = ti.field(dtype=ti.f32, shape=())
        # Debug: Track velocity statistics
        self.avg_v = ti.field(dtype=ti.f32, shape=())
        self.max_v = ti.field(dtype=ti.f32, shape=())
        self.min_v = ti.field(dtype=ti.f32, shape=())    
    
    @ti.kernel
    def simple_init(self):
        for i in range(6000):
            self.x[i] = ti.Vector([ti.random() for i in range(dim)]) * 0.4 + 0.15
            self.J[i] = 1
        self.n_particles[None] = 6000

    def initialize_box(self, box_min: np.ndarray, box_max: np.ndarray, 
                       density: float, 
                       initial_velocity: Optional[np.ndarray] = None):
        """
        保持原签名一致，通过规则采样初始化球体，确保物理合理性。
        """
        if initial_velocity is None:
            initial_velocity = np.array([0.0, 0.0, 0.0])
        
        # 1. 几何参数计算
        box_center = (box_min + box_max) / 2.0
        # 以最大边长的一半作为球体半径
        sphere_radius = 0.5 * np.max(box_max - box_min)
        
        # 2. 核心物理关联：根据网格分辨率 dx 定义粒子间距
        # MPM 黄金准则：每个网格轴向放 2 个粒子（每单元 8 粒子）
        dx = self.grid.grid_params[2] # 获取 dx
        ppc = 2.0 
        particle_spacing = dx / ppc 
        
        # 计算单个粒子的物理体积和质量
        particle_volume = particle_spacing ** 3
        particle_mass = density * particle_volume
        
        # 3. 规则采样（替代随机采样，防止初始重叠导致的爆炸）
        # 创建一个覆盖球体的立方体网格点阵
        r_range = np.arange(-sphere_radius, sphere_radius, particle_spacing)
        x, y, z = np.meshgrid(r_range, r_range, r_range)
        candidate_pos = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1) + box_center
        
        # 4. 球体过滤
        dists_sq = np.sum((candidate_pos - box_center)**2, axis=1)
        positions = candidate_pos[dists_sq <= (sphere_radius ** 2)].astype(np.float32)
        n_particles = len(positions)
        
        # 5. 状态赋值
        self.x.from_numpy(positions)
        
        # 设置初始速度
        velocities = np.tile(initial_velocity, (n_particles, 1)).astype(np.float32)
        self.v.from_numpy(velocities)
        
        # 初始化粒子属性 use default water
        self.materials.from_numpy(
            np.full(n_particles, WATER, dtype=np.int32)
        )
        
        print(f"[MPMSolver] Initialized SPHERE: r={sphere_radius:.3f}m, particles={n_particles}")
        print(f"[MPMSolver] Material type: {self.material_type_int} (0=water, 1=jelly, 2=snow)")
        print(f"[MPMSolver] PPC={ppc}^3, Particle Mass={particle_mass:.6e}kg")

    @ti.kernel
    def clear_grid(self):
        """Clear grid momentum and mass."""
        for I in ti.grouped(self.grid_v):
            self.grid_v[I] = ti.Vector.zero(ti.f32, 3)
            self.grid_m[I] = 0.0

    @ti.kernel
    def particle_to_grid(self):
        """P2G: Transfer particle data to grid (momentum and mass) and apply stress."""
        for p in self.x:
            if self.is_used[p]:
                base = (self.x[p] * inv_dx - 0.5).cast(int)
                fx = self.x[p] * inv_dx - base.cast(float)
                w = [0.5 * (1.5 - fx)**2,
                 0.75 - (fx - 1.0)**2,
                 0.5 * (fx - 0.5)**2]
                # Update deformation gradient
                self.F[p] = (ti.Matrix.identity(float, 3) + self.dt * self.C[p]) @ self.F[p]

                # Material-specific hardening and parameters
                mat = self.materials[p]
                h = ti.exp(10 * (1.0 - self.J[p]))
                if mat == JELLY:
                    h = 0.3
                mu, la = self.mu_0 * h, self.lambda_0 * h
                if mat == WATER:
                    mu = 0.0
                U, sig, V = ti.svd(self.F[p])
                J = 1.0
                for d in ti.static(range(3)):
                    new_sig = sig[d, d]
                    if mat == SNOW:
                        new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
                    self.J[p] *= sig[d, d] / new_sig
                    sig[d, d] = new_sig
                    J *= new_sig
                # Material-specific F reconstruction
                if mat == WATER:
                    self.F[p] = ti.Matrix.identity(float, 3) * ti.pow(J, 1.0 / 3.0)
                elif mat == SNOW:
                    self.F[p] = U @ sig @ V.transpose()
                # Stress computation (ORIGINAL formula, unchanged)
                stress = (2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose()
                      + ti.Matrix.identity(float, 3) * la * J * (J - 1))
                stress = (-self.dt * p_vol * 4 * inv_dx * inv_dx) * stress
                affine = stress + p_mass * self.C[p]

                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (offset.cast(float) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    self.grid_v[base + offset] += weight * (p_mass * self.v[p] + affine @ dpos)
                    self.grid_m[base + offset] += weight * p_mass
    
    @ti.kernel
    def grid_operations(self):
        """Update grid velocities (convert momentum to velocity, add gravity)."""
        # Grid operations with enhanced boundary conditions
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] = (1.0 / self.grid_m[I]) * self.grid_v[I]
                self.grid_v[I][1] += self.dt * self.gravity

            # Enhanced boundary condition handling
            # Process each dimension
            bound = 3  # Boundary thickness
            restitution = 0.5  # For elastic collisions
            for d in ti.static(range(3)):
                # Check lower boundary
                if I[d] < bound:
                    if self.grid_v[I][d] < 0:
                        if self.boundary_mode == STICKY:
                            # Sticky: clamp to zero
                            self.grid_v[I][d] = 0.0
                        elif self.boundary_mode == ELASTIC:
                            # Elastic: reflect with restitution
                            self.grid_v[I][d] *= -restitution
                        elif self.boundary_mode == FRICTION:
                            # Friction: only normal component gets clamped
                            # Tangential components affected by friction
                            normal_comp = self.grid_v[I][d]
                            self.grid_v[I][d] = 0.0  # Kill normal component
                            # Reduce tangential motion (implicit via energy dissipation)
                # Check upper boundary
                if I[d] > n_grid - bound:
                    if self.grid_v[I][d] > 0:
                        if self.boundary_mode == STICKY:
                            self.grid_v[I][d] = 0.0
                        elif self.boundary_mode == ELASTIC:
                            self.grid_v[I][d] *= -restitution
                        elif self.boundary_mode == FRICTION:
                            self.grid_v[I][d] = 0.0

    @ti.kernel
    def grid_to_particle(self):
        """G2P: Transfer grid velocities back to particles and update positions."""
        for p in self.x:
          if self.is_used[p]:
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2,
                 0.75 - (fx - 1.0)**2,
                 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                dpos = (ti.Vector([i, j, k]).cast(float) - fx)
                g_v = self.grid_v[base + ti.Vector([i, j, k])]
                weight = w[i][0] * w[j][1] * w[k][2]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * self.v[p]
    
    
    def step(self):
        """Advance simulation by one time step."""
        self.clear_grid()
        self.particle_to_grid()
        self.grid_operations()  # Convert momentum->velocity, add gravity
        #self.apply_static_collisions()  # Apply static mesh collisions ONLY
        # DISABLED: Domain boundaries removed to allow free expansion
        # self.apply_domain_boundaries()
        self.grid_to_particle()
        
        # Debug output every N steps
        self.step_count += 1
        if self.step_count % self.debug_interval == 0:
            n_particles = self.n_particles[None]
            leaked = self.count_leaked_particles()
            leak_pct = (leaked / n_particles * 100.0) if n_particles > 0 else 0.0
            
            # Calculate particle bounds
            min_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
            max_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.get_particle_bounds(min_pos, max_pos)
            
            # Calculate J statistics for elastic materials
            self.compute_j_stats()
            
            # Calculate velocity statistics
            self.compute_velocity_stats()
            
            # Grid info
            dx = self.dx
            
            print(f"[MPM Step {self.step_count}] Particles: {n_particles}, Leaked: {leaked} ({leak_pct:.2f}%)")
            print(f"  Deformation: J_avg={self.avg_J[None]:.3f}, J_min={self.min_J[None]:.3f}, J_max={self.max_J[None]:.3f}")
            print(f"  Velocity: v_avg={self.avg_v[None]:.3f}m/s, v_min={self.min_v[None]:.3f}m/s, v_max={self.max_v[None]:.3f}m/s")
            print(f"  Grid: dx={dx:.4f}m ({n_grid}³ cells), Domain: [{self.domain_min:.1f}, {self.domain_max:.1f}]")
            if n_particles > 0:
                pmin = min_pos[None]
                pmax = max_pos[None]
                print(f"  Particles range: X[{pmin[0]:.3f}, {pmax[0]:.3f}], Y[{pmin[1]:.3f}, {pmax[1]:.3f}], Z[{pmin[2]:.3f}, {pmax[2]:.3f}]")
    


    @ti.kernel
    def compute_j_stats(self):
        """Compute statistics of J (volume ratio) for debugging."""
        n = self.n_particles[None]
        if n > 0:
            sum_j = 0.0
            min_j = 1e10
            max_j = -1e10
            
            for p in range(n):
                j = self.J[p]
                sum_j += j
                min_j = ti.min(min_j, j)
                max_j = ti.max(max_j, j)
            
            self.avg_J[None] = sum_j / n
            self.min_J[None] = min_j
            self.max_J[None] = max_j

    @ti.kernel
    def count_leaked_particles(self) -> ti.i32:
        """Count particles with z < -0 (leaked through floor)."""
        count = 0
        for p in range(self.n_particles[None]):
            if self.x[p].z < -0.0:
                count += 1
        return count
    
 
    @ti.kernel
    def compute_velocity_stats(self):
        """Compute statistics of velocity magnitudes for debugging."""
        n = self.n_particles[None]
        if n > 0:
            sum_v = 0.0
            min_v = 1e10
            max_v = -1e10
            
            for p in range(n):
                v_mag = self.v[p].norm()
                sum_v += v_mag
                min_v = ti.min(min_v, v_mag)
                max_v = ti.max(max_v, v_mag)
            
            self.avg_v[None] = sum_v / n
            self.min_v[None] = min_v
            self.max_v[None] = max_v
    
    @ti.kernel
    def get_particle_bounds(self, min_pos: ti.template(), max_pos: ti.template()):
        """Calculate bounding box of all particles."""
        n = self.n_particles[None]
        if n > 0:
            # Initialize with first particle
            min_pos[None] = self.x[0]
            max_pos[None] = self.x[0]
            
            # Find min/max
            for p in range(1, n):
                for d in ti.static(range(3)):
                    ti.atomic_min(min_pos[None][d], self.x[p][d])
                    ti.atomic_max(max_pos[None][d], self.x[p][d])

    
    def load_static_bodies(self, static_bodies: list):
        """
        Load static mesh obstacles for collision detection.
        
        Args:
            static_bodies: List of StaticBodyState objects
        """
        if not static_bodies:
            print("[MPMSolver] No static bodies to load")
            return
        
        print(f"[MPMSolver] Loading {len(static_bodies)} static bodies...")
        
        for body in static_bodies:
            print(f"[MPMSolver] Loading static body: {body.name}")
            print(f"  Vertices: {len(body.vertices)}, Faces: {len(body.faces)}")
            
            # Convert to numpy arrays
            vertices = np.array(body.vertices, dtype=np.float32)
            triangles = np.array(body.faces, dtype=np.int32)
            
            # Load into boundary handler (voxelize onto grid)
            self.boundary.load_static_mesh(
                vertices=vertices,
                triangles=triangles,
                domain_min=self.grid.domain_min,
                domain_max=self.grid.domain_max
            )
        
        print(f"[MPMSolver] Static bodies loaded successfully")

def T(a):
    if dim == 2:
        return a

    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return np.array([u, v]).swapaxes(0, 1) + 0.5

def main():
    MPM = MPMSolver()
    MPM.simple_init()
    gui = ti.GUI("MPM3D", background_color=0x112F41)
    while gui.running and not gui.get_event(gui.ESCAPE):
        for s in range(6000):
            MPM.step()
        pos = MPM.x.to_numpy()
        if export_file:
            writer = ti.tools.PLYWriter(num_vertices=6000)
            writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
            writer.export_frame(gui.frame, export_file)
        gui.circles(T(pos), radius=1.5, color=0x66CCFF)
        gui.show()


if __name__ == "__main__":
    main()
