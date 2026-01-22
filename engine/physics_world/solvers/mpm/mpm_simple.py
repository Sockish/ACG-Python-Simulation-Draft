export_file = ""  # use '/tmp/mpm3d.ply' for exporting result to disk
import numpy as np
import taichi as ti
import trimesh
ti.init(arch=ti.gpu)
# dim, n_grid, steps, dt = 2, 128, 20, 2e-4
# dim, n_grid, steps, dt = 2, 256, 32, 1e-4
#dim, n_grid, steps, dt = 3, 32, 25, 4e-4
# dim, n_grid, steps, dt = 3, 64, 25, 2e-4
# dim, n_grid, steps, dt = 3, 128, 25, 8e-5
# dim = 3
#quality = 1  # Use a larger value for higher-res simulations
# n_particles, n_grid = 8192 * quality**3, 32 * quality
# #dt = 2e-4 / quality
# dx = 1.0 / n_grid
# inv_dx = float(n_grid)
# p_rho = 1
# p_vol = (dx * 0.5) ** 2
# p_mass = p_vol * p_rho
bound = 3
E = 1000
nu =0.2
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters
# enumerations for materials
WATER = 0
JELLY = 1
SNOW = 2
STATIC = 3
domain_min = 0
domain_max = 1.0

@ti.data_oriented
class MPMSolver:
    """Main MPM solver using Taichi for GPU acceleration."""
    
    def __init__(self, 
                 max_particles: int = 50000,
                 grid_resolution: int = 32,
                 domain_min: float = -4.0,
                 domain_max: float = 4.0,
                 dt: float = 2e-4,
                 gravity: tuple = (0.0, 0.0, -9.81),
                 bulk_modulus: float = 500.0,
                 youngs_modulus: float = 0.0,
                 poisson_ratio: float = 0.2,
                 materials: int = 0,
                 boundary_mode: str = 'bounce'):
        """
        Args:
            max_particles: Maximum number of particles
            grid_resolution: Grid resolution (e.g., 64 for 64^3 grid)
            domain_min: Minimum coordinate of simulation domain
            domain_max: Maximum coordinate of simulation domain
            material_type: Material type ('water', 'jelly', 'snow')
            boundary_mode: Boundary condition mode ('sticky', 'slip', 'separate', 'bounce')
        """
        
        print(f"[MPMSolver] Initializing with max_particles={max_particles}, grid_resolution={grid_resolution}³")
        print(f"[MPMSolver] Domain: [{domain_min}, {domain_max}], dt={dt}s, gravity={gravity}")
        print(f"[MPMSolver] Boundary mode: {boundary_mode}")
        self.dt = dt
        self.quality = dt / 2e-4
        self.n_grid = grid_resolution
        self.domain_min, self.domain_max = domain_min, domain_max
        self.domain_length = domain_max - domain_min
        # We do simulation in [0, 1]³ space and map to actual domain during rendering (world->[0,1]->world)
        self.dx = 1.0 / grid_resolution
        self.inv_dx = float(grid_resolution) / 1.0
        p_rho = 1
        self.p_vol = (self.dx * 0.5) ** 3
        self.p_mass = self.p_vol * p_rho
        self.dim = 3
        
        self.gravity = gravity[2]  # Use z-component for 3D gravity
        
        self.x = ti.Vector.field(self.dim, float, max_particles)
        self.v = ti.Vector.field(self.dim, float, max_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, float, max_particles)
        self.J = ti.field(float, max_particles)
        
        self.F = ti.Matrix.field(self.dim, self.dim, float, max_particles) # deformation gradient
        self.materials = ti.field(dtype=int, shape=max_particles)  # material id
        self.is_used = ti.field(dtype=ti.i32, shape=max_particles)  # whether particle is used
        
        self.grid_v = ti.Vector.field(self.dim, float, (grid_resolution,) * self.dim)
        self.grid_m = ti.field(float, (grid_resolution,) * self.dim)
        
        self.neighbour = (3,) * self.dim
        #Debug use
        self.num_used_particles = ti.field(dtype=ti.i32, shape=())
        
		
    def substep(self):
        self.particle_to_grid()
        self.grid_operation()
        self.grid_to_particle()
        
    @ti.kernel
    def particle_to_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.zero(self.grid_v[I])
            self.grid_m[I] = 0
        ti.loop_config(block_dim=self.n_grid)
        for p in self.x:
          if self.is_used[p]:
            mat = self.materials[p]
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2,
                 0.75 - (fx - 1.0)**2,
                 0.5 * (fx - 0.5)**2]
            
            # STATIC 材质：只传递极大质量，不传递动量
            if mat == STATIC:
                static_mass = self.p_mass * 1e6  # 极大质量，相当于无限重
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    weight = w[i][0] * w[j][1] * w[k][2]
                    self.grid_m[base + offset] += weight * static_mass
                    # 不传递动量（grid_v 保持为 0）
                continue
            
            # 其他材质的正常处理
            self.F[p] = (ti.Matrix.identity(float, 3) + self.dt * self.C[p]) @ self.F[p]
            h = ti.exp(10 * (1.0 - self.J[p]))
            # ORIGINAL logic for standard materials (WATER, JELLY, SNOW)
            if mat == JELLY: ########################！！！！！！！！！！！！！
                h = 0.3
            # if mat == STATIC: ########################！！！！！！！！！！！！！
            #     h = 100.0
            mu, la = mu_0 * h, lambda_0 * h
            if mat == WATER: ########################！！！！！！！！！！！！！
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
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            affine = stress + self.p_mass * self.C[p]
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass
        
    @ti.kernel
    def grid_operation(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] /= self.grid_m[I]
            self.grid_v[I][1] += self.dt * self.gravity # Apply gravity in z-direction
            # Boundary conditions (simple clamping to domain)
            cond = (I < bound) & (self.grid_v[I] < 0) | (I > self.n_grid - bound) & (self.grid_v[I] > 0)
            self.grid_v[I] = ti.select(cond, 0, self.grid_v[I])
        ti.loop_config(block_dim=self.n_grid)
        
    @ti.kernel
    def grid_to_particle(self):
        for p in self.x:
          if self.is_used[p]:
            if self.materials[p] == STATIC: ## skip static particles
                self.v[p] = ti.Vector.zero(float, 3)
                self.C[p] = ti.Matrix.zero(float, 3, 3)
                continue
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
    
		
    @ti.kernel
    def init_particles_from_mesh(self, num_particles: int, particles: ti.types.ndarray(), material_type: int):
        for i in range(num_particles):
            self.x[i] = ti.Vector([particles[i, 0], particles[i, 1], particles[i, 2]])
            self.J[i] = 1
            self.F[i] = ti.Matrix.identity(float, self.dim)  # Move identity initialization here
            self.v[i] = ti.Vector([0.0, 0.0, 0.0])
            self.materials[i] = material_type
            self.is_used[i] = 1
            
    @ti.kernel
    def init_particles_from_mesh_offset(self, offset: int, num_particles: int, particles: ti.types.ndarray(), material_type: int):
        """Initialize particles starting from a specific offset index."""
        for i in range(num_particles):
            idx = offset + i
            self.x[idx] = ti.Vector([particles[i, 0], particles[i, 1], particles[i, 2]])
            self.J[idx] = 1
            self.F[idx] = ti.Matrix.identity(float, self.dim)
            self.v[idx] = ti.Vector([0.0 for _ in range(self.dim)])
            self.materials[idx] = material_type
            self.is_used[idx] = 1
            
    def load_obj_and_init_particles(self, obj_path: str, material_type: int, particle_density: float = 0.1, translation: tuple = (0.0, 0.0, 0.0)):
        # Load the mesh using trimesh
        mesh = trimesh.load(obj_path)
        # if not mesh.is_watertight:
        #     raise ValueError("The mesh is not watertight. Ensure the OBJ file represents a closed surface.")
        mesh.apply_translation(translation)
        mesh.apply_transform([[1, 0, 0, 0],  # X remains X
                          [0, 0, 1, 0],  # Z becomes Y
                          [0, -1, 0, 0], # Y becomes -Z
                          [0, 0, 0, 1]]) # Homogeneous coordinate
        # Get the bounding box of the mesh
        bbox_min, bbox_max = mesh.bounds
        print(f"Mesh bounding box (after translation): {bbox_min} to {bbox_max}")
        # Generate particles within the bounding box
        x_range = np.arange(bbox_min[0], bbox_max[0], particle_density)
        y_range = np.arange(bbox_min[1], bbox_max[1], particle_density)
        z_range = np.arange(bbox_min[2], bbox_max[2], particle_density)
        grid = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)
        print(f"Generated {len(grid)} candidate particles in bounding box.")
        # Filter particles inside the mesh
        bbox = mesh.bounding_box_oriented
        rough_inside = bbox.contains(grid)
        grid = grid[rough_inside]
        inside = mesh.contains(grid)
        particles = grid[inside].astype(np.float32)
        print(f"Generated {len(particles)} candidate particles inside bounding box, {np.sum(inside)} inside the mesh.")
        # Filter particles within the domain range
        domain_min_vec = np.array([self.domain_min] * 3, dtype=np.float32)
        domain_max_vec = np.array([self.domain_max] * 3, dtype=np.float32)
        in_domain = np.all((particles >= domain_min_vec) & (particles <= domain_max_vec), axis=1)
        particles = particles[in_domain]
        print(f"{len(particles)} particles remain after domain filtering.")
        # Map particles to relative space [0, 1]³
        particles_relative = (particles - self.domain_min) / self.domain_length        
        # Get current number of particles to append new ones
        current_num = self.num_used_particles[None]
        num_new_particles = min(len(particles_relative), self.x.shape[0] - current_num)
        if num_new_particles <= 0:
            print(f"Warning: Cannot add more particles. Current: {current_num}, Max: {self.x.shape[0]}")
            return
        # Initialize particles in Taichi fields starting from current_num
        self.init_particles_from_mesh_offset(current_num, num_new_particles, particles_relative, material_type)
        self.num_used_particles[None] = current_num + num_new_particles
        print(f"Initialized {num_new_particles} particles from OBJ mesh. Total particles: {self.num_used_particles[None]}")

    @ti.kernel
    def init_rectangles(self, box1_min_x: float, box1_min_y: float, box1_min_z: float,
                    box1_max_x: float, box1_max_y: float, box1_max_z: float,
                    box2_min_x: float, box2_min_y: float, box2_min_z: float,
                    box2_max_x: float, box2_max_y: float, box2_max_z: float):
        box1_min = ti.Vector([box1_min_x, box1_min_y, box1_min_z])
        box1_max = ti.Vector([box1_max_x, box1_max_y, box1_max_z])
        box2_min = ti.Vector([box2_min_x, box2_min_y, box2_min_z])
        box2_max = ti.Vector([box2_max_x, box2_max_y, box2_max_z])
        for i in range(4000):  # Assume 4000 particles for the first box
            pos = ti.Vector([ti.random() for _ in range(self.dim)]) * (box1_max - box1_min) + box1_min
            self.x[i] = (pos - self.domain_min) / self.domain_length  # Map to relative space
            self.J[i] = 1
            self.F[i] = ti.Matrix.identity(float, self.dim)
            self.v[i] = ti.Vector([0.0 for _ in range(self.dim)])
            self.materials[i] = WATER
            self.is_used[i] = 1
        for i in range(4000, 8000):  # Assume 4000 particles for the second box
            pos = ti.Vector([ti.random() for _ in range(self.dim)]) * (box2_max - box2_min) + box2_min
            self.x[i] = (pos - self.domain_min) / self.domain_length  # Map to relative space
            self.J[i] = 1
            self.F[i] = ti.Matrix.identity(float, self.dim)
            self.v[i] = ti.Vector([0.0 for _ in range(self.dim)])
            self.materials[i] = JELLY
            self.is_used[i] = 1
        self.num_used_particles[None] = 8000  # Update particle count
    @ti.kernel
    def init_ball(self, center_x: float, center_y: float, center_z: float, radius: float, mat_type: int):
        center = ti.Vector([center_x, center_y, center_z])
        new_particles = 15000
        current_num = self.num_used_particles[None]
        if current_num + new_particles > self.x.shape[0]:
            new_particles = self.x.shape[0] - current_num
            print(f"Warning: Reducing new particles to {new_particles} due to max capacity.")
        self.num_used_particles[None] = current_num + new_particles
        print(f"Initializing ball with {new_particles} particles.")
        for i in range(current_num, self.num_used_particles[None]):
            pos = ti.Vector([ti.random() for _ in range(self.dim)])
            if (pos - center).norm() < radius:
                self.x[i] = pos
                self.J[i] = 1
                self.F[i] = ti.Matrix.identity(float, self.dim)
                self.v[i] = ti.Vector([0.0 for _ in range(self.dim)])
                self.materials[i] = mat_type
                self.is_used[i] = 1
            

def T(a):
    #if dim == 2:
    #    return a
    phi, theta = np.radians(45), np.radians(45)
    a = a - 0.5 
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return np.array([u, v]).swapaxes(0, 1) + 0.5

def main():
    MPM = MPMSolver()
    #MPM.init_ball(0.5, 0.6, 0.5, 0.3, JELLY)
    # MPM.init_rectangles(
    #     -0.5, -0.2, -0.5, 0.4, 0.6, 0.4,  # Box 1
    #     0.6, 0.6, 0.6, 0.8, 0.9, 0.9       # Box 2
    # )
    MPM.load_obj_and_init_particles("C:\\Users\\Furina\\Documents\\sector2-autumn\\Advanced Computer Graphics\\ACG-Python-Simulation-Draft\\Hybrid_simulation\\config\\assets\\static\\thick_landscape.obj", material_type=STATIC, particle_density=0.1, translation=(0,0, 0.0))
   # MPM.load_obj_and_init_particles("C:\\Users\\Furina\\Documents\\sector2-autumn\\Advanced Computer Graphics\\ACG-Python-Simulation-Draft\\Hybrid_simulation\\config\\assets\\nailong\\1.obj", material_type=WATER, particle_density=0.06, translation=(0.0,-1.0, 3.0))
    MPM.init_ball(0.6, 0.6, 0.7, 0.25, WATER)
    gui = ti.GUI("MPM3D", background_color=0x112F41)
    while gui.running and not gui.get_event(gui.ESCAPE):
        for s in range(15):
            MPM.substep()
        # 获取粒子位置和材料类型
        pos = MPM.x.to_numpy()
        materials = MPM.materials.to_numpy()
        # 根据材料类型分配颜色
        water_color = 0x66CCFF  # 蓝色
        jelly_color = 0xFF6666  # 红色
        static_color = 0x888888  # 灰色表示静态物体
        # 分别绘制不同材料的粒子
        water_particles = pos[materials == WATER]
        jelly_particles = pos[materials == JELLY]
        SNOW_particles = pos[materials == SNOW]
        static_particles = pos[materials == STATIC]
        if len(water_particles) > 0:
            gui.circles(T(water_particles), radius=1.5, color=water_color)
        if len(jelly_particles) > 0:
            gui.circles(T(jelly_particles), radius=1.5, color=jelly_color)
        if len(SNOW_particles) > 0:
            gui.circles(T(SNOW_particles), radius=1.5, color=0xFFFFFF)
        if len(static_particles) > 0:
            gui.circles(T(static_particles), radius=1.5, color=static_color)
        
        gui.show()


if __name__ == "__main__":
    main()
