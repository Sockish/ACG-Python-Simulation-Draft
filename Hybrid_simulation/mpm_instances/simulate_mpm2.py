export_file = ""  # use '/tmp/mpm3d.ply' for exporting result to disk
import numpy as np
import taichi as ti
import trimesh
import os
import sys

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将 scripts 目录添加到 sys.path
scripts_dir = os.path.join(project_root, "scripts")
sys.path.append(scripts_dir)

from ply_exporter import PLYExporter
from mpm_simple import WATER, MPMSolver
ti.init(arch=ti.gpu)

WATER = 0
JELLY = 1
SNOW = 2
STATIC = 3

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
	
def main_with_export():
    """Main function with PLY export and surface reconstruction functionality."""
    import os
    
    # Setup
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_base = os.path.join(workspace_dir, "config", "outputs_simulate_mpm2")
    
    # Initialize solver
    MPM = MPMSolver(max_particles=150000, domain_max=0.95, domain_min=-0.95, grid_resolution=64)
    
    # Initialize particles
    MPM.init_rectangles(
        -0.7, -0.6, -0.7, 0.7, -0.0, 0.7,  # Box 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       # Box 2
        add_particles=60000,
        material_type=WATER
    )
    MPM.load_obj_and_init_particles(
        os.path.join(workspace_dir, "config", "assets", "static", "box_20cm_thick.obj"),
        material_type=JELLY,
        particle_density=0.038,
        translation=(0.0, 0.0, 0.0),
        scale=0.92
    )

    # Define material export configurations
    # Each material has a name and particle range or material type
    material_configs = [
        {
            'name': 'fluid',           # Material name (used for folder and file naming)
            'particle_range': (0, 60000),  # Particle index range
            'material_type': WATER     # Material type ID
        },
        {
            'name': 'jelly1',
            'particle_range': (60000, 100000),
            'material_type': JELLY
        }
    ]
    
    # Initialize exporter
    exporter = PLYExporter(
        output_base, 
        target_fps=60,
        splashsurf_cmd="pysplashsurf",
        particle_radius=0.02,
        smoothing_length=2.0,
        cube_size=0.5,
        surface_threshold=0.6,
        rest_density=1000.0,
        mesh_smoothing_iters=10,
        normals=False
    )
    
    # Simulation parameters
    substeps_per_frame = 20
    max_frames = 400
    
    gui = ti.GUI("MPM3D", background_color=0x112F41)
    
    print("\n[Simulation] Starting simulation...")
    print(f"[Simulation] Max frames: {max_frames}, Substeps per frame: {substeps_per_frame}")
    print(f"[Simulation] Materials: {[config['name'] for config in material_configs]}")
    
    frame = 0
    while gui.running and not gui.get_event(gui.ESCAPE) and frame < max_frames:
        # Simulate
        for s in range(substeps_per_frame):
            MPM.substep()
        
        # Export particles for all materials
        if exporter.should_export(MPM.dt, substeps_per_frame):
            positions = MPM.x.to_numpy()
            materials = MPM.materials.to_numpy()
            is_used = MPM.is_used.to_numpy()
            
            exporter.export_particles_by_material(
                positions, materials, is_used,
                MPM.domain_min, MPM.domain_length,
                material_configs
            )
        
        exporter.increment_frame()
        
        # Render
        pos = MPM.x.to_numpy()
        materials_np = MPM.materials.to_numpy()
        
        water_particles = pos[materials_np == WATER]
        jelly_particles = pos[materials_np == JELLY]
        snow_particles = pos[materials_np == SNOW]
        static_particles = pos[materials_np == STATIC]
        
        if len(water_particles) > 0:
            gui.circles(T(water_particles), radius=1.5, color=0x66CCFF)
        if len(jelly_particles) > 0:
            gui.circles(T(jelly_particles), radius=1.5, color=0xFF6666)
        if len(snow_particles) > 0:
            gui.circles(T(snow_particles), radius=1.5, color=0xFFFFFF)
        if len(static_particles) > 0:
            gui.circles(T(static_particles), radius=1.5, color=0x888888)
        
        gui.show()
        frame += 1
    
    # Simulation complete, now perform surface reconstruction
    print(f"\n[Simulation] Simulation completed!")
    print(f"[Simulation] Total frames: {exporter.get_frame_count()}")
    print(f"[Simulation] Exported frames: {exporter.get_exported_frame_count()}")
    print(f"[Simulation] Output directory: {output_base}")
    
    # Perform surface reconstruction for all exported frames
    exporter.reconstruct_all_surfaces()
    
    print(f"\n[Simulation] All done! Check results at:")
    print(f"  Particles: {output_base}\\<material_name>\\")
    print(f"  Surfaces: {output_base}\\show_frames\\")


if __name__ == "__main__":
    # Use main_with_export() for export functionality
    # Use main() for just visualization
    main_with_export()
