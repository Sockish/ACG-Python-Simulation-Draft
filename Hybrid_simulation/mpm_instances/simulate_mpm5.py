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
    output_base = os.path.join(workspace_dir, "config", "outputs_simulate_mpm5")
    
    # Initialize solver
    MPM = MPMSolver(max_particles=150000, domain_max=5, domain_min=-5.0, grid_resolution=64, gravity=(0, 0, -2.0))
    
    # Step 1: Pre-load OBJ particles (only do this once, can be reused)
    landscape_template = MPM.load_obj_particles(
        os.path.join(workspace_dir, "config", "assets", "static", "thick_landscape.obj"),
        particle_density=0.15
    )
    nailong_template = MPM.load_obj_particles(
        os.path.join(workspace_dir, "config", "assets", "nailong", "1.obj"),
        particle_density=0.1
    )
    
    # Step 2: Initialize static particles from template
    num_static = MPM.init_particles_from_template(
        landscape_template,
        material_type=STATIC,
        translation=(0.0, 0.0, 0.0),
        scale=1.0
    )

    # Track particle ranges for each nailong instance
    nailong_instances = []  # List of (start_idx, end_idx, name)
    current_particle_idx = num_static
    
    # Define material export configurations (will be updated dynamically)
    material_configs = []
    
    # Initialize exporter
    exporter = PLYExporter(
        output_base, 
        target_fps=60,
        splashsurf_cmd="pysplashsurf",
        particle_radius=0.1,
        smoothing_length=2.0,
        cube_size=0.5,
        surface_threshold=0.6,
        rest_density=1000.0,
        mesh_smoothing_iters=10,
        normals=False
    )
    
    # Simulation parameters
    substeps_per_frame = 20
    max_frames = 400  # Increased to see more nailongs
    nailong_spawn_interval = 100  # Spawn every 100 frames
    nailong_spawn_position = (0.0, 4.0, 0.0)  # Spawn position
    nailong_scale = 1.25
    
    gui = ti.GUI("MPM3D", background_color=0x112F41)
    
    print("\n[Simulation] Starting simulation...")
    print(f"[Simulation] Max frames: {max_frames}, Substeps per frame: {substeps_per_frame}")
    print(f"[Simulation] Nailong spawn interval: {nailong_spawn_interval} frames")
    print(f"[Simulation] Nailong spawn position: {nailong_spawn_position}")
    
    frame = 0
    nailong_counter = 0
    material_counter = 0
    
    while gui.running and not gui.get_event(gui.ESCAPE) and frame < max_frames:
        # Check if it's time to spawn a new nailong
        if frame % nailong_spawn_interval == 0:
            print(f"\n[Simulation] Frame {frame}: Spawning nailong #{nailong_counter + 1}")
            
            # Initialize new nailong
            if material_counter == 0:
                add_material = SNOW
            elif material_counter == 1:
                add_material = JELLY
            else:
                add_material = WATER
            num_particles = MPM.init_particles_from_template(
                nailong_template,
                material_type=add_material,
                translation=nailong_spawn_position,
                scale=nailong_scale
            )
            material_counter += 1
            
            if num_particles > 0:
                # Track this nailong instance
                start_idx = current_particle_idx
                end_idx = current_particle_idx + num_particles
                nailong_name = f"nailong{nailong_counter + 1}"
                
                nailong_instances.append({
                    'start': start_idx,
                    'end': end_idx,
                    'name': nailong_name,
                    'spawn_frame': frame
                })
                
                # Update material configs
                material_configs.append({
                    'name': nailong_name,
                    'particle_range': (start_idx, end_idx)
                })
                
                current_particle_idx = end_idx
                nailong_counter += 1
                
                print(f"[Simulation] Nailong '{nailong_name}' spawned: particles {start_idx}-{end_idx} ({num_particles} particles)")
                print(f"[Simulation] Total nailongs: {nailong_counter}")
                print(f"[Simulation] Total particles: {current_particle_idx}")
        
        # Simulate
        for s in range(substeps_per_frame):
            MPM.substep()
        
        # Export particles for all materials (if any exist)
        if len(material_configs) > 0 and exporter.should_export(MPM.dt, substeps_per_frame):
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
    print(f"[Simulation] Total nailongs spawned: {nailong_counter}")
    print(f"[Simulation] Output directory: {output_base}")
    
    # Print summary of all nailong instances
    print(f"\n[Simulation] Nailong instances summary:")
    for instance in nailong_instances:
        print(f"  - {instance['name']}: particles {instance['start']}-{instance['end']}, spawned at frame {instance['spawn_frame']}")
    
    # Perform surface reconstruction for all exported frames
    if len(material_configs) > 0:
        exporter.reconstruct_all_surfaces()
        
        print(f"\n[Simulation] All done! Check results at:")
        print(f"  Particles: {output_base}\\<material_name>\\")
        print(f"  Surfaces: {output_base}\\show_frames\\")
    else:
        print(f"\n[Simulation] No materials to export.")


if __name__ == "__main__":
    # Use main_with_export() for export functionality
    # Use main() for just visualization
    main_with_export()
