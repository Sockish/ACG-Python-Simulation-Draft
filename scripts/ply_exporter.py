# Save this as: scripts/ply_exporter.py

import os
import numpy as np
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple


class PLYExporter:
    """Export MPM particles to PLY format with FPS control and surface reconstruction.
    Supports multiple materials with separate particle files and surface meshes.
    """
    
    def __init__(self, output_dir: str, target_fps: int = 60, 
                 splashsurf_cmd: str = "pysplashsurf",
                 particle_radius: float = 0.025,
                 smoothing_length: float = 2.0,
                 cube_size: float = 0.5,
                 surface_threshold: float = 0.6,
                 rest_density: float = 1000.0,
                 mesh_smoothing_iters: int = 10,
                 normals: bool = False):
        """
        Args:
            output_dir: Base output directory (e.g., config/outputs_simulate_mpm)
            target_fps: Target FPS for export (default 60)
            splashsurf_cmd: Command to run splashsurf (e.g., "pysplashsurf" or "splashsurf")
            particle_radius: Particle radius in meters
            smoothing_length: Smoothing length multiplier
            cube_size: Cube size multiplier for marching cubes
            surface_threshold: Surface threshold for reconstruction
            rest_density: Rest density of the fluid
            mesh_smoothing_iters: Number of mesh smoothing iterations
            normals: Whether to generate normals
        """
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.frame_counter = 0  # Total simulation frames
        self.export_counter = 0  # Exported file counter (continuous)
        self.substep_counter = 0
        
        # Material tracking: {material_name: [(particle_file, export_index), ...]}
        self.material_exports: Dict[str, List[Tuple[Path, int]]] = {}
        
        # Splashsurf parameters
        self.splashsurf_cmd = splashsurf_cmd
        self.particle_radius = particle_radius
        self.smoothing_length = smoothing_length
        self.cube_size = cube_size
        self.surface_threshold = surface_threshold
        self.rest_density = rest_density
        self.mesh_smoothing_iters = mesh_smoothing_iters
        self.normals = normals
        
        # Create base output directories
        self.show_frames_dir = self.output_dir / "show_frames"
        self.show_frames_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[PLYExporter] Initialized with output_dir={output_dir}, target_fps={target_fps}")
        print(f"[PLYExporter] Using command: {splashsurf_cmd}")
    
    def should_export(self, dt: float, substeps_per_frame: int) -> bool:
        """
        Determine if current frame should be exported based on target FPS.
        
        Args:
            dt: Simulation timestep
            substeps_per_frame: Number of substeps per render frame
            
        Returns:
            True if should export this frame
        """
        self.substep_counter += 1
        
        # Calculate simulation time per render frame
        sim_time_per_frame = dt * substeps_per_frame
        
        # Calculate target time interval for export
        target_interval = 1.0 / self.target_fps
        
        # Calculate how many render frames to skip
        frames_to_skip = max(1, int(target_interval / sim_time_per_frame))
        
        if self.frame_counter % frames_to_skip == 0:
            return True
        return False
    
    def export_particles_by_material(self, positions: np.ndarray, materials: np.ndarray, 
                                    is_used: np.ndarray, domain_min: float, domain_length: float,
                                    material_configs: List[Dict]):
        """
        Export particles for multiple materials.
        
        Args:
            positions: Particle positions in [0,1]³ normalized space
            materials: Material type array
            is_used: Whether particle is active
            domain_min: Minimum coordinate of simulation domain
            domain_length: Length of simulation domain
            material_configs: List of material configurations, each containing:
                - 'name': Material name (e.g., 'fluid', 'jelly1')
                - 'particle_range': Tuple (start_index, end_index) for particle indices
                - 'material_type': Material type ID (0=WATER, 1=JELLY, etc.)
        
        Example:
            material_configs = [
                {'name': 'fluid', 'particle_range': (0, 40000), 'material_type': 0},
                {'name': 'jelly1', 'particle_range': (40000, 70000), 'material_type': 1},
            ]
        """
        for config in material_configs:
            material_name = config['name']
            particle_range = config.get('particle_range', None)
            material_type = config.get('material_type', None)
            
            # Filter particles
            if particle_range is not None:
                # Filter by particle index range
                start_idx, end_idx = particle_range
                mask = (is_used == 1)
                indices = np.arange(len(positions))
                mask = mask & (indices >= start_idx) & (indices < end_idx)
            elif material_type is not None:
                # Filter by material type
                mask = (is_used == 1) & (materials == material_type)
            else:
                print(f"[PLYExporter] Warning: No filter specified for material '{material_name}'")
                continue
            
            filtered_positions = positions[mask]
            
            if len(filtered_positions) == 0:
                print(f"[PLYExporter] Warning: No particles for material '{material_name}' at export {self.export_counter}")
                continue
            
            # Convert from [0,1]³ space back to world space
            world_positions = filtered_positions * domain_length + domain_min
            
            # Create material directory
            material_dir = self.output_dir / material_name
            material_dir.mkdir(parents=True, exist_ok=True)
            
            # Export to PLY with continuous numbering
            output_file = material_dir / f"{material_name}_{self.export_counter:05d}.ply"
            self._write_ply(output_file, world_positions)
            
            # Track this file for later reconstruction
            if material_name not in self.material_exports:
                self.material_exports[material_name] = []
            self.material_exports[material_name].append((output_file, self.export_counter))
            
            print(f"[PLYExporter] Export {self.export_counter:05d} (Frame {self.frame_counter}): "
                  f"Material '{material_name}' exported {len(world_positions)} particles to {output_file.name}")
        
        # Increment export counter after all materials are exported
        self.export_counter += 1
    
    def export_particles(self, positions: np.ndarray, materials: np.ndarray, 
                        is_used: np.ndarray, domain_min: float, domain_length: float,
                        material_filter: int = 0, material_name: str = 'fluid'):
        """
        Export particles for a single material (backward compatibility).
        
        Args:
            positions: Particle positions in [0,1]³ normalized space
            materials: Material type array
            is_used: Whether particle is active
            domain_min: Minimum coordinate of simulation domain
            domain_length: Length of simulation domain
            material_filter: Material type to export (default 0 for WATER)
            material_name: Name for the material (default 'fluid')
        """
        material_configs = [
            {'name': material_name, 'material_type': material_filter}
        ]
        self.export_particles_by_material(positions, materials, is_used, domain_min, domain_length, material_configs)
    
    def _write_ply(self, filepath: Path, positions: np.ndarray):
        """Write positions to PLY file in binary format."""
        num_points = len(positions)
        
        with open(filepath, 'wb') as f:
            # Write header
            header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
end_header
"""
            f.write(header.encode('ascii'))
            
            # Write binary vertex data
            positions_float32 = positions.astype(np.float32)
            f.write(positions_float32.tobytes())
    
    def reconstruct_all_surfaces(self):
        """
        Perform surface reconstruction for all exported materials.
        Creates one frame folder per export index, containing all material surfaces.
        """
        if not self.material_exports:
            print("[PLYExporter] No frames to reconstruct.")
            return
        
        # Get all unique export indices
        all_export_indices = set()
        for material_name, exports in self.material_exports.items():
            for _, export_idx in exports:
                all_export_indices.add(export_idx)
        
        all_export_indices = sorted(all_export_indices)
        
        print(f"\n[PLYExporter] Starting surface reconstruction...")
        print(f"[PLYExporter] Materials: {list(self.material_exports.keys())}")
        print(f"[PLYExporter] Frames: {len(all_export_indices)}")
        print(f"[PLYExporter] Output directory: {self.show_frames_dir}")
        
        success_count = 0
        fail_count = 0
        
        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("[PLYExporter] Install tqdm for progress bar: pip install tqdm")
        
        # Iterate through all export indices
        if use_tqdm:
            export_iterator = tqdm(all_export_indices, desc="Reconstructing frames")
        else:
            export_iterator = all_export_indices
        
        for export_idx in export_iterator:
            # Create frame directory
            frame_dir = self.show_frames_dir / f"{export_idx:05d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            
            # Reconstruct all materials for this frame
            for material_name, exports in self.material_exports.items():
                # Find the export for this index
                matching_exports = [pf for pf, ei in exports if ei == export_idx]
                
                if not matching_exports:
                    continue
                
                particle_file = matching_exports[0]
                
                if not particle_file.exists():
                    print(f"[PLYExporter] Warning: Particle file not found: {particle_file}")
                    fail_count += 1
                    continue
                
                # Output surface mesh file
                surface_file = frame_dir / f"{material_name}_surface.obj"
                
                success = self._reconstruct_surface(particle_file, surface_file)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
        
        print(f"\n[PLYExporter] Surface reconstruction complete!")
        print(f"[PLYExporter] Success: {success_count}, Failed: {fail_count}")
        print(f"[PLYExporter] Meshes saved to: {self.show_frames_dir}")
    
    def _reconstruct_surface(self, particle_file: Path, surface_file: Path) -> bool:
        """
        Perform surface reconstruction using splashsurf for a single material.
        
        Args:
            particle_file: Path to the PLY particle file
            surface_file: Path to the output surface mesh file
            
        Returns:
            True if successful, False otherwise
        """
        # Build splashsurf command (same format as reconstructor.py)
        command = [self.splashsurf_cmd, "reconstruct", str(particle_file)]
        command += ["--output-file", str(surface_file)]
        command += ["--particle-radius", f"{self.particle_radius:.6f}"]
        command += ["--smoothing-length", f"{self.smoothing_length:.6f}"]
        command += ["--cube-size", f"{self.cube_size:.6f}"]
        command += ["--surface-threshold", f"{self.surface_threshold:.6f}"]
        command += ["--rest-density", f"{self.rest_density:.6f}"]
        
        if self.mesh_smoothing_iters > 0:
            command += ["--mesh-smoothing-iters", str(self.mesh_smoothing_iters)]
            command += ["--mesh-smoothing-weights=on"]
        else:
            command += ["--mesh-smoothing-iters", "0"]
        
        command += [f"--normals={'on' if self.normals else 'off'}"]
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n[PLYExporter] Warning: Surface reconstruction failed for {surface_file.name}")
            if e.stderr:
                print(f"  Error: {e.stderr[:200]}")
            return False
        except FileNotFoundError:
            print(f"\n[PLYExporter] Error: {self.splashsurf_cmd} not found.")
            print(f"  Make sure pysplashsurf is installed and in your PATH")
            return False
    
    def increment_frame(self):
        """Increment frame counter."""
        self.frame_counter += 1
    
    def get_frame_count(self):
        """Get current frame count."""
        return self.frame_counter
    
    def get_exported_frame_count(self):
        """Get number of exported frames."""
        return self.export_counter
    
    def reset(self):
        """Reset frame counters."""
        self.frame_counter = 0
        self.export_counter = 0
        self.substep_counter = 0
        self.material_exports = {}