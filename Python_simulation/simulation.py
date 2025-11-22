"""High-level façade tying together SPH simulation, exports, and rendering."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import SimulationConfig
from io_utils.exporter import write_manifest, write_particles_ply, write_rigid_transforms
from pipeline.blender_bridge import launch_blender
from surface_reconstruction import reconstruct_sequence
from hybrid.simulation import HybridSimulator


@dataclass
class RunOptions:
    scene: str = "dam_break"
    export_particles: bool = True
    reconstruct_surface: bool = False
    render_with_blender: bool = False
    render_surface: bool = False


class SimulationRunner:
    def __init__(self, config: SimulationConfig, options: RunOptions) -> None:
        self.config = config
        self.options = options
        self.solver = HybridSimulator(config, options.scene)
        self.exported_frames = 0

    def _export_frame(self, step_idx: int) -> None:
        if not self.options.export_particles:
            return
        frame_idx = self.exported_frames
        write_particles_ply(frame_idx, self.solver.fluid, self.config.ply_dir)
        write_rigid_transforms(frame_idx, self.solver.rigid_world, self.config.rigid_dir)
        self.exported_frames += 1

    def run(self) -> None:
        def callback(step_idx, _):
            self._export_frame(step_idx)

        self.solver.run(callback)
        if self.options.reconstruct_surface:
            reconstruct_sequence(
                self.config.ply_dir,
                self.config.mesh_dir,
                self.config,
                self.options.scene,
                self.config.rigid_dir,
            )
        if self.options.render_with_blender:
            self.render_with_blender()

    def render_with_blender(self) -> None:
        if self.exported_frames == 0:
            raise RuntimeError("No frames were exported; enable particle export before rendering")
        manifest = write_manifest(
            total_frames=self.exported_frames,
            ply_dir=self.config.ply_dir,
            mesh_dir=self.config.mesh_dir if self.options.render_surface else None,
            rigid_dir=self.config.rigid_dir,
            render_dir=self.config.render_dir,
        )
        script = (
            self.config.blender_script_surface if self.options.render_surface else self.config.blender_script_particles
        )
        launch_blender(
            blender_executable=self.config.blender_executable,
            blender_script=script,
            manifest_path=manifest,
        )

