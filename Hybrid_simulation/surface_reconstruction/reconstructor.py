"""Splashsurf-based surface reconstruction utilities."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.configuration import SceneConfig
from tqdm import tqdm

## example: .venv/Scripts/python.exe ./scripts/reconstruct.py --config config/scene_config.yaml --target-fps 60
## .venv/Scripts/python.exe ./scripts/reconstruct.py \
##   --config config/scene_config.yaml \
##   --splashsurf-cmd pysplashsurf \
##   --target-fps 60


@dataclass
class ReconstructionJob:
    """Represents a single particle->surface reconstruction call."""

    step_index: int
    particle_file: Path
    surface_file: Path


@dataclass
class SplashsurfReconstructor:
    config: SceneConfig
    splashsurf_cmd: str = "splashsurf"
    output_dir: Path | None = None
    output_format: str = "obj"
    smoothing_length_multiplier: float | None = None  # multiples of particle radius
    particle_radius: float | None = None  # meters
    cube_size_multiplier: float = 0.5
    surface_threshold: float = 0.6
    rest_density: float | None = None
    mesh_smoothing_iters: int = 10
    mesh_smoothing_weights: bool = True
    normals: bool = False
    normals_smoothing_iters: int = 10
    extra_args: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        export = self.config.export
        if export is None:
            output_root = Path("outputs")
            self.fluid_dir = output_root / "fluid"
        else:
            output_root = export.output_root
            self.fluid_dir = export.fluid_dir()
        self.surface_dir = (self.output_dir or (output_root / "surface")).resolve()
        self.surface_dir.mkdir(parents=True, exist_ok=True)
        spacing = float(self.config.liquid_box.particle_spacing)
        if self.particle_radius is None:
            self.particle_radius = spacing * 0.5
        if self.rest_density is None:
            self.rest_density = float(self.config.liquid_box.rest_density)
        if self.smoothing_length_multiplier is None:
            if self.particle_radius <= 0:
                self.smoothing_length_multiplier = 2.0
            else:
                smoothing_len = float(self.config.liquid_box.smoothing_length)
                self.smoothing_length_multiplier = max(smoothing_len / self.particle_radius, 0.5)
        self.cube_size_multiplier = max(self.cube_size_multiplier, 0.1)
        self.surface_threshold = max(self.surface_threshold, 1e-3)

    def available_steps(self) -> List[int]:
        if not self.fluid_dir.exists():
            return []
        steps: List[int] = []
        for file in sorted(self.fluid_dir.glob("fluid_*.ply")):
            try:
                step = int(file.stem.split("_")[-1])
            except ValueError:
                continue
            steps.append(step)
        return steps

    def reconstruct(
        self,
        steps: Sequence[int] | None = None,
        *,
        frame_stride: int | None = None,
        target_fps: float | None = None,
    ) -> List[ReconstructionJob]:
        """Run Splashsurf for selected steps, optionally downsampling by stride or FPS."""

        indices = list(steps) if steps else self.available_steps()
        if not indices:
            raise FileNotFoundError(
                "No fluid particle dumps were found. Export particles before running reconstruction."
            )
        if target_fps is not None:
            if target_fps <= 0:
                raise ValueError("target_fps must be positive if provided.")
            sim_dt = float(self.config.simulation.time_step)
            if sim_dt <= 0:
                raise ValueError("Scene configuration uses a non-positive time_step, cannot derive FPS.")
            sim_fps = 1.0 / sim_dt
            stride_from_fps = max(1, round(sim_fps / target_fps))
            frame_stride = max(frame_stride or 1, stride_from_fps)
        if frame_stride is not None:
            if frame_stride <= 0:
                raise ValueError("frame_stride must be positive if provided.")
            indices = indices[::frame_stride]

        jobs = [self._build_job(step) for step in indices]
        for job in tqdm(jobs, desc="Reconstructing surfaces"):
            self._run_splashsurf(job)
        return jobs

    def _build_job(self, step: int) -> ReconstructionJob:
        particle = self.fluid_dir / f"fluid_{step:05d}.ply"
        if not particle.exists():
            raise FileNotFoundError(f"Particle file not found for step {step}: {particle}")
        suffix = f".{self.output_format.lstrip('.') or 'obj'}"
        surface = self.surface_dir / f"liquid_surface_{step:05d}{suffix}"
        return ReconstructionJob(step, particle, surface)

    def _run_splashsurf(self, job: ReconstructionJob) -> None:
        command = [self.splashsurf_cmd, "reconstruct", str(job.particle_file)]
        command += ["--output-file", str(job.surface_file)]
        command += ["--particle-radius", f"{self.particle_radius:.6f}"]
        command += ["--smoothing-length", f"{self.smoothing_length_multiplier:.6f}"]
        command += ["--cube-size", f"{self.cube_size_multiplier:.6f}"]
        command += ["--surface-threshold", f"{self.surface_threshold:.6f}"]
        command += ["--rest-density", f"{self.rest_density:.6f}"]


        if self.mesh_smoothing_iters > 0:
            command += ["--mesh-smoothing-iters", str(self.mesh_smoothing_iters)]
            command += [f"--mesh-smoothing-weights={'on' if self.mesh_smoothing_weights else 'off'}"]
        else:
            command += ["--mesh-smoothing-iters", "0"]
        command += [f"--normals={'on' if self.normals else 'off'}"]
        if self.normals and self.normals_smoothing_iters > 0:
            command += ["--normals-smoothing-iters", str(self.normals_smoothing_iters)]

        command += list(self.extra_args)
        subprocess.run(command, check=True)