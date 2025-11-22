"""Splashsurf-based surface reconstruction utilities."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from engine.configuration import SceneConfig


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
    mesh_smoothing_iters: int = 15
    mesh_smoothing_weights: bool = True
    normals: bool = True
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

    def reconstruct(self, steps: Sequence[int] | None = None) -> List[ReconstructionJob]:
        indices = list(steps) if steps else self.available_steps()
        if not indices:
            raise FileNotFoundError(
                "No fluid particle dumps were found. Export particles before running reconstruction."
            )
        jobs = [self._build_job(step) for step in indices]
        for job in jobs:
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

        min_corner = self.config.liquid_box.min_corner
        max_corner = self.config.liquid_box.max_corner
        command += [
            "--particle-aabb-min",
            str(min_corner[0]),
            str(min_corner[1]),
            str(min_corner[2]),
        ]
        command += [
            "--particle-aabb-max",
            str(max_corner[0]),
            str(max_corner[1]),
            str(max_corner[2]),
        ]

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