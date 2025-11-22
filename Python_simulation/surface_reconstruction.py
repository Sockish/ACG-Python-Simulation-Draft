"""Batch wrapper around pysplashsurf reconstruction.

This module offers two primary entry points:

- ``reconstruct_sequence`` integrates with ``SimulationRunner`` so the
  reconstruction stage fits seamlessly into the existing Python pipeline.
- ``main`` exposes a small CLI so you can invoke the same logic manually,
  e.g. ``python surface_reconstruction.py output/ply output/mesh``.

Both paths iterate over all ``.ply`` particle caches, call the external
``pysplashsurf`` tool with consistent parameters, and emit meshes to
``output/mesh`` (or any folder you choose).
"""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_PYSPLASHSURF_CMD = "pysplashsurf"


@dataclass(frozen=True)
class SplashSurfParams:
	"""Parameter bundle mirroring the pysplashsurf CLI options we care about."""

	particle_radius: float = 0.025
	smoothing_length: float = 2.0
	cube_size: float = 0.5
	threshold: float = 0.6
	mesh_smoothing_weights: str = "on"
	mesh_smoothing_iters: int = 15
	normals: str = "on"
	normals_smoothing_iters: int = 10

	def as_cli_args(self) -> list[str]:
		return [
			"-r",
			str(self.particle_radius),
			"-l",
			str(self.smoothing_length),
			"-c",
			str(self.cube_size),
			"-t",
			str(self.threshold),
			f"--mesh-smoothing-weights={self.mesh_smoothing_weights}",
			"--mesh-smoothing-iters",
			str(self.mesh_smoothing_iters),
			f"--normals={self.normals}",
			"--normals-smoothing-iters",
			str(self.normals_smoothing_iters),
		]


def _discover_ply_frames(ply_dir: Path) -> list[Path]:
	if not ply_dir.exists():
		raise FileNotFoundError(f"Missing particle directory: {ply_dir}")
	return sorted(ply_dir.glob("*.ply"))


def _build_command(
	executable: str,
	input_file: Path,
	output_file: Path,
	params: SplashSurfParams,
) -> list[str]:
	return [
		executable,
		"reconstruct",
		str(input_file),
		*params.as_cli_args(),
		"-o",
		str(output_file),
	]


def _run_command(command: Sequence[str]) -> None:
	print("[pysplashsurf]", " ".join(command))
	try:
		subprocess.run(command, check=True)
	except FileNotFoundError as exc:
		raise RuntimeError(
			f"Could not find '{command[0]}'. Ensure pysplashsurf is installed and on PATH."
		) from exc
	except subprocess.CalledProcessError as exc:
		raise RuntimeError(f"pysplashsurf failed with exit code {exc.returncode}") from exc


def reconstruct_sequence(
	ply_dir: Path | str,
	mesh_dir: Path | str,
	config=None,
	scene_name: str | None = None,
	rigid_dir: Path | str | None = None,
	*,
	pysplashsurf_cmd: str = DEFAULT_PYSPLASHSURF_CMD,
	params: SplashSurfParams | None = None,
	overwrite: bool = True,
	output_suffix: str = "_liquid",
) -> list[Path]:
	"""Iterate over every PLY cache and reconstruct a surface mesh.

	Parameters besides ``ply_dir``/``mesh_dir`` match what the simulation
	pipeline expects, so this function can be called both interactively and
	from ``SimulationRunner`` without extra glue code.
	"""

	del config, scene_name, rigid_dir  # Reserved for future integration hooks.

	ply_dir = Path(ply_dir)
	mesh_dir = Path(mesh_dir)
	mesh_dir.mkdir(parents=True, exist_ok=True)

	frames = _discover_ply_frames(ply_dir)
	if not frames:
		print(f"No PLY frames found in {ply_dir}, skipping reconstruction")
		return []

	params = params or SplashSurfParams()
	produced: list[Path] = []

	for index, ply_file in enumerate(frames, start=1):
		mesh_name = f"{ply_file.stem}{output_suffix}.obj"
		mesh_path = mesh_dir / mesh_name
		if mesh_path.exists() and not overwrite:
			print(f"[{index}/{len(frames)}] Skipping existing {mesh_name}")
			continue

		print(f"[{index}/{len(frames)}] Reconstructing {ply_file.name} -> {mesh_name}")
		command = _build_command(pysplashsurf_cmd, ply_file, mesh_path, params)
		_run_command(command)
		produced.append(mesh_path)

	return produced


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Batch surface reconstruction via pysplashsurf")
	parser.add_argument("input_dir", type=Path, help="Directory containing frame_XXXXX.ply files")
	parser.add_argument("output_dir", type=Path, help="Destination directory for OBJ meshes")
	parser.add_argument("--pysplashsurf", default=DEFAULT_PYSPLASHSURF_CMD, help="Executable to invoke")
	parser.add_argument("--radius", "-r", type=float, default=SplashSurfParams().particle_radius)
	parser.add_argument("--smoothing-length", "-l", type=float, default=SplashSurfParams().smoothing_length)
	parser.add_argument("--cube-size", "-c", type=float, default=SplashSurfParams().cube_size)
	parser.add_argument("--threshold", "-t", type=float, default=SplashSurfParams().threshold)
	parser.add_argument(
		"--mesh-smoothing-weights",
		choices=["on", "off"],
		default=SplashSurfParams().mesh_smoothing_weights,
	)
	parser.add_argument(
		"--mesh-smoothing-iters",
		type=int,
		default=SplashSurfParams().mesh_smoothing_iters,
	)
	parser.add_argument("--normals", choices=["on", "off"], default=SplashSurfParams().normals)
	parser.add_argument(
		"--normals-smoothing-iters",
		type=int,
		default=SplashSurfParams().normals_smoothing_iters,
	)
	parser.add_argument(
		"--skip-existing",
		action="store_true",
		help="Do not re-run pysplashsurf for meshes that already exist",
	)
	parser.add_argument(
		"--suffix",
		default="_liquid",
		help="Suffix appended to frame_XXXXX when naming OBJ files (default: _liquid)",
	)
	return parser


def main(argv: Iterable[str] | None = None) -> None:
	parser = _build_parser()
	args = parser.parse_args(list(argv) if argv is not None else None)

	params = SplashSurfParams(
		particle_radius=args.radius,
		smoothing_length=args.smoothing_length,
		cube_size=args.cube_size,
		threshold=args.threshold,
		mesh_smoothing_weights=args.mesh_smoothing_weights,
		mesh_smoothing_iters=args.mesh_smoothing_iters,
		normals=args.normals,
		normals_smoothing_iters=args.normals_smoothing_iters,
	)

	reconstruct_sequence(
		args.input_dir,
		args.output_dir,
		pysplashsurf_cmd=args.pysplashsurf,
		params=params,
		overwrite=not args.skip_existing,
		output_suffix=args.suffix,
	)


if __name__ == "__main__":
	main()