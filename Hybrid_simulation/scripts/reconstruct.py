"""CLI entry point to reconstruct water surfaces using Splashsurf."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.configuration import load_scene_config
from surface_reconstruction import SplashsurfReconstructor

## .venv/Scripts/python.exe ./scripts/reconstruct.py \
##   --config config/scene_config.yaml \
##   --splashsurf-cmd pysplashsurf \
##   --target-fps 60


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct liquid surfaces from particle dumps")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/scene_config.yaml"),
        help="Scene configuration file (YAML)",
    )
    parser.add_argument(
        "--splashsurf-cmd",
        type=str,
        default="pysplashsurf",
        help="Executable used to run splashsurf",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for reconstructed meshes (defaults to outputs/surface)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="obj",
        help="Surface mesh format supported by splashsurf (e.g., obj, ply)",
    )
    parser.add_argument(
        "--particle-radius",
        type=float,
        default=None,
        help="Particle radius in meters (defaults to half the particle spacing)",
    )
    parser.add_argument(
        "--smoothing-multiplier",
        type=float,
        default=None,
        help="Smoothing length multiplier (in multiples of the particle radius)",
    )
    parser.add_argument(
        "--cube-size",
        type=float,
        default=0.5,
        help="Cube edge length for marching cubes (multiples of particle radius)",
    )
    parser.add_argument(
        "--surface-threshold",
        type=float,
        default=0.6,
        help="Iso-surface threshold for marching cubes",
    )
    parser.add_argument(
        "--rest-density",
        type=float,
        default=None,
        help="Fluid rest density (kg/m^3); defaults to liquid box value",
    )
    parser.add_argument(
        "--mesh-smoothing-iters",
        type=int,
        default=15,
        help="Mesh smoothing iterations (0 disables smoothing)",
    )
    parser.add_argument(
        "--no-mesh-smoothing-weights",
        action="store_true",
        help="Disable mesh smoothing feature weights",
    )
    parser.add_argument(
        "--normals",
        dest="normals",
        action="store_true",
        help="Enable normal generation (default)",
    )
    parser.add_argument(
        "--no-normals",
        dest="normals",
        action="store_false",
        help="Disable normal generation",
    )
    parser.set_defaults(normals=True)
    parser.add_argument(
        "--normals-smoothing-iters",
        type=int,
        default=10,
        help="Number of smoothing iterations for normals",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Use every Nth particle dump (e.g. 25 for ~100fps if sim is 2500fps)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Downsample to the closest stride that approximates this playback FPS",
    )
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        dest="steps",
        default=None,
        help="Specific simulation step to reconstruct (repeatable)",
    )
    parser.add_argument(
        "--splashsurf-arg",
        action="append",
        dest="extra_args",
        default=None,
        help="Additional argument forwarded to splashsurf (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_scene_config(args.config)
    reconstructor = SplashsurfReconstructor(
        config,
        splashsurf_cmd=args.splashsurf_cmd,
        output_dir=args.output_dir,
        output_format=args.format,
        particle_radius=args.particle_radius,
        smoothing_length_multiplier=args.smoothing_multiplier,
        cube_size_multiplier=args.cube_size,
        surface_threshold=args.surface_threshold,
        rest_density=args.rest_density,
        mesh_smoothing_iters=args.mesh_smoothing_iters,
        mesh_smoothing_weights=not args.no_mesh_smoothing_weights,
        normals=args.normals,
        normals_smoothing_iters=args.normals_smoothing_iters,
        extra_args=args.extra_args or [],
    )
    steps: List[int] | None = args.steps
    jobs = reconstructor.reconstruct(
        steps,
        frame_stride=args.frame_stride,
        target_fps=args.target_fps,
    )
    for job in jobs:
        print(f"Reconstructed step {job.step_index} -> {job.surface_file}")


if __name__ == "__main__":
    main()
