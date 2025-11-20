"""Command-line entry point for the SPH simulation + Blender pipeline."""
from __future__ import annotations

import argparse

from config import SimulationConfig
from simulation import RunOptions, SimulationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPH fluid simulation and Blender exporter")
    parser.add_argument("--scene", choices=["dam_break", "drop"], default="dam_break")
    parser.add_argument("--steps", type=int, default=1200, help="Total simulation steps")
    parser.add_argument("--particles", type=int, default=5000, help="Number of particles")
    parser.add_argument("--render", action="store_true", help="Invoke Blender once the sim finishes")
    parser.add_argument("--surface", action="store_true", help="Render reconstructed surfaces instead of cubes")
    parser.add_argument("--reconstruct", action="store_true", help="Run marching cubes on exported frames")
    parser.add_argument("--no-export", action="store_true", help="Skip writing PLY files")
    parser.add_argument("--blender", default="blender", help="Path to Blender executable")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = SimulationConfig(
        particle_count=args.particles,
        total_steps=args.steps,
        blender_executable=args.blender,
    )

    options = RunOptions(
        scene=args.scene,
        export_particles=not args.no_export,
        reconstruct_surface=args.reconstruct,
        render_with_blender=args.render,
        render_surface=args.surface,
    )

    runner = SimulationRunner(config, options)
    runner.run()


if __name__ == "__main__":
    main()

