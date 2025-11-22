"""CLI entry point to run the hybrid simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from engine import WorldContainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid rigid/fluid simulation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/scene_config.yaml"),
        help="Path to the scene configuration YAML file.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Optional override for number of steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    container = WorldContainer.from_config_file(args.config)
    container.run(steps=args.steps)


if __name__ == "__main__":
    main()
