"""CLI entry point to run the hybrid simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path to find engine module
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
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
    
    steps = args.steps if args.steps is not None else container.config.simulation.total_steps
    for _ in tqdm(range(steps), desc="Simulating"):
        container.step()


if __name__ == "__main__":
    main()
