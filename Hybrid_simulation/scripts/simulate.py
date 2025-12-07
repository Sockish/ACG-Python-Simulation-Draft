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
    parser.add_argument(
        "--use-taichi",
        action="store_true",
        help="Use Taichi GPU-accelerated SPH solver (much faster for large particle counts)"
    )
    parser.add_argument(
        "--taichi-cpu",
        action="store_true", 
        help="Use Taichi with CPU backend instead of GPU"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Set Taichi backend if requested
    if args.use_taichi:
        import taichi as ti
        import os
        os.environ['TI_LOG_LEVEL'] = 'error'  # Suppress Taichi logs
        if args.taichi_cpu:
            ti.init(arch=ti.cpu)
            print("🚀 Using Taichi SPH solver (CPU backend)")
        else:
            try:
                ti.init(arch=ti.gpu)
                print("🚀 Using Taichi SPH solver (GPU backend)")
            except Exception as e:
                print(f"⚠ GPU init failed: {e}, falling back to CPU")
                ti.init(arch=ti.cpu)
    
    container = WorldContainer.from_config_file(args.config, use_taichi=args.use_taichi)
    
    steps = args.steps if args.steps is not None else container.config.simulation.total_steps
    for _ in tqdm(range(steps), desc="Simulating"):
        container.step()


if __name__ == "__main__":
    main()
