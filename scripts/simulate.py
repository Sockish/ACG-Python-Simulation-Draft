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
    parser.add_argument(
        "--use-mpm",
        action="store_true",
        help="Use MPM (Material Point Method) solver instead of SPH"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Set Taichi backend if requested (for both SPH and MPM)
    if args.use_taichi or args.use_mpm:
        import taichi as ti
        import os
        os.environ['TI_LOG_LEVEL'] = 'error'  # Suppress Taichi logs
        if args.taichi_cpu:
            ti.init(arch=ti.cpu)
            solver_name = "MPM" if args.use_mpm else "Taichi SPH"
            print(f"🚀 Using {solver_name} solver (CPU backend)")
        else:
            try:
                ti.init(arch=ti.gpu)
                solver_name = "MPM" if args.use_mpm else "Taichi SPH"
                print(f"🚀 Using {solver_name} solver (GPU backend)")
            except Exception as e:
                print(f"⚠ GPU init failed: {e}, falling back to CPU")
                ti.init(arch=ti.cpu)
    
    # Create world container with appropriate solver
    use_mpm = args.use_mpm
    use_taichi_sph = args.use_taichi and not args.use_mpm
    
    container = WorldContainer.from_config_file(
        args.config, 
        use_taichi=use_taichi_sph,
        use_mpm=use_mpm
    )
    
    steps = args.steps if args.steps is not None else container.config.simulation.total_steps
    for _ in tqdm(range(steps), desc="Simulating"):
        container.step()


if __name__ == "__main__":
    main()
