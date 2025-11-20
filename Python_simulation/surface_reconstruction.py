"""Simple surface reconstruction from particle dumps.

This implementation intentionally stays lightweight and dependency free
except for the optional scikit-image marching cubes helper. If the
library is missing we raise a friendly error telling the user how to
install it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

try:
    from skimage import measure
except ImportError as exc:  # pragma: no cover - import guard
    measure = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from config import SimulationConfig
from kernels import SmoothingKernels


def load_ascii_ply(path: Path) -> np.ndarray:
    """Parse the ASCII ply files produced by io_utils.exporter."""

    with path.open("r", encoding="utf-8") as handle:
        header = []
        while True:
            line = handle.readline().strip()
            header.append(line)
            if line == "end_header":
                break
        count = next(int(x.split()[2]) for x in header if x.startswith("element vertex"))
        data = np.loadtxt(handle, max_rows=count)
    return data[:, :3]


def build_scalar_field(
    positions: np.ndarray,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate SPH kernels onto a dense grid to approximate a surface."""

    res = config.reconstruction_grid_resolution
    domain_min = np.asarray(config.domain_min, dtype=np.float32)
    domain_max = np.asarray(config.domain_max, dtype=np.float32)
    grid = np.zeros((res, res, res), dtype=np.float32)
    xs = np.linspace(domain_min[0], domain_max[0], res)
    ys = np.linspace(domain_min[1], domain_max[1], res)
    zs = np.linspace(domain_min[2], domain_max[2], res)
    kernels = SmoothingKernels(config.kernel_radius * config.reconstruction_radius_scale)
    cell_size = (domain_max - domain_min) / (res - 1)
    influence_radius = int(
        np.ceil(config.reconstruction_radius_scale * config.kernel_radius / cell_size.max())
    )
    for point in positions:
        center_idx = ((point - domain_min) / cell_size).astype(int)
        center_idx = np.clip(center_idx, 0, res - 1)
        for ix in range(max(center_idx[0] - influence_radius, 0), min(center_idx[0] + influence_radius + 1, res)):
            for iy in range(max(center_idx[1] - influence_radius, 0), min(center_idx[1] + influence_radius + 1, res)):
                for iz in range(max(center_idx[2] - influence_radius, 0), min(center_idx[2] + influence_radius + 1, res)):
                    grid_pt = np.array([xs[ix], ys[iy], zs[iz]])
                    r = np.linalg.norm(point - grid_pt)
                    grid[ix, iy, iz] += kernels.poly6(r)
    return grid, xs, ys, zs


def reconstruct_frame(ply_path: Path, output_path: Path, config: SimulationConfig) -> None:
    if measure is None:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "scikit-image is required for surface reconstruction. Install it via `pip install scikit-image`"
        ) from IMPORT_ERROR
    positions = load_ascii_ply(ply_path)
    field, xs, ys, zs = build_scalar_field(positions, config)
    verts, faces, normals, _ = measure.marching_cubes(field, level=np.percentile(field, 60))
    scale = np.array([
        xs[-1] - xs[0],
        ys[-1] - ys[0],
        zs[-1] - zs[0],
    ]) / field.shape[0]
    verts = verts * scale + np.array([xs[0], ys[0], zs[0]])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as obj:
        for v in verts:
            obj.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces:
            obj.write(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}\n")


def reconstruct_sequence(ply_dir: Path, mesh_dir: Path, config: SimulationConfig, limit: int | None = None) -> None:
    ply_files = sorted(ply_dir.glob("frame_*.ply"))
    if limit is not None:
        ply_files = ply_files[:limit]
    for ply_path in tqdm(ply_files, desc="Reconstructing", unit="frame"):
        mesh_path = mesh_dir / (ply_path.stem + ".obj")
        reconstruct_frame(ply_path, mesh_path, config)


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Reconstruct surfaces from exported SPH frames.")
    parser.add_argument("input", type=Path, help="Directory with .ply frames")
    parser.add_argument("output", type=Path, help="Directory to write .obj meshes")
    parser.add_argument("--grid", type=int, default=64, help="Grid resolution for marching cubes")
    args = parser.parse_args()
    config = SimulationConfig(reconstruction_grid_resolution=args.grid)
    reconstruct_sequence(args.input, args.output, config)


if __name__ == "__main__":  # pragma: no cover
    main()

