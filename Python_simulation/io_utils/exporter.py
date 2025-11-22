"""Frame export helpers for Blender rendering and post-processing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fluid.particles import ParticleSystem
from rigid.world import RigidWorld


def write_particles_ply(frame_idx: int, particles: ParticleSystem, directory: Path) -> Path:
    """Dump particle positions and velocities to a simple ASCII PLY file."""

    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / f"frame_{frame_idx:05d}.ply"
    with filepath.open("w", encoding="utf-8") as ply:
        ply.write("ply\nformat ascii 1.0\n")
        ply.write(f"element vertex {particles.count}\n")
        ply.write("property float x\nproperty float y\nproperty float z\n")
        ply.write("property float vx\nproperty float vy\nproperty float vz\n")
        ply.write("end_header\n")
        data = np.hstack((particles.positions, particles.velocities))
        np.savetxt(ply, data, fmt="%.6f")
    return filepath


def write_manifest(
    total_frames: int,
    ply_dir: Path,
    render_dir: Path,
    mesh_dir: Path | None = None,
    rigid_dir: Path | None = None,
    camera: dict | None = None,
) -> Path:
    """Produce a JSON manifest Blender scripts can rely on."""

    manifest = {
        "total_frames": total_frames,
        "ply_dir": str(ply_dir),
        "mesh_dir": str(mesh_dir) if mesh_dir else None,
        "rigid_dir": str(rigid_dir) if rigid_dir else None,
        "render_dir": str(render_dir),
        "camera": camera or {},
    }
    render_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = render_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_rigid_transforms(frame_idx: int, rigid_world: RigidWorld, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    data = []
    for body in rigid_world.bodies:
        data.append(
            {
                "name": body.name,
                "position": body.position.tolist(),
                "orientation": body.orientation.tolist(),
            }
        )
    path = directory / f"rigid_{frame_idx:05d}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path

