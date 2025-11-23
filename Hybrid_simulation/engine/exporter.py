"""Exporter that writes fluid particles (PLY) and meshes (OBJ) each timestep."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .configuration import ExportConfig
from .mesh_utils import OBJMesh, load_obj_mesh
from .physics_world.math_utils import quaternion_to_matrix, transform_point
from .physics_world.state import RigidBodyState, StaticBodyState, WorldSnapshot


@dataclass
class SimulationExporter:
    output_root: Path
    fluid_dirname: str = "fluid"
    rigid_dirname: str = "rigid"
    static_dirname: str = "static"
    _mesh_cache: Dict[Path, OBJMesh] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_config(cls, config: Optional[ExportConfig]) -> "SimulationExporter":
        if config is None:
            output_root = Path("outputs")
            fluid_dirname = "fluid"
            rigid_dirname = "rigid"
            static_dirname = "static"
        else:
            output_root = config.output_root
            fluid_dirname = config.fluid_subdir
            rigid_dirname = config.rigid_subdir
            static_dirname = config.static_subdir

        exporter = cls(
            output_root=output_root,
            fluid_dirname=fluid_dirname,
            rigid_dirname=rigid_dirname,
            static_dirname=static_dirname,
        )
        exporter._ensure_directories()
        return exporter

    def _ensure_directories(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / self.fluid_dirname).mkdir(parents=True, exist_ok=True)
        (self.output_root / self.rigid_dirname).mkdir(parents=True, exist_ok=True)
        (self.output_root / self.static_dirname).mkdir(parents=True, exist_ok=True)

    def export_step(self, step_index: int, snapshot: WorldSnapshot) -> None:
        rigid_path = self.output_root / self.rigid_dirname / f"rigid_{step_index:05d}.obj"
        static_path = self.output_root / self.static_dirname / f"static_{step_index:05d}.obj"

        # Only export fluid if it exists
        if snapshot.fluids is not None:
            fluid_path = self.output_root / self.fluid_dirname / f"fluid_{step_index:05d}.ply"
            self._write_fluid_ply(fluid_path, snapshot)
        
        self._write_obj(rigid_path, snapshot.rigids)
        self._write_obj(static_path, snapshot.statics)

    def _write_fluid_ply(self, path: Path, snapshot: WorldSnapshot) -> None:
        fluid = snapshot.fluids
        if fluid is None:
            return
        count = fluid.particle_count()
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {count}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property float vx\nproperty float vy\nproperty float vz\n")
            handle.write("property float density\n")
            handle.write("end_header\n")
            for pos, vel, density in zip(fluid.positions, fluid.velocities, fluid.densities):
                handle.write(
                    f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                    f"{vel[0]:.6f} {vel[1]:.6f} {vel[2]:.6f} {density:.6f}\n"
                )

    def _write_obj(self, path: Path, bodies: Sequence[RigidBodyState | StaticBodyState]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# Exported by SimulationExporter\n")
            vertex_offset = 0
            for body in bodies:
                mesh = self._get_mesh(body.mesh_path)
                rotation = quaternion_to_matrix(body.orientation)
                
                # RigidBody: use centered_vertices; StaticBody: use original vertices
                if hasattr(body, 'centered_vertices'):
                    # Rigid body - transform centered vertices
                    transformed = [transform_point(v, rotation, body.position) for v in body.centered_vertices]
                else:
                    # Static body - use absolute coordinates from OBJ
                    transformed = mesh.vertices
                
                handle.write(f"o {body.name}\n")
                for vx, vy, vz in transformed:
                    handle.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
                
                for face in mesh.faces:
                    if len(face) < 3:
                        continue
                    for tri in self._triangulate(face):
                        indices = [str(idx + 1 + vertex_offset) for idx in tri]
                        handle.write(f"f {' '.join(indices)}\n")
                vertex_offset += len(transformed)

    def _triangulate(self, face: Sequence[int]) -> Iterable[Sequence[int]]:
        if len(face) == 3:
            yield face
            return
        root = face[0]
        for i in range(1, len(face) - 1):
            yield (root, face[i], face[i + 1])

    def _get_mesh(self, path: Path) -> OBJMesh:
        resolved = path.expanduser().resolve()
        if resolved in self._mesh_cache:
            return self._mesh_cache[resolved]
        mesh = load_obj_mesh(resolved)
        self._mesh_cache[resolved] = mesh
        return mesh
