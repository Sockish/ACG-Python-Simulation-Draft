"""Interactive Blender helper to preview reconstructed surface frames.

How to use inside the Blender GUI:
1. Open this script in the Text Editor and press *Run Script*.
2. A file browser will pop up. Select the directory containing
   `frame_XXXXX.obj` files (e.g. `output/mesh`).
3. Press *Load* and then use the timeline to scrub or play the loaded
   animation. No video rendering is required.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import bpy  # type: ignore[import-not-found]
from bpy.props import StringProperty  # type: ignore[import-not-found]
from bpy.types import Operator  # type: ignore[import-not-found]


def load_obj(path: Path) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                indices = [int(part.split("/")[0]) - 1 for part in line.strip().split()[1:4]]
                faces.append(tuple(indices))
    return verts, faces


def build_mesh_sequence(mesh_dir: Path) -> list[bpy.types.Mesh]:
    meshes: list[bpy.types.Mesh] = []
    for obj_path in sorted(mesh_dir.glob("rigid_*.obj")):
        verts, faces = load_obj(obj_path)
        mesh = bpy.data.meshes.new(obj_path.stem)
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        meshes.append(mesh)
    if not meshes:
        raise RuntimeError(f"No OBJ files found in {mesh_dir}")
    return meshes


def ensure_object(mesh_data: bpy.types.Mesh) -> bpy.types.Object:
    scene = bpy.context.scene
    obj = next((obj for obj in scene.objects if obj.name == "SPH_Surface"), None)
    if obj is None:
        obj = bpy.data.objects.new("SPH_Surface", mesh_data)
        scene.collection.objects.link(obj)
    else:
        obj.data = mesh_data
    return obj


def register_frame_handler(meshes: list[bpy.types.Mesh], obj: bpy.types.Object) -> None:
    frame_count = len(meshes)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count

    def _handler(scene):  # type: ignore[override]
        idx = min(max(scene.frame_current - 1, 0), frame_count - 1)
        obj.data = meshes[idx]
    _handler.__sph_handler__ = True  # type: ignore[attr-defined]

    # remove existing handler instances to avoid duplicates
    bpy.app.handlers.frame_change_pre[:] = [h for h in bpy.app.handlers.frame_change_pre if not getattr(h, "__sph_handler__", False)]
    bpy.app.handlers.frame_change_pre.append(_handler)


class SPH_OT_load_surface_animation(Operator):
    """Load OBJ surface frames and hook them to the timeline."""

    bl_idname = "sph.load_surface_animation"
    bl_label = "Load SPH Surface Animation"

    directory: StringProperty(name="Mesh Directory", subtype="DIR_PATH")  # type: ignore[assignment]

    def execute(self, context):
        mesh_dir = Path(self.directory)
        if not mesh_dir.exists():
            self.report({"ERROR"}, f"Directory not found: {mesh_dir}")
            return {"CANCELLED"}
        try:
            meshes = build_mesh_sequence(mesh_dir)
        except Exception as exc:  # pragma: no cover - Blender UI feedback
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        obj = ensure_object(meshes[0])
        register_frame_handler(meshes, obj)
        self.report({"INFO"}, f"Loaded {len(meshes)} frames from {mesh_dir}")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


classes = (SPH_OT_load_surface_animation,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
    bpy.ops.sph.load_surface_animation('INVOKE_DEFAULT')
