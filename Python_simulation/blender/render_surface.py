"""Blender renderer for reconstructed meshes (.obj).

Invoke via:
    blender -b -P blender/render_surface.py -- path/to/manifest.json
"""
import json
from pathlib import Path
import sys

import bpy
import mathutils


def read_manifest() -> dict:
    try:
        idx = sys.argv.index("--") + 1
    except ValueError as exc:
        raise RuntimeError("Manifest path required") from exc
    manifest_path = Path(sys.argv[idx])
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def ensure_camera(camera_settings: dict | None) -> None:
    scene = bpy.context.scene
    cam_obj = next((obj for obj in scene.objects if obj.type == "CAMERA"), None)
    if cam_obj is None:
        camera = bpy.data.cameras.new("FluidCamera")
        cam_obj = bpy.data.objects.new("FluidCamera", camera)
        scene.collection.objects.link(cam_obj)
    cam_obj.location = camera_settings.get("location", [1.5, 1.5, 1.0])
    look = mathutils.Vector(camera_settings.get("look_at", [0.0, 0.3, 0.0]))
    direction = look - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam_obj


def ensure_light() -> None:
    scene = bpy.context.scene
    if "FluidKeyLight" in bpy.data.objects:
        return
    light_data = bpy.data.lights.new(name="FluidKeyLight", type="SUN")
    light_data.energy = 5.0
    light_obj = bpy.data.objects.new(name="FluidKeyLight", object_data=light_data)
    light_obj.location = (2.0, 3.0, 2.0)
    scene.collection.objects.link(light_obj)


def render_sequence(manifest: dict) -> None:
    scene = bpy.context.scene
    mesh_dir = Path(manifest.get("mesh_dir") or "")
    if not mesh_dir.exists():
        raise RuntimeError("Manifest missing mesh_dir or path does not exist")
    render_dir = Path(manifest["render_dir"])
    render_dir.mkdir(parents=True, exist_ok=True)
    ensure_camera(manifest.get("camera", {}))
    ensure_light()
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
    scene.render.image_settings.file_format = "PNG"
    mesh_files = sorted(mesh_dir.glob("frame_*.obj"))
    for frame_idx, mesh_path in enumerate(mesh_files):
        verts, faces = load_obj(mesh_path)
        mesh = bpy.data.meshes.new(f"FluidMesh{frame_idx}")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new("FluidMesh", mesh)
        scene.collection.objects.link(obj)
        scene.render.filepath = str(render_dir / f"frame_{frame_idx:05d}.png")
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def load_obj(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                parts = [int(part.split("/")[0]) - 1 for part in line.strip().split()[1:4]]
                faces.append(tuple(parts))
    return verts, faces


if __name__ == "__main__":
    render_sequence(read_manifest())

