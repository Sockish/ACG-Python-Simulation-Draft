"""Blender batch renderer for particle point-cloud exports.

Usage (inside Blender):
    blender -b -P blender/render_particles.py -- path/to/manifest.json
"""
import json
import math
from pathlib import Path
import sys

import bpy
import mathutils


def read_manifest() -> dict:
    try:
        manifest_arg_index = sys.argv.index("--") + 1
    except ValueError as exc:
        raise RuntimeError("Manifest path must be supplied after --") from exc
    manifest_path = Path(sys.argv[manifest_arg_index])
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_particle_positions(ply_path: Path):
    with ply_path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if line.strip() == "end_header":
                break
        data = []
        for raw in handle:
            if raw.strip():
                values = [float(x) for x in raw.split()[:3]]
                data.append(values)
    return data


def cubes_from_points(points, half_size=0.005):
    verts = []
    faces = []
    offsets = [
        (-1, -1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, -1, -1),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 1),
    ]
    cube_faces = [
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    ]
    for point in points:
        base = len(verts)
        for dx, dy, dz in offsets:
            verts.append(
                (
                    point[0] + dx * half_size,
                    point[1] + dy * half_size,
                    point[2] + dz * half_size,
                )
            )
        for face in cube_faces:
            faces.append(tuple(base + idx for idx in face))
    return verts, faces


def ensure_camera(camera_settings: dict | None) -> None:
    scene = bpy.context.scene
    cam_obj = next((obj for obj in scene.objects if obj.type == "CAMERA"), None)
    if cam_obj is None:
        camera = bpy.data.cameras.new("FluidCamera")
        cam_obj = bpy.data.objects.new("FluidCamera", camera)
        scene.collection.objects.link(cam_obj)
    cam_obj.location = camera_settings.get("location", [2.0, 1.5, 2.0])
    target = mathutils.Vector(camera_settings.get("look_at", [0.0, 0.4, 0.0]))
    direction = target - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam_obj


def ensure_light():
    scene = bpy.context.scene
    if "FluidKeyLight" in bpy.data.objects:
        return
    light_data = bpy.data.lights.new(name="FluidKeyLight", type="AREA")
    light_data.energy = 3000
    light_obj = bpy.data.objects.new(name="FluidKeyLight", object_data=light_data)
    light_obj.location = (2.0, 3.0, 2.0)
    scene.collection.objects.link(light_obj)


def render_sequence(manifest: dict) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.image_settings.file_format = "PNG"
    render_dir = Path(manifest["render_dir"])
    render_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = Path(manifest["ply_dir"])
    files = sorted(ply_dir.glob("frame_*.ply"))
    ensure_camera(manifest.get("camera", {}))
    ensure_light()
    for frame_idx, ply_path in enumerate(files):
        points = load_particle_positions(ply_path)
        mesh = bpy.data.meshes.new(f"FluidFrame{frame_idx}")
        verts, faces = cubes_from_points(points, half_size=0.004)
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new("Fluid", mesh)
        bpy.context.collection.objects.link(obj)
        scene.render.filepath = str(render_dir / f"frame_{frame_idx:05d}.png")
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


if __name__ == "__main__":
    render_sequence(read_manifest())

