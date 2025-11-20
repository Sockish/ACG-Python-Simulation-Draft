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


def clear_default_scene() -> None:
    """Remove Blender's default Cube, Camera, and Light objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def create_water_material() -> bpy.types.Material:
    """Create a visible water material with color and transparency."""
    mat = bpy.data.materials.new(name="WaterMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)
    
    # Principled BSDF for better visibility
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    # Water-like properties
    bsdf.inputs['Base Color'].default_value = (0.1, 0.4, 0.8, 1.0)  # Blue color
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.1  # Slightly rough
    bsdf.inputs['Transmission Weight'].default_value = 0.7  # Semi-transparent
    bsdf.inputs['IOR'].default_value = 1.333  # Water IOR
    bsdf.inputs['Alpha'].default_value = 0.85  # Slight transparency
    
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Enable transparency
    mat.blend_method = 'BLEND'
    mat.show_transparent_back = True
    
    return mat


def create_ground_plane() -> bpy.types.Object:
    """Create a simple ground plane for visual reference."""
    bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    
    # Ground material (simple gray diffuse)
    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.5
    ground.data.materials.append(mat)
    
    return ground


def ensure_camera(camera_settings: dict | None) -> None:
    scene = bpy.context.scene
    cam_obj = next((obj for obj in scene.objects if obj.type == "CAMERA"), None)
    if cam_obj is None:
        camera = bpy.data.cameras.new("FluidCamera")
        cam_obj = bpy.data.objects.new("FluidCamera", camera)
        scene.collection.objects.link(cam_obj)
        camera.lens = 50  # Standard focal length
    cam_obj.location = camera_settings.get("location", [3.2, 3.0, 3.4])
    look = mathutils.Vector(camera_settings.get("look_at", [0.0, 0.4, 0.0]))
    direction = look - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam_obj


def ensure_light() -> None:
    scene = bpy.context.scene
    
    # Main key light (Sun)
    if "FluidKeyLight" not in bpy.data.objects:
        key_light = bpy.data.lights.new(name="FluidKeyLight", type="SUN")
        key_light.energy = 3.0
        key_light.angle = 0.05  # Sharper shadows
        key_obj = bpy.data.objects.new(name="FluidKeyLight", object_data=key_light)
        key_obj.location = (3.0, 3.0, 4.0)
        key_obj.rotation_euler = (0.8, 0.2, 1.0)
        scene.collection.objects.link(key_obj)
    
    # Fill light (softer, from opposite side)
    if "FluidFillLight" not in bpy.data.objects:
        fill_light = bpy.data.lights.new(name="FluidFillLight", type="AREA")
        fill_light.energy = 500
        fill_light.size = 2.0
        fill_obj = bpy.data.objects.new(name="FluidFillLight", object_data=fill_light)
        fill_obj.location = (-2.0, 2.0, 1.5)
        scene.collection.objects.link(fill_obj)
    
    # Setup world background (darker for better contrast)
    world = bpy.data.worlds.get("World")
    if world:
        world.use_nodes = True
        bg = world.node_tree.nodes.get('Background')
        if bg:
            bg.inputs['Color'].default_value = (0.3, 0.35, 0.4, 1.0)  # Dark gray
            bg.inputs['Strength'].default_value = 0.3


def render_sequence(manifest: dict) -> None:
    scene = bpy.context.scene
    clear_default_scene()
    mesh_dir = Path(manifest.get("mesh_dir") or "")
    if not mesh_dir.exists():
        raise RuntimeError("Manifest missing mesh_dir or path does not exist")
    render_dir = Path(manifest["render_dir"])
    render_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup scene
    ensure_camera(manifest.get("camera", {}))
    ensure_light()
    ground = create_ground_plane()
    water_mat = create_water_material()
    
    # Rendering settings
    scene.render.engine = "CYCLES"
    
    # Enable GPU rendering if available
    try:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # or 'OPTIX' for RTX cards
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            device.use = True
        scene.cycles.device = 'GPU'
        print(f"GPU rendering enabled: {[d.name for d in bpy.context.preferences.addons['cycles'].preferences.devices if d.use]}")
    except Exception as e:
        print(f"GPU rendering not available, using CPU: {e}")
        scene.cycles.device = 'CPU'
    
    scene.cycles.samples = 16  # Higher samples for transparency
    scene.cycles.max_bounces = 8
    scene.cycles.transparent_max_bounces = 12  # More bounces for transparency
    scene.cycles.use_denoising = True  # Enable denoising for cleaner result
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.film_transparent = False  # Solid background
    
    mesh_files = sorted(mesh_dir.glob("frame_*.obj"))
    print(f"Found {len(mesh_files)} mesh files to render")
    
    for frame_idx, mesh_path in enumerate(mesh_files):
        verts, faces = load_obj(mesh_path)
        print(f"Frame {frame_idx}: {len(verts)} vertices, {len(faces)} faces")
        
        if len(verts) == 0 or len(faces) == 0:
            print(f"WARNING: Empty mesh at frame {frame_idx}, skipping")
            continue
            
        mesh = bpy.data.meshes.new(f"FluidMesh{frame_idx}")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        
        obj = bpy.data.objects.new("FluidMesh", mesh)
        obj.data.materials.append(water_mat)  # Apply water material
        scene.collection.objects.link(obj)
        
        # Enable smooth shading for better appearance
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        
        scene.render.filepath = str(render_dir / f"frame_{frame_idx:05d}.png")
        bpy.ops.render.render(write_still=True)
        
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
    
    # Cleanup
    bpy.data.objects.remove(ground, do_unlink=True)


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

