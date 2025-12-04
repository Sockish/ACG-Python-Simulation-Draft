"""Check if mesh face normals point outward or inward."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.mesh_utils import load_obj_mesh
from engine.physics_world.math_utils import sub, cross, normalize, dot, add, mul


def compute_mesh_center(vertices):
    """Compute geometric center of mesh."""
    if not vertices:
        return (0.0, 0.0, 0.0)
    
    sum_x = sum(v[0] for v in vertices)
    sum_y = sum(v[1] for v in vertices)
    sum_z = sum(v[2] for v in vertices)
    count = len(vertices)
    
    return (sum_x / count, sum_y / count, sum_z / count)


def check_face_normal_direction(mesh_path: Path):
    """Check if face normals point outward or inward."""
    
    mesh = load_obj_mesh(mesh_path)
    print(f"\n📊 Analyzing mesh: {mesh_path.name}")
    print(f"   Vertices: {len(mesh.vertices)}")
    print(f"   Faces: {len(mesh.faces)}")
    
    # Compute mesh center (approximate for convex meshes)
    center = compute_mesh_center(mesh.vertices)
    print(f"   Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    
    # Check each face
    outward_count = 0
    inward_count = 0
    
    for i, face in enumerate(mesh.faces):
        if len(face) < 3:
            continue
        
        # Get first 3 vertices of face (handles quads too)
        v0 = mesh.vertices[face[0]]
        v1 = mesh.vertices[face[1]]
        v2 = mesh.vertices[face[2]]
        
        # Compute face normal
        edge0 = sub(v1, v0)
        edge1 = sub(v2, v0)
        normal = normalize(cross(edge0, edge1))
        
        # Compute face center
        face_center = (
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0,
        )
        
        # Vector from mesh center to face center
        outward_direction = normalize(sub(face_center, center))
        
        # Check if normal aligns with outward direction
        alignment = dot(normal, outward_direction)
        
        if alignment > 0:
            outward_count += 1
        else:
            inward_count += 1
        
        # Print first few faces for inspection
        if i < 3:
            direction = "OUTWARD ✓" if alignment > 0 else "INWARD ✗"
            print(f"\n   Face {i+1} (vertices {face[0]}, {face[1]}, {face[2]})")
            print(f"      Normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
            print(f"      Outward dir: ({outward_direction[0]:.3f}, {outward_direction[1]:.3f}, {outward_direction[2]:.3f})")
            print(f"      Alignment: {alignment:.3f} → {direction}")
    
    # Summary
    print(f"\n📈 Summary:")
    print(f"   Outward-facing: {outward_count} ({outward_count/len(mesh.faces)*100:.1f}%)")
    print(f"   Inward-facing:  {inward_count} ({inward_count/len(mesh.faces)*100:.1f}%)")
    
    if inward_count == 0:
        print(f"   ✅ All normals point OUTWARD - mesh is correctly oriented!")
    elif outward_count == 0:
        print(f"   ⚠️  All normals point INWARD - mesh is inside-out!")
    else:
        print(f"   ⚠️  Mixed orientations - mesh has inconsistent winding!")
    
    return outward_count, inward_count


if __name__ == "__main__":
    # Check the sphere mesh
    sphere_path = Path("config/assets/rigid/sphere.obj")
    
    if not sphere_path.exists():
        print(f"❌ File not found: {sphere_path}")
        sys.exit(1)
    
    check_face_normal_direction(sphere_path)
