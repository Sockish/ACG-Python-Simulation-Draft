import bpy
import sys
import os

# Set rendering device to OPTIX
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'
bpy.context.preferences.addons['cycles'].preferences.get_devices()

# set denoiser
denoiser = 'OPTIX'  # Replace 'OPTIX' with your desired denoiser
bpy.context.scene.cycles.use_denoising = True
bpy.context.scene.cycles.denoiser = denoiser
bpy.context.scene.cycles.denoising_optix_input_passes = 'RGB_ALBEDO_NORMAL'

rendering_device_type = sys.argv[-4]  # Assumes the rendering device type is the fourth last argument
gpu_id = int(sys.argv[-3])  # Assumes the optix id is the third last argument
device_id_count = 0

for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    if device.type == rendering_device_type:
        if device_id_count == gpu_id:
            device.use = True
            print(f"using device: {device.name} id: {gpu_id} type: {device.type}")
        else:
            device.use = False
        device_id_count += 1
    else:
        device.use = False

# Get the .obj file path from command line arguments
frame_dir = sys.argv[-2] # Assumes the frame directory is the second last argument
output_image_path = sys.argv[-1]  # Assumes the output image path is the last argument

# Set scene frame to drive camera/animation based on folder name (e.g., 00012)
# frame_idx = int(os.path.basename(frame_dir))
# bpy.context.scene.frame_set(frame_idx)

# Print initial camera state before any object replacement
print("\n[Initial Camera State]")
camera = bpy.data.objects.get('Camera')
if camera:
    # Get evaluated (constraint-applied) transform
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera_eval = camera.evaluated_get(depsgraph)
    print(f"  Original Location: {camera.location}")
    print(f"  Original Rotation: {camera.rotation_euler}")
    print(f"  Evaluated Matrix Translation: {camera_eval.matrix_world.translation}")
    print(f"  Evaluated Matrix Rotation: {camera_eval.matrix_world.to_euler()}")
    if camera.constraints:
        for constraint in camera.constraints:
            print(f"  Constraint '{constraint.name}': Target = {constraint.target.name if constraint.target else 'None'}")
else:
    print("  No camera found!")


for file in os.listdir(frame_dir):
    if file.endswith(".obj"):
        obj_path = os.path.join(frame_dir, file)
        target_obj_name = file.split(".")[0]
        
        # Use Blender's built-in OBJ importer to preserve UVs and normals
        # This replaces the manual load_obj_raw + import_obj_no_transform approach
        bpy.ops.wm.obj_import(
            filepath=obj_path,
            forward_axis='X',
            up_axis='Z'
        )
        
        # The importer creates objects with a suffix; find the newly imported object
        imported_obj = None
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                imported_obj = obj
                break
        
        if imported_obj is None:
            print(f"Warning: Could not find imported object for {obj_path}")
            continue
        
        # Check if the target object exists in the scene
        if target_obj_name in bpy.data.objects:
            target_obj = bpy.data.objects[target_obj_name]

            # Transfer materials from the target object to the new object
            imported_obj.data.materials.clear()
            for mat in target_obj.data.materials:
                imported_obj.data.materials.append(mat)
            
            print(f"old object name: {target_obj.name}, imported object name: {imported_obj.name}")

            # Store reference to old object before any renaming
            old_obj_ref = target_obj
            
            # STEP 1: Update all constraints pointing to old object BEFORE deletion
            constraint_updated_count = 0
            for obj in bpy.data.objects:
                if obj.constraints:
                    for constraint in obj.constraints:
                        # Check if constraint targets the old object
                        if hasattr(constraint, 'target') and constraint.target == old_obj_ref:
                            print(f"[Constraint Update] '{obj.name}'.'{constraint.name}' from old object : {constraint.target.name} -> imported object : {imported_obj.name}")
                            constraint.target = imported_obj
                            constraint_updated_count += 1
                            print(f"    Updated constraint '{constraint.name}' on object '{obj.name}' to point to '{imported_obj.name}'")
                        
                        # Handle bone constraints with subtarget
                        if hasattr(constraint, 'subtarget') and constraint.subtarget == old_obj_ref.name:
                            constraint.subtarget = imported_obj.name
            
            print(f"[Constraint Summary] Updated {constraint_updated_count} constraint(s)")
            
            # STEP 2: Remove ALL old objects with this base name (including .001, .002, etc.)
            objects_to_remove = []
            for obj in bpy.data.objects:
                # Check if object name matches pattern BUT exclude the newly imported object
                if obj != imported_obj and (obj.name == target_obj_name or obj.name.startswith(target_obj_name + ".")):
                    objects_to_remove.append(obj)
                    print(f"[Cleanup] Marking '{obj.name}' for removal")
            
            # Remove all matching objects
            for obj in objects_to_remove:
                bpy.data.objects.remove(obj, do_unlink=True)
            
            # STEP 3: Now rename new object to target name (no conflicts anymore)
            print(f"[Rename] Imported object {imported_obj.name} renamed to '{target_obj_name}'")
            imported_obj.name = target_obj_name
            
            # CRITICAL: Set object origin to geometry center WITHOUT moving vertices
            # This makes Track To constraint follow the visual center instead of (0,0,0)
            if imported_obj.type == 'MESH' and imported_obj.data.vertices:
                # Calculate geometry center in world space
                mesh = imported_obj.data
                center_local = sum((v.co for v in mesh.vertices), start=bpy.context.scene.cursor.location * 0) / len(mesh.vertices)
                center_world = imported_obj.matrix_world @ center_local
                
                # Move vertices so they stay at same world position after origin change
                for v in mesh.vertices:
                    v.co -= center_local
                
                # Update object location to the geometry center
                imported_obj.location = center_world
                print(f"  [Origin Update] Object '{target_obj_name}' origin at: {imported_obj.location}")
            else:
                print(f"  [Origin Warning] Object '{target_obj_name}' has no mesh data")
        else:
            # No existing object, just rename
            imported_obj.name = target_obj_name
            print(f"[Object Add] No existing object named '{target_obj_name}'. Imported object named as is.")

# Force update the scene to re-evaluate all constraints (including camera Track To)
print("\n[Scene Update] Forcing dependency graph update...")
dg = bpy.context.evaluated_depsgraph_get()
dg.update()
bpy.context.view_layer.update()

# CRITICAL: Force scene frame update to trigger constraint evaluation
bpy.context.scene.frame_set(bpy.context.scene.frame_current)

## Print the constaint on the camera for debugging
print("\n[Final Camera State After Updates]")
camera = bpy.data.objects.get('Camera')
if camera and camera.constraints:
    for constraint in camera.constraints:
        print(f"  Constraint '{constraint.name}': Target = {constraint.target.name if constraint.target else 'None'}")
    # Get evaluated (constraint-applied) transform after updates
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera_eval = camera.evaluated_get(depsgraph)
    print(f"  Original Location: {camera.location}")
    print(f"  Original Rotation: {camera.rotation_euler}")
    print(f"  Evaluated Matrix Translation: {camera_eval.matrix_world.translation}")
    print(f"  Evaluated Matrix Rotation: {camera_eval.matrix_world.to_euler()}")
    
    # Also print target object location for comparison
    for constraint in camera.constraints:
        if hasattr(constraint, 'target') and constraint.target:
            target = constraint.target
            print(f"  Target '{target.name}' location: {target.location}")

# Render the scene to the specified file
bpy.context.scene.render.filepath = output_image_path
bpy.ops.render.render(write_still=True)
