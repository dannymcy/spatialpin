import os
import numpy as np
import pathlib
import bpy
import mathutils
import math
import open3d as o3d
from scipy.stats import zscore


def extract_obj_pc(obj):
    # Extract world coordinates of vertices
    world_vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    world_vertices = np.array(world_vertices)
    return world_vertices


def transform_coordinates_3d(coordinates, RT):
    """
    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates


def find_convex_hull_vol_ratio(points_3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    hull_mesh, _ = pcd.compute_convex_hull()
    hull_volume = hull_mesh.get_volume()

    # obb = pcd.get_minimal_oriented_bounding_box()
    # obb_volume = obb.volume()
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb_volume = aabb.volume()

    # ratio = hull_volume / obb_volume
    # print(hull_volume, obb_volume, ratio)
    ratio = hull_volume / aabb_volume
    # print(hull_volume, aabb_volume, ratio)
    
    return ratio


def find_oriented_bounding_box(points_3d, scale, center_3d):
    ratio = find_convex_hull_vol_ratio(points_3d)

    # Remove outliers, ratio threshold is finetuned
    threshold = 3 if ratio > 0.45 else 1
    z_scores = np.abs(zscore(points_3d, axis=0))
    filtered_indices = (z_scores < threshold).all(axis=1)
    points_3d = points_3d[filtered_indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    # https://www.open3d.org/docs/latest/python_api/open3d.geometry.OrientedBoundingBox.html
    obb = pcd.get_minimal_oriented_bounding_box()

    # Create the 6D pose matrix (4x4 transformation matrix)
    pose_matrix, scale_matrix = np.eye(4), np.eye(4)
    pose_matrix[:3, :3] = obb.R
    pose_matrix[:3, 3] = obb.center
    scale_matrix[0, 0], scale_matrix[1, 1], scale_matrix[2, 2] = scale, scale, scale

    xyz_axis = 0.3 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).transpose()
    sRT = pose_matrix @ scale_matrix
    transformed_axes = transform_coordinates_3d(xyz_axis, sRT)
    for i in range(1, 4):
        transformed_axes[:, i] = transformed_axes[:, i] - transformed_axes[:, 0]  # Subtract the origin
        transformed_axes[:, i] = transformed_axes[:, i] / np.linalg.norm(transformed_axes[:, i])  # Normalize

    # Identify the indices of the maximum absolute X, Y, and Z
    x_index = np.argmax(np.abs(transformed_axes[0, 1:])) + 1
    y_index = np.argmax(np.abs(transformed_axes[1, 1:])) + 1
    z_index = np.argmax(np.abs(transformed_axes[2, 1:])) + 1

    # Reorder the rotation matrix columns based on the changes in transformed_axes
    new_R = pose_matrix[:3, :3].copy()  # Extract and copy the rotation matrix
    new_R[:, 0] = pose_matrix[:3, x_index - 1]
    new_R[:, 1] = pose_matrix[:3, y_index - 1]
    new_R[:, 2] = pose_matrix[:3, z_index - 1]

    # Ensure the dominant component in each axis column is positive
    for i in range(3):
        if new_R[i, i] < 0:
            new_R[:, i] *= -1

    # Update the pose matrix with the reordered and flipped rotation matrix
    pose_matrix[:3, :3] = new_R

    # Recalculate the transformed_axes using the updated pose_matrix
    sRT_updated = pose_matrix @ scale_matrix
    transformed_axes_updated = transform_coordinates_3d(xyz_axis, sRT_updated)

    # Update to projected 3D centers
    pose_matrix[:3, 3] = center_3d
    for i in range(1, 4):
        transformed_axes_updated[:, i] = transformed_axes_updated[:, i] - transformed_axes_updated[:, 0] + center_3d
    transformed_axes_updated[:, 0] = center_3d

    # Now, transformed_axes_updated contains the verified axes
    return transformed_axes_updated, pose_matrix, scale_matrix


def create_thick_line(name, start, end, thickness, color):
    """Create a thicker line (cylinder) between two points."""

    # Calculate the midpoint and vector between start and end
    mid = (mathutils.Vector(start) + mathutils.Vector(end)) / 2
    vec = mathutils.Vector(end) - mathutils.Vector(start)
    length = vec.length

    # Create a cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=thickness, depth=length, location=mid)

    # Get the cylinder object and adjust its orientation to align with the start and end points
    cylinder = bpy.context.object
    cylinder.name = name
    vec.normalize()
    up = mathutils.Vector((0, 0, 1))
    angle = math.acos(vec.dot(up))
    axis = up.cross(vec)
    cylinder.rotation_mode = 'AXIS_ANGLE'
    cylinder.rotation_axis_angle = [angle, *axis]

    # Create a material with the specified color and assign it to the cylinder
    mat = bpy.data.materials.new(name=name + "_Material")
    mat.diffuse_color = color  # RGBA
    mat.use_nodes = False  # Optional: disable for simple color assignment
    cylinder.data.materials.append(mat)


def init_blender(blender_output_dir):
    # Clear existing mesh objects
    bpy.ops.wm.open_mainfile(filepath=str(blender_output_dir / "default.blend"))
    bpy.ops.preferences.addon_enable(module="io_import_images_as_planes")
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.delete()


def set_light(camera_matrix, light_type='POINT'):
    # bg_light_data = bpy.data.lights.new(name="Light", type=light_type)  # 'POINT', 'AREA', 'SUN', 'Spot'
    # bg_light_obj = bpy.data.objects.new(name="Light", object_data=bg_light_data)
    # bpy.context.collection.objects.link(bg_light_obj)
    # bg_light_obj.location = [-0.0101, 1.6313, 1.68921]
    # bg_light_data.energy = 250
  
    obj_light_data = bpy.data.lights.new(name="Light", type=light_type)  # 'POINT', 'AREA', 'SUN', 'Spot'
    obj_light_obj = bpy.data.objects.new(name="Light", object_data=obj_light_data)
    bpy.context.collection.objects.link(obj_light_obj)
    obj_light_scale = 1
    obj_light_energy = 100
    obj_light_obj.location = [obj_light_scale*camera_matrix[:3, -1][0], obj_light_scale*camera_matrix[:3, -1][1], obj_light_scale*camera_matrix[:3, -1][2]]
    obj_light_data.energy = obj_light_energy

    if light_type == 'AREA':
        obj_light_data.size = 2


def set_camera(input_camera_matrix, camera_matrix, blender_center_3d, depth_plane=False, cam_type="perspective"):
    # In Blender, the camera looks along its local -Z axis
    if depth_plane == False:
        direction_vector = -input_camera_matrix[:3, :3] @ np.array([0, 0, 1])
    else:
        depth_plane_matrix = np.array([[ 1.,  0,   0],
                                       [ 0,   0,  -1.],
                                       [ 0,   1.,  0]])
        direction_vector = depth_plane_matrix @ np.array([0, 0, 1])

    # Normalize the direction vector
    direction_vector /= np.linalg.norm(direction_vector)

    # Distance to move the camera back
    distance = 0.0  # Adjust this value as needed

    # Calculate new camera position
    camera_matrix[:3, 3] = input_camera_matrix[:3, 3] + blender_center_3d
    camera_matrix[:3, 3] = camera_matrix[:3, 3] - distance * direction_vector

    # Convert the rotation matrix to Euler angles (in radians), X, Y, Z in Blender
    if depth_plane == False:
        rotation_euler = mathutils.Matrix(camera_matrix[:3, :3]).to_euler('XYZ')
    else:
        rotation_euler = mathutils.Matrix(depth_plane_matrix).to_euler('XYZ')
    rotation_euler = [angle for angle in rotation_euler]

    # Create a new camera object
    if cam_type == "perspective":
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        cam_obj.location = camera_matrix[:3, 3]
        cam_obj.rotation_euler = rotation_euler
        cam_data.lens = 80
        cam_data.sensor_width = 36.0
    elif cam_type == "orthographic":
        cam_data = bpy.data.cameras.new("OrthoCamera")
        cam_obj = bpy.data.objects.new("OrthoCamera", cam_data)
        cam_obj.location = camera_matrix[:3, 3]
        cam_obj.rotation_euler = rotation_euler
        cam_data.type = 'ORTHO'
        cam_data.ortho_scale = 640/480 - 0.005
        cam_data.sensor_width = 36.0
    cam_data.dof.use_dof = False
    # print(cam_obj.location, cam_obj.rotation_euler)

    # Set this camera as the active camera for the scene
    bpy.context.scene.camera = cam_obj
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.view_layer.objects.active = cam_obj

    return camera_matrix, cam_obj, direction_vector, rotation_euler


def set_bg_plane(lama_inpainted_bg_mask_dir, i, direction_vector, rotation_euler, blender_center_3d, cam_obj=None, cam_type="perspective", black=False):
    # Add the background
    lama_inpainted_bg_mask_path = str(lama_inpainted_bg_mask_dir / f"image{i}_mask000.png")
    if not black:
        background_img = bpy.data.images.load(lama_inpainted_bg_mask_path)
    # Import the background image as a plane
    bpy.ops.import_image.to_plane(files=[{'name': os.path.basename(lama_inpainted_bg_mask_path)}], directory=os.path.dirname(lama_inpainted_bg_mask_path))
    # Get the imported plane and adjust its position and orientation
    bg_plane = bpy.context.selected_objects[-1]  # Assuming the imported plane is selected
    
    if cam_type == "perspective":
        moving_dis = 0.95  # finetuned
    elif cam_type == "orthographic":
        moving_dis = 1.0

    if cam_obj is not None:
        bg_plane.location = cam_obj.location
    else:
        bg_plane.location = blender_center_3d + moving_dis * direction_vector
    bg_plane.rotation_euler = rotation_euler  # Adjust as needed to orient towards the camera
    # print(bg_plane.location)
    # print(bg_plane.rotation_euler)

    if not black:
        bg_plane.active_material.shadow_method = 'NONE'  # 'OPAQUE'
        bg_plane.cycles.is_shadow_catcher = False  # True
    else:
        bg_plane.active_material.shadow_method = 'NONE'
        bg_plane.cycles.is_shadow_catcher = False        

    # Get or create a material for the background plane
    if bg_plane.data.materials:
        mat = bg_plane.data.materials[0]  # Use the existing material
    else:
        mat = bpy.data.materials.new(name="BackgroundMaterial")  # Create a new material
        bg_plane.data.materials.append(mat)  # Append the new material to the plane

    # Get the nodes of the material
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()  # Clear existing nodes

    # Add necessary nodes
    if not black:
        # shader_node = nodes.new(type='ShaderNodeBsdfDiffuse')
        shader_node = nodes.new(type='ShaderNodeEmission')
    else:
        shader_node = nodes.new(type='ShaderNodeEmission')
    output_node = nodes.new(type='ShaderNodeOutputMaterial')

    if not black:
        texture_node = nodes.new(type='ShaderNodeTexImage')
        texture_node.image = background_img  # Set the image
        # Link texture to Shader
        links = mat.node_tree.links
        links.new(texture_node.outputs[0], shader_node.inputs[0])
    else:
        # Set the shader color to black
        shader_node.inputs['Color'].default_value = (0, 0, 0, 1)  # RGBA, here A is 1 for no transparency

    # Link the nodes
    links = mat.node_tree.links
    links.new(shader_node.outputs[0], output_node.inputs[0])  # Shader to Output

    # Set blend method to 'CLIP' and shadow mode to 'NONE' for transparency
    mat.blend_method = 'CLIP'
    if not black:
        mat.shadow_method = 'NONE'  # 'OPAQUE'
    else:
        mat.shadow_method = 'NONE'

    # Use Z Offset to ensure the plane is rendered at the very back
    mat.use_screen_refraction = True
    mat.refraction_depth = -0.1  # Negative value to ensure it's rendered at the back
    bpy.context.view_layer.update()

    return bg_plane


# https://area-51.blog/2021/05/26/getting-blender-to-use-cuda-in-a-render-farm/
def blender_use_gpu(cuda_use_list):
    # bpy.context.scene.render.engine = 'CYCLES'
    
    # # Set the device_type
    # bpy.context.preferences.addons[
    #     "cycles"
    # ].preferences.compute_device_type = "CUDA"

    # # Set the device and feature set
    # bpy.context.scene.cycles.device = "GPU"

    # # get_devices() to let Blender detects GPU device
    # bpy.context.preferences.addons["cycles"].preferences.get_devices()
    # # print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    
    # for i, d in enumerate(bpy.context.preferences.addons["cycles"].preferences.devices):
    #     d["use"] = 0
    #     if (d["name"][:6] == 'NVIDIA' and i in cuda_use_list) or d["name"][:6] != 'NVIDIA':
    #         d["use"] = 1
    #     # print(i, d["name"], d["use"])
    return