import blenderproc as bproc
import bpy
import trimesh 

import os
import cv2
import math
import mathutils
import imageio
import argparse
import pathlib
import copy
import time
import shutil
import numpy as np
import json
import sys
from scipy import ndimage
from pyvirtualdisplay import Display

# blender path
# print(sys.exec_prefix)
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(script_dir)
sys.path.append(parent_dir)
export_dir = os.path.join(script_dir, "export")
sys.path.append(export_dir)

from simnet.lib import camera
from simnet.lib.transform import Pose

from render.utils.blender import *
from render.utils.transform import *
from utils.motion_utils import *

from gpt_4.prompts.prompt_motion_planning import plan_motion
from gpt_4.prompts.prompt_code_generation import generate_code
from gpt_4.prompts.utils import load_response, extract_useful_object, extract_code


def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')
    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)
    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)
    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))
    return 1 / (2*radius)


def load_best_cam_pose(best_match_idx):
    cnos_cam_fpath_0 = "/hdd2/chenyang/spatialpin/results/render/cam_poses_level0.npy"
    cnos_cam_fpath_1 = "/hdd2/chenyang/spatialpin/results/render/cam_poses_level1.npy"
    cnos_cam_fpath_2 = "/hdd2/chenyang/spatialpin/results/render/cam_poses_level2.npy"
    cam_poses_0 = np.load(cnos_cam_fpath_0)
    cam_poses_1 = np.load(cnos_cam_fpath_1)
    cam_poses_2 = np.load(cnos_cam_fpath_2)
    cam_poses = np.concatenate((cam_poses_0, cam_poses_1), axis=0)

    cam_poses_list = []
    for idx, cam_pose in enumerate(cam_poses):
        cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
        cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
        cam_poses_list.append(cam_pose)
        
    return cam_poses_list[best_match_idx]


def render_views(cad_path, output_dir):
    # cad_path: The path of CAD model
    cam_poses_list = []
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)  # The path to save CAD templates
    normalize = True  # Whether to normalize CAD model or not
    colorize = False  # Whether to colorize CAD model or not
    base_color = 0.05  # The base color used in CAD model

    # set the cnos camera path
    cnos_cam_fpath_0 = "/hdd2/chenyang/spatialpin/results/render/cam_poses_level0.npy"
    cnos_cam_fpath_1 = "/hdd2/chenyang/spatialpin/results/render/cam_poses_level1.npy"
    cnos_cam_fpath_2 = "/hdd2/chenyang/spatialpin/results/render/cam_poses_level2.npy"
    bproc.init()

    # load cnos camera pose
    cam_poses_0 = np.load(cnos_cam_fpath_0)
    cam_poses_1 = np.load(cnos_cam_fpath_1)
    cam_poses_2 = np.load(cnos_cam_fpath_2)
    # print(cam_poses_0.shape)  # (42, 4, 4)
    # print(cam_poses_1.shape)  # (162, 4, 4)
    # print(cam_poses_2.shape)  # (642, 4, 4)
    # cam_poses = np.concatenate((cam_poses_0, cam_poses_1, cam_poses_2), axis=0)
    cam_poses = np.concatenate((cam_poses_0, cam_poses_1), axis=0)

    # calculating the scale of CAD model
    if normalize:
        scale = get_norm_info(cad_path)
    else:
        scale = 1

    for idx, cam_pose in enumerate(cam_poses):
        bproc.clean_up()

        # load object
        obj = bproc.loader.load_obj(cad_path)[0]
        obj.set_scale([scale, scale, scale])
        obj.set_cp("category_id", 1)

        # assigning material colors to untextured objects
        if colorize:
            color = [base_color, base_color, base_color, 0.]
            material = bproc.material.create('obj')
            material.set_principled_shader_value('Base Color', color)
            obj.set_material(0, material)

        # convert cnos camera poses to blender camera poses
        cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
        cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
        cam_poses_list.append(cam_pose)
        bproc.camera.add_camera_pose(cam_pose)
        
        # set light
        light_scale = 2
        light_energy = 1000
        light1 = bproc.types.Light()
        light1.set_type("POINT")
        light1.set_location([light_scale*cam_pose[:3, -1][0], light_scale*cam_pose[:3, -1][1], light_scale*cam_pose[:3, -1][2]])
        light1.set_energy(light_energy)

        bproc.renderer.set_max_amount_of_samples(50)
        # render the whole pipeline
        data = bproc.renderer.render()
        # render nocs
        data.update(bproc.renderer.render_nocs())
        
        # check save folder
        save_fpath = os.path.join(output_dir, "templates")
        if not os.path.exists(save_fpath):
            os.makedirs(save_fpath)

        # save rgb image
        color_bgr_0 = data["colors"][0]
        color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
        cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(idx)+'.png'), color_bgr_0)

        # save mask
        mask_0 = data["nocs"][0][..., -1]
        cv2.imwrite(os.path.join(save_fpath,'mask_'+str(idx)+'.png'), mask_0*255)
        
        # save nocs
        xyz_0 = 2*(data["nocs"][0][..., :3] - 0.5)
        np.save(os.path.join(save_fpath,'xyz_'+str(idx)+'.npy'), xyz_0.astype(np.float16))

    return cam_poses_list


def extract_main_contour(mask):
    """Extract the main contour from a mask image."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming the largest contour is the main object
    main_contour = max(contours, key=cv2.contourArea)
    return main_contour


# https://subscription.packtpub.com/book/data/9781789344912/10/ch10lvl1sec94/matching-contours
def compare_contours(contour1, contour2):
    """Compare two contours and return a similarity score."""
    return cv2.matchShapes(calculate_hu_moments(contour1), calculate_hu_moments(contour2), cv2.CONTOURS_MATCH_I1, 0)


def mask_size_score(mask1, mask2):
    """Lower values indicate greater similarity in mask sizes."""
    size1 = np.sum(normalize_contour(mask1, True))
    size2 = np.sum(normalize_contour(mask2, True))
    return 1 - min(size1, size2) / max(size1, size2)


def calculate_hu_moments(contour):
    # Calculate Hu moments of the contour
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    # Convert the Hu moments to a 1D array for easy comparison
    return -np.sign(hu_moments) * np.log10(np.abs(hu_moments))


# https://stackoverflow.com/questions/55529371/opencv-shape-matching-between-two-similar-shapes/55530040#55530040
def normalize_contour(img, resize=False):
    # Find contours on the 2D image
    contours = extract_main_contour(img)
    bounding_rect = cv2.boundingRect(contours)

    # Crop the image using the bounding rectangle
    img_cropped_bounding_rect = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
                                    bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]

    if resize:
        # Resize the cropped image
        new_height = int((1.0 * img.shape[0]) / img.shape[1] * 300.0)
        img_resized = cv2.resize(img_cropped_bounding_rect, (300, new_height))
        return img_resized
    else:
        return img_cropped_bounding_rect


def find_best_matching_view(gt_mask_path, masks_folder):
    """Find the mask in the folder that best matches the ground truth mask."""
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = normalize_contour(gt_mask)
    gt_contour = extract_main_contour(gt_mask)

    best_match_idx = -1
    min_match_score = float('inf')

    # List all mask files in the given folder
    mask_files = sorted([f for f in os.listdir(masks_folder) if f.startswith("mask") and f.endswith('.png')])

    for idx, mask_file in enumerate(mask_files):
        mask_path = os.path.join(masks_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = normalize_contour(mask)
        mask_contour = extract_main_contour(mask)

        match_score = compare_contours(gt_contour, mask_contour)
        size_score = mask_size_score(mask, gt_mask)
        print(mask_file, match_score + size_score, match_score, size_score)

        if (match_score + size_score) < min_match_score:
            min_match_score = (match_score + size_score)
            best_match_idx = idx

    # return int(mask_files[best_match_idx].split('_')[-1].split('.')[0])
    # return 34
    return 166


def calculate_blender_centers_3d(data_dir, output_dir, camera_matrix):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    centers_2d_output_dir = pathlib.Path(output_dir) / "centers_2d"

    obj_centers_3d_all_img = []
    img_centers_3d_all_img = []
    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        obj_centers_3d = []
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        color_path = img_full_path + '_color.png'
        depth_path = img_full_path + '_depth_DA.png'
        _, depth_vis, _ = load_img_NOCS(color_path, depth_path)

        # Segmented mask 2D center to 3D center back projection
        filename = str(centers_2d_output_dir / f"centers_2d_{i}.json")
        with open(filename, 'r') as json_file:
            centers_2d = json.load(json_file)
        centers_2d = [[int(row), int(col)] for row, col in centers_2d]
        
        for obj_idx, center_2d in enumerate(centers_2d):
            center_3d, _ = backproject(depth_vis, camera_matrix, create_masks([center_2d], depth_vis))
            obj_centers_3d.append([center_3d[0][0], center_3d[0][2], -center_3d[0][1]])

        img_centers_3d, _ = backproject(depth_vis, camera_matrix, create_masks(None, depth_vis, img_center=True))
        img_centers_3d_all_img.append([img_centers_3d[0][0], img_centers_3d[0][2], -img_centers_3d[0][1]])
        obj_centers_3d_all_img.append(obj_centers_3d)

    return obj_centers_3d_all_img, img_centers_3d_all_img


def calculate_object_scales(data_dir, output_dir, camera_matrix):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    obj_binary_mask_dir = pathlib.Path(output_dir) / "binary_mask"

    obj_scales_all_img = []
    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        obj_scales = []
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        color_path = img_full_path + '_color.png'
        depth_path = img_full_path + '_depth_DA.png'
        _, depth_vis, _ = load_img_NOCS(color_path, depth_path)

        file_count = sum(1 for f in obj_binary_mask_dir.iterdir() if f.is_file() and f.name.startswith(f"masks_{i}_obj_"))
        for obj_idx in range(file_count):
            obj_mask_path = obj_binary_mask_dir / f"masks_{i}_obj_{obj_idx}.png"
            points_3d, idxs = backproject(depth_vis, camera_matrix, cv2.imread(str(obj_mask_path), cv2.IMREAD_GRAYSCALE))
            # points_2d = np.vstack((idxs[0], idxs[1])).T
            obj_scales.append(calculate_scale(points_3d, cv2.imread(str(obj_mask_path), cv2.IMREAD_GRAYSCALE)))

        obj_scales_all_img.append(normalize_scale(obj_scales))
    
    return obj_scales_all_img


def raycast_centers_3d_plane(bg_coords_3d_list, cam_coords_3d_list, loc_list, kz_init=2):
    refined_bg_coords_3d_list = []
    for obj_idx, bg_coords_3d in enumerate(bg_coords_3d_list):
        cam_coords_3d = cam_coords_3d_list[obj_idx]
        direction_vector = (bg_coords_3d - cam_coords_3d) / (bg_coords_3d - cam_coords_3d).length
        refined_bg_coords_3d = np.array(list(cam_coords_3d)) + np.array(list(loc_list[obj_idx][1] * kz_init * direction_vector))
        refined_bg_coords_3d_list.append(refined_bg_coords_3d)
        
    return refined_bg_coords_3d_list


def raycast_centers_3d_pinhole(bg_coords_3d_list, cam_obj_loc, loc_list, tolerance=1e-2):
    refined_bg_coords_3d_list = []
    
    for obj_idx, bg_coords_3d in enumerate(bg_coords_3d_list):
        direction_vector = (bg_coords_3d - cam_obj_loc) / np.linalg.norm(bg_coords_3d - cam_obj_loc)
        scale_factor = 0
        
        # Initial calculation for refined_bg_coords_3d
        refined_bg_coords_3d = np.array(list(cam_obj_loc)) + scale_factor * direction_vector
        
        # Iterate until the depth (y-coordinate) of refined_bg_coords_3d equals loc_list[obj_idx][1]
        while not np.isclose(refined_bg_coords_3d[1], loc_list[obj_idx][1], atol=tolerance):
            # Calculate the adjustment needed in the scale_factor
            depth_difference = loc_list[obj_idx][1] - refined_bg_coords_3d[1]
            scale_adjustment = depth_difference / direction_vector[1]  # Adjust based on the y component of the direction vector
            
            # Update the scale_factor and refined_bg_coords_3d
            scale_factor += scale_adjustment
            refined_bg_coords_3d = np.array(list(cam_obj_loc)) + scale_factor * direction_vector
        
        refined_bg_coords_3d_list.append(refined_bg_coords_3d)
        
    return refined_bg_coords_3d_list


def refine_obj_3d_coords_plane(data_dir, output_dir, obj_centers_3d_all_img, img_centers_3d_all_img, input_camera_matrix, res_xy, cuda_use_list=[0, 1, 2, 3]):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    centers_2d_output_dir = pathlib.Path(output_dir) / "lang_sam" / "centers_2d"
    blender_mask_output_dir = pathlib.Path(output_dir) / "blender_mask"
    lama_inpainted_bg_mask_dir = pathlib.Path(output_dir) / "lama" / "inpainted_bg_mask"
    blender_mask_output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blender_mask_output_dir / "default.blend"))

    refined_obj_centers_3d_all_img = []
    for i, img_path in enumerate(data_path):
        # if i != 0: continue
        filename = str(centers_2d_output_dir / f"centers_2d_{i}.json")
        with open(filename, 'r') as json_file:
            centers_2d = json.load(json_file)
        centers_2d = [[int(row), int(col)] for row, col in centers_2d]

        camera_matrix = input_camera_matrix.copy()
        loc_list = obj_centers_3d_all_img[i]
        blender_center_3d = img_centers_3d_all_img[i]
        
        with Display():
            # Initialize blender
            init_blender(blender_mask_output_dir)
            blender_use_gpu(cuda_use_list)

            # Set camera
            camera_matrix, cam_obj, direction_vector, rotation_euler = set_camera(input_camera_matrix, camera_matrix, blender_center_3d, depth_plane=True, cam_type="orthographic")

            # Set background
            bg_plane = set_bg_plane(lama_inpainted_bg_mask_dir, i, direction_vector, rotation_euler, blender_center_3d, black=True, cam_type="orthographic")

            # Set camera plane
            cam_plane = set_bg_plane(lama_inpainted_bg_mask_dir, i, direction_vector, rotation_euler, blender_center_3d, cam_obj=cam_obj, black=True, cam_type="orthographic")

            # Calculate refined objects 3d centers
            plane_width, plane_height = bg_plane.dimensions.x, bg_plane.dimensions.y
            bg_coords_3d_list, cam_coords_3d_list = [], []
            for y, x in centers_2d:
                x_norm = x / res_xy[0]
                y_norm = y / res_xy[1]
                x_local = (x_norm - 0.5) * plane_width
                y_local = -(y_norm - 0.5) * plane_height
                z_local = 0  
                local_point = mathutils.Vector((x_local, y_local, z_local))
                bg_world_point = bg_plane.matrix_world @ local_point
                cam_world_point = cam_plane.matrix_world @ local_point
                bg_coords_3d_list.append(bg_world_point)
                cam_coords_3d_list.append(cam_world_point)
            
            # kz_init finetuned
            refined_loc_list = raycast_centers_3d_plane(bg_coords_3d_list, cam_coords_3d_list, loc_list, kz_init=3.75)  # 2.5
 
        refined_obj_centers_3d_all_img.append(refined_loc_list)
        # print(refined_loc_list)
        # print()

    return refined_obj_centers_3d_all_img


def refine_obj_3d_coords_pinhole(data_dir, output_dir, obj_centers_3d_all_img, obj_scales_all_img, img_centers_3d_all_img, input_camera_matrix, res_xy, render=True, cuda_use_list=[0, 1, 2, 3]):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    centers_2d_output_dir = pathlib.Path(output_dir) / "lang_sam" / "centers_2d"
    obj_binary_mask_dir = pathlib.Path(output_dir) / "lang_sam" / "binary_mask"
    blender_mask_output_dir = pathlib.Path(output_dir) / "blender_mask"
    lama_inpainted_bg_mask_dir = pathlib.Path(output_dir) / "lama" / "inpainted_bg_mask"
    blender_mask_output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blender_mask_output_dir / "default.blend"))

    refined_obj_centers_3d_all_img, refined_obj_scales_all_img = [], []
    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        color_path = img_full_path + '_color.png'
        cad_path = pathlib.Path(output_dir) / "one2345_plus" / str(i)
        file_count = sum(1 for f in cad_path.iterdir() if f.is_file() and f.suffix == '.glb')
        object_paths = [str(cad_path / f"{obj_idx}.glb") for obj_idx in range(file_count)]

        filename = str(centers_2d_output_dir / f"centers_2d_{i}.json")
        with open(filename, 'r') as json_file:
            centers_2d = json.load(json_file)
        centers_2d = [[int(row), int(col)] for row, col in centers_2d]

        camera_matrix = input_camera_matrix.copy()
        loc_list = obj_centers_3d_all_img[i]
        scale_list = obj_scales_all_img[i]
        blender_center_3d = img_centers_3d_all_img[i]
        
        refined_scale_list = []
        for obj_idx in range(len(loc_list)):
            with Display():
                # Initialize blender
                init_blender(blender_mask_output_dir)
                blender_use_gpu(cuda_use_list)

                # Set camera
                camera_matrix, cam_obj, direction_vector, rotation_euler = set_camera(input_camera_matrix, camera_matrix, blender_center_3d)

                # Set background
                bg_plane = set_bg_plane(lama_inpainted_bg_mask_dir, i, direction_vector, rotation_euler, blender_center_3d, black=True)

                # Calculate refined objects 3d centers
                plane_width, plane_height = bg_plane.dimensions.x, bg_plane.dimensions.y
                bg_coords_3d_list = []
                for y, x in centers_2d:
                    x_norm = x / res_xy[0]
                    y_norm = y / res_xy[1]
                    x_local = (x_norm - 0.5) * plane_width
                    y_local = -(y_norm - 0.5) * plane_height
                    z_local = 0  
                    local_point = mathutils.Vector((x_local, y_local, z_local))
                    world_point = bg_plane.matrix_world @ local_point
                    bg_coords_3d_list.append(world_point)
                
                refined_loc_list = raycast_centers_3d_pinhole(bg_coords_3d_list, cam_obj.location, loc_list)

                # Import GLB object
                glb_path = object_paths[obj_idx]
                bpy.ops.import_scene.gltf(filepath=glb_path)
                obj = bpy.context.selected_objects[-1]  # Assuming the imported object is selected
                obj.scale = (scale_list[obj_idx], scale_list[obj_idx], scale_list[obj_idx])
                obj.location = refined_loc_list[obj_idx]

                # Create a new white material
                white_mat = bpy.data.materials.new(name="WhiteMaterial")
                white_mat.use_nodes = True
                nodes = white_mat.node_tree.nodes
                nodes.clear()

                # Add a Diffuse BSDF node (you can also use an Emission node if you don't want shading)
                diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')
                diffuse_node.inputs[0].default_value = (1, 1, 1, 1)  # Set color to white (R, G, B, A)

                # Add a Material Output node and link it to the Diffuse BSDF node
                output_node = nodes.new(type='ShaderNodeOutputMaterial')
                links = white_mat.node_tree.links
                links.new(diffuse_node.outputs[0], output_node.inputs[0])

                # Apply the white material to the imported object
                if obj.data.materials:
                    # If the object has any existing materials, replace the first one
                    obj.data.materials[0] = white_mat
                else:
                    # If the object has no materials, append the new one
                    obj.data.materials.append(white_mat)
                bpy.context.view_layer.update()

                # Set rendering parameters
                bpy.context.scene.render.resolution_x = res_xy[0] * 1
                bpy.context.scene.render.resolution_y = res_xy[1] * 1
                bpy.context.scene.render.resolution_percentage = 100
                bpy.context.scene.render.image_settings.file_format = 'PNG'
                blender_mask_path = str(blender_mask_output_dir / f"masks_{i}_obj_{obj_idx}.png")
                bpy.context.scene.render.filepath = blender_mask_path

                # Render the scene
                if render:
                    bpy.ops.render.render(write_still=True)

            # Post-process to 1-channel black and white mask
            if render:
                mask = cv2.imread(blender_mask_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                _, blender_obj_mask = cv2.threshold(mask, np.max(mask) // 2, 255, cv2.THRESH_BINARY)
                cv2.imwrite(blender_mask_path, blender_obj_mask)

            obj_mask_path = obj_binary_mask_dir / f"masks_{i}_obj_{obj_idx}.png"
            gt_obj_mask = cv2.imread(str(obj_mask_path), cv2.IMREAD_GRAYSCALE)
            blender_obj_mask = cv2.imread(blender_mask_path, cv2.IMREAD_GRAYSCALE)
            refined_scale_list.append(calculate_contour_ratio(gt_obj_mask, blender_obj_mask) * scale_list[obj_idx])

        refined_obj_centers_3d_all_img.append(refined_loc_list)
        refined_obj_scales_all_img.append(refined_scale_list)
        # print(refined_loc_list)
        # print(refined_scale_list)
        # print()

    return refined_obj_centers_3d_all_img, refined_obj_scales_all_img


def generate_inital_scene(data_dir, output_dir, obj_centers_3d_all_img, obj_scales_all_img, img_centers_3d_all_img, input_camera_matrix, res_xy, render=True, cuda_use_list=[0, 1, 2, 3]):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    centers_2d_output_dir = pathlib.Path(output_dir) / "lang_sam" / "centers_2d"
    blender_scene_output_dir = pathlib.Path(output_dir) / "blender_scene"
    lama_inpainted_bg_mask_dir = pathlib.Path(output_dir) / "lama" / "inpainted_bg_mask"
    blender_scene_output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blender_scene_output_dir / "default.blend"))

    abs_pose_opt_all_img, transformed_axes_opt_all_img, pc_list_all_img, gpt4_info_all_img = [], [], [], []
    for i, img_path in enumerate(data_path):
        # if i != 0: continue
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        color_path = img_full_path + '_color.png'
        cad_path = pathlib.Path(output_dir) / "one2345_plus" / str(i)
        file_count = sum(1 for f in cad_path.iterdir() if f.is_file() and f.suffix == '.glb')
        object_paths = [str(cad_path / f"{obj_idx}.glb") for obj_idx in range(file_count)]

        filename = str(centers_2d_output_dir / f"centers_2d_{i}.json")
        with open(filename, 'r') as json_file:
            centers_2d = json.load(json_file)
        centers_2d = [[int(row), int(col)] for row, col in centers_2d]

        camera_matrix = input_camera_matrix.copy()
        loc_list = obj_centers_3d_all_img[i]
        scale_list = obj_scales_all_img[i]
        blender_center_3d = img_centers_3d_all_img[i]
        print(loc_list)
        print(scale_list)
        print()
        
        with Display():
            # Initialize blender
            init_blender(blender_scene_output_dir)
            blender_use_gpu(cuda_use_list)

            # Set camera
            camera_matrix, cam_obj, direction_vector, rotation_euler = set_camera(input_camera_matrix, camera_matrix, blender_center_3d)
            print(camera_matrix)

            # Set background
            bg_plane = set_bg_plane(lama_inpainted_bg_mask_dir, i, direction_vector, rotation_euler, blender_center_3d)

            # Import GLB objects
            abs_pose_opt, transformed_axes_opt, pc_list = [], [], []
            for obj_idx, glb_path in enumerate(object_paths):
                bpy.ops.import_scene.gltf(filepath=glb_path)
                obj = bpy.context.selected_objects[-1]  # Assuming the imported object is selected
                obj.scale = (scale_list[obj_idx], scale_list[obj_idx], scale_list[obj_idx])
                obj.location = loc_list[obj_idx]
                bpy.context.view_layer.update()
                points_3d = extract_obj_pc(obj)
                obj_axes, camera_T_object, scale_matrix = find_oriented_bounding_box(points_3d, scale_list[obj_idx], loc_list[obj_idx])

                # Visualize axes
                # for j, direction in enumerate(['X', 'Y', 'Z']):
                #     end_point = loc_list[obj_idx] + (obj_axes[:, j + 1] - loc_list[obj_idx]) / np.linalg.norm(obj_axes[:, j + 1] - loc_list[obj_idx]) * 0.085
                #     # X red, Y green, Z blue
                #     color = (1, 0, 0, 1) if direction == 'X' else ((0, 1, 0, 1) if direction == 'Y' else (0, 0, 1, 1))
                #     create_thick_line(f"{obj.name}_{direction}_Axis", loc_list[obj_idx], end_point, 0.005, color)
                
                # Objects have zero rotation initially
                camera_T_object[:3, :3] = np.eye(3)
                obj_pose = Pose(camera_T_object)
                obj_pose.scale_matrix = scale_matrix
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_3d)
                abs_pose_opt.append(obj_pose)
                transformed_axes_opt.append(obj_axes)
                pc_list.append(pcd)
            
            abs_pose_opt_all_img.append(abs_pose_opt)
            transformed_axes_opt_all_img.append(transformed_axes_opt)
            pc_list_all_img.append(pc_list)

            # Information for ChatGPT-4
            results_3d = print_local_axis_center(abs_pose_opt, transformed_axes_opt, _CAMERA.K_matrix[:3,:3], print_only_2d=False)
            info_obj_size = print_obj_size(abs_pose_opt, pc_list)
            info_spatial_relation = print_spatial_relation(abs_pose_opt, print_only_closest=False)
            gpt4_info_all_img.append(results_3d + info_obj_size + info_spatial_relation)

            if render:
                # Set light
                set_light(camera_matrix)

                # Set rendering parameters
                bpy.context.scene.render.resolution_x = res_xy[0] * 1
                bpy.context.scene.render.resolution_y = res_xy[1] * 1
                bpy.context.scene.render.resolution_percentage = 100
                bpy.context.scene.render.image_settings.file_format = 'PNG'
                bpy.context.scene.render.filepath = str(blender_scene_output_dir / f"scene_{i}.png")

                # Render the scene
                bpy.ops.render.render(write_still=True)
    
    return abs_pose_opt_all_img, transformed_axes_opt_all_img, pc_list_all_img, gpt4_info_all_img


def motion_planning_gpt4(data_dir, output_dir, gpt4_info_all_img, temperature_dict, model_dict, start_over=False):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    blender_scene_dir = pathlib.Path(output_dir) / "blender_scene"

    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        conversation_hist = []
        blender_scene_path = blender_scene_dir / f"scene_{i}.png"
        obj_finding_path = pathlib.Path(output_dir) / "gpt4_response" / "prompts/obj_finding" / str(i)
        task_proposal_path = pathlib.Path(output_dir) / "gpt4_response" / "prompts/task_proposal" / str(i)
        motion_planning_path = pathlib.Path(output_dir) / "gpt4_response" / "prompts/motion_planning" / str(i)
        gpt4_info = gpt4_info_all_img[i]
        obj_text_list = extract_useful_object("obj_finding", obj_finding_path)

        blender_scene_path = pathlib.Path(output_dir) / "blender_scene" / f"scene_{i}.png"
        binding_box_vis = cv2.imread(str(blender_scene_path))
        binding_box_vis = cv2.cvtColor(binding_box_vis, cv2.COLOR_BGR2RGB)

        subdirs = [d for d in os.listdir(obj_finding_path) if os.path.isdir(obj_finding_path / d)]
        subdirs.sort()
        latest_subdir = max(subdirs, key=lambda d: (obj_finding_path / d).stat().st_mtime)
        json_file_path = obj_finding_path / latest_subdir / "obj_finding.json"
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
            user = json_data["user"]
            res = json_data["res"]
        conversation_hist.append([user, res])

        for obj_idx, _ in enumerate(obj_text_list):
            # if obj_idx != 0: continue
            subdirs = [d for d in os.listdir(task_proposal_path) if os.path.isdir(task_proposal_path / d)]
            obj_subdir = [d for d in subdirs if d.startswith(f"obj_{obj_idx}")][0]
            json_file_path = task_proposal_path / obj_subdir / "task_proposal.json"
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                user = json_data["user"]
                res = json_data["res"]
            conversation_hist.append([user, res])
            conversation_hist.append(["", gpt4_info])

            if start_over:
                plan_motion(None, binding_box_vis, obj_idx, motion_planning_path, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
                time.sleep(30)
            else: 
                plan_motion(None, binding_box_vis, obj_idx, motion_planning_path, existing_response=load_response("motion_planning", motion_planning_path, file_idx=obj_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
            
            conversation_hist = conversation_hist[:-2]


def code_generation_gpt4(data_dir, output_dir, temperature_dict, model_dict, start_over=False):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    generated_codes_all_img, video_filenames_all_img = [], []

    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        blender_video_output_dir = pathlib.Path(output_dir) / "blender_video" / str(i)
        blender_video_output_dir.mkdir(parents=True, exist_ok=True)

        conversation_hist = []
        obj_finding_path = pathlib.Path(output_dir) / "gpt4_response" / "prompts/obj_finding" / str(i)
        motion_planning_path = pathlib.Path(output_dir) / "gpt4_response" / "prompts/motion_planning" / str(i)
        code_generation_path = pathlib.Path(output_dir) / "gpt4_response" / "prompts/code_generation" / str(i)
        obj_text_list = extract_useful_object("obj_finding", obj_finding_path)

        for obj_idx, _ in enumerate(obj_text_list):
            conversation_hist = []
            subdirs = [d for d in os.listdir(motion_planning_path) if os.path.isdir(motion_planning_path / d)]
            obj_subdir = [d for d in subdirs if d.startswith(f"obj_{obj_idx}")][0]
            json_file_path = motion_planning_path / obj_subdir / "motion_planning.json"
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                user = json_data["user"]
                res = json_data["res"]
            conversation_hist.append([user, res])

            if start_over:
                generate_code(None, None, obj_idx, code_generation_path, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
                time.sleep(20)
            else: 
                generate_code(None, None, obj_idx, code_generation_path, existing_response=load_response("code_generation", code_generation_path, file_idx=obj_idx), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)

        generated_codes, video_filenames = extract_code("code_generation", code_generation_path, blender_video_output_dir, i)
        generated_codes_all_img.append(generated_codes)
        video_filenames_all_img.append(video_filenames)
    
    return generated_codes_all_img, video_filenames_all_img


def generate_video(data_dir, output_dir, obj_centers_3d_all_img, obj_scales_all_img, img_centers_3d_all_img, abs_pose_opt_all_img, transformed_axes_opt_all_img, pc_list_all_img, generated_codes_all_img, video_filenames_all_img, input_camera_matrix, intrinsics, res_xy, cuda_use_list=[0, 1, 2, 3], planned=True):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    blender_scene_dir = pathlib.Path(output_dir) / "blender_scene"
    lama_inpainted_bg_mask_dir = pathlib.Path(output_dir) / "lama" / "inpainted_bg_mask"
    hand_path = str(pathlib.Path(output_dir) / "one2345_plus" / "human_hand.glb")
 
    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        blender_video_output_dir = pathlib.Path(output_dir) / "blender_video" / str(i)
        cad_path = pathlib.Path(output_dir) / "one2345_plus" / str(i)
        file_count = sum(1 for f in cad_path.iterdir() if f.is_file() and f.suffix == '.glb')
        object_paths = [str(cad_path / f"{obj_idx}.glb") for obj_idx in range(file_count)]

        camera_matrix = input_camera_matrix.copy()
        loc_list = obj_centers_3d_all_img[i]
        scale_list = obj_scales_all_img[i]
        blender_center_3d = img_centers_3d_all_img[i]

        abs_pose_opt = abs_pose_opt_all_img[i]
        transformed_axes_opt = transformed_axes_opt_all_img[i]
        pc_list = pc_list_all_img[i]
        generated_codes = generated_codes_all_img[i]
        video_filenames = video_filenames_all_img[i]

        for j, _ in enumerate(object_paths):
            # if j != 0: continue
            planned_motion_tuples = [item for item in generated_codes if item[0] == j]
            video_names = [filename for filename in video_filenames if filename.startswith(f"video_{i}_obj_{j}")]

            for k in range(len(video_names)):
                # if k != 0: continue
                print(f"--------------Generating video for image {i} object {j} task {k}--------------")
                mani_obj_idx = planned_motion_tuples[k][0]
                motion_list = planned_motion_tuples[k][1]
                blender_video_output_path = blender_video_output_dir / video_names[k]
                trajectory_json_output_path = str(blender_video_output_path.parent) + '/' + blender_video_output_path.name.replace('video', 'trajectory', 1).rsplit(".", 1)[0] + ".json"
                if planned is False:
                    motion_steps = generate_seq_motion_physical_rrt(mani_obj_idx, motion_list, abs_pose_opt, transformed_axes_opt, pc_list, intrinsics)

                with Display():
                    # Initialize blender
                    init_blender(blender_scene_dir)
                    blender_use_gpu(cuda_use_list)

                    # Import GLB objects
                    imported_objects = []
                    for obj_idx, glb_path in enumerate(object_paths):
                        bpy.ops.import_scene.gltf(filepath=glb_path)
                        # in bpy.context.selected_objects, first is [bpy.data.objects['world.00x'], second is bpy.data.objects['geometry_0.00x']]
                        # if change both, results are accumulative in Blender
                        obj = bpy.context.selected_objects[-1]  # Assuming the imported object is selected
                        obj.scale = (scale_list[obj_idx], scale_list[obj_idx], scale_list[obj_idx])
                        obj.location = loc_list[obj_idx]
                        obj.rotation_mode = 'XYZ'  # default rotation mode for .glb is in WXYZ
                        bpy.context.view_layer.update()
                        imported_objects.append(bpy.context.selected_objects[-1])
                    
                    # Set camera
                    camera_matrix, cam_obj, direction_vector, rotation_euler = set_camera(input_camera_matrix, camera_matrix, blender_center_3d)

                    # Set background
                    bg_plane = set_bg_plane(lama_inpainted_bg_mask_dir, i, direction_vector, rotation_euler, blender_center_3d)

                    # Set light
                    set_light(camera_matrix)
                    
                    # Start animation
                    bpy.context.view_layer.update()
                    obj_to_animate = imported_objects[j]

                    # Import and set GLB hand
                    bpy.ops.import_scene.gltf(filepath=hand_path)
                    hand_obj = bpy.context.selected_objects[0]

                    hand_obj.rotation_mode = 'XYZ'
                    hand_obj.rotation_euler = [math.radians(angle) for angle in [0, 120, 110]]  # finetuned
                    obj_to_animate_pts = np.asarray(pc_list[j].points)
                    # obj_to_animate_size = calculate_obj_size(abs_pose_opt, pc_list)[j]
                    # obj_to_animate_width = obj_to_animate_size[0] / 100
                    # hand_obj.location = [loc_list[j][0] + obj_to_animate_width/2, loc_list[j][1], loc_list[j][2]]
                    # hand_obj.location = obj_to_animate_pts[np.argmax(obj_to_animate_pts[:, 0])]  # point with the largest x
                    hand_obj.location = find_surface_point(obj_to_animate_pts, loc_list[j])
                    hand_scale = 0.0005  # finetuned, 0.0005
                    hand_obj.scale = (hand_scale, hand_scale, hand_scale)  

                    hand_obj.parent = obj_to_animate
                    hand_obj.matrix_parent_inverse = obj_to_animate.matrix_world.inverted()

                    # Create a new material or use the existing one
                    if not hand_obj.children[0].children[0].data.materials:
                        mat = bpy.data.materials.new(name="Hand_Material")
                        hand_obj.children[0].children[0].data.materials.append(mat)
                    else:
                        mat = hand_obj.children[0].children[0].data.materials[0]

                    # Set the material's base color
                    rgb_color = (255 / 255.0, 228 / 255.0, 201 / 255.0)
                    mat.use_nodes = True
                    bsdf = mat.node_tree.nodes.get('Principled BSDF')
                    if bsdf is not None:
                        bsdf.inputs['Base Color'].default_value = (*rgb_color, 1)  # RGB + Alpha
                    bpy.context.view_layer.update()
                    bsdf.inputs['Roughness'].default_value = 0.5  # Adjust for shininess
                    bsdf.inputs['Metallic'].default_value = 0  # Non-metallic by default

                    # Set the animation frame range
                    if planned is False:
                        pose_trajectory, scale_trajectory = [], []
                    else:
                        with open(trajectory_json_output_path, 'r') as file:
                            json_data = json.load(file)
                            pose_trajectory = np.array(json_data['pose_trajectory'])
                            motion_steps = pose_trajectory
                            # motion_steps = [1, 2] # debug

                    bpy.context.scene.frame_start = 1
                    bpy.context.scene.frame_end = len(motion_steps)

                    # Iterate over the motion steps and set keyframes
                    for frame_num, motion_step in enumerate(motion_steps, start=1):
                        # Set the frame number
                        bpy.context.scene.frame_set(frame_num)

                        # Decompose the 4x4 transformation matrix into location and rotation
                        if planned is False:
                            loc, rot = motion_step.camera_T_object[:3, 3], motion_step.camera_T_object[:3, :3] 
                            pose_trajectory.append(motion_step.camera_T_object.tolist())
                            scale_trajectory.append(motion_step.scale_matrix.tolist())
                        else:
                            loc, rot = pose_trajectory[frame_num - 1][:3, 3], pose_trajectory[frame_num - 1][:3, :3]

                        # Apply the location and rotation to the object
                        obj_to_animate.location = loc
                        rotation_euler = mathutils.Matrix(rot).to_euler('XYZ')
                        rotation_euler = [angle for angle in rotation_euler]
                        obj_to_animate.rotation_euler = rotation_euler

                        # Insert keyframes for location and rotation
                        obj_to_animate.keyframe_insert(data_path="location", frame=frame_num)
                        obj_to_animate.keyframe_insert(data_path="rotation_euler", frame=frame_num)
                        bpy.context.view_layer.update()

                    # Set rendering parameters
                    bpy.context.scene.render.resolution_x = res_xy[0] * 4
                    bpy.context.scene.render.resolution_y = res_xy[1] * 4
                    bpy.context.scene.render.resolution_percentage = 100
                    current_fps = bpy.context.scene.render.fps
                    bpy.context.scene.render.fps = 8  # Set FPS (int), default 24

                    # Set output settings
                    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
                    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
                    bpy.context.scene.render.ffmpeg.codec = 'H264'
                    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
                    bpy.context.scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
                    bpy.context.scene.render.filepath = str(blender_video_output_path)

                    # Render the animation
                    bpy.ops.render.render(animation=True)
                
                if planned is False:
                    with open(trajectory_json_output_path, 'w') as json_file:
                        json.dump({'pose_trajectory': pose_trajectory, 'scale_trajectory': scale_trajectory}, json_file)
    


# blenderproc run main/render_animate.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--app_output', default='main', type=str)
    app_group.add_argument('--result_name', default='/hdd2/chenyang/spatialpin/results', type=str)
    app_group.add_argument('--data_dir', default='/hdd2/chenyang/spatialpin/test_data', type=str)

    hparams = parser.parse_args()
    output_dir = hparams.result_name
    
    gpu_device = 'cuda:0'
    _CAMERA = camera.NOCS_Real()

    # NOCS camera
    # _CAMERA.K_matrix[0, 2] = 320
    # _CAMERA.K_matrix[1, 2] = 240

    # iPhone 12 Pro Max camera
    # _CAMERA.K_matrix[0, 0] = 452.37
    # _CAMERA.K_matrix[1, 1] = 453.49
    # _CAMERA.K_matrix[0, 2] = 320
    # _CAMERA.K_matrix[1, 2] = 240
    # print(_CAMERA.K_matrix[:3,:3])

    # Communicating to ChatGPT-4 API
    temperature_dict = {
      "obj_finding": 0.2,
      "task_proposal": 0.7,
      "spatial_understanding": 0.2,
      "motion_planning": 0.2,
      "code_generation": 0.2
    }
    # GPT-4 1106-preview is GPT-4 Turbo (https://openai.com/pricing)
    model_dict = {
      "obj_finding": "gpt-4-vision-preview",
      "task_proposal": "gpt-4-vision-preview",
      "spatial_understanding": "gpt-4-vision-preview",
      "motion_planning": "gpt-4-vision-preview",
      "code_generation": "gpt-4-vision-preview"
    }

    lang_sam_output_dir = pathlib.Path(output_dir) / "lang_sam"
    obj_centers_3d_all_img, img_centers_3d_all_img = calculate_blender_centers_3d(hparams.data_dir, str(lang_sam_output_dir), _CAMERA.K_matrix[:3,:3])
    obj_scales_all_img = calculate_object_scales(hparams.data_dir, str(lang_sam_output_dir), _CAMERA.K_matrix[:3,:3])

    # cad_path = pathlib.Path(output_dir) / "one2345_plus" / str(0) / "0.glb"
    # render_output_dir = pathlib.Path(output_dir) / "render"
    # cam_poses_list = render_views(str(cad_path), str(render_output_dir))
    
    # gt_mask_path = pathlib.Path(output_dir) / "lang_sam" / "binary_mask" / "masks_0_obj_0.png"
    # templates_folder = pathlib.Path(output_dir) / "render" / "templates"
    # best_match_idx = find_best_matching_view(str(gt_mask_path), str(templates_folder))
    # best_camera_T_object = load_best_cam_pose(best_match_idx)
    # print(best_camera_T_object)
    # camera_matrix = np.array([[ 0.95105479,  0.16246467, -0.26286883, -0.52573764],
    #                           [-0.30902232,  0.50000532, -0.80901167, -1.61802328],
    #                           [ 0.,          0.85064676,  0.52573766,  1.05147529],
    #                           [ 0.,          0.,          0.,          1.        ]])

    # This camera pos is the object's camera pose used during reconstruction by One-2-3-45++
    camera_matrix = np.array([[ 0.99996682, -0.00411941,  0.00702707,  0.01405415],
                              [ 0.00814551,  0.50571068, -0.86266468, -1.7253294 ],
                              [ 0.,          0.8626933,   0.50572746,  1.01145494],
                              [ 0.,          0.,          0.,          1.        ]])
    
    cuda_use_list = [0, 1]
    # obj_centers_3d_all_img = refine_obj_3d_coords_plane(hparams.data_dir, output_dir, obj_centers_3d_all_img, img_centers_3d_all_img, camera_matrix, (640, 480), cuda_use_list=cuda_use_list)
    obj_centers_3d_all_img, obj_scales_all_img = refine_obj_3d_coords_pinhole(hparams.data_dir, output_dir, obj_centers_3d_all_img, obj_scales_all_img, img_centers_3d_all_img, camera_matrix, (640, 480), render=False, cuda_use_list=cuda_use_list)
    abs_pose_opt_all_img, transformed_axes_opt_all_img, pc_list_all_img, gpt4_info_all_img = generate_inital_scene(hparams.data_dir, output_dir, obj_centers_3d_all_img, obj_scales_all_img, img_centers_3d_all_img, camera_matrix, (640, 480), render=True, cuda_use_list=cuda_use_list)
    
    motion_planning_gpt4(hparams.data_dir, output_dir, gpt4_info_all_img, temperature_dict, model_dict, start_over=True)

    generated_codes_all_img, video_filenames_all_img = code_generation_gpt4(hparams.data_dir, output_dir, temperature_dict, model_dict, start_over=True)

    generate_video(hparams.data_dir, output_dir, obj_centers_3d_all_img, obj_scales_all_img, img_centers_3d_all_img, abs_pose_opt_all_img, transformed_axes_opt_all_img, pc_list_all_img, generated_codes_all_img, video_filenames_all_img, camera_matrix, _CAMERA.K_matrix[:3,:3], (640, 480), cuda_use_list=cuda_use_list)