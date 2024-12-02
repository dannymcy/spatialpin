import numpy as np
import random
import cv2
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d, CubicSpline
from scipy.stats import zscore
# import bezier
from sklearn.cluster import KMeans

from rrt_src.rrt.rrt_star import RRTStar
from rrt_src.search_space.search_space import SearchSpace
from rrt_src.utilities.plotting import Plot


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


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


def return_major_directions():
    down = np.array([0, 0, -1])
    up = np.array([0, 0, 1])
    right = np.array([1, 0, 0])
    left = np.array([-1, 0, 0])
    back = np.array([0, 1, 0])
    front = np.array([0, -1, 0])
    front_right = (right + front) / 2
    front_left = (left + front) / 2
    back_right = (right + back) / 2
    back_left = (left + back) / 2
    front_right_normalized = front_right / np.linalg.norm(front_right)
    front_left_normalized = front_left / np.linalg.norm(front_left)
    back_right_normalized = back_right / np.linalg.norm(back_right)
    back_left_normalized = back_left / np.linalg.norm(back_left)
    
    directions = {
        "down": down,
        "up": up,
        "right": right,
        "left": left,
        "back": back,
        "front": front,
        "front_right": front_right_normalized,
        "front_left": front_left_normalized,
        "back_right": back_right_normalized,
        "back_left": back_left_normalized
    }
    return directions


def print_local_axis_center(abs_pose_opt, transformed_axes_opt, intrinsics, print_only_2d=False):
    directions = return_major_directions()
    result = ""

    for i, abs_pose in enumerate(abs_pose_opt):
        center = abs_pose.camera_T_object[:3, 3].reshape(3,)
        center_image_plane = calculate_2d_projections(center.reshape(3, 1), intrinsics).reshape(2,)

        local_axis_x = transformed_axes_opt[i][:, 1] - transformed_axes_opt[i][:, 0]
        local_axis_x_normalized = local_axis_x / np.linalg.norm(local_axis_x)
        axis_x_image_plane = calculate_2d_projections(local_axis_x_normalized.reshape(3, 1), intrinsics)
        axis_x_image_plane = (axis_x_image_plane / np.linalg.norm(axis_x_image_plane)).reshape(2,)

        local_axis_y = transformed_axes_opt[i][:, 2] - transformed_axes_opt[i][:, 0]
        local_axis_y_normalized = local_axis_y / np.linalg.norm(local_axis_y)
        axis_y_image_plane = calculate_2d_projections(local_axis_y_normalized.reshape(3, 1), intrinsics)
        axis_y_image_plane = (axis_y_image_plane / np.linalg.norm(axis_y_image_plane)).reshape(2,)

        local_axis_z = transformed_axes_opt[i][:, 3] - transformed_axes_opt[i][:, 0]
        local_axis_z_normalized = local_axis_z / np.linalg.norm(local_axis_z)
        axis_z_image_plane = calculate_2d_projections(local_axis_z_normalized.reshape(3, 1), intrinsics)
        axis_z_image_plane = (axis_z_image_plane / np.linalg.norm(axis_z_image_plane)).reshape(2,)
    
        # Function to find the closest major direction to a given axis
        def closest_direction(axis_normalized):
            return min(directions.keys(), key=lambda k: np.linalg.norm(directions[k] - axis_normalized))

        # Determine the closest major directions for each local axis
        closest_direction_x = closest_direction(local_axis_x_normalized)
        closest_direction_y = closest_direction(local_axis_y_normalized)
        closest_direction_z = closest_direction(local_axis_z_normalized)

        if print_only_2d:
            print(f"Obj {i}: 2D center in image plane: {center_image_plane}")
            # print(f"2D local xyz axes in image plane: {axis_x_image_plane}, {axis_y_image_plane}, {axis_z_image_plane}")
            result += f"Obj {i}: {center_image_plane}\n"
        else:
            center_3d = [round(coord, 2) * 100 for coord in center]
            axis_x_3d = [round(coord, 4) for coord in local_axis_x_normalized]
            axis_y_3d = [round(coord, 4) for coord in local_axis_y_normalized]
            axis_z_3d = [round(coord, 4) for coord in local_axis_z_normalized]

            print(f"Obj {i} spatial context:\n"
                f"3D center: {center_3d} cm\n"
                f"Local x-axis (towards '{closest_direction_x}'): {axis_x_3d}\n"
                f"Local y-axis (towards '{closest_direction_y}'): {axis_y_3d}\n"
                f"Local z-axis (towards '{closest_direction_z}'): {axis_z_3d}")
            
            result += (f"Obj {i} spatial context:\n"
                    f"3D center: {center_3d} cm\n"
                    f"Local x-axis (towards '{closest_direction_x}'): {axis_x_3d}\n"
                    f"Local y-axis (towards '{closest_direction_y}'): {axis_y_3d}\n"
                    f"Local z-axis (towards '{closest_direction_z}'): {axis_z_3d}\n")
    
    print("")    
    return result


def print_obj_size(abs_pose_opt, pc_list):
    boxes_3d = generate_obj_3d_boxes(abs_pose_opt, pc_list, 0.1)
    result = ""

    for i, box in enumerate(boxes_3d): 
        min_point, max_point = box[:3], box[3:]
        sizes_cm = [calculate_physical_distance_in_cm(min_point[dim], max_point[dim]) for dim in range(3)]
        sizes_str = " x ".join([f"{size:.2f} cm" for size in sizes_cm])

        print(f"Obj {i} size: {sizes_str} (WxDxH)")
        result += f"Obj {i} size: {sizes_str} (WxDxH)\n"
    
    print("")
    return result


def calculate_obj_size(abs_pose_opt, pc_list):
    boxes_3d = generate_obj_3d_boxes(abs_pose_opt, pc_list, 0.1)
    obj_sizes = []
    for i, box in enumerate(boxes_3d): 
        min_point, max_point = box[:3], box[3:]

        # Calculate the size in each dimension
        width_cm = calculate_physical_distance_in_cm(min_point[0], max_point[0])
        height_cm = calculate_physical_distance_in_cm(min_point[1], max_point[1])
        depth_cm = calculate_physical_distance_in_cm(min_point[2], max_point[2])
        obj_sizes.append([width_cm, height_cm, depth_cm])
    
    return obj_sizes


def print_spatial_relation(abs_pose_opt, print_only_closest=False):
    directions = return_major_directions()
    result = ""

    all_obj_relation = []
    for i, abs_pose in enumerate(abs_pose_opt):
        obj_relation = []
        for j, target_pose in enumerate(abs_pose_opt):
            if i != j:
                distance_cm = calculate_physical_distance_in_cm(abs_pose.camera_T_object[:3, 3].reshape(3,), target_pose.camera_T_object[:3, 3].reshape(3,))
                direc_axis_normalized = compute_vec_diff_normalized(abs_pose, target_pose, 1)

                # Determine the closest major direction
                closest_direction = min(directions.keys(), key=lambda k: np.linalg.norm(directions[k] - direc_axis_normalized))
                obj_relation.append((j, distance_cm, closest_direction))
        all_obj_relation.append(obj_relation)

    if print_only_closest == False:
        # Process to find the closest object in each major direction
        closest_obj_in_each_direction = []
        for relations in all_obj_relation:
            closest = {}
            for relation in relations:
                target_idx, distance, direction = relation
                if direction not in closest or closest[direction][1] > distance:
                    closest[direction] = (target_idx, distance)
            closest_obj_in_each_direction.append(closest)
        
        # Print the spatial relationships
        for i, relations in enumerate(closest_obj_in_each_direction):
            spatial_str = "; ".join([f"{direction}: Obj {target_idx} ({distance:.2f} cm)" for direction, (target_idx, distance) in relations.items()])
            print(f"Obj {i} - Closest per direction: {spatial_str}")
            result += f"Obj {i} - Closest per direction: {spatial_str}\n"
    else:
        # Process to find the overall closest object
        closest_obj_overall = []
        for relations in all_obj_relation:
            if relations:
                closest = min(relations, key=lambda x: x[1])
                closest_obj_overall.append(closest)
        
        # Print the closest object for each object
        for i, (target_idx, distance, direction) in enumerate(closest_obj_overall):
            print(f"Obj {i} - Closest: Obj {target_idx} ({direction}, {distance:.2f} cm)")
            result += f"Obj {i} - Closest: Obj {target_idx} ({direction}, {distance:.2f} cm)\n"

    print("")
    return result


def calculate_transformed_axes(abs_pose_opt):
    xyz_axis = 0.3*np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).transpose()
    sRT = abs_pose_opt.camera_T_object @ abs_pose_opt.scale_matrix
    transformed_axes = transform_coordinates_3d(xyz_axis, sRT)
    return transformed_axes


def compute_vec_diff_normalized(abs_pose, target_pose, dis_ang):
    # Determine the direction of vector
    local_axis = target_pose.camera_T_object[:3, 3] - abs_pose.camera_T_object[:3, 3]

    # Normalize the direction vector
    local_axis = local_axis * np.sign(dis_ang)
    local_axis_normalized = local_axis / np.linalg.norm(local_axis)
    return local_axis_normalized


def compute_local_axis(xyz, transformed_axes):
    # Determine the local axis for rotation based on the 'xyz' parameter.
    # Positive rotation is counterclockwise
    if xyz == "local_x":
        local_axis = transformed_axes[:, 1] - transformed_axes[:, 0]
    elif xyz == "local_y":
        local_axis = transformed_axes[:, 2] - transformed_axes[:, 0]
    elif xyz == "local_z":
        local_axis = transformed_axes[:, 3] - transformed_axes[:, 0]
    elif xyz == "world_x":
        local_axis = np.array([1, 0, 0])
    elif xyz == "world_y":
        local_axis = np.array([0, 1, 0])
    elif xyz == "world_z":
        local_axis = np.array([0, 0, 1])

    # Normalize the local axis to get a unit vector.
    local_axis_normalized = local_axis / np.linalg.norm(local_axis)
    return local_axis_normalized


def rotate_around_axis(angle, xyz, transformed_axes, abs_pose):
    local_axis_normalized = compute_local_axis(xyz, transformed_axes)
    
    new_pose_list = []
    accumulated_angle = 0 
    interpolated_angles = interpolate_rotation(angle)
    for interpolated_angle in interpolated_angles: 
        # Create a rotation matrix for the given angle around the normalized axis.
        accumulated_angle += interpolated_angle
        rotation_matrix = R.from_rotvec(np.radians(accumulated_angle) * local_axis_normalized).as_matrix()

        # Apply the rotation to the existing rotation part of the pose.
        new_rotation = rotation_matrix @ abs_pose.camera_T_object[:3, :3]

        # Construct the new pose with the updated rotation and the original translation.
        new_pose = copy.deepcopy(abs_pose)
        new_pose.camera_T_object[:3, :3] = new_rotation
        new_pose_list.append(new_pose)
    
    return new_pose_list


def rotate_wref_target_obj(angle, xyz, abs_pose, target_pose, obj_axes, target_axes):
    # Manipulating object local local x and z-axes
    obj_local_x_axis = obj_axes[:, 1] - obj_axes[:, 0]
    obj_local_x_axis_normalized = obj_local_x_axis / np.linalg.norm(obj_local_x_axis)
    obj_local_z_axis = obj_axes[:, 3] - obj_axes[:, 0]
    obj_local_z_axis_normalized = obj_local_z_axis / np.linalg.norm(obj_local_z_axis)

    # Target object local x and z-axes
    target_local_x_axis = target_axes[:, 1] - target_axes[:, 0]
    target_local_x_axis_normalized = target_local_x_axis / np.linalg.norm(target_local_x_axis)
    target_local_z_axis = target_axes[:, 3] - target_axes[:, 0]
    target_local_z_axis_normalized = target_local_z_axis / np.linalg.norm(target_local_z_axis)

    # Directional vector from the manipulating object to the target object
    if angle == "fixed_towards" or angle >= 0: 
        direc_axis_normalized = compute_vec_diff_normalized(abs_pose, target_pose, 1)
    elif angle == "fixed_back" or angle < 0: 
        direc_axis_normalized = compute_vec_diff_normalized(abs_pose, target_pose, -1)
    original_direc_axis_normalized = compute_vec_diff_normalized(abs_pose, target_pose, 1)
    
    # Calculate rotational axis
    if xyz == "pitch":
        rot_axis = np.cross(target_local_z_axis_normalized, direc_axis_normalized)
        rot_axis_normalized = rot_axis / np.linalg.norm(rot_axis)
        orientation_cross = np.cross(obj_local_z_axis_normalized, direc_axis_normalized)
        if angle == "fixed_towards": 
            angle = np.degrees(np.arccos(np.dot(obj_local_z_axis_normalized, direc_axis_normalized)))
            if angle > 90: angle = 180 - angle
        elif angle == "fixed_back": 
            angle = np.degrees(np.arccos(np.dot(obj_local_z_axis_normalized, direc_axis_normalized)))
            if angle > 90: angle = 180 - angle
            angle = -angle

    elif xyz == "yaw":
        pitch_rot_axis = np.cross(target_local_z_axis_normalized, original_direc_axis_normalized)
        rot_axis = np.cross(pitch_rot_axis, direc_axis_normalized)
        rot_axis_normalized = rot_axis / np.linalg.norm(rot_axis)
        orientation_cross = np.cross(obj_local_z_axis_normalized, direc_axis_normalized)
        if angle == "fixed_towards": 
            angle = np.degrees(np.arccos(np.dot(obj_local_x_axis_normalized, direc_axis_normalized)))
            if angle > 90: angle = 180 - angle
        elif angle == "fixed_back": 
            angle = np.degrees(np.arccos(np.dot(obj_local_x_axis_normalized, direc_axis_normalized)))
            if angle > 90: angle = 180 - angle
            angle = -angle

    elif xyz == "roll":
        rot_axis_normalized, orientation_cross = direc_axis_normalized, direc_axis_normalized
        if angle == "fixed_towards": 
            angle = np.degrees(np.arccos(np.dot(obj_local_z_axis_normalized, target_local_z_axis_normalized)))
            if angle < 90: angle = 180 - angle
            angle = 360
        elif angle == "fixed_back": 
            angle = np.degrees(np.arccos(np.dot(obj_local_z_axis_normalized, target_local_z_axis_normalized)))
            if angle < 90: angle = 180 - angle
            angle = -angle
            angle = -360
            
    # Determine the object's orientation relative to the directional vector
    sign_ = 1 if np.dot(orientation_cross, rot_axis_normalized) > 0 else -1

    new_pose_list = []
    accumulated_angle = 0
    interpolated_angles = interpolate_rotation(angle)
    for interpolated_angle in interpolated_angles: 
        # Create a rotation matrix for the given angle around the normalized axis.
        accumulated_angle += interpolated_angle
        rotation_matrix = R.from_rotvec(sign_ * np.radians(np.abs(accumulated_angle)) * rot_axis_normalized).as_matrix()

        # Apply the rotation to the existing rotation part of the pose.
        new_rotation = rotation_matrix @ abs_pose.camera_T_object[:3, :3]

        # Construct the new pose with the updated rotation and the original translation.
        new_pose = copy.deepcopy(abs_pose)
        new_pose.camera_T_object[:3, :3] = new_rotation
        new_pose_list.append(new_pose)
    
    return new_pose_list


def calculate_new_3d_coord_physical(physical_movement_cm, abs_pose, axis_normalized, intrinsics, scaling_coefficient=1.0):
    """
    Calculate the new 3D coordinate for a physical movement in the object's local space in a 3D scene.

    Parameters:
    - physical_movement_cm: Physical movement in centimeters along the object's local axis.
    - abs_pose: The absolute pose matrix of the object, which includes rotation and translation.
    - axis_normalized: Normalized direction axis to apply the translation.
    - intrinsics: The camera calibration matrix including intrinsic parameters.
    - scaling coefficient: The coefficient to scale the distance.

    Returns:
    - New 3D coordinate.
    """    
    # Decompose the abs_pose matrix into rotation (R) and translation (t)
    R = abs_pose.camera_T_object[:3, :3]
    t = abs_pose.camera_T_object[:3, 3]
    
    # Convert the physical movement in cm to meters (if your pose is in meters)
    movement_meters = np.abs(physical_movement_cm) * scaling_coefficient / 100.0
    
    # Movement is in the direction of the normalized axis
    local_movement_vector = axis_normalized * movement_meters
    
    # Transform the local movement to the world coordinate system using the rotation part of the abs_pose.
    # Don't need the below line, everything is already defined in the world coordinate system.
    # world_movement_vector = R @ local_movement_vector
    world_movement_vector = local_movement_vector
    
    # Translate the object's position by the world movement vector
    # The new position of the object in the camera coordinate system after applying the physical movement.
    new_world_position = t + world_movement_vector

    return new_world_position


def calculate_physical_distance_in_cm(coord1, coord2, scaling_coefficient=1.0):
    """
    Calculate the physical distance between two 3D coordinates, taking into account a scaling factor to convert the distance to centimeters.

    Parameters:
    - coord1: The first 3D coordinate as a numpy array or list.
    - coord2: The second 3D coordinate as a numpy array or list.
    - scaling coefficient: The coefficient to scale the distance.

    Returns:
    - Physical distance in centimeters after applying the scaling factor.
    """
    # Ensure the coordinates are numpy arrays for vector operations
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    
    # Calculate the Euclidean distance between the two points in meters
    distance_meters = np.linalg.norm(coord1 - coord2)
    
    # Apply the scaling factor to convert the distance to centimeters
    distance_cm = distance_meters * 100.0 / scaling_coefficient
    return distance_cm


def translate_along_axis(distance, xyz, transformed_axes, abs_pose, idx_tuple, obj_sizes, intrinsics, grounding=False):
    if distance == 0 and grounding == False:
        return copy.deepcopy(abs_pose)

    new_pose = copy.deepcopy(abs_pose)
    if distance != 0:
        local_axis_normalized = compute_local_axis(xyz, transformed_axes) * np.sign(distance)
        new_pose.camera_T_object[:3, 3] = calculate_new_3d_coord_physical(distance, abs_pose, local_axis_normalized, intrinsics)

    obj_idx, target_obj_idx = idx_tuple[0], idx_tuple[1]
    obj_size, target_obj_size = obj_sizes[obj_idx], obj_sizes[target_obj_idx]
    
    if grounding and obj_idx != target_obj_idx:
        grounding_distance = -(target_obj_size[2] - obj_size[2]) / 2
        local_axis_normalized = compute_local_axis("local_z", calculate_transformed_axes(new_pose)) * np.sign(grounding_distance)
        new_pose.camera_T_object[:3, 3] = calculate_new_3d_coord_physical(grounding_distance, new_pose, local_axis_normalized, intrinsics)

    return new_pose


def translate_along_vec_diff(distance, abs_pose, target_pose, idx_tuple, obj_sizes, intrinsics):
    if distance == 0:
        return copy.deepcopy(abs_pose)

    obj_idx, ref_obj_1_idx, ref_obj_2_idx = idx_tuple[0], idx_tuple[1], idx_tuple[2]
    obj_size, ref_obj_1_size, ref_obj_2_size = obj_sizes[obj_idx], obj_sizes[ref_obj_1_idx], obj_sizes[ref_obj_2_idx]

    new_abs_pose, new_target_pose = copy.deepcopy(abs_pose), copy.deepcopy(target_pose)
    if obj_idx == ref_obj_1_idx:
        grounding_distance = -(ref_obj_2_size[2] - obj_size[2]) / 2
        local_axis_normalized = compute_local_axis("local_z", calculate_transformed_axes(target_pose)) * np.sign(grounding_distance)
        new_target_pose.camera_T_object[:3, 3] = calculate_new_3d_coord_physical(grounding_distance, target_pose, local_axis_normalized, intrinsics)
    elif obj_idx == ref_obj_2_idx:
        grounding_distance = -(ref_obj_1_size[2] - obj_size[2]) / 2
        local_axis_normalized = compute_local_axis("local_z", calculate_transformed_axes(abs_pose)) * np.sign(grounding_distance)
        new_abs_pose.camera_T_object[:3, 3] = calculate_new_3d_coord_physical(grounding_distance, abs_pose, local_axis_normalized, intrinsics)
    else: 
        grounding_distance_1 = -(ref_obj_1_size[2] - obj_size[2]) / 2
        grounding_distance_2 = -(ref_obj_2_size[2] - obj_size[2]) / 2
        local_axis_normalized_1 = compute_local_axis("local_z", calculate_transformed_axes(abs_pose)) * np.sign(grounding_distance_1)
        local_axis_normalized_2 = compute_local_axis("local_z", calculate_transformed_axes(target_pose)) * np.sign(grounding_distance_2)
        new_abs_pose.camera_T_object[:3, 3] = calculate_new_3d_coord_physical(grounding_distance_1, abs_pose, local_axis_normalized_1, intrinsics)
        new_target_pose.camera_T_object[:3, 3] = calculate_new_3d_coord_physical(grounding_distance_2, target_pose, local_axis_normalized_2, intrinsics) 

    local_axis_normalized = compute_vec_diff_normalized(new_abs_pose, new_target_pose, distance)
    new_pose = copy.deepcopy(abs_pose)
    new_pose.camera_T_object[:3, 3] = calculate_new_3d_coord_physical(distance, abs_pose, local_axis_normalized, intrinsics)
    return new_pose


def generate_seq_motion_physical_rrt(obj_idx, motion_list, abs_pose_opt, transformed_axes_opt, pc_list, intrinsics):
    # obj_idx: Index of the manipulating object.
    # motion_list: List of motions, each a tuple defining a specific action:
    # - 'rotate_self': Axial rotation of the object around its own local axis.
    #   Tuple structure: ("rotate_self", angle_degrees, axis_type, axis_obj_idx)
    #   - angle_degrees: Rotation angle in degrees.
    #   - axis_type: Axis of obj_idx for rotation ('local_x', 'local_y', 'local_z').
    #   - axis_obj_idx: Always obj_idx, indicating self-rotation.
    #
    # - 'rotate_wref': Rotation of the object with reference to a target object.
    #   Tuple structure: ("rotate_wref", angle_degrees, axis_type, target_obj_idx)
    #   - angle_degrees: Rotation angle in degrees, or 'fixed_towards'/'fixed_back'.
    #   - axis_type: Axis for rotation ('pitch', 'yaw', 'roll').
    #   - target_obj_idx: Index of the reference object for rotation.
    #
    # - 'translate_tar_obj': Translation towards a goal relative to a target object.
    #   Tuple structure: ("translate_tar_obj", translation_vector, target_obj_idx)
    #   - translation_vector: Distances [dx, dy, dz] to move along each target axis.
    #   - target_obj_idx: Index of the target object for relative translation.
    #
    # - 'translate_direc_axis': Translation between two reference objects.
    #   Tuple structure: ("translate_direc_axis", distance, ref_obj_1_idx, ref_obj_2_idx)
    #   - distance: Distance in cm to travel from ref_obj_1_idx to ref_obj_2_idx.
    #   - ref_obj_1_idx: Index of the starting reference object.
    #   - ref_obj_2_idx: Index of the destination reference object.
    #
    # Sequential motions in motion_list are accumulative.
    # Examples of motion tuples:
    # - ("rotate_self", 90, "local_x", axis_obj_idx)
    # - ("rotate_wref", 90, "pitch", target_obj_idx)
    # - ("translate_tar_obj", [2, 1, 0], target_obj_idx)
    # - ("translate_direc_axis", 3, ref_obj_1_idx, ref_obj_2_idx)
    
    # seq_motion is a list of pose object
    seq_motion = []
    motion_step_0 = copy.deepcopy(abs_pose_opt[obj_idx])
    seq_motion.append(motion_step_0)

    for i, motion_tup in enumerate(motion_list):
        if motion_tup[0] == "rotate_self":
            new_motion_step = rotate_around_axis(motion_tup[1], motion_tup[2], transformed_axes_opt[motion_tup[3]], seq_motion[-1])
            if isinstance(new_motion_step, list):
                seq_motion += new_motion_step
            else:
                seq_motion.append(new_motion_step)
        
        elif motion_tup[0] == "rotate_wref":
            new_motion_step = rotate_wref_target_obj(motion_tup[1], motion_tup[2], seq_motion[-1], abs_pose_opt[motion_tup[3]], transformed_axes_opt[obj_idx], transformed_axes_opt[motion_tup[3]])
            if isinstance(new_motion_step, list):
                seq_motion += new_motion_step
            else:
                seq_motion.append(new_motion_step)

        elif motion_tup[0] == "translate_tar_obj" or "translate_direc_axis":
            obj_sizes = calculate_obj_size(abs_pose_opt, pc_list)
            if motion_tup[0] == "translate_tar_obj":
                goal_pose = translate_along_axis(motion_tup[1][0], "local_x", transformed_axes_opt[motion_tup[2]], abs_pose_opt[motion_tup[2]], (obj_idx, motion_tup[2]), obj_sizes, intrinsics)
                goal_pose = translate_along_axis(motion_tup[1][1], "local_y", transformed_axes_opt[motion_tup[2]], goal_pose, (obj_idx, motion_tup[2]), obj_sizes, intrinsics)
                goal_pose = translate_along_axis(motion_tup[1][2], "local_z", transformed_axes_opt[motion_tup[2]], goal_pose, (obj_idx, motion_tup[2]), obj_sizes, intrinsics, True)
            elif motion_tup[0] == "translate_direc_axis":
                if motion_tup[2] == obj_idx:
                    goal_pose = translate_along_vec_diff(motion_tup[1], seq_motion[-1], abs_pose_opt[motion_tup[3]], (obj_idx, motion_tup[2], motion_tup[3]), obj_sizes, intrinsics)
                elif motion_tup[3] == obj_idx:
                    goal_pose = translate_along_vec_diff(motion_tup[1], abs_pose_opt[motion_tup[2]], seq_motion[-1], (obj_idx, motion_tup[2], motion_tup[3]), obj_sizes, intrinsics)
                else:
                    goal_pose = translate_along_vec_diff(motion_tup[1], abs_pose_opt[motion_tup[2]], abs_pose_opt[motion_tup[3]], (obj_idx, motion_tup[2], motion_tup[3]), obj_sizes, intrinsics)
            
            start_point = seq_motion[-1].camera_T_object[:3, 3].reshape(3,)
            end_point = goal_pose.camera_T_object[:3, 3].reshape(3,)
            obstacles = return_obstacles(obj_idx, abs_pose_opt, pc_list)
            rrt_planned = rrt_star_planning(abs_pose_opt, pc_list, obj_idx, start_point, end_point, seq_motion[-1], obstacles)

            # Start point is in obstacle
            sampled_point = None
            if rrt_planned == -1:
                print("Collison detected at start point!")
                print("")
                # Sample points around until start point is not in obstacle
                range_ = 0.0125
                num_samples = 25
                while rrt_planned == -1:
                    points_fluc = np.random.uniform(-range_, range_, (num_samples, 3))
                    # std_val = range_
                    # points_fluc = np.random.normal(0, std_val, (num_samples, 3))
                    for point_fluc in points_fluc:
                        sampled_point = start_point + np.array(point_fluc)
                        rrt_planned = rrt_star_planning(abs_pose_opt, pc_list, obj_idx, sampled_point, end_point, seq_motion[-1], obstacles)
                        if rrt_planned != -1: break
                    range_ += 0.0125
                
                interpolated_coordinate = interpolate_coordinates(start_point, sampled_point, method='spline', num_points=generate_pts_prop_to_dis(start_point, sampled_point))
                for point in interpolated_coordinate:
                    new_motion_step = copy.deepcopy(seq_motion[-1])
                    new_motion_step.camera_T_object[:3, 3] = point
                    seq_motion.append(new_motion_step)
            
            # No path found or start point is in obstacle
            original_start_point = start_point
            if sampled_point is not None:
                start_point = sampled_point

            if (rrt_planned is None) or (sampled_point is not None):
                rrt_planned = rrt_star_planning(abs_pose_opt, pc_list, obj_idx, start_point, end_point, seq_motion[-1], obstacles)

                # The below parameters are experimented
                prev_std_val = 0.0
                std_val = 0.1 # 0.1cm
                step_val = 0.5 # 0.5cm
                growth_rate = 0.5  # 50% growth rate
                num_samples = 10
                end_points = []
                
                if motion_tup[0] == "translate_tar_obj":
                    dx, dy, dz = motion_tup[1][0], motion_tup[1][1], motion_tup[1][2]
                    # print(dx, dy, dz)
                elif motion_tup[0] == "translate_direc_axis":
                    d_ = motion_tup[1]
                
                rrt_iteration = 0
                while rrt_planned is None:
                    if rrt_iteration > 50:
                        return seq_motion
                    print("rrt_iteration:", rrt_iteration)

                    end_points = []
                    if motion_tup[0] == "translate_tar_obj":
                        if dx == 0 and dy == 0 and dz == 0:
                            scalars_x = np.random.normal(0, std_val, num_samples)
                            scalars_y = np.random.normal(0, std_val, num_samples)
                            scalars_z = np.random.normal(0, std_val, num_samples)
                        elif dx == 0 and dy == 0 and dz != 0:
                            scalars_x = np.zeros(num_samples)
                            scalars_y = np.zeros(num_samples)
                            scalars_z = np.concatenate([np.random.uniform(-std_val, -prev_std_val, num_samples // 2), np.random.uniform(prev_std_val, std_val, num_samples // 2)])
                        else:
                            scalars_x = np.random.normal(0, std_val, num_samples)
                            scalars_y = np.random.normal(0, std_val, num_samples)
                            scalars_z = np.zeros(num_samples) if np.sign(dz) == 0 else np.random.normal(0, std_val, num_samples)                        
                        if dz != 0:
                            # > 0 because positive direction of local z-axis is upwards
                            scalars_x, scalars_y, scalars_z = scalars_x[scalars_z * dz >= 0], scalars_y[scalars_z * dz >= 0], scalars_z[scalars_z * dz >= 0]
                            if scalars_z.size == 0: continue
                            # print(scalars_z)
                        # print(scalars_x, scalars_y, scalars_z)

                        for j in range(len(scalars_z)):
                            fluc_pose = translate_along_axis(motion_tup[1][0] + scalars_x[j], "local_x", transformed_axes_opt[motion_tup[2]], abs_pose_opt[motion_tup[2]], (obj_idx, motion_tup[2]), obj_sizes, intrinsics)
                            fluc_pose = translate_along_axis(motion_tup[1][1] + scalars_y[j], "local_y", transformed_axes_opt[motion_tup[2]], fluc_pose, (obj_idx, motion_tup[2]), obj_sizes, intrinsics)
                            fluc_pose = translate_along_axis(motion_tup[1][2] + scalars_z[j], "local_z", transformed_axes_opt[motion_tup[2]], fluc_pose, (obj_idx, motion_tup[2]), obj_sizes, intrinsics, True)
                            end_points.append(fluc_pose.camera_T_object[:3, 3].reshape(3,))

                    elif motion_tup[0] == "translate_direc_axis":
                        scalars = np.concatenate([np.random.uniform(-std_val, -prev_std_val, num_samples // 2), np.random.uniform(prev_std_val, std_val, num_samples // 2)])
                        # scalars = scalars[scalars * d_ >= 0]
                        # print(scalars)
                        if scalars.size == 0: continue

                        for j in range(len(scalars)):
                            if motion_tup[2] == obj_idx:
                                fluc_pose = translate_along_vec_diff(motion_tup[1] + scalars[j], seq_motion[-1], abs_pose_opt[motion_tup[3]], (obj_idx, motion_tup[2], motion_tup[3]), obj_sizes, intrinsics)
                            elif motion_tup[3] == obj_idx:
                                fluc_pose = translate_along_vec_diff(motion_tup[1] + scalars[j], abs_pose_opt[motion_tup[2]], seq_motion[-1], (obj_idx, motion_tup[2], motion_tup[3]), obj_sizes, intrinsics)
                            else:
                                fluc_pose = translate_along_vec_diff(motion_tup[1] + scalars[j], abs_pose_opt[motion_tup[2]], abs_pose_opt[motion_tup[3]], (obj_idx, motion_tup[2], motion_tup[3]), obj_sizes, intrinsics)
                            end_points.append(fluc_pose.camera_T_object[:3, 3].reshape(3,))
                    
                    closest_rrt_planned = None
                    min_distance_to_goal = np.inf
                    for end_point in end_points:
                        # rrt_planned = rrt_star_planning(abs_pose_opt, pc_list, obj_idx, start_point, end_point, seq_motion[-1], obstacles, scale_factor=max(rrt_iteration // 2 + 1, 1))
                        rrt_planned = rrt_star_planning(abs_pose_opt, pc_list, obj_idx, start_point, end_point, seq_motion[-1], obstacles, scale_factor=max(rrt_iteration // 5 + 1, 1))
                        if rrt_planned is not None and rrt_planned != -1:
                            # Calculate the distance from the end point of this path to the goal
                            distance_to_goal = np.linalg.norm(end_point - goal_pose.camera_T_object[:3, 3].reshape(3,))
                            
                            # Update if this path is closer to the goal
                            if distance_to_goal < min_distance_to_goal:
                                min_distance_to_goal = distance_to_goal
                                closest_rrt_planned = rrt_planned

                    rrt_iteration += 1
                    rrt_planned = closest_rrt_planned
                    prev_std_val = std_val
                    std_val += step_val # linear increment
                    # std_val *= (1 + growth_rate) # geometric progression
            
            seq_motion += rrt_planned
            print("translate done: sequential motion", i)
        
    return seq_motion


def interpolate_coordinates(start_point, end_point, method='linear', num_points=10):
    """
    Interpolates points between start and end points using specified method.
    :param start_point: Starting point (3, )
    :param end_point: Ending point (3, )
    :param method: Method of interpolation ('linear', 'spline')
    :param num_points: Number of points to interpolate
    :return: List of interpolated points
    """
    t = np.linspace(0, 1, num_points)
    if method == 'linear':
        interpolated_points = np.outer(t, end_point - start_point) + start_point

    elif method == 'spline':
        # Using CubicSpline for cubic spline interpolation with default boundary conditions
        f = CubicSpline([0, 1], np.vstack([start_point, end_point]), bc_type='natural')
        interpolated_points = f(t)

    else:
        raise ValueError("Invalid interpolation method.")

    return interpolated_points


def smooth_trajectory(path, num_points=10, bc_type='natural'):
    """
    Smooths a trajectory using cubic spline interpolation.
    
    :param interpolated_path: A numpy array of shape (n, 3) representing the trajectory.
    :param num_points: Number of points to use for the smooth trajectory.
    :param bc_type: Boundary condition type ('natural' or 'clamped'). 'natural' is the default.
    :return: A numpy array of shape (num_points, 3) representing the smoothed trajectory.
    """
    # Define the parameter t for the original points
    t = np.linspace(0, 1, len(path))
    
    # Create a cubic spline interpolation
    cs = CubicSpline(t, path, bc_type=bc_type)
    
    # Generate more points on the spline curve
    fine_t = np.linspace(0, 1, num_points)
    smooth_path = cs(fine_t)
    
    return smooth_path


def generate_pts_prop_to_dis(start_point, end_point, proportionality_factor=50):
    """
    Generates points between start_point and end_point. The number of points is proportional 
    to the distance between these points, determined by a proportionality factor.
    """
    # Calculate the distance between start and end points
    distance = np.linalg.norm(np.array(end_point) - np.array(start_point))

    # Determine the number of points based on the distance
    num_points = max(4, int(distance * proportionality_factor))
    # print(num_points)

    return num_points


def interpolate_rotation(total_degrees):
    if total_degrees == 0:
        return [0]
    
    # Determine the sign of the rotation (positive or negative)
    sign_ = 1 if total_degrees >= 0 else -1

    # Calculate the number of steps needed
    step_degrees = 10
    num_steps = int(np.abs(total_degrees) // step_degrees)
    remainder = np.abs(total_degrees) % step_degrees

    # Create the list of interpolated tuples
    interpolated_degrees = [sign_ * step_degrees for _ in range(num_steps)]

    # Add the remainder step if there is any
    if remainder > 0:
        interpolated_degrees.append(sign_ * remainder)

    return interpolated_degrees


def generate_obj_3d_boxes(abs_pose_opt, pc_list, shrink_factor_=0.25):
    boxes_3d = []

    for i, pc in enumerate(pc_list):
        points = np.asarray(pc.points).T

        # coordinates: [3, N]
        if points.shape[0] != 3 or len(points.shape) != 2:
            raise ValueError("Point cloud must be of shape (3, N)")

        # Calculate the Z-score for each point (along columns)
        # Remove points with a Z-score greater than the threshold
        threshold = 2
        z_scores = np.abs(zscore(points, axis=1)) # about 95%, 99.7% of the data falls within ±2, ±3 std from the mean
        points = points[:, (z_scores < threshold).all(axis=0)]
                
        min_x, max_x = np.min(points[0, :]), np.max(points[0, :])
        min_y, max_y = np.min(points[1, :]), np.max(points[1, :])
        min_z, max_z = np.min(points[2, :]), np.max(points[2, :])
        # print((min_x, min_y, min_z, max_x, max_y, max_z))

        # Get the object center or the box center
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        center_z = (max_z + min_z) / 2
        # center = abs_pose_opt[i].camera_T_object[:3, 3].reshape(3,) 
        # center_x = center[0]
        # center_y = center[1]
        # center_z = center[2]

        # Calculate new min and max coordinates after shrinking
        # a shrink_factor of 0.1 would shrink the box by 10%, a negative factor expands the box
        shrink_factor = shrink_factor_
        new_min_x = center_x + (min_x - center_x) * (1 - shrink_factor)
        new_max_x = center_x + (max_x - center_x) * (1 - shrink_factor)
        new_min_y = center_y + (min_y - center_y) * (1 - shrink_factor)
        new_max_y = center_y + (max_y - center_y) * (1 - shrink_factor)
        new_min_z = center_z + (min_z - center_z) * (1 - shrink_factor)
        new_max_z = center_z + (max_z - center_z) * (1 - shrink_factor)

        boxes_3d.append((new_min_x, new_min_y, new_min_z, new_max_x, new_max_y, new_max_z))
    return boxes_3d


def generate_segmented_3d_boxes(obj_idx, abs_pose_opt, pc_list, num_clusters=300):
    small_boxes_3d = []

    for i, pc in enumerate(pc_list):
        if i != obj_idx:
            points = np.asarray(pc.points).T
            if points.shape[0] != 3 or len(points.shape) != 2:
                raise ValueError("Point cloud must be of shape (3, N)")

            # Remove outliers
            threshold = 2
            z_scores = np.abs(zscore(points, axis=1))
            points = points[:, (z_scores < threshold).all(axis=0)]

            # Cluster the points
            num_clusters = int(points.shape[1] / 3 + 0.5)
            kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
            labels = kmeans.fit_predict(points.T)

            # Generate bounding boxes for each cluster
            for cluster_idx in range(num_clusters):
                cluster_points = points[:, labels == cluster_idx]

                if cluster_points.size == 0:
                    continue

                min_x, max_x = np.min(cluster_points[0, :]), np.max(cluster_points[0, :])
                min_y, max_y = np.min(cluster_points[1, :]), np.max(cluster_points[1, :])
                min_z, max_z = np.min(cluster_points[2, :]), np.max(cluster_points[2, :])
                center = [np.mean(cluster_points[axis, :]) for axis in range(3)]

                # Calculate box center and apply shrink factor
                shrink_factor = 0.0
                new_min = [center[axis] + (np.min(cluster_points[axis, :]) - center[axis]) * (1 - shrink_factor) for axis in range(3)]
                new_max = [center[axis] + (np.max(cluster_points[axis, :]) - center[axis]) * (1 - shrink_factor) for axis in range(3)]
                small_boxes_3d.append((new_min[0], new_min[1], new_min[2], new_max[0], new_max[1], new_max[2]))

    return small_boxes_3d


def grow_obj_3d_boxes(obj_idx, boxes_3d, obj_box=None):
    if obj_box is None:
        obj_box = boxes_3d[obj_idx]
    else:
        obj_box = obj_box[0]
    width = obj_box[3] - obj_box[0]  # max_x - min_x
    height = obj_box[4] - obj_box[1]  # max_y - min_y
    depth = obj_box[5] - obj_box[2]  # max_z - min_z
    
    expanded_boxes_3d = []
    for i, box in enumerate(boxes_3d):
        if  i != obj_idx:
            expanded_box = (
            box[0] - width / 2,  # min_x
            box[1] - height / 2,  # min_y
            box[2] - depth / 2,  # min_z
            box[3] + width / 2,  # max_x
            box[4] + height / 2,  # max_y
            box[5] + depth / 2   # max_z
            )
            expanded_boxes_3d.append(expanded_box)

    return expanded_boxes_3d


def return_obstacles(obj_idx, abs_pose_opt, pc_list):
    # Treating object as a single 3D binding box
    # obstacles = generate_obj_3d_boxes(abs_pose_opt, pc_list)
    # obstacles = grow_obj_3d_boxes(obj_idx, obstacles)

    # Segment object into a high number of 3D binding boxes
    obj_box = generate_obj_3d_boxes([abs_pose_opt[obj_idx]], [pc_list[obj_idx]], 0.1)
    obstacles = generate_segmented_3d_boxes(obj_idx, abs_pose_opt, pc_list)
    obstacles = grow_obj_3d_boxes(-1, obstacles, obj_box)
    return obstacles


def rrt_star_planning(abs_pose_opt, pc_list, obj_idx, start_point, end_point, initial_pose, obstacles, scale_factor=1):
    X_dimensions = np.array([(-2, 2), (-2, 2), (-1, 2)])  # dimensions of Search Space
    x_init = tuple(start_point)  # starting location
    x_goal = tuple(end_point)  # goal location

    # The below parameters are experimented
    Q = np.array([(0.05 * scale_factor, 8)])  # [(0.01, 8)], (length of tree edges, iterate over number of edges of given length to add)
    r = 0.0125 * scale_factor  # 0.0025, length of smallest edge to check for intersection with obstacles
    max_samples = 2048  # 2048, max number of samples to take before timing out
    rewire_count = 32  # optional, number of nearby branches to rewire
    prc = 0.75  # probability of checking for a connection to goal

    # create Search Space
    X = SearchSpace(X_dimensions, obstacles)

    # create rrt_search
    rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star()
    if path is None or path == -1:
        return path

    # plot if path found
    # rrt_plot(rrt, X, path, x_init, x_goal, obstacles)

    # interpolated_path = []
    # for i in range(len(path) - 1):
    #     start_point = path[i]
    #     end_point = path[i + 1]
    #     interpolated_points = interpolate_coordinates(start_point, end_point, method='spline', num_points=generate_pts_prop_to_dis(start_point, end_point))
    #     interpolated_path.extend(interpolated_points[:-1])
    # interpolated_path.append(path[-1])

    num_points = generate_pts_prop_to_dis(start_point, end_point, proportionality_factor=75)
    interpolated_path = smooth_trajectory(path, num_points=num_points)

    return seq_motion_from_planned_path(initial_pose, interpolated_path)


def rrt_plot(rrt, X, path, x_init, x_goal, obstacles):
    plot = Plot("rrt_star_3d")
    plot.plot_tree(X, rrt.trees)
    if path is not None:
        plot.plot_path(X, path)
    plot.plot_obstacles(X, obstacles)
    plot.plot_start(X, x_init)
    plot.plot_goal(X, x_goal)
    plot.draw(auto_open=True)


def seq_motion_from_planned_path(initial_pose, path):
    new_obj_motion_steps = []
    for coord in path:
        new_motion_step = copy.deepcopy(initial_pose)
        new_motion_step.camera_T_object[0, 3] = coord[0]
        new_motion_step.camera_T_object[1, 3] = coord[1] 
        new_motion_step.camera_T_object[2, 3] = coord[2]
        new_obj_motion_steps.append(new_motion_step)

    del new_obj_motion_steps[0]
    return new_obj_motion_steps
