import os
import cv2
import numpy as np
import pathlib
import copy
from scipy.stats import zscore
from sklearn.decomposition import PCA
from itertools import combinations


def create_masks(loc_2d, depth_img, img_center=False):
    if img_center:
        loc_2d = [[depth_img.shape[0] // 2, depth_img.shape[1] // 2]]
    else:
        loc_2d = np.array(loc_2d)
    
    mask = np.zeros(depth_img.shape[:2], dtype=np.uint8)
    for point in loc_2d:
        if 0 <= point[0] < mask.shape[0] and 0 <= point[1] < mask.shape[1]:
            mask[point[0], point[1]] = 255
    return mask


def fluctuate_pixel(mask, dist):
    """ Move the white pixel in the mask by 'dist' pixels in all directions. """
    rows, cols = np.where(mask)
    new_masks = []
    row, col = rows[0], cols[0]
    new_positions = [(row + r, col + c) for r in range(-dist, dist + 1) for c in range(-dist, dist + 1) if not (r == 0 and c == 0)]

    for new_row, new_col in new_positions:
        if 0 <= new_row < mask.shape[0] and 0 <= new_col < mask.shape[1]:
            new_mask = np.zeros_like(mask)
            new_mask[new_row, new_col] = 1
            new_masks.append(new_mask)

    return new_masks


def calculate_scale(points_3d, mask):
    # Remove outliers
    # threshold = 3
    # z_scores = np.abs(zscore(points_3d, axis=0))
    # filtered_indices = (z_scores < threshold).all(axis=1)
    # points_3d = points_3d[filtered_indices]

    # min_vals = np.min(points_3d, axis=0)
    # max_vals = np.max(points_3d, axis=0)
    # area_xy = (max_vals[0] - min_vals[0]) * (max_vals[1] - min_vals[1])
    # area_xz = (max_vals[0] - min_vals[0]) * (max_vals[2] - min_vals[2])
    # area_yz = (max_vals[1] - min_vals[1]) * (max_vals[2] - min_vals[2])

    # One-2-3-45++ fits the reconstructed 3D model in a sphere of diameter 2
    # edge_lengths = max_vals - min_vals
    # return np.max(edge_lengths)
    # largest_area = max(area_xy, area_xz, area_yz)
    # return np.sqrt(largest_area)
    # size = max_vals - min_vals
    # return np.cbrt(size[0] * size[1]* size[2])
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    length = cv2.arcLength(main_contour, True)
    return length
    

def normalize_scale(scale_list):
    # scale factor needs to be finetuned
    scale_factor = 0.00025  # NOCS
    # scale_factor = 0.5  # iPhone 12 Pro Max
    scale_list = [item * scale_factor for item in scale_list]
    return scale_list


def calculate_contour_ratio(mask_1, mask_2):
    contours_1, _ = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour_1 = max(contours_1, key=cv2.contourArea)
    length_1 = cv2.arcLength(main_contour_1, True)

    contours_2, _ = cv2.findContours(mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour_2 = max(contours_2, key=cv2.contourArea)
    length_2 = cv2.arcLength(main_contour_2, True)
    
    # finetuned
    ratio = length_1 / length_2
    return ratio
    # return ratio + 0.5 * (1 - ratio) if (ratio < 0.7 or ratio > 1.3) else 1


def find_surface_point(obj_to_animate_pts, centroid, initial_tolerance_z=0.01, max_tolerance=0.1):
    # Initial tolerance value for z
    tolerance_z = initial_tolerance_z

    # Initialize surface_point as None
    surface_point = None

    while surface_point is None and tolerance_z <= max_tolerance:
        # Find points that are within the tolerance range for z only
        mask = np.abs(obj_to_animate_pts[:, 2] - centroid[2]) <= tolerance_z

        # Filtered points that satisfy the condition
        filtered_points = obj_to_animate_pts[mask]

        # If any points satisfy the condition, find the one with the maximum x value
        if len(filtered_points) > 0:
            surface_point = filtered_points[np.argmax(filtered_points[:, 0])]
            break  # Exit the loop if a point is found
        else:
            # Double the tolerance for the next iteration
            tolerance_z *= 2

    if surface_point is not None:
        return surface_point
    else:
        print("No surface point found within the maximum tolerance")
        return None  # No surface point found within the maximum tolerance


def load_depth(depth_path, depth_estimate):
    """ Load depth image from img_path. """
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        if depth_estimate:
            depth = 1 - (depth - depth.min()) / (depth.max() - depth.min()) if depth.max() != depth.min() else depth
        else:
            depth = (depth - depth.min()) / (depth.max() - depth.min()) if depth.max() != depth.min() else depth
        depth16 = depth        
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def load_img_NOCS(color, depth, depth_estimate=True):
  left_img = cv2.imread(color)
  actual_depth = load_depth(depth, depth_estimate)
  right_img = np.array(actual_depth, dtype=np.float32)
  return left_img, right_img, actual_depth


# https://gist.github.com/gavrielstate/8c855eb3b4b1f23e2990bc02c534792e
# https://www.open3d.org/docs/0.6.0/python_api/open3d.geometry.create_point_cloud_from_rgbd_image.html
def backproject(depth, intrinsics, instance_mask):
    """ Back-projection, use opencv camera coordinate frame."""
    cam_fx = intrinsics[0, 0]
    cam_fy = intrinsics[1, 1]
    cam_cx = intrinsics[0, 2]
    cam_cy = intrinsics[1, 2]

    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    # Solving the case when invalid depth measurement meets object 2D center
    if not np.any(final_instance_mask):
        # Fluctuate the pixel position if the mask is all zeros
        max_fluctuation = 25
        for dist in range(1, max_fluctuation + 1):
            new_masks = fluctuate_pixel(instance_mask, dist)
            for new_mask in new_masks:
                final_instance_mask = np.logical_and(new_mask, non_zero_mask)
                if np.any(final_instance_mask):
                    break
            if np.any(final_instance_mask):
                break

    idxs = np.where(final_instance_mask)
    z = depth[idxs[0], idxs[1]]
    x = (idxs[1] - cam_cx) * z / cam_fx 
    y = (idxs[0] - cam_cy) * z / cam_fy
    pts = np.stack((x, y, z), axis=1)

    return pts, idxs
