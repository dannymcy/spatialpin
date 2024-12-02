import numpy as np
import cv2
import open3d as o3d
import copy
import re
from scipy.spatial.transform import Rotation as R

from utils.transform_utils import calculate_2d_projections, transform_coordinates_3d


def save_projected_mask(pc, intrinsics, input_image, filename, erosion_pixels=20):
    """
    Create and erode a black mask for image inpainting.

    :param pc: Open3D point cloud object.
    :param intrinsics: Camera intrinsic matrix.
    :param input_image: Numpy array of the image.
    :param filename: Filename to save the mask.
    :param erosion_pixels: Number of pixels to erode (expand) the mask.
    :return: Eroded black mask with points from the point cloud.
    """
    image_masked = input_image.copy()
    # Extract points the point cloud
    points = np.asarray(pc.points).T

    # Transform points to camera frame and project to 2D image coordinates
    points_image_plane = calculate_2d_projections(points, intrinsics)

    # Start with a white mask
    mask = np.ones_like(input_image, dtype=np.uint8) * 255  

    for point in points_image_plane:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < input_image.shape[1] and 0 <= y < input_image.shape[0]:
            mask[y, x] = 0  # Set pixel to black

    # Define the erosion kernel
    kernel = np.ones((erosion_pixels, erosion_pixels), np.uint8)

    # Erode the mask to enlarge the black areas
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # Set pixels to black where the mask is black
    image_masked[eroded_mask == 0] = 0  

    # Reverse the mask (convert black to white and vice versa)
    eroded_mask = 255 - eroded_mask

    # Name format according to LaMa
    matches = re.findall(r'\d+', filename)
    i, k = matches  # Assuming there are always two numbers
    formatted_k = "{:03d}".format(int(k))
    cv2.imwrite("/hdd2/chenyang/lama/test_data" + "/image" + str(i) + "_mask" + str(formatted_k) + '.png', eroded_mask)
    cv2.imwrite("/hdd2/chenyang/lama/test_data" + "/image" + str(i) + "_mask*img" + str(formatted_k) + '.png', image_masked)
    cv2.imwrite("/hdd2/chenyang/lama/test_data" + "/image" + str(i) + '.png', input_image)


def load_inpainted_img(img_id, pose_id):
    formatted_pose_id = "{:03d}".format(int(pose_id))
    filename = "/hdd2/chenyang/lama/results" + "/image" + str(img_id) + "_mask" + str(formatted_pose_id) + '.png'
    inpainted_img = cv2.imread(filename)
    return inpainted_img


def inpaint_img_from_impainted(input_image, img_id, pose_id_list, pc_list, intrinsics, erosion_pixels=20):
    image_masked = input_image.copy()

    for i, pose_id in enumerate(pose_id_list):
        formatted_pose_id = "{:03d}".format(int(pose_id))
        filename = "/hdd2/chenyang/lama/results" + "/image" + str(img_id) + "_mask" + str(formatted_pose_id) + '.png'
        inpainted_img = cv2.imread(filename)

        pc = pc_list[int(pose_id)]
        points = np.asarray(pc.points).T

        # Transform points to camera frame and project to 2D image coordinates
        points_image_plane = calculate_2d_projections(points, intrinsics)
        mask = np.ones_like(input_image, dtype=np.uint8) * 255 

        for point in points_image_plane:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < input_image.shape[1] and 0 <= y < input_image.shape[0]:
                mask[y, x] = 0  # Set pixel to black
 
        kernel = np.ones((erosion_pixels, erosion_pixels), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)

        image_masked[eroded_mask == 0] = inpainted_img[eroded_mask == 0]

    return image_masked