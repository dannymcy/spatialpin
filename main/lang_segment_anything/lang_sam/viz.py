import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import cv2
from skimage.feature import canny
from skimage.morphology import binary_dilation
from scipy import ndimage


def save_mask_centers(masks, output_path):
    # Calculate the center of each mask
    mask_centers = [ndimage.center_of_mass(mask) for mask in masks]
    mask_centers = [list(map(int, map(round, center))) for center in mask_centers]

    with open(output_path, 'w') as json_file:
        json.dump(mask_centers, json_file)
    
    return mask_centers


# https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb
def save_mask(mask, ax, random_color=False):
    """ Renders a single mask on the given axis. """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image[edge, :] = [1, 1, 1, 1]  # White in RGBA (normalized)
    ax.imshow(mask_image)


def save_masks_on_image(raw_image, masks, output_path):
    """ Applies all masks on the image and saves the output. """
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for mask in masks:
        save_mask(mask, ax=ax, random_color=True)
    plt.axis("off")
    plt.savefig(f"{output_path}.png", bbox_inches='tight', pad_inches=0)
    plt.close()


def save_separate_masks_white(raw_image, masks, output_path):
    """
    Saves each mask separately. Inside the mask, the original object is shown,
    while the area outside of the mask is white.
    """
    for i, mask in enumerate(masks):
        # Create an all-white image of the same shape as the raw image
        white_image = np.ones_like(raw_image) * 255

        # Apply the mask: Copy pixels from the raw image where the mask is true
        white_image[mask] = raw_image[mask]

        # Save the masked image
        individual_output_path = f"{output_path}_obj_{i}.png"
        plt.imsave(individual_output_path, white_image.astype(np.uint8))


def save_separate_masks_binary(masks, output_path):
    """
    Saves each mask separately in a binary format (0 or 1).
    """
    for i, mask in enumerate(masks):
        # Create a binary mask with 0s for the background and 1s for the object
        binary_mask = (mask > 0).astype(np.uint8)
        individual_output_path = f"{output_path}_obj_{i}.png"
        plt.imsave(individual_output_path, binary_mask, cmap='gray')


def save_separate_masks_inpaint(img_id, raw_image, binary_mask_dir, binary_mask_kept_path, 
                                guide_mask_output_path, inpaint_mask_output_path, 
                                dilation_pixels=5, background=False):
    # Create an initial mask filled with 0s (for the binary mask)
    binary_accumulated_mask = np.zeros(raw_image.shape[:2], dtype=np.uint8)

    # Iterate over all binary mask files in the directory
    for mask_filename in os.listdir(binary_mask_dir):
        if mask_filename.startswith(f"masks_{img_id}_obj_"):
            mask_path = os.path.join(binary_mask_dir, mask_filename)

            # Skip the binary_mask_kept_path if background is False
            if background is False:
                if mask_path == binary_mask_kept_path:
                    continue
            else:
                dilation_pixels = 17

            # Load the binary mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Define the dilation kernel
            kernel = np.ones((dilation_pixels, dilation_pixels), np.uint8)
            # Dilate the mask to enlarge the white areas
            mask = cv2.dilate(mask, kernel, iterations=1)

            # Update the accumulated masks
            object_area = mask == 255
            binary_accumulated_mask[object_area] = 1

    # Create the RGB mask by making holes (white) in the raw_image
    rgb_mask = raw_image.copy()
    holes_mask = binary_accumulated_mask == 1
    rgb_mask[holes_mask] = [255, 255, 255]

    # Save the RGB modified image with original colors and black holes
    cv2.imwrite(guide_mask_output_path, rgb_mask)

    # Invert the binary mask for saving: holes as 0, rest as 1
    cv2.imwrite(inpaint_mask_output_path, (binary_accumulated_mask * 255).astype(np.uint8))
  

def save_separate_masks_transparent(raw_image_or_list, masks, output_path):
    """
    Saves each mask separately. Inside the mask, the original object is shown,
    while the area outside of the mask is set to white.
    """
    for i, mask in enumerate(masks):
        raw_image = raw_image_or_list[i] if isinstance(raw_image_or_list, list) else raw_image_or_list

        # Ensure raw_image is in RGBA format
        if raw_image.shape[2] == 3:  # RGB
            raw_image_rgba = np.concatenate([raw_image, np.ones((raw_image.shape[0], raw_image.shape[1], 1)) * 255], axis=-1)
        else:  # Already RGBA
            raw_image_rgba = raw_image.copy()

        # Set RGB channels to white (255, 255, 255) where the mask is False
        raw_image_rgba[~mask, :3] = 255

        # Set alpha channel to 0 (transparent) where the mask is False
        raw_image_rgba[~mask, 3] = 0

        # Save the masked image with white background outside the mask
        individual_output_path = f"{output_path}_obj_{i}.png"
        plt.imsave(individual_output_path, raw_image_rgba.astype(np.uint8), format='png')



