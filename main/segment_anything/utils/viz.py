import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.morphology import binary_dilation
from scipy import ndimage
from scipy.stats import zscore


def calculate_scale(points_3d):
    # Remove outliers
    threshold = 2
    z_scores = np.abs(zscore(points_3d, axis=0))
    filtered_indices = (z_scores < threshold).all(axis=1)
    points_3d = points_3d[filtered_indices]

    min_vals = np.min(points_3d, axis=0)
    max_vals = np.max(points_3d, axis=0)
    size = max_vals - min_vals
    return size[0] * size[1] * size[2]


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


def filter_masks(masks):
    # Calculate the pixel size and center of each mask
    mask_sizes_and_centers = [(np.sum(mask["segmentation"]), ndimage.center_of_mass(mask["segmentation"])) for mask in masks]

    # Determine the lower and upper percentile sizes, needs to be finetined
    lower_threshold = np.percentile([size for size, _ in mask_sizes_and_centers], 60.0)
    upper_threshold = np.percentile([size for size, _ in mask_sizes_and_centers], 97.5)

    # Filter masks that are between the 60th and 97.5th percentile sizes and gather their centers
    filtered_masks = [mask for mask, (size, _) in zip(masks, mask_sizes_and_centers) if lower_threshold <= size <= upper_threshold]
    filtered_centers = [list(map(int, map(round, center))) for size, center in mask_sizes_and_centers if lower_threshold <= size <= upper_threshold]

    return filtered_masks, filtered_centers


# https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb
def save_mask(mask, ax, random_color=False):
    """ Renders a single mask on the given axis. """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask["segmentation"].shape[-2:]
    edge = canny(mask["segmentation"])
    edge = binary_dilation(edge, np.ones((2, 2)))
    mask_image = mask["segmentation"].reshape(h, w, 1) * color.reshape(1, 1, -1)
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

        # Get the mask
        mask_img = mask["segmentation"]

        # Apply the mask: Copy pixels from the raw image where the mask is true
        white_image[mask_img] = raw_image[mask_img]

        # Save the masked image
        individual_output_path = f"{output_path}_obj_{i}.png"
        plt.imsave(individual_output_path, white_image.astype(np.uint8))


def save_separate_masks_binary(masks, output_path):
    """
    Saves each mask separately in a binary format (0 or 1).
    """
    for i, mask in enumerate(masks):
        # Create a binary mask with 0s for the background and 1s for the object
        binary_mask = (mask["segmentation"] > 0).astype(np.uint8)
        individual_output_path = f"{output_path}_obj_{i}.png"
        plt.imsave(individual_output_path, binary_mask, cmap='gray')
    

def save_separate_masks_transparent(raw_image, masks, output_path):
    """
    Saves each mask separately. Inside the mask, the original object is shown,
    while the area outside of the mask is transparent.
    """
    for i, mask in enumerate(masks):
        # Ensure raw_image is in RGBA format
        if raw_image.shape[2] == 3:  # RGB
            raw_image_rgba = np.concatenate([raw_image, np.ones((raw_image.shape[0], raw_image.shape[1], 1)) * 255], axis=-1)
        else:  # Already RGBA
            raw_image_rgba = raw_image.copy()

        # Get the mask
        mask_img = mask["segmentation"]

        # Set alpha channel to 0 (transparent) where the mask is False
        raw_image_rgba[~mask_img, 3] = 0

        # Save the masked image with transparent background
        individual_output_path = f"{output_path}_obj_{i}.png"
        plt.imsave(individual_output_path, raw_image_rgba.astype(np.uint8), format='png')


