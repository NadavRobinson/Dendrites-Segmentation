"""
Classic CV pipeline for SEM dendrite segmentation.

Four-stage pipeline:
  A. Pre-processing  — histogram normalization, CLAHE, bilateral filter
  B. Segmentation    — adaptive thresholding (primary), Otsu (fallback)
  C. Post-processing — morphological reconstruction, closing, small component removal
  D. Separation      — distance transform + watershed for touching branches

Plus skeletonization via Zhang-Suen thinning.
"""

import argparse
import cv2
import numpy as np
import os
import sys

from skimage.morphology import reconstruction, skeletonize

# Add project directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_image,
    save_image,
    list_images,
    remove_scale_bar,
    create_overlay,
    create_comparison_strip,
)

# ---------------------------------------------------------------------------
# Tunable parameters (all constants at top for easy adjustment)
# ---------------------------------------------------------------------------

# Stage A: Pre-processing
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 50
BILATERAL_SIGMA_SPACE = 50

# # Stage B: Segmentation
ADAPTIVE_BLOCK_SIZE = 67
ADAPTIVE_C = -12

# Stage C: Post-processing
EROSION_KERNEL_SIZE = 3
EROSION_ITERATIONS = 1
CLOSING_KERNEL_SIZE = 3
MIN_COMPONENT_AREA = 250

# Stage D: Separation
DISTANCE_THRESHOLD = 0.3  # fraction of max distance for watershed markers

# Parameter profiles (edit EASY_PARAMS/HARD_PARAMS independently as needed)
DEFAULT_PARAMS = {
    "CLAHE_CLIP_LIMIT": 2.0,
    "CLAHE_TILE_SIZE": 8,
    "BILATERAL_D": 9,
    "BILATERAL_SIGMA_COLOR": 50,
    "BILATERAL_SIGMA_SPACE": 50,
    "ADAPTIVE_BLOCK_SIZE": 67,
    "ADAPTIVE_C": -12,
    "EROSION_KERNEL_SIZE": 3,
    "EROSION_ITERATIONS": 1,
    "CLOSING_KERNEL_SIZE": 3,
    "MIN_COMPONENT_AREA": 250,
    "DISTANCE_THRESHOLD": 0.07,
}
EASY_PARAMS = DEFAULT_PARAMS.copy()
HARD_PARAMS = DEFAULT_PARAMS.copy()
EASY_PARAMS.update({
    "MIN_COMPONENT_AREA": 250,
})

PARAMETER_PROFILES = {
    "default": DEFAULT_PARAMS,
    "easy": EASY_PARAMS,
    "hard": HARD_PARAMS,
}


def apply_parameter_profile(profile_name):
    """
    Apply a named parameter profile by updating global constants.
    """
    profile = PARAMETER_PROFILES[profile_name]

    global CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE
    global BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
    global ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
    global EROSION_KERNEL_SIZE, EROSION_ITERATIONS
    global CLOSING_KERNEL_SIZE, MIN_COMPONENT_AREA, DISTANCE_THRESHOLD

    CLAHE_CLIP_LIMIT = profile["CLAHE_CLIP_LIMIT"]
    CLAHE_TILE_SIZE = profile["CLAHE_TILE_SIZE"]
    BILATERAL_D = profile["BILATERAL_D"]
    BILATERAL_SIGMA_COLOR = profile["BILATERAL_SIGMA_COLOR"]
    BILATERAL_SIGMA_SPACE = profile["BILATERAL_SIGMA_SPACE"]
    ADAPTIVE_BLOCK_SIZE = profile["ADAPTIVE_BLOCK_SIZE"]
    ADAPTIVE_C = profile["ADAPTIVE_C"]
    EROSION_KERNEL_SIZE = profile["EROSION_KERNEL_SIZE"]
    EROSION_ITERATIONS = profile["EROSION_ITERATIONS"]
    CLOSING_KERNEL_SIZE = profile["CLOSING_KERNEL_SIZE"]
    MIN_COMPONENT_AREA = profile["MIN_COMPONENT_AREA"]
    DISTANCE_THRESHOLD = profile["DISTANCE_THRESHOLD"]


# ===========================================================================
# Stage A: Pre-processing
# ===========================================================================

def normalize_histogram(image):
    """
    Linear stretch of pixel values to the full [0, 255] range.
    Ensures a common basis across images with different exposures.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W), dtype uint8.

    Returns
    -------
    normalized : np.ndarray
        Image with values stretched to [0, 255].
    """
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val == min_val:
        return np.zeros_like(image)
    normalized = ((image.astype(np.float64) - min_val) / (max_val - min_val) * 255)
    return normalized.astype(np.uint8)


def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    Divides the image into tiles and equalizes each independently with
    a clip limit to prevent noise amplification.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W), dtype uint8.

    Returns
    -------
    enhanced : np.ndarray
        Contrast-enhanced image.
    """
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE)
    )
    return clahe.apply(image)


def apply_bilateral_filter(image):
    """
    Edge-preserving denoising via bilateral filtering.
    Smooths flat regions while preserving sharp dendrite edges.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W), dtype uint8.

    Returns
    -------
    filtered : np.ndarray
        Denoised image with edges preserved.
    """
    return cv2.bilateralFilter(
        image, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
    )


def preprocess(image):
    """
    Full pre-processing pipeline: clean → normalize → CLAHE → bilateral.

    Parameters
    ----------
    image : np.ndarray
        Raw grayscale SEM image (H, W).

    Returns
    -------
    result : np.ndarray
        Pre-processed image ready for segmentation.
    intermediates : dict
        Dictionary of intermediate images for visualization.
    """
    removed = remove_scale_bar(image)
    normalized = normalize_histogram(removed)
    clahe_img = apply_clahe(normalized)
    bilateral_img = apply_bilateral_filter(clahe_img)

    intermediates = {
        "01_original": image,
        "02_cleaned": removed,
        "03_normalized": normalized,
        "04_clahe": clahe_img,
        "05_bilateral": bilateral_img,
    }
    return bilateral_img, intermediates


# ===========================================================================
# Stage B: Segmentation
# ===========================================================================

def segment_adaptive(image, block_size=ADAPTIVE_BLOCK_SIZE, c=ADAPTIVE_C):
    """
    Adaptive thresholding — computes a local threshold per pixel based on
    neighborhood mean. Preferred for SEM images with non-uniform illumination.

    Parameters
    ----------
    image : np.ndarray
        Pre-processed grayscale image (H, W).

    Returns
    -------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.
    """

    # Adaptive threshold block size must be odd and >= 3
    block_size = int(block_size)
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1

    mask = cv2.adaptiveThreshold(
        image, maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c
    )
    return mask


# ===========================================================================
# Stage C: Post-processing
# ===========================================================================

def morphological_reconstruction(mask, kernel_size=None, iterations=None):
    """
    Geodesic dilation-based reconstruction to remove noise while preserving
    dendrite structure.

    Process:
      1. Aggressively erode the mask to keep only thick branch cores (marker)
      2. Use the original mask as the limit (mask image)
      3. Dilate the marker iteratively within the mask boundaries

    This removes noise without damaging thin dendrite branches (unlike
    standard morphological opening).

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.
    kernel_size : int or None
        Erosion kernel size. Uses EROSION_KERNEL_SIZE if None.
    iterations : int or None
        Number of erosion iterations. Uses EROSION_ITERATIONS if None.

    Returns
    -------
    reconstructed : np.ndarray
        Cleaned binary mask (0 or 255), dtype uint8.
    """
    if kernel_size is None:
        kernel_size = EROSION_KERNEL_SIZE
    if iterations is None:
        iterations = EROSION_ITERATIONS

    kernel_size = int(kernel_size)
    if kernel_size < 1:
        kernel_size = 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    iterations = int(iterations)
    if iterations < 0:
        iterations = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    # Create marker by aggressive erosion — only thick cores remain
    marker = cv2.erode(mask, kernel, iterations=iterations)

    # skimage reconstruction expects float images in [0, 1]
    marker_f = (marker / 255.0).astype(np.float64)
    mask_f = (mask / 255.0).astype(np.float64)

    # Geodesic dilation: grow marker within mask boundaries
    reconstructed_f = reconstruction(marker_f, mask_f, method='dilation')

    reconstructed = (reconstructed_f * 255).astype(np.uint8)
    return reconstructed


def apply_closing(mask):
    """
    Morphological closing to fill small holes and ensure branch continuity.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.

    Returns
    -------
    closed : np.ndarray
        Closed binary mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def remove_small_components(mask, min_area=MIN_COMPONENT_AREA):
    """
    Remove connected components smaller than min_area pixels.
    Based on the physical assumption that dendrites are large,
    continuous structures.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.
    min_area : int or None
        Minimum component area in pixels. Uses MIN_COMPONENT_AREA if None.
    Returns
    -------
    cleaned : np.ndarray
        Mask with small components removed.
    """
    if min_area is None:
        min_area = MIN_COMPONENT_AREA

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def postprocess(mask, min_area=MIN_COMPONENT_AREA):
    """
    Full post-processing pipeline: reconstruction -> closing -> small component removal.

    Parameters
    ----------
    mask : np.ndarray
        Raw binary segmentation mask.

    Returns
    -------
    result : np.ndarray
        Cleaned binary mask.
    intermediates : dict
        Dictionary of intermediate masks.
    """
    recon = morphological_reconstruction(mask)

    closed = apply_closing(recon)

    small_removed = remove_small_components(closed, min_area=min_area)

    intermediates = {
        "07_reconstructed": recon,
        "08_closed": closed,
        "09_small_removed": small_removed,
    }
    return small_removed, intermediates

# ===========================================================================
# Stage D: Separation (Distance Transform + Watershed)
# ===========================================================================

def separate_branches(mask):
    """
    Separate touching dendrite branches using distance transform and watershed.

    Process:
      1. Compute distance transform of the binary mask
      2. Threshold at a fraction of the maximum distance → foreground markers
      3. Identify background (far from any foreground)
      4. Label markers with connected components
      5. Run watershed to find boundaries between touching branches

    Parameters
    ----------
    mask : np.ndarray
        Clean binary mask (0 or 255), dtype uint8.

    Returns
    -------
    separated : np.ndarray
        Binary mask with touching branches separated.
    """
    if np.sum(mask) == 0:
        return mask.copy()

    # Distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Threshold to find sure foreground (branch cores)
    _, sure_fg = cv2.threshold(
        dist, DISTANCE_THRESHOLD * dist.max(), 255, cv2.THRESH_BINARY
    )
    sure_fg = sure_fg.astype(np.uint8)

    # Sure background — region far from any foreground
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    # Unknown region — between sure foreground and sure background
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers for watershed
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # background = 1, not 0
    markers[unknown == 255] = 0  # unknown = 0 (watershed will determine)

    # Watershed needs 3-channel input
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask_color, markers)

    # Build separated mask: watershed boundaries are marked as -1
    separated = mask.copy()
    separated[markers == -1] = 0

    return separated


# ===========================================================================
# Skeletonization
# ===========================================================================

def skeletonize_mask(mask):
    """
    Extract single-pixel-width centerline skeleton from binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.

    Returns
    -------
    skeleton : np.ndarray
        Skeleton image (0 or 255), dtype uint8.
    """
    binary = (mask > 0).astype(bool)
    skel = skeletonize(binary)
    return (skel.astype(np.uint8) * 255)


def restore_mask_to_original_canvas(mask, original_image):
    """
    Restore cropped mask-like outputs to the original image size.
    Missing area is filled with black.
    """
    if mask is None or original_image is None:
        return mask
    oh, ow = original_image.shape[:2]
    mh, mw = mask.shape[:2]
    if mh == oh and mw == ow:
        return mask

    restored = np.zeros((oh, ow), dtype=mask.dtype)
    copy_h = min(mh, oh)
    copy_w = min(mw, ow)
    restored[:copy_h, :copy_w] = mask[:copy_h, :copy_w]
    return restored


# ===========================================================================
# Pipeline orchestration
# ===========================================================================

def run_classic_pipeline(image_path, output_dir=None, save_intermediates=True):
    """
    Run the full classic segmentation pipeline on a single SEM image.

    Parameters
    ----------
    image_path : str
        Path to input SEM image.
    output_dir : str or None
        Directory to save results. If None, results are not saved.
    save_intermediates : bool
        If True, save every intermediate image for analysis/reporting.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'mask': final binary segmentation mask
        - 'skeleton': single-pixel skeleton
        - 'separated': mask after branch separation
        - 'intermediates': dict of all intermediate images
    """
    # Load image
    image = load_image(image_path, grayscale=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing: {basename} ({image.shape[1]}x{image.shape[0]})")

    # Stage A: Pre-processing
    preprocessed, preprocess_ints = preprocess(image)

    # Stage B: Segmentation
    seg_mask = segment_adaptive(
        preprocessed,
        block_size=ADAPTIVE_BLOCK_SIZE,
        c=ADAPTIVE_C,
    )
    preprocess_ints["06_segmented"] = seg_mask

    # Stage C: Post-processing
    clean_mask, postprocess_ints = postprocess(seg_mask, min_area=MIN_COMPONENT_AREA)

    # Stage D: Separation
    separated = separate_branches(clean_mask)

    # Skeletonization
    skeleton = skeletonize_mask(separated)

    # Restore final mask outputs to original size for preview alignment.
    separated = restore_mask_to_original_canvas(separated, image)
    skeleton = restore_mask_to_original_canvas(skeleton, image)

    # Preview strip: Source | Mask | Mask Overlay | Skeleton Overlay
    mask_overlay = create_overlay(image, separated, color=(0, 255, 0), alpha=0.55)
    skeleton_overlay = create_overlay(image, skeleton, color=(0, 0, 255), alpha=0.70)
    preview = create_comparison_strip(
        [image, separated, mask_overlay, skeleton_overlay],
        ["Source", "Mask", "Mask Overlay", "Skeleton Overlay"],
        height=320,
    )

    # Collect all intermediates
    all_intermediates = {}
    all_intermediates.update(preprocess_ints)
    all_intermediates.update(postprocess_ints)
    all_intermediates["10_separated"] = separated
    all_intermediates["11_skeleton"] = skeleton

    # Save results
    if output_dir and save_intermediates:
        img_out_dir = os.path.join(output_dir, basename)
        os.makedirs(img_out_dir, exist_ok=True)
        for name, img in all_intermediates.items():
            save_image(img, os.path.join(img_out_dir, f"{name}.png"))
        save_image(preview, os.path.join(img_out_dir, "12_preview.png"))
        print(
            f"  Saved {len(all_intermediates) + 1} intermediate images to "
            f"{img_out_dir}/"
        )
    elif output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_image(separated, os.path.join(output_dir, f"{basename}_mask.png"))
        save_image(skeleton, os.path.join(output_dir, f"{basename}_skeleton.png"))
        save_image(preview, os.path.join(output_dir, f"{basename}_preview.png"))

    results = {
        "mask": separated,
        "skeleton": skeleton,
        "separated": separated,
        "intermediates": all_intermediates,
    }
    return results


def process_all_images(input_dir, output_dir, save_intermediates=True):
    """
    Batch-process all SEM images in a directory through the classic pipeline.

    Parameters
    ----------
    input_dir : str
        Directory containing input SEM images.
    output_dir : str
        Directory to save all outputs.
    save_intermediates : bool
        If True, save all intermediate pipeline stages.

    Returns
    -------
    all_results : dict
        Mapping of image basename to pipeline results.
    """
    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return {}

    print(f"Found {len(image_paths)} images in {input_dir}\n")
    all_results = {}
    for path in image_paths:
        results = run_classic_pipeline(
            path,
            output_dir,
            save_intermediates=save_intermediates,
        )
        basename = os.path.splitext(os.path.basename(path))[0]
        all_results[basename] = results
        print()

    print(f"Batch processing complete. Results saved to {output_dir}/")
    return all_results


# ===========================================================================
# CLI entry point
# ===========================================================================

"""Argparse CLI for the classic segmentation pipeline."""
parser = argparse.ArgumentParser(
    description="Classic CV pipeline for SEM dendrite segmentation"
)
parser.add_argument(
    "image", nargs="?", default=None,
    help="Path to a single SEM image (omit for batch mode with --input)"
)
parser.add_argument(
    "--input", default=None,
    help="Directory of SEM images for batch processing"
)
parser.add_argument(
    "--output", default=None,
    help=(
        "Output directory. If omitted, defaults to output/easy with --easy, "
        "output/hard with --hard, otherwise output/classic."
    )
)
parser.add_argument(
    "--no-intermediates", action="store_true",
    help="Only save final mask and skeleton, not intermediate stages"
)
profile_group = parser.add_mutually_exclusive_group()
profile_group.add_argument(
    "--easy", action="store_true",
    help=(
        "Use EASY parameter profile. In batch mode without --input, defaults "
        "to dataset/Easy and output/easy."
    )
)
profile_group.add_argument(
    "--hard", action="store_true",
    help=(
        "Use HARD parameter profile. In batch mode without --input, defaults "
        "to dataset/Hard and output/hard."
    )
)

args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

if args.easy:
    selected_profile = "easy"
elif args.hard:
    selected_profile = "hard"
else:
    selected_profile = "default"
apply_parameter_profile(selected_profile)
print(f"Using parameter profile: {selected_profile}")

if args.output:
    output_dir = args.output
elif selected_profile == "easy":
    output_dir = os.path.join(repo_root, "output", "easy")
elif selected_profile == "hard":
    output_dir = os.path.join(repo_root, "output", "hard")
else:
    output_dir = os.path.join(repo_root, "output", "classic")

if args.image:
    # Single image mode
    if not os.path.isfile(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    run_classic_pipeline(
        args.image, output_dir,
        save_intermediates=not args.no_intermediates
    )
else:
    input_dir = args.input
    if input_dir is None and selected_profile == "easy":
        input_dir = os.path.join(repo_root, "dataset", "Easy")
    elif input_dir is None and selected_profile == "hard":
        input_dir = os.path.join(repo_root, "dataset", "Hard")

    if not input_dir:
        parser.print_help()
        sys.exit(1)

    # Batch mode
    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)
    process_all_images(
        input_dir,
        output_dir,
        save_intermediates=not args.no_intermediates,
    )
