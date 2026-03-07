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
RECON_MIN_KEEP_RATIO = 0.75
RECON_FALLBACK_MIN_KEEP_RATIO = 0.70
RECON_FALLBACK_KERNEL_SIZE = 3
RECON_FALLBACK_ITERATIONS = 1
CLOSING_KERNEL_SIZE = 3
MIN_COMPONENT_AREA = 450
BASELINE_DETECT_MIN_ROW_RATIO = 0.80
BASELINE_DETECT_SEARCH_START_RATIO = 0.6
SMALL_TREE_BAND_HEIGHT = 30

# Stage D: Separation
DISTANCE_THRESHOLD = 0.3  # fraction of max distance for watershed markers


def detect_baseline_row(mask):
    """
    Detect baseline row (horizontal bright strip under the forest) in a mask.
    """
    if mask is None or mask.ndim != 2 or mask.size == 0:
        return None

    h = mask.shape[0]
    y_start = int(round(h * BASELINE_DETECT_SEARCH_START_RATIO))
    y_start = min(h - 1, y_start)

    row_ratio = np.mean(mask > 0, axis=1)
    candidates = row_ratio[y_start:] >= BASELINE_DETECT_MIN_ROW_RATIO
    true_idx = np.flatnonzero(candidates)
    if true_idx.size == 0:
        return None

    return y_start + int(true_idx[0])


def zero_below_baseline(mask, baseline_row):
    """
    Set baseline row and everything below it to zero.
    """
    if baseline_row is None:
        return mask
    h = mask.shape[0]
    y0 = max(0, min(h, int(baseline_row)))
    out = mask.copy()
    out[y0:h, :] = 0
    return out


def restore_band_components_from_reference(target_mask, reference_mask, baseline_row, band_height=None):
    """
    Restore connected components from reference_mask whose bottom lies in the
    baseline band into target_mask.
    """
    if (
        baseline_row is None
        or target_mask is None
        or reference_mask is None
        or target_mask.shape != reference_mask.shape
    ):
        return target_mask

    if band_height is None:
        band_height = SMALL_TREE_BAND_HEIGHT

    band_top = max(0, int(baseline_row) - int(band_height))
    band_bottom = int(baseline_row)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        reference_mask, connectivity=8
    )
    out = target_mask.copy()
    for i in range(1, num_labels):
        top = int(stats[i, cv2.CC_STAT_TOP])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        bottom = top + height - 1
        if band_top <= bottom <= band_bottom and top <= band_bottom:
            out[labels == i] = 255
    return out


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


def remove_small_components(mask, min_area=MIN_COMPONENT_AREA, baseline_row=None):
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
    baseline_row : int or None
        If provided, preserve any component whose bottom lies in the
        50-pixel band directly above the baseline.

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
        top = int(stats[i, cv2.CC_STAT_TOP])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        bottom = top + height - 1
        if baseline_row is not None:
            band_top = max(0, int(baseline_row) - SMALL_TREE_BAND_HEIGHT)
            band_bottom = int(baseline_row)
            if band_top <= bottom <= band_bottom and top <= band_bottom:
                cleaned[labels == i] = 255
                continue

        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def postprocess(mask, min_area=MIN_COMPONENT_AREA, baseline_row=None):
    """
    Full post-processing pipeline: small component removal → reconstruction → closing .

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
    small_removed = remove_small_components(
        mask, min_area=min_area, baseline_row=baseline_row
    )
    small_removed = zero_below_baseline(small_removed, baseline_row)

    # Reconstruction is intentionally skipped.
    # recon = morphological_reconstruction(small_removed)
    # recon = restore_band_components_from_reference(
    #     recon, small_removed, baseline_row
    # )
    # recon = zero_below_baseline(recon, baseline_row)
    recon = small_removed.copy()

    # Closing is intentionally skipped.
    # closed = apply_closing(recon)
    # closed = restore_band_components_from_reference(
    #     closed, small_removed, baseline_row
    # )
    # closed = zero_below_baseline(closed, baseline_row)
    closed = recon.copy()
    closed = zero_below_baseline(closed, baseline_row)

    intermediates = {
        "07_small_removed": small_removed,
        "08_reconstructed": recon,
        "09_closed": closed,
    }
    return closed, intermediates


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

    baseline_row = detect_baseline_row(seg_mask)
    if baseline_row is not None:
        print(f"  Baseline detected at row={baseline_row}")
    seg_mask = zero_below_baseline(seg_mask, baseline_row)
    preprocess_ints["06b_baseline_cut"] = seg_mask

    # Stage C: Post-processing
    clean_mask, postprocess_ints = postprocess(
        seg_mask, min_area=MIN_COMPONENT_AREA, baseline_row=baseline_row
    )
    clean_mask = zero_below_baseline(clean_mask, baseline_row)

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


def process_all_images(input_dir, output_dir):
    """
    Batch-process all SEM images in a directory through the classic pipeline.

    Parameters
    ----------
    input_dir : str
        Directory containing input SEM images.
    output_dir : str
        Directory to save all outputs.

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
            save_intermediates=True,
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
    help="Output directory (default: output/classic/)"
)
parser.add_argument(
    "--no-intermediates", action="store_true",
    help="Only save final mask and skeleton, not intermediate stages"
)

args = parser.parse_args()

project_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = args.output or os.path.join(project_dir, "output", "classic")

if args.image:
    # Single image mode
    if not os.path.isfile(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    run_classic_pipeline(
        args.image, output_dir,
        save_intermediates=not args.no_intermediates
    )
elif args.input:
    # Batch mode
    if not os.path.isdir(args.input):
        print(f"Error: Directory not found: {args.input}")
        sys.exit(1)
    process_all_images(args.input, output_dir)
else:
    parser.print_help()
    sys.exit(1)
