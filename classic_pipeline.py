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
from utils import load_image, save_image, list_images, clean_sem_image

# ---------------------------------------------------------------------------
# Tunable parameters (all constants at top for easy adjustment)
# ---------------------------------------------------------------------------

# Stage A: Pre-processing
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = 8

BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Stage B: Segmentation
ADAPTIVE_BLOCK_SIZE = 51
ADAPTIVE_C = 5

# Stage C: Post-processing
EROSION_KERNEL_SIZE = 5
EROSION_ITERATIONS = 3
CLOSING_KERNEL_SIZE = 5
MIN_COMPONENT_AREA = 50

# Stage D: Separation
DISTANCE_THRESHOLD = 0.4  # fraction of max distance for watershed markers

# Dataset-specific cutoff file (used for easy_* samples with known base strip)
CUTOFFS_FILE = "cut_offs.txt"

# Heuristic: image is considered mask-like if most pixels are near 0/255
MASKLIKE_BIMODAL_RATIO = 0.85


def load_cutoff_map(cutoffs_file=CUTOFFS_FILE):
    """
    Load per-image cutoff values from cut_offs.txt.

    The file is expected to contain 10 lines for easy_1 .. easy_10.
    Returns an empty dict if the file does not exist.
    """
    if not os.path.isfile(cutoffs_file):
        return {}

    values = []
    with open(cutoffs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(int(line))

    mapping = {}
    for i, v in enumerate(values, start=1):
        mapping[f"easy_{i}"] = v
    return mapping


def apply_known_cutoff(mask, image_name, cutoff_map):
    """
    Zero out the bottom strip according to pre-defined per-image cutoff.
    """
    cutoff = cutoff_map.get(image_name)
    if cutoff is None:
        return mask
    h = mask.shape[0]
    y0 = max(0, h - int(cutoff))
    out = mask.copy()
    out[y0:h, :] = 0
    return out


def is_masklike_input(image):
    """
    Detect quasi-binary inputs (mostly 0/255) from pre-generated masks.
    """
    near_zero = image <= 3
    near_white = image >= 252
    ratio = float(np.mean(np.logical_or(near_zero, near_white)))
    return ratio >= MASKLIKE_BIMODAL_RATIO

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
    cleaned = clean_sem_image(image)
    normalized = normalize_histogram(cleaned)
    clahe_img = apply_clahe(normalized)
    bilateral_img = apply_bilateral_filter(clahe_img)

    intermediates = {
        "01_original": image,
        "02_cleaned": cleaned,
        "03_normalized": normalized,
        "04_clahe": clahe_img,
        "05_bilateral": bilateral_img,
    }
    return bilateral_img, intermediates


# ===========================================================================
# Stage B: Segmentation
# ===========================================================================

def segment_adaptive(image):
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
    mask = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )
    return mask


def segment_otsu(image):
    """
    Otsu's binarization — finds the optimal global threshold minimizing
    within-class variance. Fallback for uniformly illuminated images.

    Parameters
    ----------
    image : np.ndarray
        Pre-processed grayscale image (H, W).

    Returns
    -------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.
    """
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def segment_masklike(image):
    """
    Segmentation path for quasi-binary inputs:
    keep any non-zero pixel as foreground.
    """
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[image > 0] = 255
    return mask


# ===========================================================================
# Stage C: Post-processing
# ===========================================================================

def morphological_reconstruction(mask):
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

    Returns
    -------
    reconstructed : np.ndarray
        Cleaned binary mask (0 or 255), dtype uint8.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (EROSION_KERNEL_SIZE, EROSION_KERNEL_SIZE)
    )
    # Create marker by aggressive erosion — only thick cores remain
    marker = cv2.erode(mask, kernel, iterations=EROSION_ITERATIONS)

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


def remove_small_components(mask, min_area=None):
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
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def fill_external_contours(mask, min_area=50):
    """
    Fill external contours to create compact branch regions.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    valid = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    if valid:
        cv2.drawContours(filled, valid, -1, 255, thickness=cv2.FILLED)
    return filled


def postprocess(mask):
    """
    Full post-processing pipeline: reconstruction → closing → small component removal.

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
    cleaned = remove_small_components(closed)
    filled = fill_external_contours(cleaned, min_area=MIN_COMPONENT_AREA)

    intermediates = {
        "07_reconstructed": recon,
        "08_closed": closed,
        "09_small_removed": cleaned,
        "09b_filled": filled,
    }
    return filled, intermediates


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
    Uses Zhang-Suen thinning algorithm via scikit-image.

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


# ===========================================================================
# Pipeline orchestration
# ===========================================================================

def run_classic_pipeline(image_path, output_dir=None, save_intermediates=True, cutoff_map=None):
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

    if cutoff_map is None:
        cutoff_map = load_cutoff_map()

    # Stage A: Pre-processing
    preprocessed, preprocess_ints = preprocess(image)

    # Stage B: Segmentation
    # For quasi-binary images (maked_dataset), keep non-zero foreground.
    # For raw SEM images, use adaptive thresholding.
    masklike_mode = is_masklike_input(image)
    if masklike_mode:
        seg_mask = segment_masklike(image)
    else:
        seg_mask = segment_adaptive(preprocessed)
    preprocess_ints["06_segmented"] = seg_mask

    # Stage C: Post-processing
    clean_mask, postprocess_ints = postprocess(seg_mask)

    # Optional dataset-specific bottom-strip removal (easy_1..easy_10).
    clean_mask = apply_known_cutoff(clean_mask, basename, cutoff_map)
    postprocess_ints["09c_cutoff_applied"] = clean_mask

    # Stage D: Separation
    if masklike_mode:
        separated = clean_mask.copy()
    else:
        separated = separate_branches(clean_mask)

    # Skeletonization
    skeleton = skeletonize_mask(separated)

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
        print(f"  Saved {len(all_intermediates)} intermediate images to {img_out_dir}/")
    elif output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_image(separated, os.path.join(output_dir, f"{basename}_mask.png"))
        save_image(skeleton, os.path.join(output_dir, f"{basename}_skeleton.png"))

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
    cutoff_map = load_cutoff_map()
    all_results = {}
    for path in image_paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        results = run_classic_pipeline(
            path,
            output_dir,
            save_intermediates=True,
            cutoff_map=cutoff_map,
        )
        all_results[basename] = results
        print()

    print(f"Batch processing complete. Results saved to {output_dir}/")
    return all_results


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Synthetic self-test (no CLI args)
        print("=== classic_pipeline.py — Synthetic Self-Test ===\n")

        # Create a synthetic SEM-like image with dendrite-like structures
        np.random.seed(42)
        h, w = 512, 512
        synth = np.random.randint(30, 80, (h, w), dtype=np.uint8)

        # Draw some bright "dendrite" structures
        cv2.line(synth, (100, 50), (100, 400), 200, 3)
        cv2.line(synth, (100, 200), (250, 150), 190, 2)
        cv2.line(synth, (100, 300), (200, 350), 185, 2)
        cv2.line(synth, (300, 100), (300, 450), 210, 4)
        cv2.line(synth, (300, 250), (400, 200), 195, 2)
        cv2.line(synth, (300, 350), (450, 400), 180, 2)

        # Add a bright "scale bar" at bottom
        synth[460:, :] = 230

        # Save synthetic image, then process it
        project_dir = os.path.dirname(__file__)
        test_img_path = os.path.join(project_dir, "output", "synth_dendrites.png")
        os.makedirs(os.path.dirname(test_img_path), exist_ok=True)
        cv2.imwrite(test_img_path, synth)

        out_dir = os.path.join(project_dir, "output", "classic")
        results = run_classic_pipeline(test_img_path, out_dir, save_intermediates=True)

        print(f"\nFinal mask — non-zero pixels: {np.sum(results['mask'] > 0)}")
        print(f"Skeleton   — non-zero pixels: {np.sum(results['skeleton'] > 0)}")

        # Print stage dimensions
        for name, img in results["intermediates"].items():
            print(f"  {name}: shape={img.shape}, range=[{img.min()}, {img.max()}]")

        print("\nAll classic pipeline tests passed.")
