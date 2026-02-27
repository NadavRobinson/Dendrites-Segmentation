"""
Shared utilities for SEM dendrite segmentation project.
I/O helpers, SEM image cleaning, and visualization functions.
"""

import cv2
import numpy as np
import os

# Supported image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

# Scale bar region — bottom fraction of SEM image containing instrument metadata
SCALE_BAR_FRACTION = 0.12


def load_image(path, grayscale=True):
    """
    Load an image from disk.

    Parameters
    ----------
    path : str
        Path to the image file.
    grayscale : bool
        If True, load as single-channel grayscale.

    Returns
    -------
    image : np.ndarray
        Loaded image (H, W) if grayscale, (H, W, 3) if color.
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, flag)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def save_image(image, path):
    """
    Save an image to disk, creating parent directories if needed.

    Parameters
    ----------
    image : np.ndarray
        Image to save.
    path : str
        Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def list_images(directory):
    """
    List image files in a directory, sorted alphabetically.

    Parameters
    ----------
    directory : str
        Path to directory.

    Returns
    -------
    paths : list of str
        Sorted list of full paths to image files.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = []
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(IMAGE_EXTENSIONS):
            files.append(os.path.join(directory, f))
    return files


def remove_scale_bar(image):
    """
    Mask the bottom region of an SEM image (instrument metadata / scale bar).
    Replaces the region with the median intensity of the area just above it.

    Parameters
    ----------
    image : np.ndarray
        Grayscale SEM image (H, W).

    Returns
    -------
    cleaned : np.ndarray
        Image with bottom metadata region replaced.
    """
    h, w = image.shape[:2]
    cutoff = int(h * (1 - SCALE_BAR_FRACTION))

    # Compute fill value from the strip just above the metadata region
    ref_strip = image[max(0, cutoff - 20):cutoff, :]
    fill_value = int(np.median(ref_strip))

    cleaned = image.copy()
    cleaned[cutoff:, :] = fill_value
    return cleaned


def remove_text_overlay(image):
    """
    Detect and inpaint bright text overlays on SEM images.
    Uses thresholding to find very bright pixels (text/annotations),
    filters small connected components, and inpaints them.

    Parameters
    ----------
    image : np.ndarray
        Grayscale SEM image (H, W).

    Returns
    -------
    cleaned : np.ndarray
        Image with text overlays inpainted.
    """
    # Bright text is typically near-white on SEM images
    thresh = int(np.percentile(image, 99.5))
    text_mask = (image >= thresh).astype(np.uint8) * 255

    # Dilate slightly to cover text edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)

    # Keep only small connected components (text characters, not large bright regions)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
    filtered_mask = np.zeros_like(text_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Text components are small (< 2000 px) but not tiny noise (> 5 px)
        if 5 < area < 2000:
            filtered_mask[labels == i] = 255

    if np.sum(filtered_mask) == 0:
        return image

    # Inpaint the detected text regions
    cleaned = cv2.inpaint(image, filtered_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return cleaned


def clean_sem_image(image):
    """
    Full SEM image cleaning: remove scale bar then text overlays.

    Parameters
    ----------
    image : np.ndarray
        Grayscale SEM image (H, W).

    Returns
    -------
    cleaned : np.ndarray
        Cleaned image.
    """
    cleaned = remove_scale_bar(image)
    cleaned = remove_text_overlay(cleaned)
    return cleaned


def create_overlay(image, mask, color=(0, 255, 0), alpha=0.4):
    """
    Create a semi-transparent colored overlay of a mask on a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W) or color image (H, W, 3).
    mask : np.ndarray
        Binary mask (H, W), values 0 or 255.
    color : tuple
        BGR color for the overlay.
    alpha : float
        Overlay transparency (0 = invisible, 1 = opaque).

    Returns
    -------
    overlay : np.ndarray
        Color image (H, W, 3) with mask overlay.
    """
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    overlay = base.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = color

    result = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)
    return result


def create_comparison_strip(images, titles, height=400):
    """
    Create a horizontal strip of images with titles for visual comparison.
    All images are resized to the same height and stacked horizontally.

    Parameters
    ----------
    images : list of np.ndarray
        Images to display (grayscale or color).
    titles : list of str
        Title for each image.
    height : int
        Target height for all panels.

    Returns
    -------
    strip : np.ndarray
        Horizontally concatenated comparison image (H, W, 3).
    """
    panels = []
    for img, title in zip(images, titles):
        # Convert to color if needed
        if img.ndim == 2:
            panel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            panel = img.copy()

        # Resize to target height, preserve aspect ratio
        h, w = panel.shape[:2]
        scale = height / h
        new_w = int(w * scale)
        panel = cv2.resize(panel, (new_w, height))

        # Add title bar at top
        title_bar = np.zeros((40, new_w, 3), dtype=np.uint8)
        cv2.putText(title_bar, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)

        panel = np.vstack([title_bar, panel])
        panels.append(panel)

    # Pad panels to same height before hstacking
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    strip = np.hstack(padded)
    return strip


if __name__ == "__main__":
    # Synthetic test: create a fake SEM image and test all utilities
    print("=== utils.py — Synthetic Self-Test ===\n")

    # Create a synthetic SEM-like image (512x512)
    np.random.seed(42)
    synth = np.random.randint(40, 180, (512, 512), dtype=np.uint8)
    # Add a bright "scale bar" region at bottom
    synth[450:, :] = 220
    # Add some bright "text" pixels
    synth[30:35, 100:130] = 255
    synth[30:35, 140:160] = 255

    print(f"Synthetic image shape: {synth.shape}")
    print(f"Pixel range: [{synth.min()}, {synth.max()}]")

    # Test cleaning
    cleaned = clean_sem_image(synth)
    print(f"After cleaning — bottom row mean: {cleaned[500, :].mean():.1f} "
          f"(was {synth[500, :].mean():.1f})")

    # Test overlay
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 255
    overlay = create_overlay(synth, mask)
    print(f"Overlay shape: {overlay.shape}, dtype: {overlay.dtype}")

    # Test comparison strip
    strip = create_comparison_strip(
        [synth, cleaned, mask],
        ["Original", "Cleaned", "Mask"]
    )
    print(f"Comparison strip shape: {strip.shape}")

    # Save outputs
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    save_image(synth, os.path.join(out_dir, "synth_original.png"))
    save_image(cleaned, os.path.join(out_dir, "synth_cleaned.png"))
    save_image(strip, os.path.join(out_dir, "synth_comparison.png"))
    print(f"\nSaved test outputs to {out_dir}/")
    print("All utils tests passed.")
