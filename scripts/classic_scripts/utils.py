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
SCALE_BAR_FRACTION = 0.06398537477148080438756855575868


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
    Crops out the bottom region.

    Parameters
    ----------
    image : np.ndarray
        Grayscale SEM image (H, W).

    Returns
    -------
    cleaned : np.ndarray
        Image with bottom metadata region removed.
    """
    h, w = image.shape[:2]
    cutoff = int(h * (1 - SCALE_BAR_FRACTION))

    cleaned = image.copy()
    cleaned_cutoff = cleaned[:cutoff, :]
    return cleaned_cutoff


def _to_display_uint8(image):
    """
    Convert arbitrary image dtypes/ranges to uint8 for visualization.
    """
    arr = np.asarray(image)

    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.bool_:
        return (arr.astype(np.uint8) * 255)

    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    min_val = float(arr[finite].min())
    max_val = float(arr[finite].max())
    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = (arr - min_val) * (255.0 / (max_val - min_val))
    return np.clip(scaled, 0, 255).astype(np.uint8)


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
    elif image.ndim == 3 and image.shape[2] == 1:
        base = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        base = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        base = image.copy()

    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    if base.ndim == 3 and base.shape[2] == 1:
        base = np.repeat(base, 3, axis=2)
    base = _to_display_uint8(base)

    # Accept HxW, HxWx1, or other singleton-expanded masks.
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim > 2:
        # Last-resort fallback for unusual layouts.
        mask = mask.reshape(mask.shape[0], mask.shape[1])
    if mask.shape[:2] != base.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_u8 = np.zeros(mask.shape[:2], dtype=np.uint8)
    mask_u8[mask > 0] = 255

    color_layer = np.zeros_like(base, dtype=np.float32)
    color_layer[:, :, 0] = float(color[0])
    color_layer[:, :, 1] = float(color[1])
    color_layer[:, :, 2] = float(color[2])

    base_f = base.astype(np.float32)
    alpha_map = ((mask_u8.astype(np.float32) / 255.0) * float(alpha))[:, :, None]
    result = (1.0 - alpha_map) * base_f + alpha_map * color_layer
    return np.clip(result, 0, 255).astype(np.uint8)


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
        img = _to_display_uint8(img)

        # Convert to color if needed
        if img.ndim == 2:
            panel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            panel = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            panel = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            panel = img.copy()

        # Resize to target height, preserve aspect ratio
        h, w = panel.shape[:2]
        scale = height / h
        new_w = int(w * scale)
        panel = cv2.resize(panel, (new_w, height))
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        elif panel.ndim == 3 and panel.shape[2] == 1:
            panel = cv2.cvtColor(panel[:, :, 0], cv2.COLOR_GRAY2BGR)

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

    strip = np.hstack(padded).astype(np.uint8, copy=False)
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
