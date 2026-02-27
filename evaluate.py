"""
Evaluation module for SEM dendrite segmentation.

Computes pixel-level metrics (Dice, IoU, Precision, Recall) between
predicted and ground truth masks, generates comparison figures, and
produces summary reports for both classic and YOLO pipelines.
"""

import cv2
import numpy as np
import os
import sys

# Add project directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_image, save_image, list_images, create_comparison_strip, create_overlay


# ===========================================================================
# Metrics
# ===========================================================================

def compute_dice(pred, gt):
    """
    Dice similarity coefficient between two binary masks.

    Dice = 2 * |pred AND gt| / (|pred| + |gt|)

    Parameters
    ----------
    pred : np.ndarray
        Predicted binary mask (H, W), values 0 or 255.
    gt : np.ndarray
        Ground truth binary mask (H, W), values 0 or 255.

    Returns
    -------
    dice : float
        Dice score in [0, 1]. Returns 1.0 if both masks are empty.
    """
    pred_bin = (pred > 0).astype(np.float64)
    gt_bin = (gt > 0).astype(np.float64)

    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()

    if pred_sum + gt_sum == 0:
        return 1.0

    intersection = (pred_bin * gt_bin).sum()
    return 2.0 * intersection / (pred_sum + gt_sum)


def compute_iou(pred, gt):
    """
    Intersection over Union (Jaccard index) between two binary masks.

    IoU = |pred AND gt| / |pred OR gt|

    Parameters
    ----------
    pred : np.ndarray
        Predicted binary mask (H, W), values 0 or 255.
    gt : np.ndarray
        Ground truth binary mask (H, W), values 0 or 255.

    Returns
    -------
    iou : float
        IoU score in [0, 1]. Returns 1.0 if both masks are empty.
    """
    pred_bin = (pred > 0).astype(np.float64)
    gt_bin = (gt > 0).astype(np.float64)

    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection

    if union == 0:
        return 1.0

    return intersection / union


def compute_precision_recall(pred, gt):
    """
    Pixel-level precision and recall.

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)

    Parameters
    ----------
    pred : np.ndarray
        Predicted binary mask (H, W), values 0 or 255.
    gt : np.ndarray
        Ground truth binary mask (H, W), values 0 or 255.

    Returns
    -------
    precision : float
        Precision in [0, 1]. Returns 1.0 if no positive predictions.
    recall : float
        Recall in [0, 1]. Returns 1.0 if no ground truth positives.
    """
    pred_bin = (pred > 0)
    gt_bin = (gt > 0)

    tp = float(np.logical_and(pred_bin, gt_bin).sum())
    fp = float(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = float(np.logical_and(~pred_bin, gt_bin).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    return precision, recall


def evaluate_single(pred, gt):
    """
    Compute all metrics for a single prediction/ground-truth pair.

    Parameters
    ----------
    pred : np.ndarray
        Predicted binary mask (H, W), values 0 or 255.
    gt : np.ndarray
        Ground truth binary mask (H, W), values 0 or 255.

    Returns
    -------
    metrics : dict
        Dictionary with keys: dice, iou, precision, recall.
    """
    dice = compute_dice(pred, gt)
    iou = compute_iou(pred, gt)
    precision, recall = compute_precision_recall(pred, gt)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


# ===========================================================================
# Visualization
# ===========================================================================

def create_comparison_figure(source_path, classic_mask, yolo_mask, skeleton,
                             output_path):
    """
    Generate a 4-panel comparison figure:
    Source | Classic Mask | YOLO Mask | Skeleton overlay.

    Parameters
    ----------
    source_path : str
        Path to the original SEM image.
    classic_mask : np.ndarray
        Binary mask from classic pipeline (H, W), values 0 or 255.
    yolo_mask : np.ndarray
        Binary mask from YOLO pipeline (H, W), values 0 or 255.
    skeleton : np.ndarray
        Skeleton image (H, W), values 0 or 255.
    output_path : str
        Path to save the comparison figure.
    """
    source = load_image(source_path, grayscale=True)

    # Create skeleton overlay on source image
    skeleton_overlay = create_overlay(source, skeleton, color=(0, 0, 255), alpha=0.6)

    images = [source, classic_mask, yolo_mask, skeleton_overlay]
    titles = ["Source", "Classic Mask", "YOLO Mask", "Skeleton"]

    strip = create_comparison_strip(images, titles)
    save_image(strip, output_path)


def generate_metrics_summary(results, output_path):
    """
    Format and save a metrics summary for all evaluated images.

    Parameters
    ----------
    results : dict
        Mapping of image name to dict with keys:
        - 'classic': metrics dict (dice, iou, precision, recall)
        - 'yolo': metrics dict (dice, iou, precision, recall)
    output_path : str
        Path to save the summary text file.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("SEM Dendrite Segmentation — Evaluation Summary")
    lines.append("=" * 70)
    lines.append("")

    # Column header
    header = f"{'Image':<25} {'Method':<10} {'Dice':>6} {'IoU':>6} {'Prec':>6} {'Rec':>6}"
    lines.append(header)
    lines.append("-" * 70)

    classic_totals = {"dice": [], "iou": [], "precision": [], "recall": []}
    yolo_totals = {"dice": [], "iou": [], "precision": [], "recall": []}

    for name in sorted(results.keys()):
        entry = results[name]

        if "classic" in entry:
            m = entry["classic"]
            lines.append(
                f"{name:<25} {'Classic':<10} "
                f"{m['dice']:>6.3f} {m['iou']:>6.3f} "
                f"{m['precision']:>6.3f} {m['recall']:>6.3f}"
            )
            for k in classic_totals:
                classic_totals[k].append(m[k])

        if "yolo" in entry:
            m = entry["yolo"]
            lines.append(
                f"{name:<25} {'YOLO':<10} "
                f"{m['dice']:>6.3f} {m['iou']:>6.3f} "
                f"{m['precision']:>6.3f} {m['recall']:>6.3f}"
            )
            for k in yolo_totals:
                yolo_totals[k].append(m[k])

        lines.append("")

    # Averages
    lines.append("-" * 70)
    lines.append("AVERAGES")
    lines.append("-" * 70)

    if classic_totals["dice"]:
        n = len(classic_totals["dice"])
        lines.append(
            f"{'Classic (n=' + str(n) + ')':<25} {'':10} "
            f"{np.mean(classic_totals['dice']):>6.3f} "
            f"{np.mean(classic_totals['iou']):>6.3f} "
            f"{np.mean(classic_totals['precision']):>6.3f} "
            f"{np.mean(classic_totals['recall']):>6.3f}"
        )

    if yolo_totals["dice"]:
        n = len(yolo_totals["dice"])
        lines.append(
            f"{'YOLO (n=' + str(n) + ')':<25} {'':10} "
            f"{np.mean(yolo_totals['dice']):>6.3f} "
            f"{np.mean(yolo_totals['iou']):>6.3f} "
            f"{np.mean(yolo_totals['precision']):>6.3f} "
            f"{np.mean(yolo_totals['recall']):>6.3f}"
        )

    lines.append("")
    lines.append("=" * 70)

    # Failure analysis
    failures = analyze_failures(results)
    failure_report = format_failure_report(failures)
    lines.append(failure_report)

    summary = "\n".join(lines)
    print(summary)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {output_path}")


# ===========================================================================
# Failure analysis
# ===========================================================================

def analyze_failures(results, threshold=0.5):
    """
    Identify and characterize segmentation failures based on metric patterns.

    Root cause characterization rules:
      - Low precision + high recall  → over-segmentation (noise as dendrite)
      - High precision + low recall  → under-segmentation (thin branches missed)
      - Low precision + low recall   → fundamental mismatch (wrong region / artifacts)
      - Classic fails but YOLO ok    → non-uniform illumination (threshold sensitivity)
      - YOLO fails but classic ok    → insufficient training data / OOD sample

    Parameters
    ----------
    results : dict
        Mapping of image name to dict with 'classic' and/or 'yolo' metric dicts.
        Each metric dict has keys: dice, iou, precision, recall.
    threshold : float
        Dice score below this value is considered a failure.

    Returns
    -------
    failures : list of dict
        Each entry: {name, method, dice, precision, recall, cause}.
    """
    prec_threshold = 0.6
    rec_threshold = 0.6

    failures = []

    for name in sorted(results.keys()):
        entry = results[name]
        classic_dice = entry.get("classic", {}).get("dice")
        yolo_dice = entry.get("yolo", {}).get("dice")

        for method_key, method_label in [("classic", "Classic"), ("yolo", "YOLO")]:
            if method_key not in entry:
                continue
            m = entry[method_key]
            if m["dice"] >= threshold:
                continue

            prec = m["precision"]
            rec = m["recall"]

            # Determine root cause
            if prec < prec_threshold and rec >= rec_threshold:
                cause = "Over-segmentation — noise included as dendrite"
            elif prec >= prec_threshold and rec < rec_threshold:
                cause = "Under-segmentation — thin branches missed"
            elif prec < prec_threshold and rec < rec_threshold:
                cause = "Fundamental mismatch — wrong region or severe artifacts"
            else:
                cause = "Marginal failure — metrics near threshold"

            # Cross-method insight
            other_key = "yolo" if method_key == "classic" else "classic"
            other_dice = entry.get(other_key, {}).get("dice")
            if other_dice is not None and other_dice >= threshold:
                if method_key == "classic":
                    cause += " (YOLO succeeds → likely non-uniform illumination)"
                else:
                    cause += " (Classic succeeds → likely OOD sample for YOLO)"

            failures.append({
                "name": name,
                "method": method_label,
                "dice": m["dice"],
                "precision": prec,
                "recall": rec,
                "cause": cause,
            })

    return failures


def format_failure_report(failures):
    """
    Format a list of failure dicts into a readable text report.

    Parameters
    ----------
    failures : list of dict
        Output from analyze_failures().

    Returns
    -------
    report : str
        Formatted failure analysis text.
    """
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("Failure Analysis (Dice < threshold)")
    lines.append("=" * 70)

    if not failures:
        lines.append("  No failures detected — all images above threshold.")
    else:
        lines.append(f"  {len(failures)} failure(s) detected:\n")
        for f in failures:
            lines.append(f"  Image: {f['name']}")
            lines.append(f"    Method:    {f['method']}")
            lines.append(f"    Dice:      {f['dice']:.3f}")
            lines.append(f"    Precision: {f['precision']:.3f}")
            lines.append(f"    Recall:    {f['recall']:.3f}")
            lines.append(f"    Cause:     {f['cause']}")
            lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


# ===========================================================================
# Batch evaluation
# ===========================================================================

def evaluate_all(classic_dir, yolo_dir, gt_dir, image_dir, output_dir):
    """
    Batch evaluation: load masks by matching filenames, compute metrics,
    generate comparison figures, and write summary.

    Expected file naming convention:
        image_dir/<name>.png   — source image
        classic_dir/<name>_mask.png  — classic pipeline mask
        yolo_dir/<name>_mask.png     — YOLO pipeline mask
        gt_dir/<name>.png      — ground truth mask

    Parameters
    ----------
    classic_dir : str
        Directory containing classic pipeline masks (*_mask.png).
    yolo_dir : str
        Directory containing YOLO pipeline masks (*_mask.png).
    gt_dir : str
        Directory containing ground truth binary masks.
    image_dir : str
        Directory containing original source images.
    output_dir : str
        Directory to save comparison figures and summary.

    Returns
    -------
    results : dict
        Evaluation results for all images.
    """
    gt_paths = list_images(gt_dir)
    if not gt_paths:
        print(f"No ground truth images found in {gt_dir}")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for gt_path in gt_paths:
        name = os.path.splitext(os.path.basename(gt_path))[0]
        gt_mask = load_image(gt_path, grayscale=True)

        entry = {}

        # Look for classic mask
        classic_path = os.path.join(classic_dir, f"{name}_mask.png")
        if os.path.exists(classic_path):
            classic_mask = load_image(classic_path, grayscale=True)
            entry["classic"] = evaluate_single(classic_mask, gt_mask)
        else:
            classic_mask = None

        # Look for YOLO mask
        yolo_path = os.path.join(yolo_dir, f"{name}_mask.png")
        if os.path.exists(yolo_path):
            yolo_mask = load_image(yolo_path, grayscale=True)
            entry["yolo"] = evaluate_single(yolo_mask, gt_mask)
        else:
            yolo_mask = None

        # Generate comparison figure if source image exists
        source_path = None
        for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
            candidate = os.path.join(image_dir, f"{name}{ext}")
            if os.path.exists(candidate):
                source_path = candidate
                break

        if source_path and classic_mask is not None and yolo_mask is not None:
            from skimage.morphology import skeletonize
            skel = skeletonize((classic_mask > 0).astype(bool))
            skeleton = (skel.astype(np.uint8) * 255)

            fig_path = os.path.join(output_dir, f"{name}_comparison.png")
            create_comparison_figure(
                source_path, classic_mask, yolo_mask, skeleton, fig_path
            )

        if entry:
            results[name] = entry
            print(f"  {name}: ", end="")
            if "classic" in entry:
                print(f"Classic Dice={entry['classic']['dice']:.3f}  ", end="")
            if "yolo" in entry:
                print(f"YOLO Dice={entry['yolo']['dice']:.3f}", end="")
            print()

    # Write summary
    if results:
        summary_path = os.path.join(output_dir, "metrics_summary.txt")
        generate_metrics_summary(results, summary_path)

    return results


# ===========================================================================
# Synthetic self-test
# ===========================================================================

if __name__ == "__main__":
    print("=== evaluate.py — Synthetic Self-Test ===\n")

    # --- Test 1: Perfect overlap → Dice=1.0, IoU=1.0 ---
    mask_a = np.zeros((100, 100), dtype=np.uint8)
    mask_a[20:80, 20:80] = 255
    mask_b = mask_a.copy()

    metrics = evaluate_single(mask_a, mask_b)
    print(f"Test 1 — Perfect overlap:")
    print(f"  Dice={metrics['dice']:.4f}  IoU={metrics['iou']:.4f}  "
          f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}")
    assert abs(metrics["dice"] - 1.0) < 1e-9, f"Expected Dice=1.0, got {metrics['dice']}"
    assert abs(metrics["iou"] - 1.0) < 1e-9, f"Expected IoU=1.0, got {metrics['iou']}"

    # --- Test 2: Partial overlap → Dice ~0.667, IoU ~0.5 ---
    # pred: rows 0-59 (6000 px), gt: rows 20-79 (6000 px), overlap: rows 20-59 (4000 px)
    # Dice = 2*4000/(6000+6000) = 0.667, IoU = 4000/8000 = 0.5
    mask_c = np.zeros((100, 100), dtype=np.uint8)
    mask_c[0:60, 0:100] = 255
    mask_d = np.zeros((100, 100), dtype=np.uint8)
    mask_d[20:80, 0:100] = 255

    metrics2 = evaluate_single(mask_c, mask_d)
    print(f"\nTest 2 — Partial overlap:")
    print(f"  Dice={metrics2['dice']:.4f}  IoU={metrics2['iou']:.4f}  "
          f"Prec={metrics2['precision']:.4f}  Rec={metrics2['recall']:.4f}")
    assert abs(metrics2["dice"] - 2/3) < 0.01, f"Expected Dice~0.667, got {metrics2['dice']}"
    assert abs(metrics2["iou"] - 0.5) < 0.01, f"Expected IoU~0.5, got {metrics2['iou']}"

    # --- Test 3: No overlap → Dice=0.0, IoU=0.0 ---
    mask_e = np.zeros((100, 100), dtype=np.uint8)
    mask_e[0:30, :] = 255
    mask_f = np.zeros((100, 100), dtype=np.uint8)
    mask_f[70:100, :] = 255

    metrics3 = evaluate_single(mask_e, mask_f)
    print(f"\nTest 3 — No overlap:")
    print(f"  Dice={metrics3['dice']:.4f}  IoU={metrics3['iou']:.4f}  "
          f"Prec={metrics3['precision']:.4f}  Rec={metrics3['recall']:.4f}")
    assert abs(metrics3["dice"] - 0.0) < 1e-9, f"Expected Dice=0.0, got {metrics3['dice']}"
    assert abs(metrics3["iou"] - 0.0) < 1e-9, f"Expected IoU=0.0, got {metrics3['iou']}"

    # --- Test 4: Both empty → Dice=1.0, IoU=1.0 ---
    empty = np.zeros((100, 100), dtype=np.uint8)
    metrics4 = evaluate_single(empty, empty)
    print(f"\nTest 4 — Both empty:")
    print(f"  Dice={metrics4['dice']:.4f}  IoU={metrics4['iou']:.4f}")
    assert abs(metrics4["dice"] - 1.0) < 1e-9
    assert abs(metrics4["iou"] - 1.0) < 1e-9

    # --- Test 5: Generate synthetic comparison figure ---
    print("\nTest 5 — Comparison figure generation:")
    project_dir = os.path.dirname(__file__)
    out_dir = os.path.join(project_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    # Create a synthetic source image and save it
    np.random.seed(42)
    synth_source = np.random.randint(40, 180, (256, 256), dtype=np.uint8)
    cv2.line(synth_source, (60, 20), (60, 230), 200, 3)
    cv2.line(synth_source, (60, 120), (180, 80), 190, 2)
    source_path = os.path.join(out_dir, "synth_eval_source.png")
    save_image(synth_source, source_path)

    # Synthetic masks
    classic_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(classic_mask, (58, 20), (58, 230), 255, 5)
    cv2.line(classic_mask, (58, 120), (178, 80), 255, 4)

    yolo_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(yolo_mask, (62, 20), (62, 230), 255, 6)
    cv2.line(yolo_mask, (62, 120), (182, 80), 255, 5)

    skeleton = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(skeleton, (60, 20), (60, 230), 255, 1)
    cv2.line(skeleton, (60, 120), (180, 80), 255, 1)

    fig_path = os.path.join(out_dir, "synth_comparison.png")
    create_comparison_figure(source_path, classic_mask, yolo_mask, skeleton, fig_path)
    print(f"  Saved comparison figure: {fig_path}")

    # --- Test 6: Metrics summary ---
    print("\nTest 6 — Metrics summary:")
    summary_results = {
        "image_001": {
            "classic": {"dice": 0.85, "iou": 0.74, "precision": 0.90, "recall": 0.81},
            "yolo": {"dice": 0.91, "iou": 0.84, "precision": 0.93, "recall": 0.89},
        },
        "image_002": {
            "classic": {"dice": 0.78, "iou": 0.64, "precision": 0.82, "recall": 0.75},
            "yolo": {"dice": 0.88, "iou": 0.79, "precision": 0.91, "recall": 0.86},
        },
    }
    summary_path = os.path.join(out_dir, "synth_metrics_summary.txt")
    generate_metrics_summary(summary_results, summary_path)

    # --- Test 7: Failure analysis ---
    print("\nTest 7 — Failure analysis:")
    failure_results = {
        "img_good": {
            "classic": {"dice": 0.85, "iou": 0.74, "precision": 0.90, "recall": 0.81},
            "yolo": {"dice": 0.91, "iou": 0.84, "precision": 0.93, "recall": 0.89},
        },
        "img_over_seg": {
            "classic": {"dice": 0.35, "iou": 0.21, "precision": 0.30, "recall": 0.80},
            "yolo": {"dice": 0.88, "iou": 0.79, "precision": 0.91, "recall": 0.86},
        },
        "img_under_seg": {
            "classic": {"dice": 0.40, "iou": 0.25, "precision": 0.85, "recall": 0.30},
        },
        "img_mismatch": {
            "yolo": {"dice": 0.20, "iou": 0.11, "precision": 0.15, "recall": 0.25},
            "classic": {"dice": 0.75, "iou": 0.60, "precision": 0.80, "recall": 0.70},
        },
    }

    failures = analyze_failures(failure_results, threshold=0.5)
    report = format_failure_report(failures)
    print(report)

    # Verify expected failures
    failure_names = [(f["name"], f["method"]) for f in failures]
    assert ("img_good", "Classic") not in failure_names, "img_good Classic should not fail"
    assert ("img_good", "YOLO") not in failure_names, "img_good YOLO should not fail"
    assert ("img_over_seg", "Classic") in failure_names, "img_over_seg Classic should fail"
    assert ("img_under_seg", "Classic") in failure_names, "img_under_seg Classic should fail"
    assert ("img_mismatch", "YOLO") in failure_names, "img_mismatch YOLO should fail"

    # Verify root cause patterns
    over_seg = [f for f in failures if f["name"] == "img_over_seg"][0]
    assert "Over-segmentation" in over_seg["cause"], \
        f"Expected over-segmentation cause, got: {over_seg['cause']}"
    under_seg = [f for f in failures if f["name"] == "img_under_seg"][0]
    assert "Under-segmentation" in under_seg["cause"], \
        f"Expected under-segmentation cause, got: {under_seg['cause']}"
    mismatch = [f for f in failures if f["name"] == "img_mismatch"][0]
    assert "Fundamental mismatch" in mismatch["cause"], \
        f"Expected fundamental mismatch cause, got: {mismatch['cause']}"

    print("  All failure analysis assertions passed.")

    print("\nAll evaluation tests passed.")
