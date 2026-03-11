"""
Evaluate precomputed Classic/YOLO masks on a dataset split for report artifacts.

This script computes overlap metrics against ground truth masks:
- Dice
- IoU
- Precision
- Recall

It writes report-friendly outputs:
- metrics_per_image.csv
- metrics_averages.csv
- failures.csv
- metrics_summary.txt
- optional comparison figures (Source | Classic | YOLO | Skeleton)
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
METRIC_KEYS = ("dice", "iou", "precision", "recall")


def list_images(directory: str) -> List[str]:
    """Return sorted image paths in a directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    paths: List[str] = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            paths.append(os.path.join(directory, name))
    return paths


def to_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Convert any mask-like array to uint8 {0,255}."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    out = np.zeros(mask.shape[:2], dtype=np.uint8)
    out[mask > 0] = 255
    return out


def load_binary_mask(path: Optional[str]) -> Optional[np.ndarray]:
    """Load a mask path as binary uint8 mask."""
    if not path or not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return to_binary_mask(img)


def find_mask_path(mask_dir: str, name: str) -> Optional[str]:
    """Find predicted mask path by name in multiple common layouts."""
    if not os.path.isdir(mask_dir):
        return None

    direct_candidates = [
        os.path.join(mask_dir, f"{name}_mask.png"),
        os.path.join(mask_dir, f"{name}.png"),
        os.path.join(mask_dir, name, "10_separated.png"),
        os.path.join(mask_dir, name, "06_segmented.png"),
    ]
    for candidate in direct_candidates:
        if os.path.isfile(candidate):
            return candidate

    # First prefer files under a folder named exactly like the image stem.
    for root, _, files in os.walk(mask_dir):
        if os.path.basename(root) != name:
            continue
        for filename in ("10_separated.png", "06_segmented.png", f"{name}_mask.png", f"{name}.png"):
            if filename in files:
                return os.path.join(root, filename)

    # Fallback: search globally for a canonical <name>_mask.png.
    target = f"{name}_mask.png"
    for root, _, files in os.walk(mask_dir):
        if target in files:
            return os.path.join(root, target)

    return None


def find_gt_path(gt_dir: str, name: str) -> Optional[str]:
    """Find ground-truth mask by stem in root or nested folders."""
    if not os.path.isdir(gt_dir):
        return None

    # Common direct locations.
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(gt_dir, f"{name}{ext}")
        if os.path.isfile(candidate):
            return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(gt_dir, "test", f"{name}{ext}")
        if os.path.isfile(candidate):
            return candidate

    # Recursive fallback.
    valid_names = {f"{name}{ext}" for ext in IMAGE_EXTENSIONS}
    for root, _, files in os.walk(gt_dir):
        for filename in files:
            if filename in valid_names:
                return os.path.join(root, filename)

    return None


def rasterize_yolo_label(label_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize YOLO polygon TXT into a binary mask."""
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = np.asarray(parts[1:], dtype=np.float32)
        if coords.size < 6 or coords.size % 2 != 0:
            continue

        xs = coords[0::2]
        ys = coords[1::2]
        xpix = np.clip(np.round(xs * (width - 1)), 0, width - 1).astype(np.int32)
        ypix = np.clip(np.round(ys * (height - 1)), 0, height - 1).astype(np.int32)
        pts = np.stack([xpix, ypix], axis=1).reshape(-1, 1, 2)
        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts], 255)

    return mask


def resolve_gt_mask(
    name: str,
    gt_dir: str,
    source_path: str,
    labels_dir: Optional[str],
    synth_out_dir: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[str], str]:
    """
    Resolve GT mask from gt_dir, optionally falling back to YOLO labels.

    Returns:
        (mask, reference_path, source_type)
        source_type in {"gt_file", "label_rasterized", "missing"}
    """
    gt_path = find_gt_path(gt_dir, name)
    if gt_path:
        gt_mask = load_binary_mask(gt_path)
        if gt_mask is not None:
            return gt_mask, gt_path, "gt_file"

    if labels_dir and os.path.isdir(labels_dir):
        label_path = os.path.join(labels_dir, f"{name}.txt")
        if os.path.isfile(label_path):
            src = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
            if src is not None:
                gt_mask = rasterize_yolo_label(label_path, src.shape[:2])
                if synth_out_dir:
                    os.makedirs(synth_out_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(synth_out_dir, f"{name}.png"), gt_mask)
                return gt_mask, label_path, "label_rasterized"

    return None, None, "missing"


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient."""
    pred_bin = (pred > 0).astype(np.float64)
    gt_bin = (gt > 0).astype(np.float64)
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return 1.0
    intersection = (pred_bin * gt_bin).sum()
    return float((2.0 * intersection) / denom)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union."""
    pred_bin = (pred > 0).astype(np.float64)
    gt_bin = (gt > 0).astype(np.float64)
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Pixel precision and recall."""
    pred_bin = pred > 0
    gt_bin = gt > 0
    tp = float(np.logical_and(pred_bin, gt_bin).sum())
    fp = float(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = float(np.logical_and(~pred_bin, gt_bin).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return precision, recall


def evaluate_single(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Evaluate one predicted mask against one GT mask."""
    precision, recall = compute_precision_recall(pred, gt)
    return {
        "dice": compute_dice(pred, gt),
        "iou": compute_iou(pred, gt),
        "precision": precision,
        "recall": recall,
    }


def analyze_failures(
    rows: List[Dict[str, object]],
    threshold: float,
) -> List[Dict[str, object]]:
    """Classify failed cases using precision/recall patterns."""
    failures: List[Dict[str, object]] = []
    prec_threshold = 0.6
    rec_threshold = 0.6

    for row in rows:
        name = row["name"]
        for method_key, method_label in (("classic", "Classic"), ("yolo", "YOLO")):
            metrics = row.get(method_key)
            if not isinstance(metrics, dict):
                continue
            dice = float(metrics["dice"])
            prec = float(metrics["precision"])
            rec = float(metrics["recall"])
            if dice >= threshold:
                continue

            if prec < prec_threshold and rec >= rec_threshold:
                cause = "Over-segmentation - noise included as dendrite"
            elif prec >= prec_threshold and rec < rec_threshold:
                cause = "Under-segmentation - thin branches missed"
            elif prec < prec_threshold and rec < rec_threshold:
                cause = "Fundamental mismatch - wrong region or severe artifacts"
            else:
                cause = "Marginal failure - metrics near threshold"

            other_key = "yolo" if method_key == "classic" else "classic"
            other = row.get(other_key)
            if isinstance(other, dict) and float(other["dice"]) >= threshold:
                if method_key == "classic":
                    cause += " (YOLO succeeds -> likely illumination sensitivity)"
                else:
                    cause += " (Classic succeeds -> likely OOD sample for YOLO)"

            failures.append(
                {
                    "name": name,
                    "method": method_label,
                    "dice": dice,
                    "precision": prec,
                    "recall": rec,
                    "cause": cause,
                }
            )

    failures.sort(key=lambda f: (f["dice"], f["name"], f["method"]))
    return failures


def mean_metrics(rows: List[Dict[str, object]], method: str) -> Dict[str, Optional[float]]:
    """Mean metrics for one method over available rows."""
    vals = {key: [] for key in METRIC_KEYS}
    for row in rows:
        metrics = row.get(method)
        if not isinstance(metrics, dict):
            continue
        for key in METRIC_KEYS:
            vals[key].append(float(metrics[key]))

    out: Dict[str, Optional[float]] = {}
    for key in METRIC_KEYS:
        out[key] = float(np.mean(vals[key])) if vals[key] else None
    return out


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure BGR output."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def make_titled_panel(image: np.ndarray, title: str, target_height: int = 320) -> np.ndarray:
    """Resize image and add title bar."""
    panel = to_bgr(image)
    h, w = panel.shape[:2]
    scale = target_height / max(1, h)
    new_w = max(1, int(round(w * scale)))
    panel = cv2.resize(panel, (new_w, target_height), interpolation=cv2.INTER_AREA)

    title_bar = np.zeros((36, new_w, 3), dtype=np.uint8)
    cv2.putText(
        title_bar,
        title,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([title_bar, panel])


def make_comparison_figure(
    source_path: str,
    classic_mask: Optional[np.ndarray],
    yolo_mask: Optional[np.ndarray],
    output_path: str,
) -> bool:
    """Create Source | Classic | YOLO | Skeleton strip."""
    source = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    if source is None:
        return False

    h, w = source.shape[:2]
    if classic_mask is None:
        classic_mask = np.zeros((h, w), dtype=np.uint8)
    if yolo_mask is None:
        yolo_mask = np.zeros((h, w), dtype=np.uint8)

    if classic_mask.shape != (h, w):
        classic_mask = cv2.resize(classic_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if yolo_mask.shape != (h, w):
        yolo_mask = cv2.resize(yolo_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    skel = (skeletonize(classic_mask > 0).astype(np.uint8) * 255)
    overlay = to_bgr(source)
    overlay[skel > 0] = (0, 0, 255)

    panels = [
        make_titled_panel(source, "Source"),
        make_titled_panel(classic_mask, "Classic Mask"),
        make_titled_panel(yolo_mask, "YOLO Mask"),
        make_titled_panel(overlay, "Skeleton"),
    ]

    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    strip = np.hstack(padded)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, strip)
    return True


def write_outputs(
    rows: List[Dict[str, object]],
    failures: List[Dict[str, object]],
    out_dir: str,
    failure_threshold: float,
    stats: Dict[str, int],
) -> Dict[str, str]:
    """Write CSV and text report outputs."""
    os.makedirs(out_dir, exist_ok=True)

    per_image_csv = os.path.join(out_dir, "metrics_per_image.csv")
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "classic_dice",
                "classic_iou",
                "classic_precision",
                "classic_recall",
                "yolo_dice",
                "yolo_iou",
                "yolo_precision",
                "yolo_recall",
                "delta_dice_yolo_minus_classic",
            ]
        )
        for row in rows:
            classic = row.get("classic")
            yolo = row.get("yolo")
            delta_dice = None
            if isinstance(classic, dict) and isinstance(yolo, dict):
                delta_dice = float(yolo["dice"]) - float(classic["dice"])
            writer.writerow(
                [
                    row["name"],
                    "" if not isinstance(classic, dict) else f"{classic['dice']:.6f}",
                    "" if not isinstance(classic, dict) else f"{classic['iou']:.6f}",
                    "" if not isinstance(classic, dict) else f"{classic['precision']:.6f}",
                    "" if not isinstance(classic, dict) else f"{classic['recall']:.6f}",
                    "" if not isinstance(yolo, dict) else f"{yolo['dice']:.6f}",
                    "" if not isinstance(yolo, dict) else f"{yolo['iou']:.6f}",
                    "" if not isinstance(yolo, dict) else f"{yolo['precision']:.6f}",
                    "" if not isinstance(yolo, dict) else f"{yolo['recall']:.6f}",
                    "" if delta_dice is None else f"{delta_dice:.6f}",
                ]
            )

    classic_avg = mean_metrics(rows, "classic")
    yolo_avg = mean_metrics(rows, "yolo")
    classic_n = sum(1 for r in rows if isinstance(r.get("classic"), dict))
    yolo_n = sum(1 for r in rows if isinstance(r.get("yolo"), dict))

    averages_csv = os.path.join(out_dir, "metrics_averages.csv")
    with open(averages_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n", "dice", "iou", "precision", "recall"])
        writer.writerow(
            [
                "classic",
                classic_n,
                "" if classic_avg["dice"] is None else f"{classic_avg['dice']:.6f}",
                "" if classic_avg["iou"] is None else f"{classic_avg['iou']:.6f}",
                "" if classic_avg["precision"] is None else f"{classic_avg['precision']:.6f}",
                "" if classic_avg["recall"] is None else f"{classic_avg['recall']:.6f}",
            ]
        )
        writer.writerow(
            [
                "yolo",
                yolo_n,
                "" if yolo_avg["dice"] is None else f"{yolo_avg['dice']:.6f}",
                "" if yolo_avg["iou"] is None else f"{yolo_avg['iou']:.6f}",
                "" if yolo_avg["precision"] is None else f"{yolo_avg['precision']:.6f}",
                "" if yolo_avg["recall"] is None else f"{yolo_avg['recall']:.6f}",
            ]
        )

    failures_csv = os.path.join(out_dir, "failures.csv")
    with open(failures_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "method", "dice", "precision", "recall", "cause"])
        for failure in failures:
            writer.writerow(
                [
                    failure["name"],
                    failure["method"],
                    f"{failure['dice']:.6f}",
                    f"{failure['precision']:.6f}",
                    f"{failure['recall']:.6f}",
                    failure["cause"],
                ]
            )

    summary_path = os.path.join(out_dir, "metrics_summary.txt")
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("SEM Dendrite Segmentation - Test Evaluation Summary")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Total test images: {stats['num_test_images']}")
    lines.append(f"Images with GT: {stats['num_with_gt']}")
    lines.append(f"Classic predictions found: {stats['num_with_classic_pred']}")
    lines.append(f"YOLO predictions found: {stats['num_with_yolo_pred']}")
    lines.append(f"Classic images evaluated: {classic_n}")
    lines.append(f"YOLO images evaluated: {yolo_n}")
    lines.append(f"GT rasterized from labels: {stats['num_gt_from_labels']}")
    lines.append("")
    lines.append("Averages")
    lines.append("-" * 72)
    lines.append(
        "Classic: Dice={0} IoU={1} Precision={2} Recall={3}".format(
            "--" if classic_avg["dice"] is None else f"{classic_avg['dice']:.3f}",
            "--" if classic_avg["iou"] is None else f"{classic_avg['iou']:.3f}",
            "--" if classic_avg["precision"] is None else f"{classic_avg['precision']:.3f}",
            "--" if classic_avg["recall"] is None else f"{classic_avg['recall']:.3f}",
        )
    )
    lines.append(
        "YOLO:    Dice={0} IoU={1} Precision={2} Recall={3}".format(
            "--" if yolo_avg["dice"] is None else f"{yolo_avg['dice']:.3f}",
            "--" if yolo_avg["iou"] is None else f"{yolo_avg['iou']:.3f}",
            "--" if yolo_avg["precision"] is None else f"{yolo_avg['precision']:.3f}",
            "--" if yolo_avg["recall"] is None else f"{yolo_avg['recall']:.3f}",
        )
    )
    lines.append("")
    lines.append(f"Failure threshold (Dice): {failure_threshold:.3f}")
    lines.append(f"Failures detected: {len(failures)}")
    lines.append("=" * 72)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {
        "per_image_csv": per_image_csv,
        "averages_csv": averages_csv,
        "failures_csv": failures_csv,
        "summary_txt": summary_path,
    }


def parse_args() -> argparse.Namespace:
    """CLI args."""
    parser = argparse.ArgumentParser(
        description="Evaluate precomputed Classic/YOLO masks and export report-ready metrics."
    )
    parser.add_argument(
        "--images-dir",
        default=os.path.join("dataset", "yolo_dataset", "test", "images"),
        help="Directory of test source images.",
    )
    parser.add_argument(
        "--gt-dir",
        default="ground_truth_masks",
        help="Ground-truth mask directory.",
    )
    parser.add_argument(
        "--classic-dir",
        required=True,
        help="Classic prediction masks directory.",
    )
    parser.add_argument(
        "--yolo-dir",
        required=True,
        help="YOLO prediction masks directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("output", "evaluation", "test_report"),
        help="Output directory for metrics artifacts.",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="Optional YOLO label directory to rasterize missing GT masks.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.5,
        help="Dice threshold for failure analysis.",
    )
    parser.add_argument(
        "--no-comparisons",
        action="store_true",
        help="Disable comparison figure generation.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    if not os.path.isdir(args.classic_dir):
        raise FileNotFoundError(f"Classic directory not found: {args.classic_dir}")
    if not os.path.isdir(args.yolo_dir):
        raise FileNotFoundError(f"YOLO directory not found: {args.yolo_dir}")
    if args.labels_dir and not os.path.isdir(args.labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")
    if not os.path.isdir(args.gt_dir):
        raise FileNotFoundError(f"GT directory not found: {args.gt_dir}")

    image_paths = list_images(args.images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {args.images_dir}")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    comparisons_dir = os.path.join(out_dir, "comparisons")
    synth_gt_dir = os.path.join(out_dir, "synthesized_gt")

    rows: List[Dict[str, object]] = []
    missing_gt: List[str] = []
    missing_classic_pred: List[str] = []
    missing_yolo_pred: List[str] = []

    num_with_gt = 0
    num_with_classic_pred = 0
    num_with_yolo_pred = 0
    num_gt_from_labels = 0
    num_comparisons = 0

    for source_path in image_paths:
        name = os.path.splitext(os.path.basename(source_path))[0]
        classic_path = find_mask_path(args.classic_dir, name)
        yolo_path = find_mask_path(args.yolo_dir, name)

        gt_mask, gt_ref, gt_source = resolve_gt_mask(
            name=name,
            gt_dir=args.gt_dir,
            source_path=source_path,
            labels_dir=args.labels_dir,
            synth_out_dir=synth_gt_dir if args.labels_dir else None,
        )
        classic_mask = load_binary_mask(classic_path)
        yolo_mask = load_binary_mask(yolo_path)

        if gt_mask is None:
            missing_gt.append(name)
        else:
            num_with_gt += 1
            if gt_source == "label_rasterized":
                num_gt_from_labels += 1

        if classic_mask is None:
            missing_classic_pred.append(name)
        else:
            num_with_classic_pred += 1

        if yolo_mask is None:
            missing_yolo_pred.append(name)
        else:
            num_with_yolo_pred += 1

        row: Dict[str, object] = {
            "name": name,
            "source_path": source_path,
            "gt_ref": gt_ref,
            "classic_path": classic_path,
            "yolo_path": yolo_path,
            "classic": None,
            "yolo": None,
        }

        if gt_mask is not None and classic_mask is not None:
            row["classic"] = evaluate_single(classic_mask, gt_mask)
        if gt_mask is not None and yolo_mask is not None:
            row["yolo"] = evaluate_single(yolo_mask, gt_mask)

        rows.append(row)

        if not args.no_comparisons:
            out_path = os.path.join(comparisons_dir, f"{name}_comparison.png")
            ok = make_comparison_figure(source_path, classic_mask, yolo_mask, out_path)
            if ok:
                num_comparisons += 1

    failures = analyze_failures(rows, threshold=args.failure_threshold)
    stats = {
        "num_test_images": len(image_paths),
        "num_with_gt": num_with_gt,
        "num_with_classic_pred": num_with_classic_pred,
        "num_with_yolo_pred": num_with_yolo_pred,
        "num_gt_from_labels": num_gt_from_labels,
    }
    output_paths = write_outputs(
        rows=rows,
        failures=failures,
        out_dir=out_dir,
        failure_threshold=args.failure_threshold,
        stats=stats,
    )

    if missing_gt:
        print(f"Missing GT for {len(missing_gt)} image(s): {', '.join(sorted(missing_gt))}")
    if missing_classic_pred:
        print(
            f"Missing Classic predictions for {len(missing_classic_pred)} image(s): "
            f"{', '.join(sorted(missing_classic_pred))}"
        )
    if missing_yolo_pred:
        print(
            f"Missing YOLO predictions for {len(missing_yolo_pred)} image(s): "
            f"{', '.join(sorted(missing_yolo_pred))}"
        )

    print(f"Evaluated test images: {len(image_paths)}")
    print(f"Classic evaluated pairs: {sum(1 for r in rows if isinstance(r.get('classic'), dict))}")
    print(f"YOLO evaluated pairs: {sum(1 for r in rows if isinstance(r.get('yolo'), dict))}")
    print(f"Failures detected: {len(failures)}")
    print(f"Comparison figures: {num_comparisons}")
    print(f"Per-image CSV: {output_paths['per_image_csv']}")
    print(f"Averages CSV: {output_paths['averages_csv']}")
    print(f"Failures CSV: {output_paths['failures_csv']}")
    print(f"Summary TXT: {output_paths['summary_txt']}")


if __name__ == "__main__":
    main()
