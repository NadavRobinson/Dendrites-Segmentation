"""
End-to-end orchestrator for SEM dendrite segmentation.

Runs both pipelines on the same images and produces all required
deliverables: masks, skeletons, comparison figures, metrics summary
with failure analysis.

Usage:
    python run_all.py --images <dir> [--gt <dir>] [--yolo-model <path>] [--output <dir>]
"""

import argparse
import cv2
import numpy as np
import os
import sys

# Add project/script directories to path for imports.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "classic_scripts"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "yolo_scripts"))

from classic_pipeline import apply_parameter_profile, process_all_images, skeletonize_mask
from evaluate import create_comparison_figure, evaluate_all
from utils import save_image, list_images
from yolo_pipeline import predict_batch


def _find_split_subdirs(images_dir):
    """
    Detect split layout:
      images_dir/easy + images_dir/hard (case-insensitive first letter).
    """
    easy_dir = None
    hard_dir = None

    for candidate in ("easy", "Easy"):
        path = os.path.join(images_dir, candidate)
        if os.path.isdir(path):
            easy_dir = path
            break

    for candidate in ("hard", "Hard"):
        path = os.path.join(images_dir, candidate)
        if os.path.isdir(path):
            hard_dir = path
            break

    if easy_dir and hard_dir:
        return {"easy": easy_dir, "hard": hard_dir}
    return {}


def _collect_image_paths(images_dir, split_subdirs):
    """Collect images from either one directory or easy/hard subdirectories."""
    if split_subdirs:
        paths = []
        for subset in ("easy", "hard"):
            paths.extend(list_images(split_subdirs[subset]))
        return sorted(paths)
    return list_images(images_dir)


def run_orchestrator(images_dir, gt_dir=None, yolo_model=None, output_dir=None):
    """
    Run the full segmentation orchestrator.

    Parameters
    ----------
    images_dir : str
        Directory containing source SEM images.
    gt_dir : str or None
        Directory containing ground truth masks (optional).
    yolo_model : str or None
        Path to trained YOLO weights (optional; skip YOLO if not provided).
    output_dir : str or None
        Output root directory. Defaults to output/.

    Returns
    -------
    summary : dict
        Summary of the orchestrator run with counts and paths.
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(project_dir, "output")

    classic_dir = os.path.join(output_dir, "classic")
    yolo_dir = os.path.join(output_dir, "yolo")
    compare_dir = os.path.join(output_dir, "comparisons")
    eval_dir = os.path.join(output_dir, "evaluation")

    split_subdirs = _find_split_subdirs(images_dir)
    image_paths = _collect_image_paths(images_dir, split_subdirs)
    num_images = len(image_paths)
    if num_images == 0:
        print(f"No images found in {images_dir}")
        return {"images_processed": 0}

    print(f"{'=' * 60}")
    print("SEM Dendrite Segmentation - Full Pipeline")
    print(f"  Images:     {images_dir} ({num_images} files)")
    print(f"  GT masks:   {gt_dir or 'not provided'}")
    print(f"  YOLO model: {yolo_model or 'not provided (skip YOLO)'}")
    print(f"  Output:     {output_dir}")
    if split_subdirs:
        print(f"  Split mode: easy={split_subdirs['easy']} | hard={split_subdirs['hard']}")
    else:
        print("  Split mode: disabled")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Stage 1: Classic pipeline
    # ------------------------------------------------------------------
    print("[Stage 1/5] Running classic pipeline...")
    classic_results = {}

    if split_subdirs:
        for profile_name in ("easy", "hard"):
            subset_dir = split_subdirs[profile_name]
            subset_paths = list_images(subset_dir)
            if not subset_paths:
                print(f"  - {profile_name.upper()}: no images found in {subset_dir}")
                continue

            print(
                f"  - {profile_name.upper()} profile on {subset_dir} "
                f"({len(subset_paths)} files)"
            )
            apply_parameter_profile(profile_name)
            subset_results = process_all_images(subset_dir, classic_dir)

            for name, result in subset_results.items():
                if name in classic_results:
                    print(f"  WARNING: duplicate basename across subsets: {name}. Overwriting.")
                classic_results[name] = result

        # Reset profile after split processing.
        apply_parameter_profile("default")
    else:
        classic_results = process_all_images(images_dir, classic_dir)

    # Save top-level mask files for evaluate_all() compatibility (<name>_mask.png)
    for name, res in classic_results.items():
        mask_path = os.path.join(classic_dir, f"{name}_mask.png")
        save_image(res["mask"], mask_path)
    print(f"  -> {len(classic_results)} images processed\n")

    # ------------------------------------------------------------------
    # Stage 2: YOLO pipeline (optional)
    # ------------------------------------------------------------------
    yolo_results = {}
    if yolo_model:
        print("[Stage 2/5] Running YOLO pipeline...")
        if not os.path.isfile(yolo_model):
            print(f"  WARNING: YOLO model not found: {yolo_model} - skipping")
        else:
            if split_subdirs:
                for subset_name in ("easy", "hard"):
                    subset_dir = split_subdirs[subset_name]
                    subset_paths = list_images(subset_dir)
                    if not subset_paths:
                        continue

                    print(
                        f"  - Predicting {subset_name} subset: {subset_dir} "
                        f"({len(subset_paths)} files)"
                    )
                    subset_preds = predict_batch(yolo_model, subset_dir, yolo_dir)
                    for name, mask in subset_preds.items():
                        if name in yolo_results:
                            print(f"  WARNING: duplicate YOLO basename across subsets: {name}. Overwriting.")
                        yolo_results[name] = mask
            else:
                yolo_results = predict_batch(yolo_model, images_dir, yolo_dir)
            print(f"  -> {len(yolo_results)} masks generated\n")
    else:
        print("[Stage 2/5] Skipping YOLO pipeline (no model provided)\n")

    # ------------------------------------------------------------------
    # Stage 3: Skeletonization
    # ------------------------------------------------------------------
    print("[Stage 3/5] Computing skeletons...")
    skeletons = {}
    for name, res in classic_results.items():
        skeletons[name] = res.get("skeleton", skeletonize_mask(res["mask"]))
    print(f"  -> {len(skeletons)} skeletons computed\n")

    # ------------------------------------------------------------------
    # Stage 4: Comparison figures
    # ------------------------------------------------------------------
    print("[Stage 4/5] Generating comparison figures...")
    os.makedirs(compare_dir, exist_ok=True)
    figures_created = 0

    for img_path in image_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        if name not in classic_results:
            continue

        classic_mask = classic_results[name]["mask"]
        skeleton = skeletons.get(name, np.zeros_like(classic_mask))

        if name in yolo_results:
            yolo_mask = yolo_results[name]
            if isinstance(yolo_mask, dict):
                yolo_mask = yolo_mask.get("mask", np.zeros_like(classic_mask))
        else:
            # Blank placeholder when YOLO is not available
            yolo_mask = np.zeros_like(classic_mask)

        fig_path = os.path.join(compare_dir, f"{name}_comparison.png")
        create_comparison_figure(img_path, classic_mask, yolo_mask, skeleton, fig_path)
        figures_created += 1

    print(f"  -> {figures_created} comparison figures saved to {compare_dir}/\n")

    # ------------------------------------------------------------------
    # Stage 5: Evaluation (if ground truth provided)
    # ------------------------------------------------------------------
    eval_results = {}
    if gt_dir and os.path.isdir(gt_dir):
        print("[Stage 5/5] Evaluating against ground truth...")
        eval_results = evaluate_all(
            classic_dir=classic_dir,
            yolo_dir=yolo_dir,
            gt_dir=gt_dir,
            image_dir=images_dir,
            output_dir=eval_dir,
        )
        print(f"  -> {len(eval_results)} images evaluated\n")
    else:
        print("[Stage 5/5] Skipping evaluation (no ground truth provided)\n")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Pipeline Complete")
    print(f"  Images processed:     {num_images}")
    print(f"  Classic masks:        {classic_dir}/")
    if yolo_results:
        print(f"  YOLO masks:           {yolo_dir}/")
    print(f"  Comparison figures:   {compare_dir}/")
    if eval_results:
        print(f"  Evaluation summary:   {eval_dir}/metrics_summary.txt")

        # Print average metrics
        for method in ["classic", "yolo"]:
            dices = [entry[method]["dice"] for entry in eval_results.values() if method in entry]
            if dices:
                label = method.capitalize()
                print(f"  {label} avg Dice:      {np.mean(dices):.3f}")
    print("=" * 60)

    return {
        "images_processed": num_images,
        "classic_dir": classic_dir,
        "yolo_dir": yolo_dir if yolo_results else None,
        "comparisons_dir": compare_dir,
        "figures_created": figures_created,
        "eval_results": eval_results,
        "split_mode": bool(split_subdirs),
    }
def main():
    """Argparse CLI for the orchestrator."""
    parser = argparse.ArgumentParser(
        description="End-to-end SEM dendrite segmentation orchestrator"
    )
    parser.add_argument(
        "--images", required=True,
        help="Directory containing source SEM images"
    )
    parser.add_argument(
        "--gt", default=None,
        help="Directory containing ground truth masks (optional)"
    )
    parser.add_argument(
        "--yolo-model", default=None,
        help="Path to trained YOLO weights .pt file (optional)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output root directory (default: output/)"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.images):
        print(f"Error: Images directory not found: {args.images}")
        sys.exit(1)

    run_orchestrator(
        images_dir=args.images,
        gt_dir=args.gt,
        yolo_model=args.yolo_model,
        output_dir=args.output,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
