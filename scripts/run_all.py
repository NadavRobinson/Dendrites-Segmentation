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

# Add project directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from classic_pipeline import run_classic_pipeline, process_all_images, skeletonize_mask
from evaluate import create_comparison_figure, evaluate_all
from utils import load_image, save_image, list_images


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

    image_paths = list_images(images_dir)
    num_images = len(image_paths)
    if num_images == 0:
        print(f"No images found in {images_dir}")
        return {"images_processed": 0}

    print(f"{'=' * 60}")
    print(f"SEM Dendrite Segmentation — Full Pipeline")
    print(f"  Images:     {images_dir} ({num_images} files)")
    print(f"  GT masks:   {gt_dir or 'not provided'}")
    print(f"  YOLO model: {yolo_model or 'not provided (skip YOLO)'}")
    print(f"  Output:     {output_dir}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Stage 1: Classic pipeline
    # ------------------------------------------------------------------
    print("[Stage 1/5] Running classic pipeline...")
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
            print(f"  WARNING: YOLO model not found: {yolo_model} — skipping")
        else:
            from yolo_pipeline import predict_batch
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
            dices = [e[method]["dice"] for e in eval_results.values() if method in e]
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
    else:
        # Synthetic end-to-end test
        import shutil

        print("=== run_all.py — Synthetic End-to-End Test ===\n")

        project_dir = os.path.dirname(os.path.abspath(__file__))
        test_root = os.path.join(project_dir, "output", "_run_all_test")
        test_images = os.path.join(test_root, "images")
        test_gt = os.path.join(test_root, "ground_truth")
        test_output = os.path.join(test_root, "output")

        os.makedirs(test_images, exist_ok=True)
        os.makedirs(test_gt, exist_ok=True)

        np.random.seed(42)
        patterns = [
            # Pattern 1: vertical + horizontal dendrites
            [(100, 50, 100, 400, 200, 3), (100, 200, 300, 200, 190, 2)],
            # Pattern 2: diagonal branches
            [(50, 50, 400, 400, 210, 4), (200, 50, 50, 300, 185, 2)],
            # Pattern 3: star pattern
            [(200, 50, 200, 400, 195, 3), (50, 200, 400, 200, 190, 3),
             (80, 80, 320, 320, 180, 2)],
        ]

        for i, branch_list in enumerate(patterns):
            # Create synthetic SEM image
            img = np.random.randint(30, 80, (512, 512), dtype=np.uint8)
            gt_mask = np.zeros((512, 512), dtype=np.uint8)
            for (x1, y1, x2, y2, intensity, thickness) in branch_list:
                cv2.line(img, (x1, y1), (x2, y2), intensity, thickness)
                cv2.line(gt_mask, (x1, y1), (x2, y2), 255, thickness + 2)

            # Add scale bar (SEM metadata)
            img[460:, :] = 230

            img_path = os.path.join(test_images, f"synth_{i:03d}.png")
            gt_path = os.path.join(test_gt, f"synth_{i:03d}.png")
            cv2.imwrite(img_path, img)
            cv2.imwrite(gt_path, gt_mask)

        print(f"Created 3 synthetic images in {test_images}/")
        print(f"Created 3 ground truth masks in {test_gt}/\n")

        # Run orchestrator (classic only, no YOLO model)
        result = run_orchestrator(
            images_dir=test_images,
            gt_dir=test_gt,
            yolo_model=None,
            output_dir=test_output,
        )

        # Verify outputs
        print("\n--- Verification ---")

        assert result["images_processed"] == 3, \
            f"Expected 3 images, got {result['images_processed']}"
        print(f"  Images processed: {result['images_processed']} (OK)")

        assert result["figures_created"] == 3, \
            f"Expected 3 figures, got {result['figures_created']}"
        print(f"  Comparison figures: {result['figures_created']} (OK)")

        # Check comparison figures exist
        compare_dir = os.path.join(test_output, "comparisons")
        compare_files = [f for f in os.listdir(compare_dir) if f.endswith(".png")]
        assert len(compare_files) == 3, \
            f"Expected 3 comparison PNGs, found {len(compare_files)}"
        print(f"  Comparison PNGs in directory: {len(compare_files)} (OK)")

        # Check metrics summary exists
        summary_path = os.path.join(test_output, "evaluation", "metrics_summary.txt")
        assert os.path.isfile(summary_path), \
            f"Metrics summary not found: {summary_path}"
        with open(summary_path, 'r') as f:
            summary_text = f.read()
        assert "Failure Analysis" in summary_text, \
            "Failure analysis section missing from summary"
        print(f"  Metrics summary with failure analysis: (OK)")

        # Check evaluation results
        assert len(result["eval_results"]) == 3, \
            f"Expected 3 eval results, got {len(result['eval_results'])}"
        for name, entry in result["eval_results"].items():
            assert "classic" in entry, f"Missing classic metrics for {name}"
            print(f"  {name}: Classic Dice={entry['classic']['dice']:.3f}")

        # Cleanup
        shutil.rmtree(test_root)
        print(f"\n  Cleaned up test directory: {test_root}")
        print("\nAll run_all.py tests passed.")
