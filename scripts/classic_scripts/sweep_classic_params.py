"""
Parameter sweep for classic SEM segmentation pipeline.

This script searches important parameters in classic_pipeline.py across:
  - dataset/classic_dataset/Easy
  - dataset/classic_dataset/Hard

For each parameter combination it:
  1. Runs the classic pipeline on all selected images.
  2. Saves outputs per combination for manual visual review.
  3. Computes no-GT quality heuristics.
  4. Ranks combinations by balanced Easy/Hard performance.

Notes
-----
- Ranking uses heuristics (not ground-truth metrics). Use it to shortlist
  candidates, then decide by visual inspection.
- To override search space, pass --space-json with a JSON mapping:
    {
      "RAW_ADAPTIVE_BLOCK_SIZE": [35, 51, 67],
      "RAW_ADAPTIVE_C": [-9, -5, -1]
    }
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import sys
from datetime import datetime

import cv2
import numpy as np

import classic_pipeline as cp
from utils import create_comparison_strip, create_overlay, list_images, load_image, save_image


# Important parameters for raw SEM images in classic_pipeline.py.
# The full cartesian product is large, so we sample with --max-combos.
DEFAULT_PARAM_SPACE = {
    "CLAHE_CLIP_LIMIT": [2.0, 3.0, 4.5],
    "BILATERAL_SIGMA_COLOR": [50, 75, 100],
    "RAW_ADAPTIVE_BLOCK_SIZE": [35, 51, 67],
    "RAW_ADAPTIVE_C": [-9, -5, -1],
    "RAW_MIN_COMPONENT_AREA": [200, 300, 450],
    "EROSION_KERNEL_SIZE": [3, 5, 7],
    "EROSION_ITERATIONS": [1, 2, 3],
    "CLOSING_KERNEL_SIZE": [3, 5, 7],
    "DISTANCE_THRESHOLD": [0.30, 0.40, 0.50],
}


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _band_score(value: float, ideal_lo: float, ideal_hi: float, hard_lo: float, hard_hi: float) -> float:
    """
    Piecewise-linear score:
      1.0 inside [ideal_lo, ideal_hi]
      0.0 outside [hard_lo, hard_hi]
      linear ramps in between.
    """
    if value < hard_lo or value > hard_hi:
        return 0.0
    if ideal_lo <= value <= ideal_hi:
        return 1.0
    if value < ideal_lo:
        denom = max(ideal_lo - hard_lo, 1e-9)
        return max(0.0, min(1.0, (value - hard_lo) / denom))
    denom = max(hard_hi - ideal_hi, 1e-9)
    return max(0.0, min(1.0, (hard_hi - value) / denom))


def _upper_better_low(value: float, ideal_max: float, hard_max: float) -> float:
    """
    Score where lower is better:
      1.0 if value <= ideal_max
      0.0 if value >= hard_max
      linear between.
    """
    if value <= ideal_max:
        return 1.0
    if value >= hard_max:
        return 0.0
    denom = max(hard_max - ideal_max, 1e-9)
    return max(0.0, min(1.0, (hard_max - value) / denom))


def compute_mask_stats(mask: np.ndarray, skeleton: np.ndarray) -> dict[str, float]:
    """
    Compute no-GT quality descriptors from final mask/skeleton.
    """
    h, w = mask.shape[:2]
    total_pixels = float(h * w)
    fg_pixels = float(np.count_nonzero(mask))
    fg_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64) if num_labels > 1 else np.array([], dtype=np.float64)
    component_count = int(areas.size)

    if fg_pixels > 0 and component_count > 0:
        largest_component_ratio = float(areas.max() / fg_pixels)
    else:
        largest_component_ratio = 0.0

    tiny_component_frac = (
        float(np.mean(areas < 64.0))
        if component_count > 0 else 0.0
    )

    skel_pixels = float(np.count_nonzero(skeleton))
    skeleton_to_mask_ratio = skel_pixels / fg_pixels if fg_pixels > 0 else 0.0

    bottom_band = max(1, int(round(0.10 * h)))
    bottom_fg_ratio = float(np.count_nonzero(mask[h - bottom_band:h, :])) / float(bottom_band * w)

    # Composite no-GT quality heuristic.
    score_fg = _band_score(fg_ratio, ideal_lo=0.03, ideal_hi=0.35, hard_lo=0.005, hard_hi=0.75)
    score_components = _band_score(component_count, ideal_lo=5, ideal_hi=220, hard_lo=1, hard_hi=700)
    score_largest = _band_score(largest_component_ratio, ideal_lo=0.08, ideal_hi=0.85, hard_lo=0.01, hard_hi=0.98)
    score_skeleton = _band_score(skeleton_to_mask_ratio, ideal_lo=0.015, ideal_hi=0.30, hard_lo=0.001, hard_hi=0.60)
    score_bottom = _upper_better_low(bottom_fg_ratio, ideal_max=0.02, hard_max=0.10)
    score_tiny = _upper_better_low(tiny_component_frac, ideal_max=0.35, hard_max=0.90)

    quality_score = float(np.mean([
        score_fg,
        score_components,
        score_largest,
        score_skeleton,
        score_bottom,
        score_tiny,
    ]))

    return {
        "fg_ratio": fg_ratio,
        "component_count": float(component_count),
        "largest_component_ratio": largest_component_ratio,
        "tiny_component_frac": tiny_component_frac,
        "skeleton_to_mask_ratio": skeleton_to_mask_ratio,
        "bottom_fg_ratio": bottom_fg_ratio,
        "quality_score": quality_score,
        "quality_pass": 1.0 if quality_score >= 0.60 else 0.0,
    }


def compute_combo_score(per_image_rows: list[dict[str, float]]) -> dict[str, float]:
    """
    Build a balanced score across Easy and Hard subsets.
    """
    easy_scores = [r["quality_score"] for r in per_image_rows if r["split"] == "Easy"]
    hard_scores = [r["quality_score"] for r in per_image_rows if r["split"] == "Hard"]

    easy_pass = [r["quality_pass"] for r in per_image_rows if r["split"] == "Easy"]
    hard_pass = [r["quality_pass"] for r in per_image_rows if r["split"] == "Hard"]

    all_scores = [r["quality_score"] for r in per_image_rows]

    easy_mean = _safe_mean(easy_scores)
    hard_mean = _safe_mean(hard_scores)
    easy_pass_rate = _safe_mean(easy_pass)
    hard_pass_rate = _safe_mean(hard_pass)

    balanced_quality = 0.5 * (easy_mean + hard_mean)
    balanced_pass = min(easy_pass_rate, hard_pass_rate)

    # Lower std is better stability across images.
    std_all = _safe_std(all_scores)
    stability = max(0.0, 1.0 - min(std_all, 0.5) / 0.5)

    overall = 0.55 * balanced_quality + 0.30 * balanced_pass + 0.15 * stability

    return {
        "overall_score": overall,
        "balanced_quality": balanced_quality,
        "balanced_pass_rate": balanced_pass,
        "stability": stability,
        "easy_mean_quality": easy_mean,
        "hard_mean_quality": hard_mean,
        "easy_pass_rate": easy_pass_rate,
        "hard_pass_rate": hard_pass_rate,
        "std_quality": std_all,
    }


def load_param_space(path: str | None) -> dict[str, list]:
    if path is None:
        return dict(DEFAULT_PARAM_SPACE)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("space-json must be a non-empty JSON object: {param: [values]}")
    parsed = {}
    for key, values in data.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Parameter '{key}' must map to a non-empty list of values.")
        parsed[str(key)] = values
    return parsed


def sample_combinations(space: dict[str, list], max_combos: int, seed: int) -> list[dict[str, object]]:
    keys = list(space.keys())
    value_lists = [space[k] for k in keys]
    total = math.prod(len(v) for v in value_lists)
    print(f"Search space size: {total} combinations")

    rng = random.Random(seed)

    # If small enough, enumerate all and subsample deterministically.
    if total <= 200000:
        all_tuples = list(itertools.product(*value_lists))
        if len(all_tuples) > max_combos:
            idxs = list(range(len(all_tuples)))
            rng.shuffle(idxs)
            all_tuples = [all_tuples[i] for i in idxs[:max_combos]]
        combos = [dict(zip(keys, vals)) for vals in all_tuples]
        return combos

    # Large search space: random unique sampling without full expansion.
    combos = []
    seen = set()
    attempts = 0
    max_attempts = max_combos * 100
    while len(combos) < max_combos and attempts < max_attempts:
        attempts += 1
        vals = tuple(rng.choice(v) for v in value_lists)
        if vals in seen:
            continue
        seen.add(vals)
        combos.append(dict(zip(keys, vals)))
    return combos


def apply_params(params: dict[str, object]) -> None:
    for name, value in params.items():
        setattr(cp, name, value)

    # Keep bilateral sigma space coupled unless explicitly set.
    if "BILATERAL_SIGMA_COLOR" in params and "BILATERAL_SIGMA_SPACE" not in params:
        cp.BILATERAL_SIGMA_SPACE = params["BILATERAL_SIGMA_COLOR"]


def capture_current_params(param_names: list[str]) -> dict[str, object]:
    captured = {}
    for name in param_names:
        if hasattr(cp, name):
            captured[name] = getattr(cp, name)

    # Save coupled parameter as well so restore is complete.
    if hasattr(cp, "BILATERAL_SIGMA_SPACE"):
        captured["BILATERAL_SIGMA_SPACE"] = getattr(cp, "BILATERAL_SIGMA_SPACE")
    return captured


def restore_params(saved: dict[str, object]) -> None:
    for name, value in saved.items():
        setattr(cp, name, value)


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep important parameters of classic_pipeline.py on classic_dataset Easy+Hard."
    )
    parser.add_argument(
        "--dataset-root",
        default=os.path.join("dataset", "classic_dataset"),
        help="Root folder containing Easy/Hard subfolders.",
    )
    parser.add_argument("--easy-subdir", default="Easy", help="Easy split subfolder name.")
    parser.add_argument("--hard-subdir", default="Hard", help="Hard split subfolder name.")
    parser.add_argument(
        "--output-root",
        default=os.path.join("output", "classic_param_sweep"),
        help="Root output folder for sweep runs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name (default: timestamped).",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=80,
        help="Number of sampled combinations to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for combination sampling.",
    )
    parser.add_argument(
        "--max-easy",
        type=int,
        default=None,
        help="Limit number of Easy images (for faster iteration).",
    )
    parser.add_argument(
        "--max-hard",
        type=int,
        default=None,
        help="Limit number of Hard images (for faster iteration).",
    )
    parser.add_argument(
        "--space-json",
        default=None,
        help="Optional JSON file overriding parameter space.",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save all stage intermediates for every combo/image (large output).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview strips (source/mask/overlays).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print sampled combinations and exit.",
    )
    args = parser.parse_args()

    easy_dir = os.path.join(args.dataset_root, args.easy_subdir)
    hard_dir = os.path.join(args.dataset_root, args.hard_subdir)

    easy_images = list_images(easy_dir)
    hard_images = list_images(hard_dir)

    if args.max_easy is not None:
        easy_images = easy_images[: max(0, args.max_easy)]
    if args.max_hard is not None:
        hard_images = hard_images[: max(0, args.max_hard)]

    if not easy_images and not hard_images:
        raise RuntimeError("No images selected from Easy/Hard splits.")

    all_images = [("Easy", p) for p in easy_images] + [("Hard", p) for p in hard_images]

    print(f"Selected images: Easy={len(easy_images)} Hard={len(hard_images)} Total={len(all_images)}")

    param_space = load_param_space(args.space_json)
    combos = sample_combinations(param_space, max_combos=args.max_combos, seed=args.seed)
    print(f"Evaluating {len(combos)} combinations")

    if args.dry_run:
        for idx, combo in enumerate(combos, start=1):
            print(f"{idx:04d}: {combo}")
        return

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    param_names = list(param_space.keys())
    original_params = capture_current_params(param_names)

    combo_rows: list[dict] = []
    image_rows: list[dict] = []
    combo_manifest: list[dict] = []

    try:
        for combo_idx, params in enumerate(combos, start=1):
            combo_id = f"combo_{combo_idx:04d}"
            combo_dir = os.path.join(run_dir, combo_id)
            os.makedirs(combo_dir, exist_ok=True)

            print(f"\n[{combo_idx}/{len(combos)}] {combo_id} {params}")
            apply_params(params)

            combo_manifest.append({
                "combo_id": combo_id,
                **params,
            })

            per_image = []
            for split_name, image_path in all_images:
                base = os.path.splitext(os.path.basename(image_path))[0]
                print(f"  - {split_name}/{base}")

                results = cp.run_classic_pipeline(
                    image_path,
                    output_dir=combo_dir,
                    save_intermediates=args.save_intermediates,
                )

                mask = results["mask"]
                skeleton = results["skeleton"]
                stats = compute_mask_stats(mask, skeleton)

                row = {
                    "combo_id": combo_id,
                    "image": base,
                    "split": split_name,
                    **stats,
                }
                image_rows.append(row)
                per_image.append(row)

                if not args.no_preview:
                    source = load_image(image_path, grayscale=True)
                    mask_overlay = create_overlay(source, mask, color=(0, 255, 0), alpha=0.55)
                    skel_overlay = create_overlay(source, skeleton, color=(0, 0, 255), alpha=0.70)
                    preview = create_comparison_strip(
                        [source, mask, mask_overlay, skel_overlay],
                        ["Source", "Mask", "Mask Overlay", "Skeleton Overlay"],
                        height=320,
                    )
                    save_image(preview, os.path.join(combo_dir, f"{base}_preview.png"))

            combo_scores = compute_combo_score(per_image)
            combo_row = {
                "combo_id": combo_id,
                **combo_scores,
                **params,
            }
            combo_rows.append(combo_row)

            with open(os.path.join(combo_dir, "params.json"), "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2)
            with open(os.path.join(combo_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(combo_row, f, indent=2)

    finally:
        restore_params(original_params)

    combo_rows.sort(key=lambda r: r["overall_score"], reverse=True)

    combo_fields = [
        "combo_id",
        "overall_score",
        "balanced_quality",
        "balanced_pass_rate",
        "stability",
        "easy_mean_quality",
        "hard_mean_quality",
        "easy_pass_rate",
        "hard_pass_rate",
        "std_quality",
    ] + param_names

    image_fields = [
        "combo_id",
        "image",
        "split",
        "quality_score",
        "quality_pass",
        "fg_ratio",
        "component_count",
        "largest_component_ratio",
        "tiny_component_frac",
        "skeleton_to_mask_ratio",
        "bottom_fg_ratio",
    ]

    write_csv(os.path.join(run_dir, "combo_ranking.csv"), combo_rows, combo_fields)
    write_csv(os.path.join(run_dir, "image_metrics.csv"), image_rows, image_fields)
    write_csv(
        os.path.join(run_dir, "combo_manifest.csv"),
        combo_manifest,
        ["combo_id"] + param_names,
    )

    best = combo_rows[0] if combo_rows else None
    if best is not None:
        best_params = {k: best[k] for k in param_names}
        with open(os.path.join(run_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

    with open(os.path.join(run_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Classic pipeline parameter sweep\n")
        f.write(f"Run directory: {run_dir}\n")
        f.write(f"Combinations: {len(combo_rows)}\n")
        f.write(f"Images: Easy={len(easy_images)} Hard={len(hard_images)}\n\n")
        f.write("Files:\n")
        f.write("- combo_ranking.csv: ranked combinations (highest overall_score first)\n")
        f.write("- image_metrics.csv: per-image quality heuristics for each combination\n")
        f.write("- combo_manifest.csv: parameter values per combination\n")
        f.write("- best_params.json: top-ranked parameter combination\n")

    print("\nSweep complete.")
    print(f"Results saved to: {run_dir}")
    if best is not None:
        print("Top combination:")
        print(f"  combo_id={best['combo_id']} overall_score={best['overall_score']:.4f}")
        for k in param_names:
            print(f"  {k}={best[k]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
