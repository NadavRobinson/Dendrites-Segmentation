import argparse
from pathlib import Path

import cv2
import numpy as np


def draw_annotation(image: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        coords = np.array(parts[1:], dtype=float)
        xs = (coords[0::2] * width).astype(np.int32)
        ys = (coords[1::2] * height).astype(np.int32)
        if len(xs) < 3 or len(ys) < 3:
            continue

        pts = np.stack((xs, ys), axis=-1).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO segmentation annotations on input images."
    )
    parser.add_argument(
        "--ann-input-folder",
        required=True,
        help="Folder with YOLO .txt annotation files.",
    )
    parser.add_argument(
        "--image-input-folder",
        required=True,
        help="Folder with source images (e.g. dataset/Hard).",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder where visualization PNGs will be written.",
    )
    parser.add_argument(
        "--image-ext",
        default=".tif",
        help="Image extension to load per annotation base name (default: .tif).",
    )
    parser.add_argument(
        "--cutoff-value",
        type=int,
        default=120,
        help="Draw red horizontal cutoff line at (height - cutoff).",
    )
    args = parser.parse_args()

    ann_dir = Path(args.ann_input_folder)
    image_dir = Path(args.image_input_folder)
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation folder not found: {ann_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    txt_files = sorted(ann_dir.glob("*.txt"))
    written = 0
    missing_images = 0

    for txt_path in txt_files:
        stem = txt_path.stem
        image_path = image_dir / f"{stem}{args.image_ext}"

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Skipping {stem}: image not found/readable at {image_path}")
            missing_images += 1
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        vis = draw_annotation(img, lines)
        h = vis.shape[0]
        y_cutoff = max(0, min(h - 1, h - int(args.cutoff_value)))
        cv2.line(vis, (0, y_cutoff), (vis.shape[1], y_cutoff), (0, 0, 255), 2)

        output_path = out_dir / f"{stem}_visualization.png"
        cv2.imwrite(str(output_path), vis)
        written += 1

    print(
        f"Done. Wrote {written} visualizations to {out_dir}. "
        f"Missing images: {missing_images}."
    )


if __name__ == "__main__":
    main()
