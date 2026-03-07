import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def mask_to_yolo_lines(mask: np.ndarray, min_area: int, class_id: int) -> list[str]:
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    lines: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        normalized_coords = []
        for point in contour:
            x = point[0][0] / width
            y = point[0][1] / height
            normalized_coords.extend([f"{x:.6f}", f"{y:.6f}"])

        if normalized_coords:
            lines.append(f"{class_id} " + " ".join(normalized_coords))

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert pipeline mask PNGs to YOLO segmentation annotation TXT files."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory containing per-image subfolders (e.g. output/hard_moved_smallremoved).",
    )
    parser.add_argument(
        "--mask-name",
        required=True,
        help="Mask filename inside each subfolder (e.g. 10_separated.png or 11_skeleton.png).",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Directory where YOLO .txt files will be written.",
    )
    parser.add_argument(
        "--cutoff-value",
        type=int,
        default=120,
        help="Zero out the last N rows from the bottom before contour extraction.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Minimum contour area to keep.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO class id.",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    subdirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    written = 0
    skipped = 0

    for subdir in subdirs:
        mask_path = subdir / args.mask_name
        if not mask_path.is_file():
            skipped += 1
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Skipping unreadable mask: {mask_path}")
            skipped += 1
            continue

        height = mask.shape[0]
        cutoff = max(0, min(args.cutoff_value, height))
        if cutoff > 0:
            mask[height - cutoff : height, :] = 0

        lines = mask_to_yolo_lines(mask, min_area=args.min_area, class_id=args.class_id)
        txt_path = output_folder / f"{subdir.name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
        written += 1

    print(
        f"Done. Wrote {written} annotation files to {output_folder}. "
        f"Skipped {skipped} subfolders."
    )


if __name__ == "__main__":
    main()
