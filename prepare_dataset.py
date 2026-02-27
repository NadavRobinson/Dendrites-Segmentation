"""
Dataset preparation for SEM dendrite segmentation.

Builds a YOLO-seg dataset split from existing polygon annotations and
creates rasterized ground-truth masks for evaluation.

Expected source layout:
  maked_dataset/Easy/*.jpg
  annotations/Easy/*.txt

Outputs:
  yolo_dataset/
    data.yaml
    train/images, train/labels
    valid/images, valid/labels
    test/images,  test/labels
  ground_truth_masks/Easy/*.png
  ground_truth_masks/test/*.png
"""

import argparse
import os
import random
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np

from utils import list_images, save_image


def read_yolo_polygons(label_path: str, width: int, height: int) -> List[np.ndarray]:
    """
    Parse YOLO-seg polygon rows and return absolute pixel contours.

    Parameters
    ----------
    label_path : str
        Path to a YOLO segmentation label file.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.

    Returns
    -------
    polygons : list of np.ndarray
        List of contours in cv2 format (N, 1, 2), dtype int32.
    """
    polygons = []
    if not os.path.isfile(label_path):
        return polygons

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
            xs = np.clip(np.round(coords[:, 0] * width), 0, width - 1).astype(np.int32)
            ys = np.clip(np.round(coords[:, 1] * height), 0, height - 1).astype(np.int32)
            contour = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            if contour.shape[0] >= 3:
                polygons.append(contour)
    return polygons


def polygons_to_mask(polygons: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize a list of polygons to a binary uint8 mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, 255)
    return mask


def collect_pairs(images_dir: str, labels_dir: str) -> List[Dict[str, str]]:
    """
    Match images and labels by stem.

    Returns
    -------
    pairs : list of dict
        [{'name': stem, 'image': image_path, 'label': label_path}, ...]
    """
    image_paths = list_images(images_dir)
    label_by_stem = {}
    for fname in os.listdir(labels_dir):
        if fname.lower().endswith(".txt"):
            stem = os.path.splitext(fname)[0]
            label_by_stem[stem] = os.path.join(labels_dir, fname)

    pairs = []
    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = label_by_stem.get(stem)
        if label_path and os.path.isfile(label_path):
            pairs.append({"name": stem, "image": image_path, "label": label_path})
    return pairs


def split_pairs(
    pairs: List[Dict[str, str]],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, str]]]:
    """Split pairs into train/valid/test with deterministic shuffle."""
    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Ratios must satisfy: train>0, valid>0, train+valid<1.")

    pairs = pairs.copy()
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = max(1, int(round(n * train_ratio)))
    n_valid = max(1, int(round(n * valid_ratio)))
    if n_train + n_valid >= n:
        n_valid = max(1, n - n_train - 1)
    n_test = n - n_train - n_valid
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_pairs = pairs[:n_train]
    valid_pairs = pairs[n_train:n_train + n_valid]
    test_pairs = pairs[n_train + n_valid:]

    return {"train": train_pairs, "valid": valid_pairs, "test": test_pairs}


def copy_split_files(splits: Dict[str, List[Dict[str, str]]], output_root: str) -> None:
    """Copy image/label pairs into YOLO split directories."""
    for split_name, pairs in splits.items():
        img_dir = os.path.join(output_root, split_name, "images")
        lbl_dir = os.path.join(output_root, split_name, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for pair in pairs:
            dst_img = os.path.join(img_dir, os.path.basename(pair["image"]))
            dst_lbl = os.path.join(lbl_dir, os.path.basename(pair["label"]))
            shutil.copy2(pair["image"], dst_img)
            shutil.copy2(pair["label"], dst_lbl)


def write_data_yaml(dataset_root: str) -> str:
    """Write YOLO `data.yaml`."""
    yaml_path = os.path.join(dataset_root, "data.yaml")
    content = (
        f"path: {os.path.abspath(dataset_root)}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: dendrite\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return yaml_path


def create_ground_truth_masks(
    pairs: List[Dict[str, str]],
    output_dir: str,
) -> None:
    """Create binary masks from polygon annotations for all provided pairs."""
    os.makedirs(output_dir, exist_ok=True)
    for pair in pairs:
        image = cv2.imread(pair["image"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        h, w = image.shape
        polygons = read_yolo_polygons(pair["label"], w, h)
        mask = polygons_to_mask(polygons, image.shape)
        save_image(mask, os.path.join(output_dir, f"{pair['name']}.png"))


def prepare(
    images_dir: str,
    labels_dir: str,
    yolo_dataset_dir: str,
    gt_root_dir: str,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Prepare YOLO split dataset and GT masks.

    Returns
    -------
    info : dict
        Paths and split counts.
    """
    pairs = collect_pairs(images_dir, labels_dir)
    if not pairs:
        raise RuntimeError("No matched image/label pairs found.")

    if os.path.isdir(yolo_dataset_dir):
        shutil.rmtree(yolo_dataset_dir)
    os.makedirs(yolo_dataset_dir, exist_ok=True)

    splits = split_pairs(pairs, train_ratio=train_ratio, valid_ratio=valid_ratio, seed=seed)
    copy_split_files(splits, yolo_dataset_dir)
    data_yaml = write_data_yaml(yolo_dataset_dir)

    easy_gt_dir = os.path.join(gt_root_dir, "Easy")
    test_gt_dir = os.path.join(gt_root_dir, "test")
    create_ground_truth_masks(pairs, easy_gt_dir)
    create_ground_truth_masks(splits["test"], test_gt_dir)

    return {
        "data_yaml": data_yaml,
        "num_pairs": str(len(pairs)),
        "num_train": str(len(splits["train"])),
        "num_valid": str(len(splits["valid"])),
        "num_test": str(len(splits["test"])),
        "gt_easy_dir": easy_gt_dir,
        "gt_test_dir": test_gt_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset and GT masks")
    parser.add_argument("--images", default="maked_dataset/Easy", help="Image directory")
    parser.add_argument("--labels", default="annotations/Easy", help="Label directory")
    parser.add_argument("--yolo-out", default="yolo_dataset", help="Output YOLO dataset directory")
    parser.add_argument("--gt-out", default="ground_truth_masks", help="Output GT mask root directory")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    info = prepare(
        images_dir=args.images,
        labels_dir=args.labels,
        yolo_dataset_dir=args.yolo_out,
        gt_root_dir=args.gt_out,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    print("Prepared dataset successfully:")
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
