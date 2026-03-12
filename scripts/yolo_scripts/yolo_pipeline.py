"""
YOLO-Seg pipeline for SEM dendrite segmentation.

Uses ultralytics YOLOv11 instance segmentation with transfer learning
from COCO pretrained weights. Handles dataset validation, training,
single/batch inference, and mask extraction.

Usage:
    python yolo_pipeline.py train --data <dataset_yaml> [--epochs 100] [--model yolo11n-seg.pt]
    python yolo_pipeline.py predict --model <weights.pt> --source <image_or_dir> [--output <dir>]
"""

import cv2
import numpy as np
import os
import sys
import argparse

from skimage.morphology import skeletonize

# Add project directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
try:
    from utils import save_image, list_images
except ModuleNotFoundError:
    # Reuse shared utility helpers from classic_scripts if local utils is absent.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classic_scripts"))
    from utils import save_image, list_images

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "yolo11x-seg.pt"   # Nano model — fast training, decent accuracy
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_PATIENCE = 0              # Early stopping patience
DEFAULT_FREEZE = 0                # Freeze first N backbone layers
DEFAULT_LR0 = 0.001                # Initial learning rate
DEFAULT_CONF = 0.25                # Inference confidence threshold
DEFAULT_WORKERS = 0 if os.name == "nt" else 8


# ===========================================================================
# Dataset preparation
# ===========================================================================

def ensure_three_channel_dataset(dataset_dir):
    """
    Normalize dataset images to 3-channel PNG files.

    Ultralytics pretrained YOLO models expect 3-channel inputs. This helper
    converts grayscale/alpha images to BGR and rewrites non-PNG formats to PNG
    to avoid TIFF metadata/decoder edge cases.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing train/valid(/test) image folders.
    """
    split_dirs = [
        os.path.join(dataset_dir, "train", "images"),
        os.path.join(dataset_dir, "valid", "images"),
        os.path.join(dataset_dir, "test", "images"),
    ]
    converted = 0
    total = 0

    for image_dir in split_dirs:
        if not os.path.isdir(image_dir):
            continue

        image_paths = list_images(image_dir)
        for image_path in image_paths:
            total += 1
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"  WARNING: Could not read image: {image_path}")
                continue

            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif image.ndim == 3 and image.shape[2] == 3:
                pass
            else:
                print(f"  WARNING: Unsupported image shape {image.shape} at {image_path}")
                continue

            dst_path = os.path.splitext(image_path)[0] + ".png"
            needs_rewrite = (os.path.splitext(image_path)[1].lower() != ".png")
            if needs_rewrite:
                ok = cv2.imwrite(dst_path, image)
                if not ok:
                    raise RuntimeError(f"Failed to write normalized image: {dst_path}")
                if os.path.normcase(dst_path) != os.path.normcase(image_path):
                    os.remove(image_path)
                converted += 1
            else:
                # Keep PNG format, but ensure 3 channels if source was grayscale/alpha.
                if image.ndim == 3 and image.shape[2] == 3:
                    ok = cv2.imwrite(image_path, image)
                    if not ok:
                        raise RuntimeError(f"Failed to update image: {image_path}")

    # Remove stale Ultralytics cache files so new image metadata is re-read.
    for split in ("train", "valid", "test"):
        cache_path = os.path.join(dataset_dir, split, "labels.cache")
        if os.path.isfile(cache_path):
            os.remove(cache_path)

    print(f"  Normalized images to 3-channel PNG: converted {converted}/{total}")


def prepare_yolo_dataset(roboflow_dir, output_yaml=None):
    """
    Validate a Roboflow YOLO-Segmentation export and create/verify dataset.yaml.

    Expected Roboflow export structure:
        roboflow_dir/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        └── test/   (optional)
            ├── images/
            └── labels/

    Parameters
    ----------
    roboflow_dir : str
        Path to the Roboflow YOLO export directory.
    output_yaml : str or None
        If provided, write a corrected dataset.yaml to this path.
        Otherwise, use roboflow_dir/data.yaml.

    Returns
    -------
    yaml_path : str
        Path to the validated dataset.yaml file.
    """
    # Check required directories exist
    required = ["train/images", "train/labels", "valid/images", "valid/labels"]
    for subdir in required:
        full_path = os.path.join(roboflow_dir, subdir)
        if not os.path.isdir(full_path):
            raise FileNotFoundError(
                f"Required directory not found: {full_path}\n"
                f"Ensure you exported from Roboflow in 'YOLOv8 Segmentation' format."
            )

    # Count images and labels
    train_imgs = len(list_images(os.path.join(roboflow_dir, "train/images")))
    valid_imgs = len(list_images(os.path.join(roboflow_dir, "valid/images")))
    train_labels = len([f for f in os.listdir(os.path.join(roboflow_dir, "train/labels"))
                        if f.endswith('.txt')])
    valid_labels = len([f for f in os.listdir(os.path.join(roboflow_dir, "valid/labels"))
                        if f.endswith('.txt')])

    print(f"Dataset validation:")
    print(f"  Train: {train_imgs} images, {train_labels} labels")
    print(f"  Valid: {valid_imgs} images, {valid_labels} labels")

    if train_imgs == 0:
        raise ValueError("No training images found.")
    if train_imgs != train_labels:
        print(f"  WARNING: Image/label count mismatch in train "
              f"({train_imgs} vs {train_labels})")

    # Ensure all dataset images are 3-channel so pretrained YOLO models can train.
    ensure_three_channel_dataset(roboflow_dir)

    # Check or create dataset.yaml
    yaml_path = output_yaml or os.path.join(roboflow_dir, "data.yaml")

    if os.path.exists(yaml_path):
        print(f"  Using existing: {yaml_path}")
    else:
        # Create a minimal dataset.yaml
        yaml_content = (
            f"path: {os.path.abspath(roboflow_dir)}\n"
            f"train: train/images\n"
            f"val: valid/images\n"
            f"\n"
            f"nc: 1\n"
            f"names:\n"
            f"  0: dendrite\n"
        )
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"  Created dataset.yaml at: {yaml_path}")

    return yaml_path


# ===========================================================================
# Training
# ===========================================================================

def train_yolo(dataset_yaml, model=DEFAULT_MODEL, epochs=DEFAULT_EPOCHS,
               imgsz=DEFAULT_IMGSZ, batch=DEFAULT_BATCH,
               patience=DEFAULT_PATIENCE, freeze=DEFAULT_FREEZE,
               lr0=DEFAULT_LR0, project=None, workers=DEFAULT_WORKERS):
    """
    Train YOLO-Seg model with transfer learning.

    Freezes the first N backbone layers and uses a low learning rate
    to fine-tune on SEM dendrite data.

    Parameters
    ----------
    dataset_yaml : str
        Path to dataset.yaml file.
    model : str
        Pretrained model name or path to .pt file.
    epochs : int
        Maximum number of training epochs.
    imgsz : int
        Input image size (square).
    batch : int
        Batch size.
    patience : int
        Early stopping patience (epochs without improvement).
    freeze : int
        Number of backbone layers to freeze.
    lr0 : float
        Initial learning rate.
    project : str or None
        Output project directory. Defaults to 'output/yolo/train'.
    workers : int
        Number of dataloader workers. Use 0 on restricted Windows setups.

    Returns
    -------
    results : ultralytics Results object
        Training results including best model path.
    """
    from ultralytics import YOLO

    if project is None:
        project = os.path.join(os.path.dirname(__file__), "output", "yolo", "train")

    print(f"\n{'='*60}")
    print(f"YOLO-Seg Training")
    print(f"  Model:    {model}")
    print(f"  Dataset:  {dataset_yaml}")
    print(f"  Epochs:   {epochs}")
    print(f"  ImgSize:  {imgsz}")
    print(f"  Batch:    {batch}")
    print(f"  Patience: {patience}")
    print(f"  Freeze:   {freeze} layers")
    print(f"  LR0:      {lr0}")
    print(f"  Workers:  {workers}")
    print(f"{'='*60}\n")

    yolo_model = YOLO(model)

    results = yolo_model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        freeze=freeze,
        lr0=lr0,
        workers=workers,
        project=project,
        name="dendrite_seg",
        exist_ok=True,
        verbose=True,
    )

    best_path = os.path.join(project, "dendrite_seg", "weights", "best.pt")
    print(f"\nTraining complete. Best weights: {best_path}")
    return results


# ===========================================================================
# Inference
# ===========================================================================

def predict_single(model_path, image_path, conf=DEFAULT_CONF):
    """
    Run YOLO-Seg inference on a single image and extract the binary mask.

    Combines all detected instance masks via logical OR into a single
    binary mask representing all dendrite pixels.

    Parameters
    ----------
    model_path : str
        Path to trained YOLO weights (.pt file).
    image_path : str
        Path to input image.
    conf : float
        Confidence threshold for detections.

    Returns
    -------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8, same size as input image.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf, verbose=False)

    # Get original image dimensions
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Combine all instance masks into one binary mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if results and results[0].masks is not None:
        masks_data = results[0].masks.data.cpu().numpy()  # (N, mH, mW)
        for instance_mask in masks_data:
            # Resize mask to original image dimensions
            resized = cv2.resize(
                instance_mask, (w, h), interpolation=cv2.INTER_LINEAR
            )
            combined_mask[resized > 0.5] = 255

    return combined_mask


def predict_batch(model_path, input_dir, output_dir, conf=DEFAULT_CONF):
    """
    Run YOLO-Seg inference on all images in a directory.

    Parameters
    ----------
    model_path : str
        Path to trained YOLO weights (.pt file).
    input_dir : str
        Directory containing input images.
    output_dir : str
        Directory to save output masks.
    conf : float
        Confidence threshold.

    Returns
    -------
    results : dict
        Mapping of image basename to binary mask.
    """
    from ultralytics import YOLO

    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return {}

    print(f"Running YOLO inference on {len(image_paths)} images...")
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for path in image_paths:
        basename = os.path.splitext(os.path.basename(path))[0]

        # Run inference
        preds = model.predict(path, conf=conf, verbose=False, save=True)
        image = cv2.imread(path)
        h, w = image.shape[:2]

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        if preds and preds[0].masks is not None:
            masks_data = preds[0].masks.data.cpu().numpy()
            for instance_mask in masks_data:
                resized = cv2.resize(instance_mask, (w, h),
                                     interpolation=cv2.INTER_LINEAR)
                combined_mask[resized > 0.5] = 255

        save_image(combined_mask, os.path.join(output_dir, f"{basename}_mask.png"))
        all_results[basename] = combined_mask
        print(f"  {basename}: {np.sum(combined_mask > 0)} foreground pixels")

    print(f"Saved {len(all_results)} masks to {output_dir}/")
    return all_results


def yolo_mask_to_skeleton(mask):
    """
    Extract skeleton from a YOLO-generated binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.

    Returns
    -------
    skeleton : np.ndarray
        Skeleton image (0 or 255), dtype uint8.
    """
    binary = (mask > 0).astype(bool)
    skel = skeletonize(binary)
    return (skel.astype(np.uint8) * 255)


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLO-Seg pipeline for SEM dendrite segmentation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train YOLO-Seg model")
    train_parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    train_parser.add_argument("--model", default=DEFAULT_MODEL,
                              help=f"Pretrained model (default: {DEFAULT_MODEL})")
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train_parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    train_parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    train_parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    train_parser.add_argument("--freeze", type=int, default=DEFAULT_FREEZE)
    train_parser.add_argument("--lr0", type=float, default=DEFAULT_LR0)
    train_parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    train_parser.add_argument("--project", default=None)

    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Run inference")
    pred_parser.add_argument("--model", required=True, help="Path to weights (.pt)")
    pred_parser.add_argument("--source", required=True,
                             help="Image path or directory")
    pred_parser.add_argument("--output", default=None,
                             help="Output directory for masks")
    pred_parser.add_argument("--conf", type=float, default=DEFAULT_CONF)

    args = parser.parse_args()

    if args.command == "train":
        yaml_path = prepare_yolo_dataset(
            os.path.dirname(args.data), output_yaml=args.data
        )
        train_yolo(
            yaml_path, model=args.model, epochs=args.epochs,
            imgsz=args.imgsz, batch=args.batch, patience=args.patience,
            freeze=args.freeze, lr0=args.lr0,
            workers=args.workers, project=args.project
        )

    elif args.command == "predict":
        if not os.path.exists(args.model):
            print(f"Error: Model not found: {args.model}")
            sys.exit(1)

        output_dir = args.output or os.path.join(
            os.path.dirname(__file__), "output", "yolo"
        )

        if os.path.isdir(args.source):
            predict_batch(args.model, args.source, output_dir, conf=args.conf)
        else:
            mask = predict_single(args.model, args.source, conf=args.conf)
            basename = os.path.splitext(os.path.basename(args.source))[0]
            save_image(mask, os.path.join(output_dir, f"{basename}_mask.png"))
            print(f"Saved mask: {np.sum(mask > 0)} foreground pixels")

    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
