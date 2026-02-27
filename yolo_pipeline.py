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
from utils import load_image, save_image, list_images

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "yolo11n-seg.pt"   # Nano model — fast training, decent accuracy
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_PATIENCE = 20              # Early stopping patience
DEFAULT_FREEZE = 10                # Freeze first N backbone layers
DEFAULT_LR0 = 0.001                # Initial learning rate
DEFAULT_CONF = 0.25                # Inference confidence threshold


# ===========================================================================
# Dataset preparation
# ===========================================================================

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
               lr0=DEFAULT_LR0, project=None):
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
        preds = model.predict(path, conf=conf, verbose=False)
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
            freeze=args.freeze, lr0=args.lr0, project=args.project
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
    else:
        # Synthetic self-test (no ultralytics needed)
        print("=== yolo_pipeline.py — Synthetic Self-Test ===\n")

        # Test dataset validation with a fake structure
        test_dir = os.path.join(os.path.dirname(__file__), "output", "_yolo_test")
        for subdir in ["train/images", "train/labels", "valid/images", "valid/labels"]:
            os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)

        # Create dummy files
        dummy_img = np.zeros((64, 64), dtype=np.uint8)
        for i in range(3):
            cv2.imwrite(os.path.join(test_dir, "train/images", f"img{i}.png"), dummy_img)
            with open(os.path.join(test_dir, "train/labels", f"img{i}.txt"), 'w') as f:
                f.write("0 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n")
        for i in range(2):
            cv2.imwrite(os.path.join(test_dir, "valid/images", f"img{i}.png"), dummy_img)
            with open(os.path.join(test_dir, "valid/labels", f"img{i}.txt"), 'w') as f:
                f.write("0 0.3 0.3 0.4 0.3 0.4 0.4 0.3 0.4\n")

        yaml_path = prepare_yolo_dataset(test_dir)
        print(f"\nDataset YAML created at: {yaml_path}")

        # Test skeleton extraction
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(test_mask, (20, 10), (20, 90), 255, 5)
        cv2.line(test_mask, (20, 50), (80, 50), 255, 3)
        skeleton = yolo_mask_to_skeleton(test_mask)
        print(f"Skeleton test — mask pixels: {np.sum(test_mask > 0)}, "
              f"skeleton pixels: {np.sum(skeleton > 0)}")

        # Cleanup test directory
        import shutil
        shutil.rmtree(test_dir)
        print("\nAll YOLO pipeline tests passed.")
