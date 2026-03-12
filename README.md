# SEM Dendrite Segmentation

This project segments lithium dendrites in SEM images with two approaches:

- Classic computer vision pipeline (`scripts/classic_scripts/classic_pipeline.py`)
- YOLO segmentation pipeline (`scripts/yolo_scripts/yolo_pipeline.py`)

Both pipelines output binary masks that can be used for evaluation and visualization.

## Environment Setup

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Repository Entry Points

- `scripts/classic_scripts/classic_pipeline.py`: classic segmentation (single image or batch)
- `scripts/yolo_scripts/yolo_pipeline.py`: YOLO dataset validation, training, and inference
- `scripts/yolo_scripts/prepare_dataset.py`: build YOLO split folders + raster ground-truth masks
- `scripts/run_all.py`: end-to-end orchestration
- `scripts/evaluate.py`: Dice/IoU/Precision/Recall evaluation

## Classic Pipeline

Run from repository root.

### 1) Easy Profile (Default Folders)

```bash
python scripts/classic_scripts/classic_pipeline.py --easy
```

Defaults in this mode:
- input: `dataset/Easy`
- output: `output/easy`

### 2) Hard Profile (Default Folders)

```bash
python scripts/classic_scripts/classic_pipeline.py --hard
```

Defaults in this mode:
- input: `dataset/Hard`
- output: `output/hard`

### 3) Single Image

```bash
python scripts/classic_scripts/classic_pipeline.py dataset/Easy/2e-9_100s_002.tif --output output/classic_single
```

### 4) Custom Batch Directory

```bash
python scripts/classic_scripts/classic_pipeline.py --input dataset/combined --output output/classic_combined
```

Optional flag:
- `--no-intermediates`: save only final mask/skeleton/preview outputs.

## YOLO Pipeline

Run from repository root.

### 1) Train

```powershell
python scripts/yolo_scripts/yolo_pipeline.py train `
  --data dataset/yolo_dataset/data.yaml `
  --model yolo11x-seg.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 8 `
  --project output/yolo/train
```

Notes:
- `--project output/yolo/train` keeps training artifacts under the repo-level `output/` folder.
- Best weights are written to `output/yolo/train/dendrite_seg/weights/best.pt`.

### 2) Predict on a Directory

```powershell
python scripts/yolo_scripts/yolo_pipeline.py predict `
  --model output/yolo/train/dendrite_seg/weights/best.pt `
  --source dataset/Easy `
  --output output/yolo/predictions
```

### 3) Predict on a Single Image

```powershell
python scripts/yolo_scripts/yolo_pipeline.py predict `
  --model output/yolo/train/dendrite_seg/weights/best.pt `
  --source dataset/Easy/2e-9_100s_002.tif `
  --output output/yolo/predictions
```

## Optional: Build YOLO Dataset + GT Masks

If you need to rebuild the YOLO split from images and YOLO polygon labels:

```powershell
python scripts/yolo_scripts/prepare_dataset.py `
  --images dataset/Easy `
  --labels annotations/Easy `
  --yolo-out dataset/yolo_dataset `
  --gt-out ground_truth_masks
```

## Dependencies

`requirements.txt` includes the required libraries for both classic and YOLO pipelines:

- `numpy`
- `opencv-python`
- `scikit-image`
- `ultralytics`
- `torch`
