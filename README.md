# SEM Dendrite Segmentation

This repository provides two runnable pipelines:

- Classic pipeline: `scripts/classic_scripts/classic_pipeline.py`
- YOLO pipeline: `scripts/yolo_scripts/yolo_pipeline.py`

## Environment Setup

Run from repository root:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Classic Pipeline (`scripts/classic_scripts/classic_pipeline.py`)

### Easy profile (default folders)

```bash
python scripts/classic_scripts/classic_pipeline.py --easy
```

- input: `dataset/Easy`
- output: `output/easy`

### Hard profile (default folders)

```bash
python scripts/classic_scripts/classic_pipeline.py --hard
```

- input: `dataset/Hard`
- output: `output/hard`

### Single image

```bash
python scripts/classic_scripts/classic_pipeline.py dataset/Easy/2e-9_100s_002.tif --output output/classic_single
```

### Custom batch directory

```bash
python scripts/classic_scripts/classic_pipeline.py --input dataset/combined --output output/classic_combined
```

Optional:
- `--no-intermediates` to save only final outputs.

## YOLO Pipeline (`scripts/yolo_scripts/yolo_pipeline.py`)

### Train

```powershell
python scripts/yolo_scripts/yolo_pipeline.py train `
  --data dataset/yolo_dataset/data.yaml `
  --model yolo11x-seg.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 8 `
  --project output/yolo/train
```

- best weights: `output/yolo/train/dendrite_seg/weights/best.pt`

### Predict on a directory

```powershell
python scripts/yolo_scripts/yolo_pipeline.py predict `
  --model output/yolo/train/dendrite_seg/weights/best.pt `
  --source dataset/Easy `
  --output output/yolo/predictions
```

### Predict on a single image

```powershell
python scripts/yolo_scripts/yolo_pipeline.py predict `
  --model output/yolo/train/dendrite_seg/weights/best.pt `
  --source dataset/Easy/2e-9_100s_002.tif `
  --output output/yolo/predictions
```
