# SEM Dendrite Segmentation (Final Project)

Automatic segmentation of lithium dendrites in SEM images using two complementary approaches:

- `Approach A`: Deep learning instance segmentation (`YOLO-Seg`)
- `Approach B`: Classic image processing (morphology + watershed)

The project includes dataset preparation, training/inference, skeleton extraction, quantitative evaluation (Dice/IoU/Precision/Recall), and failure analysis.

## 1) Project Goal

Build a robust system that segments dendrites from noisy SEM images and compares:

- learning-based segmentation (YOLO)
- deterministic classical CV segmentation

Required outputs:

- Binary mask (dendrite vs background)
- Pre-processing / cleaning pipeline
- Skeletonization (single-pixel centerline)

## 2) Repository Entry Points

- `prepare_dataset.py` - builds YOLO dataset splits + raster ground-truth masks
- `classic_pipeline.py` - classic SEM segmentation pipeline (single image or batch)
- `yolo_pipeline.py` - YOLO dataset validation, training, and inference
- `run_all.py` - end-to-end orchestrator (classic + optional YOLO + evaluation)
- `evaluate.py` - metrics, comparison figure generation, failure analysis utilities

Helper scripts for annotation/visualization:

- `process_easy_dataset.py`
- `mask_to_yolo_annotation.py`
- `visualize_annotation.py`

## 3) Environment Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`:

- `numpy`
- `opencv-python`
- `scikit-image`
- `ultralytics`

## 4) Data and Annotation Format

- Images: grayscale SEM images (e.g. under `maked_dataset/Easy`)
- Labels: YOLO segmentation polygon `.txt` files (e.g. `annotations/Easy`)
- Class mapping: `0 = dendrite`

If you start from binary masks and need polygon labels, use:

```bash
python process_easy_dataset.py
```

This script reads `cut_offs.txt`, creates polygon labels in `annotations/Easy`, and visualizations in `visualizations/Easy`.

## 5) Run Instructions

### A. Prepare YOLO Dataset + Ground Truth Masks

```bash
python prepare_dataset.py \
  --images maked_dataset/Easy \
  --labels annotations/Easy \
  --yolo-out yolo_dataset \
  --gt-out ground_truth_masks
```

Creates:

- `yolo_dataset/train|valid|test/{images,labels}`
- `yolo_dataset/data.yaml`
- `ground_truth_masks/Easy/*.png`
- `ground_truth_masks/test/*.png`

### B. Train YOLO-Seg

```bash
python yolo_pipeline.py train \
  --data yolo_dataset/data.yaml \
  --model yolo11n-seg.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 8
```

Best weights are saved at:

```text
output/yolo/train/dendrite_seg/weights/best.pt
```

### C. Run YOLO Inference

```bash
python yolo_pipeline.py predict \
  --model output/yolo/train/dendrite_seg/weights/best.pt \
  --source maked_dataset/Easy \
  --output output/yolo
```

Outputs masks as `*_mask.png` in `output/yolo`.

### D. Run Classic Pipeline

Single image:

```bash
python classic_pipeline.py path/to/image.png --output output/classic
```

Batch mode:

```bash
python classic_pipeline.py --input maked_dataset/Easy --output output/classic
```

By default, intermediate stage images are saved per image folder under `output/classic/<image_name>/`.

### E. End-to-End (Recommended)

Run both approaches + comparison figures + evaluation in one command:

```bash
python run_all.py \
  --images maked_dataset/Easy \
  --gt ground_truth_masks/Easy \
  --yolo-model output/yolo/train/dendrite_seg/weights/best.pt \
  --output output
```

Classic-only run (skip YOLO):

```bash
python run_all.py --images maked_dataset/Easy --gt ground_truth_masks/Easy
```

## 6) Outputs and Artifacts

Typical artifacts created under `output/`:

- `output/classic/` - classic masks + per-stage intermediates + skeletons
- `output/yolo/` - YOLO predicted masks
- `output/comparisons/` - 4-panel visuals: Source | Classic | YOLO | Skeleton
- `output/evaluation/metrics_summary.txt` - metrics table + failure analysis

## 7) Build Submission Report (PDF Package)

Generate a paper-style report package with:

- per-image and average metrics CSV tables
- failure-analysis CSV
- visual success/failure analysis figures
- `summary_report.tex` (and `summary_report.pdf` if `pdflatex` is installed)

```bash
python build_summary_report.py \
  --images maked_dataset/Easy \
  --gt ground_truth_masks/Easy \
  --classic output/report_inputs/classic \
  --yolo output/report_inputs/yolo \
  --yolo-train output/yolo/train/dendrite_seg \
  --out output/summary_report
```

Report outputs are saved under `output/summary_report/`.

## 8) Evaluation Criteria Implemented

- Dice score
- IoU
- Precision / Recall
- Failure analysis (under/over-segmentation diagnostics)

These are generated via `evaluate.py` functions and called by `run_all.py` when `--gt` is provided.

## 9) Suggested Submission Checklist

- Source code + comments/docstrings
- `requirements.txt`
- This `README.md` with train/inference instructions
- Best model weights (`.pt`) or a shared link
- At least 5 visual examples of:
  - Source image
  - Classic mask
  - YOLO mask
  - Skeleton
- PDF report and short demo presentation/video

## 10) Quick Repro (Minimal)

```bash
python prepare_dataset.py
python yolo_pipeline.py train --data yolo_dataset/data.yaml --model yolo11n-seg.pt
python run_all.py --images maked_dataset/Easy --gt ground_truth_masks/Easy --yolo-model output/yolo/train/dendrite_seg/weights/best.pt
```
