"""
Build a paper-style summary report for SEM dendrite segmentation.

This script:
1) Computes Classic/YOLO metrics against ground truth (Dice/IoU/Precision/Recall)
2) Writes CSV tables (per-image, averages, deltas, failures)
3) Creates visual analysis figures (success/failure examples)
4) Generates a LaTeX report with embedded tables and figures
5) Optionally compiles PDF when pdflatex is available

Usage example:
    python build_summary_report.py \
      --images maked_dataset/Easy \
      --gt ground_truth_masks/Easy \
      --classic output/report_inputs/classic \
      --yolo output/report_inputs/yolo \
      --yolo-train output/yolo/train/dendrite_seg \
      --out output/summary_report
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np
from skimage.morphology import skeletonize

# Add project directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "classic_scripts"))
from evaluate import evaluate_single, analyze_failures  # noqa: E402
from utils import list_images, load_image, save_image  # noqa: E402


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
METRIC_KEYS = ("dice", "iou", "precision", "recall")


def _latex_escape(text):
    """Escape LaTeX special characters in plain text."""
    if text is None:
        return ""
    escaped = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        escaped = escaped.replace(src, dst)
    return escaped


def _fmt(value):
    """Format metric values for tables."""
    if value is None:
        return "--"
    return f"{value:.3f}"


def _normalize_text(text):
    """Normalize common mojibake symbols and keep text LaTeX-safe."""
    if text is None:
        return ""
    out = str(text)
    replacements = {
        "â€”": "-",
        "—": "-",
        "â€“": "-",
        "–": "-",
        "â†’": "->",
        "→": "->",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _to_binary_mask(mask):
    """Normalize any mask-like image to uint8 {0,255} binary mask."""
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask > 0] = 255
    return out


def _find_source_path(images_dir, name):
    """Find source image path by basename and common image extensions."""
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(images_dir, f"{name}{ext}")
        if os.path.isfile(candidate):
            return candidate
    for subset in ("easy", "Easy", "hard", "Hard"):
        subset_dir = os.path.join(images_dir, subset)
        if not os.path.isdir(subset_dir):
            continue
        for ext in IMAGE_EXTENSIONS:
            candidate = os.path.join(subset_dir, f"{name}{ext}")
            if os.path.isfile(candidate):
                return candidate
    return None


def _find_mask_path(mask_dir, name):
    """Find a predicted mask path in the given directory tree."""
    direct_candidates = [
        os.path.join(mask_dir, f"{name}_mask.png"),
        os.path.join(mask_dir, f"{name}.png"),
        os.path.join(mask_dir, name, "10_separated.png"),
        os.path.join(mask_dir, name, "06_segmented.png"),
    ]
    for candidate in direct_candidates:
        if os.path.isfile(candidate):
            return candidate

    recursive_names = (
        f"{name}_mask.png",
        f"{name}.png",
        "10_separated.png",
        "06_segmented.png",
    )
    for root, _, files in os.walk(mask_dir):
        if os.path.basename(root) != name:
            continue
        for filename in recursive_names:
            if filename in files:
                return os.path.join(root, filename)

    for root, _, files in os.walk(mask_dir):
        target = f"{name}_mask.png"
        if target in files:
            return os.path.join(root, target)
    return None


def _load_binary_from_path(path):
    """Load image from path and return binary uint8 mask."""
    if path is None or not os.path.isfile(path):
        return None
    image = load_image(path, grayscale=True)
    return _to_binary_mask(image)


def _parse_metrics_summary(metrics_summary_path):
    """
    Parse run_all evaluation summary text into per-image metric records.

    Expected row format:
        <image_name> <Classic|YOLO> <dice> <iou> <precision> <recall>
    """
    metrics_by_name = {}
    if not metrics_summary_path or not os.path.isfile(metrics_summary_path):
        return metrics_by_name

    row_re = re.compile(
        r"^(?P<name>\S+)\s+"
        r"(?P<method>Classic|YOLO)\s+"
        r"(?P<dice>\d+(?:\.\d+)?)\s+"
        r"(?P<iou>\d+(?:\.\d+)?)\s+"
        r"(?P<precision>\d+(?:\.\d+)?)\s+"
        r"(?P<recall>\d+(?:\.\d+)?)\s*$"
    )

    with open(metrics_summary_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = row_re.match(line.strip())
            if not match:
                continue
            name = match.group("name")
            method = match.group("method").lower()
            metrics_by_name.setdefault(name, {})[method] = {
                "dice": float(match.group("dice")),
                "iou": float(match.group("iou")),
                "precision": float(match.group("precision")),
                "recall": float(match.group("recall")),
            }
    return metrics_by_name


def _to_bgr(image):
    """Convert grayscale or single-channel image to BGR."""
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return None


def _render_mask(mask):
    """Render a binary mask as a visible BGR panel."""
    if mask is None:
        return None
    mask_gray = _to_binary_mask(mask)
    return cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)


def _resize_panel(panel, width=520, height=340):
    """Resize panel to fixed dimensions for comparison grids."""
    if panel is None:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        return panel
    return cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)


def _annotate_panel(panel, title):
    """Add title strip to panel."""
    h, w = panel.shape[:2]
    title_h = 36
    out = np.zeros((h + title_h, w, 3), dtype=np.uint8)
    out[title_h:, :, :] = panel
    cv2.putText(
        out,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _error_overlay(source, pred_mask, gt_mask):
    """
    Build TP/FP/FN overlay:
      TP -> green, FP -> red, FN -> blue.
    """
    base = _to_bgr(source)
    if base is None:
        return None

    pred = _to_binary_mask(pred_mask)
    gt = _to_binary_mask(gt_mask)
    if pred is None or gt is None:
        return base

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    if base.shape[:2] != gt.shape:
        base = cv2.resize(base, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)

    tp = np.logical_and(pred > 0, gt > 0)
    fp = np.logical_and(pred > 0, gt == 0)
    fn = np.logical_and(pred == 0, gt > 0)

    overlay = base.copy()
    overlay[tp] = (0, 180, 0)
    overlay[fp] = (0, 0, 220)
    overlay[fn] = (220, 0, 0)
    return cv2.addWeighted(overlay, 0.55, base, 0.45, 0)


def _make_case_figure(source, gt, classic, yolo, output_path):
    """Create a 2x3 visual analysis figure for one image."""
    if gt is None:
        return

    if source is None:
        source = np.zeros_like(gt)

    if source.shape[:2] != gt.shape:
        source = cv2.resize(source, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)

    classic = _to_binary_mask(classic) if classic is not None else np.zeros_like(gt)
    yolo = _to_binary_mask(yolo) if yolo is not None else np.zeros_like(gt)

    if classic.shape != gt.shape:
        classic = cv2.resize(classic, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    if yolo.shape != gt.shape:
        yolo = cv2.resize(yolo, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    skel = skeletonize(classic > 0).astype(np.uint8) * 255

    panels = [
        _annotate_panel(_resize_panel(_to_bgr(source)), "Source"),
        _annotate_panel(_resize_panel(_render_mask(gt)), "Ground Truth"),
        _annotate_panel(_resize_panel(_render_mask(classic)), "Classic Mask"),
        _annotate_panel(_resize_panel(_render_mask(yolo)), "YOLO Mask"),
        _annotate_panel(_resize_panel(_render_mask(skel)), "Classic Skeleton"),
        _annotate_panel(_resize_panel(_error_overlay(source, yolo, gt)), "YOLO Error (TP/FP/FN)"),
    ]

    top = np.hstack(panels[:3])
    bottom = np.hstack(panels[3:])
    figure = np.vstack([top, bottom])
    save_image(figure, output_path)


def _collect_records(images_dir, gt_dir, classic_dir, yolo_dir, metrics_by_name=None):
    """Collect per-image metadata and metrics (from summary file or recomputed)."""
    gt_paths = list_images(gt_dir)
    records = []

    for gt_path in gt_paths:
        name = os.path.splitext(os.path.basename(gt_path))[0]
        source_path = _find_source_path(images_dir, name)
        classic_path = _find_mask_path(classic_dir, name) if classic_dir else None
        yolo_path = _find_mask_path(yolo_dir, name) if yolo_dir else None

        gt_mask = _load_binary_from_path(gt_path)
        classic_mask = _load_binary_from_path(classic_path)
        yolo_mask = _load_binary_from_path(yolo_path)

        row = {
            "name": name,
            "source_path": source_path,
            "gt_path": gt_path,
            "classic_path": classic_path,
            "yolo_path": yolo_path,
            "classic": None,
            "yolo": None,
        }

        if metrics_by_name is not None:
            metric_entry = metrics_by_name.get(name, {})
            row["classic"] = metric_entry.get("classic")
            row["yolo"] = metric_entry.get("yolo")
        else:
            if classic_mask is not None:
                row["classic"] = evaluate_single(classic_mask, gt_mask)
            if yolo_mask is not None:
                row["yolo"] = evaluate_single(yolo_mask, gt_mask)

        records.append(row)
    return sorted(records, key=lambda r: r["name"])


def _mean_metrics(records, method):
    """Compute mean metrics for one method over available rows."""
    out = {k: None for k in METRIC_KEYS}
    vals = {k: [] for k in METRIC_KEYS}
    for rec in records:
        m = rec.get(method)
        if not m:
            continue
        for key in METRIC_KEYS:
            vals[key].append(float(m[key]))

    for key in METRIC_KEYS:
        if vals[key]:
            out[key] = float(np.mean(vals[key]))
    return out


def _build_failures(records, threshold):
    """Build failure list from evaluate.analyze_failures-compatible input."""
    packed = {}
    for rec in records:
        entry = {}
        if rec.get("classic"):
            entry["classic"] = rec["classic"]
        if rec.get("yolo"):
            entry["yolo"] = rec["yolo"]
        if entry:
            packed[rec["name"]] = entry
    failures = analyze_failures(packed, threshold=threshold)
    for failure in failures:
        failure["cause"] = _normalize_text(failure.get("cause", ""))
        failure["method"] = _normalize_text(failure.get("method", ""))
    failures = sorted(failures, key=lambda f: f["dice"])
    return failures


def _combined_score(rec):
    """Combined ranking score using available Dice values."""
    dices = []
    if rec.get("classic"):
        dices.append(rec["classic"]["dice"])
    if rec.get("yolo"):
        dices.append(rec["yolo"]["dice"])
    if not dices:
        return -1.0
    return float(np.mean(dices))


def _select_cases(records, failures, num_examples):
    """Select representative success and failure image names."""
    num_each = max(1, num_examples // 2)

    failure_names = []
    for failure in failures:
        if failure["name"] not in failure_names:
            failure_names.append(failure["name"])
        if len(failure_names) >= num_each:
            break

    if len(failure_names) < num_each:
        ranked_low = sorted(records, key=_combined_score)
        for rec in ranked_low:
            if rec["name"] not in failure_names:
                failure_names.append(rec["name"])
            if len(failure_names) >= num_each:
                break

    success_names = []
    ranked_high = sorted(records, key=_combined_score, reverse=True)
    for rec in ranked_high:
        if rec["name"] in failure_names:
            continue
        success_names.append(rec["name"])
        if len(success_names) >= num_each:
            break

    return success_names, failure_names


def _write_csv_tables(records, failures, out_dir):
    """Write CSV tables for metrics, averages, deltas, and failures."""
    os.makedirs(out_dir, exist_ok=True)

    per_image_csv = os.path.join(out_dir, "metrics_per_image.csv")
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "classic_dice", "classic_iou", "classic_precision", "classic_recall",
            "yolo_dice", "yolo_iou", "yolo_precision", "yolo_recall",
            "delta_dice_yolo_minus_classic",
        ])
        for rec in records:
            c = rec.get("classic")
            y = rec.get("yolo")
            delta_dice = None
            if c and y:
                delta_dice = y["dice"] - c["dice"]
            writer.writerow([
                rec["name"],
                "" if not c else f"{c['dice']:.6f}",
                "" if not c else f"{c['iou']:.6f}",
                "" if not c else f"{c['precision']:.6f}",
                "" if not c else f"{c['recall']:.6f}",
                "" if not y else f"{y['dice']:.6f}",
                "" if not y else f"{y['iou']:.6f}",
                "" if not y else f"{y['precision']:.6f}",
                "" if not y else f"{y['recall']:.6f}",
                "" if delta_dice is None else f"{delta_dice:.6f}",
            ])

    averages_csv = os.path.join(out_dir, "metrics_averages.csv")
    classic_avg = _mean_metrics(records, "classic")
    yolo_avg = _mean_metrics(records, "yolo")
    with open(averages_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dice", "iou", "precision", "recall"])
        writer.writerow([
            "classic",
            "" if classic_avg["dice"] is None else f"{classic_avg['dice']:.6f}",
            "" if classic_avg["iou"] is None else f"{classic_avg['iou']:.6f}",
            "" if classic_avg["precision"] is None else f"{classic_avg['precision']:.6f}",
            "" if classic_avg["recall"] is None else f"{classic_avg['recall']:.6f}",
        ])
        writer.writerow([
            "yolo",
            "" if yolo_avg["dice"] is None else f"{yolo_avg['dice']:.6f}",
            "" if yolo_avg["iou"] is None else f"{yolo_avg['iou']:.6f}",
            "" if yolo_avg["precision"] is None else f"{yolo_avg['precision']:.6f}",
            "" if yolo_avg["recall"] is None else f"{yolo_avg['recall']:.6f}",
        ])

    failures_csv = os.path.join(out_dir, "failures.csv")
    with open(failures_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "method", "dice", "precision", "recall", "cause"])
        for fail in failures:
            writer.writerow([
                fail["name"],
                fail["method"],
                f"{fail['dice']:.6f}",
                f"{fail['precision']:.6f}",
                f"{fail['recall']:.6f}",
                fail["cause"],
            ])

    return {
        "per_image_csv": per_image_csv,
        "averages_csv": averages_csv,
        "failures_csv": failures_csv,
        "classic_avg": classic_avg,
        "yolo_avg": yolo_avg,
    }


def _copy_if_exists(src_path, dst_path):
    """Copy file only if source exists."""
    if src_path and os.path.isfile(src_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return True
    return False


def _build_latex_document(
    report_title,
    report_author,
    records,
    failures,
    classic_avg,
    yolo_avg,
    case_entries,
    training_entries,
    max_rows,
):
    """Build LaTeX source text for the summary report."""
    n_images = len(records)
    n_failures = len(failures)
    classic_dice = _fmt(classic_avg["dice"])
    yolo_dice = _fmt(yolo_avg["dice"])

    table_rows = []
    for rec in records[:max_rows]:
        c = rec.get("classic")
        y = rec.get("yolo")
        table_rows.append(
            " & ".join([
                _latex_escape(rec["name"]),
                _fmt(None if not c else c["dice"]),
                _fmt(None if not c else c["iou"]),
                _fmt(None if not c else c["precision"]),
                _fmt(None if not c else c["recall"]),
                _fmt(None if not y else y["dice"]),
                _fmt(None if not y else y["iou"]),
                _fmt(None if not y else y["precision"]),
                _fmt(None if not y else y["recall"]),
            ]) + r" \\"
        )
    if not table_rows:
        table_rows.append(r"\multicolumn{9}{c}{No rows available} \\")

    failure_rows = []
    for fail in failures[:max_rows]:
        failure_rows.append(
            " & ".join([
                _latex_escape(fail["name"]),
                _latex_escape(fail["method"]),
                _fmt(fail["dice"]),
                _fmt(fail["precision"]),
                _fmt(fail["recall"]),
                _latex_escape(fail["cause"]),
            ]) + r" \\"
        )
    if not failure_rows:
        failure_rows.append(r"\multicolumn{6}{c}{No failures below threshold} \\")

    case_figures = []
    for entry in case_entries:
        case_figures.append(
            "\n".join([
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.96\linewidth]{{{entry['latex_path']}}}",
                rf"\caption{{{_latex_escape(entry['caption'])}}}",
                r"\end{figure}",
            ])
        )
    if not case_figures:
        case_figures.append(r"No visual examples were generated.")

    training_figures = []
    for entry in training_entries:
        training_figures.append(
            "\n".join([
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.92\linewidth]{{{entry['latex_path']}}}",
                rf"\caption{{{_latex_escape(entry['caption'])}}}",
                r"\end{figure}",
            ])
        )

    training_block = "\n\n".join(training_figures) if training_figures else (
        "No YOLO training curves were found in the configured training directory."
    )

    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{tabularx}}
\usepackage{{array}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}

\title{{{_latex_escape(report_title)}}}
\author{{{_latex_escape(report_author)}}}
\date{{\today}}

\begin{{document}}
\maketitle

\section*{{Abstract}}
This report summarizes pixel-level semantic segmentation of lithium dendrites in SEM images using two pipelines:
(1) a deterministic classic morphology pipeline and
(2) a YOLO segmentation model.
The evaluation set includes {n_images} labeled images.
Average Dice scores are Classic={classic_dice} and YOLO={yolo_dice}.
The report includes quantitative metrics, method comparison tables, and visual success/failure analysis.

\section*{{Methodology}}
\textbf{{Classic Pipeline:}} pre-processing (normalization, CLAHE, bilateral denoising), threshold-based segmentation,
morphological cleaning/reconstruction, branch separation, and skeleton extraction.\\
\textbf{{Deep Learning Pipeline:}} transfer-learned YOLO segmentation model with binary mask output per image.\\
\textbf{{Evaluation Metrics:}} Dice, IoU, Precision, Recall measured against manual ground-truth masks.

\section*{{Results and Discussion}}
\subsection*{{Average Metrics}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
Method & Dice & IoU & Precision & Recall \\
\midrule
Classic & {_fmt(classic_avg["dice"])} & {_fmt(classic_avg["iou"])} & {_fmt(classic_avg["precision"])} & {_fmt(classic_avg["recall"])} \\
YOLO & {_fmt(yolo_avg["dice"])} & {_fmt(yolo_avg["iou"])} & {_fmt(yolo_avg["precision"])} & {_fmt(yolo_avg["recall"])} \\
\bottomrule
\end{{tabular}}
\caption{{Average segmentation metrics across available predictions.}}
\end{{table}}

\subsection*{{Per-Image Comparison}}
\begin{{table}}[H]
\centering
\scriptsize
\begin{{tabular}}{{lcccccccc}}
\toprule
Image & C-Dice & C-IoU & C-Prec & C-Rec & Y-Dice & Y-IoU & Y-Prec & Y-Rec \\
\midrule
{os.linesep.join(table_rows)}
\bottomrule
\end{{tabular}}
\caption{{Per-image metric comparison (first {min(len(records), max_rows)} rows). Full table is exported to CSV.}}
\end{{table}}

\subsection*{{Failure Analysis}}
Number of failures under Dice threshold: {n_failures}.
\begin{{table}}[H]
\centering
\scriptsize
\begin{{tabularx}}{{\linewidth}}{{llcccX}}
\toprule
Image & Method & Dice & Precision & Recall & Estimated Cause \\
\midrule
{os.linesep.join(failure_rows)}
\bottomrule
\end{{tabularx}}
\caption{{Failure characterization based on precision/recall patterns.}}
\end{{table}}

\subsection*{{Visual Analysis (Successes and Failures)}}
{os.linesep.join(case_figures)}

\subsection*{{YOLO Training Diagnostics}}
{training_block}

\section*{{Conclusions}}
The generated artifacts support the project submission requirements:
technical report format, quantitative metrics, comparison tables, and qualitative visual analysis.
Use this report as a reproducible baseline and regenerate it whenever new experiments are produced.

\end{{document}}
"""
    return tex


def _compile_latex(tex_path):
    """Compile LaTeX to PDF with pdflatex if available."""
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        return False, "pdflatex not found on PATH; skipped PDF compilation."

    out_dir = os.path.dirname(tex_path)
    tex_file = os.path.basename(tex_path)
    for _ in range(2):
        proc = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", tex_file],
            cwd=out_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            log_path = os.path.join(out_dir, "latex_build.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(proc.stdout)
            return False, f"LaTeX compilation failed. See: {log_path}"
    return True, "PDF compiled successfully."


def _build_fallback_pdf(
    out_pdf_path,
    report_title,
    records,
    failures,
    classic_avg,
    yolo_avg,
    case_entries,
    training_entries,
):
    """
    Build a fallback PDF with matplotlib when LaTeX is unavailable.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    def _new_page():
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        return fig, ax

    def _save_table_page(pdf, title, headers, rows, font_size=8):
        fig, ax = _new_page()
        ax.text(0.02, 0.98, title, va="top", ha="left", fontsize=14, weight="bold")
        table = ax.table(
            cellText=rows,
            colLabels=headers,
            cellLoc="center",
            colLoc="center",
            bbox=[0.02, 0.05, 0.96, 0.88],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1.0, 1.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    with PdfPages(out_pdf_path) as pdf:
        # Page 1: summary text
        fig, ax = _new_page()
        ax.text(0.02, 0.97, report_title, va="top", ha="left", fontsize=17, weight="bold")
        summary_lines = [
            "",
            f"Images evaluated: {len(records)}",
            f"Failures (Dice<threshold): {len(failures)}",
            "",
            "Average metrics:",
            f"  Classic  Dice={_fmt(classic_avg['dice'])}  IoU={_fmt(classic_avg['iou'])}  "
            f"Precision={_fmt(classic_avg['precision'])}  Recall={_fmt(classic_avg['recall'])}",
            f"  YOLO     Dice={_fmt(yolo_avg['dice'])}  IoU={_fmt(yolo_avg['iou'])}  "
            f"Precision={_fmt(yolo_avg['precision'])}  Recall={_fmt(yolo_avg['recall'])}",
            "",
            "This PDF was generated by fallback mode (matplotlib) because pdflatex",
            "was not available. The matching LaTeX source is summary_report.tex.",
        ]
        ax.text(0.02, 0.92, "\n".join(summary_lines), va="top", ha="left", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: per-image metrics table
        headers = [
            "Image", "C Dice", "C IoU", "C P", "C R", "Y Dice", "Y IoU", "Y P", "Y R"
        ]
        rows = []
        for rec in records:
            c = rec.get("classic")
            y = rec.get("yolo")
            rows.append([
                rec["name"],
                _fmt(None if not c else c["dice"]),
                _fmt(None if not c else c["iou"]),
                _fmt(None if not c else c["precision"]),
                _fmt(None if not c else c["recall"]),
                _fmt(None if not y else y["dice"]),
                _fmt(None if not y else y["iou"]),
                _fmt(None if not y else y["precision"]),
                _fmt(None if not y else y["recall"]),
            ])
        _save_table_page(pdf, "Per-image metrics", headers, rows, font_size=8)

        # Page 3: failure analysis table
        fail_headers = ["Image", "Method", "Dice", "Precision", "Recall", "Cause"]
        fail_rows = []
        for fail in failures:
            fail_rows.append([
                fail["name"],
                fail["method"],
                _fmt(fail["dice"]),
                _fmt(fail["precision"]),
                _fmt(fail["recall"]),
                fail["cause"][:70],
            ])
        if not fail_rows:
            fail_rows.append(["-", "-", "-", "-", "-", "No failures"])
        _save_table_page(pdf, "Failure analysis", fail_headers, fail_rows, font_size=7)

        # Case-analysis image pages
        for case in case_entries:
            abs_path = os.path.join(os.path.dirname(out_pdf_path), case["latex_path"].replace("/", os.sep))
            if not os.path.isfile(abs_path):
                continue
            image = cv2.imread(abs_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fig, ax = _new_page()
            ax.set_title(case["caption"], fontsize=13, loc="left", pad=12)
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Training artifact pages
        for item in training_entries:
            abs_path = os.path.join(os.path.dirname(out_pdf_path), item["latex_path"].replace("/", os.sep))
            if not os.path.isfile(abs_path):
                continue
            image = cv2.imread(abs_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fig, ax = _new_page()
            ax.set_title(item["caption"], fontsize=13, loc="left", pad=12)
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def build_report(args):
    """Main report build orchestration."""
    os.makedirs(args.out, exist_ok=True)
    assets_dir = os.path.join(args.out, "assets")
    cases_dir = os.path.join(assets_dir, "cases")
    train_dir = os.path.join(assets_dir, "training")
    os.makedirs(cases_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    metrics_by_name = _parse_metrics_summary(args.metrics_summary)
    if args.metrics_summary and os.path.isfile(args.metrics_summary):
        if not metrics_by_name:
            raise ValueError(
                f"Metrics summary exists but no metric rows were parsed: {args.metrics_summary}"
            )
        print(f"Using precomputed metrics from: {args.metrics_summary}")
    elif args.metrics_summary:
        print(f"Metrics summary not found, recomputing metrics from masks: {args.metrics_summary}")

    records = _collect_records(
        args.images,
        args.gt,
        args.classic,
        args.yolo,
        metrics_by_name=metrics_by_name if metrics_by_name else None,
    )
    if metrics_by_name:
        missing = [r["name"] for r in records if (r.get("classic") is None and r.get("yolo") is None)]
        if missing:
            print(
                f"WARNING: No metric rows in summary for {len(missing)} GT image(s); "
                "their CSV metric fields will be empty."
            )
    failures = _build_failures(records, args.failure_threshold)
    csv_info = _write_csv_tables(records, failures, args.out)

    name_to_record = {r["name"]: r for r in records}
    success_names, failure_names = _select_cases(records, failures, args.examples)

    case_entries = []
    ordered_names = [("Success", n) for n in success_names] + [("Failure", n) for n in failure_names]
    for label, name in ordered_names:
        rec = name_to_record.get(name)
        if not rec:
            continue

        src = load_image(rec["source_path"], grayscale=True) if rec["source_path"] else None
        gt = _load_binary_from_path(rec["gt_path"])
        classic = _load_binary_from_path(rec["classic_path"])
        yolo = _load_binary_from_path(rec["yolo_path"])

        rel_path = os.path.join("assets", "cases", f"{name}_analysis.png")
        abs_path = os.path.join(args.out, rel_path)
        _make_case_figure(src, gt, classic, yolo, abs_path)

        score = _combined_score(rec)
        caption = f"{label} case: {name} (combined Dice={score:.3f})"
        case_entries.append({
            "latex_path": rel_path.replace("\\", "/"),
            "caption": caption,
        })

    training_entries = []
    if args.yolo_train and os.path.isdir(args.yolo_train):
        training_files = [
            ("results.png", "YOLO training metrics over epochs"),
            ("MaskPR_curve.png", "Mask precision-recall curve"),
            ("confusion_matrix_normalized.png", "Normalized confusion matrix"),
        ]
        for filename, caption in training_files:
            src_path = os.path.join(args.yolo_train, filename)
            dst_rel = os.path.join("assets", "training", filename)
            dst_abs = os.path.join(args.out, dst_rel)
            if _copy_if_exists(src_path, dst_abs):
                training_entries.append({
                    "latex_path": dst_rel.replace("\\", "/"),
                    "caption": caption,
                })

    latex_text = _build_latex_document(
        report_title=args.title,
        report_author=args.author,
        records=records,
        failures=failures,
        classic_avg=csv_info["classic_avg"],
        yolo_avg=csv_info["yolo_avg"],
        case_entries=case_entries,
        training_entries=training_entries,
        max_rows=args.max_rows,
    )

    tex_path = os.path.join(args.out, "summary_report.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_text)

    compiled = False
    compile_msg = "PDF compilation skipped by user option."
    if not args.no_compile:
        compiled, compile_msg = _compile_latex(tex_path)
        if (not compiled) and (not args.no_fallback_pdf):
            fallback_pdf = os.path.join(args.out, "summary_report.pdf")
            _build_fallback_pdf(
                out_pdf_path=fallback_pdf,
                report_title=args.title,
                records=records,
                failures=failures,
                classic_avg=csv_info["classic_avg"],
                yolo_avg=csv_info["yolo_avg"],
                case_entries=case_entries,
                training_entries=training_entries,
            )
            compiled = True
            compile_msg = (
                f"{compile_msg} Fallback PDF generated with matplotlib: {fallback_pdf}"
            )

    manifest_path = os.path.join(args.out, "report_manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("Summary Report Build\n")
        f.write("====================\n\n")
        f.write(f"images_dir: {args.images}\n")
        f.write(f"gt_dir: {args.gt}\n")
        f.write(f"classic_dir: {args.classic}\n")
        f.write(f"yolo_dir: {args.yolo}\n")
        f.write(f"output_dir: {args.out}\n\n")
        f.write(f"records: {len(records)}\n")
        f.write(f"failures: {len(failures)}\n\n")
        f.write(f"per_image_csv: {csv_info['per_image_csv']}\n")
        f.write(f"averages_csv: {csv_info['averages_csv']}\n")
        f.write(f"failures_csv: {csv_info['failures_csv']}\n")
        f.write(f"latex: {tex_path}\n")
        pdf_path = os.path.join(args.out, "summary_report.pdf")
        f.write(f"pdf: {pdf_path if os.path.isfile(pdf_path) else 'not generated'}\n\n")
        f.write(f"compile_status: {compile_msg}\n")

    print(f"Records evaluated: {len(records)}")
    print(f"Failures detected: {len(failures)}")
    print(f"LaTeX report: {tex_path}")
    if compiled:
        print(f"PDF report: {os.path.join(args.out, 'summary_report.pdf')}")
    else:
        print(compile_msg)
    print(f"Manifest: {manifest_path}")


def parse_args():
    """CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate a technical PDF report with metrics and visual analysis."
    )
    parser.add_argument("--images", required=True, help="Directory with source images")
    parser.add_argument("--gt", required=True, help="Directory with ground-truth masks")
    parser.add_argument("--classic", required=True, help="Directory with classic masks")
    parser.add_argument("--yolo", required=True, help="Directory with YOLO masks")
    parser.add_argument(
        "--metrics-summary",
        default=os.path.join("run_all_output", "evaluation", "metrics_summary.txt"),
        help=(
            "Path to run_all evaluation metrics_summary.txt. "
            "If present, metrics are loaded from this file instead of recomputed."
        ),
    )
    parser.add_argument(
        "--yolo-train",
        default=None,
        help="Optional YOLO training artifacts directory (results.png, curves, confusion matrix)",
    )
    parser.add_argument("--out", default=os.path.join("output", "summary_report"), help="Output report directory")
    parser.add_argument("--title", default="SEM Dendrite Segmentation: Comparative Technical Report")
    parser.add_argument("--author", default="Automated Report Generator")
    parser.add_argument("--failure-threshold", type=float, default=0.50)
    parser.add_argument("--examples", type=int, default=6, help="Total number of visual examples (success+failure)")
    parser.add_argument("--max-rows", type=int, default=20, help="Max rows shown in LaTeX tables")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Do not attempt to compile LaTeX to PDF",
    )
    parser.add_argument(
        "--no-fallback-pdf",
        action="store_true",
        help="Disable matplotlib fallback PDF generation when pdflatex is missing",
    )
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    for path in (args.images, args.gt, args.classic, args.yolo):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Required directory not found: {path}")

    build_report(args)


if __name__ == "__main__":
    main()
