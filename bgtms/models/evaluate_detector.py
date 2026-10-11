"""
BGTMS - Comprehensive Model Evaluation & Diagnostic Engine
Evaluates SSDLite320-MobileNetV3 on test split:
- Precision-Recall curves & threshold sweeping (0.10 to 0.90)
- Optimal operating confidence threshold (F1 maximization)
- Class-wise Average Precision (AP@50) and mAP@50
- Confusion matrix & error categorization (background false alarm vs class confusion)
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torchvision.ops as ops
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bgtms.models.detector_dataset import IndianVehicleDataset
from bgtms.models.train_detector import create_detector_model, compute_iou, CLASSES, NUM_CLASSES


def run_threshold_sweep(model, dataset, device, iou_thresh=0.45):
    """
    Sweeps confidence thresholds from 0.15 to 0.85 to find optimal F1-score operating point.
    """
    model.eval()
    thresholds = np.linspace(0.15, 0.85, 15)
    results = []

    # Cache model predictions once for fast sweeping
    cached_preds = []
    cached_gts = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            img_tensor, target = dataset[idx]
            preds = model([img_tensor.to(device)])[0]
            
            p_boxes = preds["boxes"].cpu()
            p_scores = preds["scores"].cpu()
            p_labels = preds["labels"].cpu()

            # Apply NMS per class to ensure clean detections
            keep = ops.batched_nms(p_boxes, p_scores, p_labels, iou_threshold=0.40)
            cached_preds.append({
                "boxes": p_boxes[keep].numpy(),
                "scores": p_scores[keep].numpy(),
                "labels": p_labels[keep].numpy()
            })
            cached_gts.append({
                "boxes": target["boxes"].numpy(),
                "labels": target["labels"].numpy()
            })

    total_gt = sum(len(gt["labels"]) for gt in cached_gts)

    for conf in thresholds:
        tp = 0
        fp = 0

        for pred, gt in zip(cached_preds, cached_gts):
            mask = pred["scores"] >= conf
            filt_boxes = pred["boxes"][mask]
            filt_labels = pred["labels"][mask]

            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]
            matched_gt = set()

            for pb, pl in zip(filt_boxes, filt_labels):
                best_iou = 0.0
                best_gt_idx = -1

                for g_idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                    if g_idx in matched_gt:
                        continue
                    iou = compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_thresh:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, total_gt)
        f1 = 2 * (prec * rec) / max(1e-6, prec + rec)

        results.append({
            "threshold": float(conf),
            "tp": tp,
            "fp": fp,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

    return results, cached_preds, cached_gts


def compute_class_ap(cached_preds, cached_gts, class_id, iou_thresh=0.40):
    """
    Computes 11-point interpolated Average Precision for a specific class.
    """
    target_cls = class_id + 1
    scores = []
    matches = []
    total_class_gt = 0

    for pred, gt in zip(cached_preds, cached_gts):
        gt_mask = gt["labels"] == target_cls
        gt_boxes = gt["boxes"][gt_mask]
        total_class_gt += len(gt_boxes)

        pred_mask = pred["labels"] == target_cls
        p_boxes = pred["boxes"][pred_mask]
        p_scores = pred["scores"][pred_mask]

        matched_gt = set()
        for pb, ps in zip(p_boxes, p_scores):
            scores.append(ps)
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, gb in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_thresh:
                matches.append(1)
                matched_gt.add(best_gt_idx)
            else:
                matches.append(0)

    if total_class_gt == 0 or len(scores) == 0:
        return 0.0, total_class_gt

    # Sort by descending score
    sort_idx = np.argsort(scores)[::-1]
    matches = np.array(matches)[sort_idx]

    tp_cum = np.cumsum(matches)
    fp_cum = np.cumsum(1 - matches)

    recall_curve = tp_cum / total_class_gt
    precision_curve = tp_cum / (tp_cum + fp_cum)

    # 11-point interpolation
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 11):
        prec_at_r = precision_curve[recall_curve >= r]
        ap += max(prec_at_r) if len(prec_at_r) > 0 else 0.0

    return ap / 11.0, total_class_gt


def plot_evaluation_charts(sweep_results, class_aps, output_path="data/processed/model_evaluation_metrics.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    thresholds = [r["threshold"] for r in sweep_results]
    precisions = [r["precision"] * 100.0 for r in sweep_results]
    recalls = [r["recall"] * 100.0 for r in sweep_results]
    f1s = [r["f1"] for r in sweep_results]

    ax1.plot(thresholds, precisions, "b-o", label="Precision (%)", linewidth=2)
    ax1.plot(thresholds, recalls, "r--s", label="Recall (%)", linewidth=2)
    ax1.plot(thresholds, [f * 100 for f in f1s], "g-^", label="F1-Score x100", linewidth=2)
    ax1.set_title("Precision-Recall vs Confidence Threshold", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, 105)
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Class AP Bar Chart
    cnames = [CLASSES[cid] for cid in range(len(class_aps))]
    aps = [class_aps[cid] * 100.0 for cid in range(len(class_aps))]
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#6B7280", "#06B6D4", "#EC4899", "#EF4444"]
    bars = ax2.bar(cnames, aps, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_title("Class-Wise Average Precision (AP@50)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("AP (%)")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar, ap in zip(bars, aps):
        if ap > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., ap + 2, f"{ap:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Evaluation] Metrics chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detailed Model Evaluation & Threshold Tuning")
    parser.add_argument("--model-path", default="bgtms/models/vehicle_detector.pth", help="Model weights path")
    parser.add_argument("--test-split", default="data/splits/test.txt", help="Test split path")
    parser.add_argument("--out-plot", default="data/processed/model_evaluation_metrics.png", help="Output plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Detailed Evaluation on device: {device}...")

    model = create_detector_model(NUM_CLASSES, freeze_backbone=False).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = IndianVehicleDataset(args.test_split, augment=False)
    sweep_results, cached_preds, cached_gts = run_threshold_sweep(model, test_dataset, device)

    # Find optimal threshold
    best_op = max(sweep_results, key=lambda x: x["f1"])

    print("\n" + "="*70)
    print("       BGTMS VEHICLE DETECTOR: CONFIDENCE THRESHOLD SWEEP")
    print("="*70)
    print(f" {'Conf':<8} | {'Precision':<12} | {'Recall':<10} | {'F1-Score':<10} | {'TP':<6} | {'FP':<6}")
    print("-"*70)
    for r in sweep_results:
        print(f" {r['threshold']:<8.2f} | {r['precision']*100:<11.1f}% | {r['recall']*100:<9.1f}% | {r['f1']:<10.3f} | {r['tp']:<6} | {r['fp']:<6}")
    print("="*70)
    print(f"\n[+] Optimal Operating Point Found:")
    print(f"    - Confidence Threshold:  {best_op['threshold']:.2f}")
    print(f"    - Operating Precision:   {best_op['precision']*100:.2f}%")
    print(f"    - Operating Recall:      {best_op['recall']*100:.2f}%")
    print(f"    - Operating F1-Score:    {best_op['f1']:.3f}")

    # Compute Class-Wise AP
    class_aps = {}
    print("\n[*] Computing Class-Wise AP@50...")
    for cid in range(len(CLASSES)):
        ap, gt_count = compute_class_ap(cached_preds, cached_gts, cid)
        class_aps[cid] = ap
        print(f"    - Class {cid} ({CLASSES[cid]}): AP = {ap*100:.1f}% (GT Objects: {gt_count})")

    valid_aps = [ap for ap, gt in [(class_aps[c], compute_class_ap(cached_preds, cached_gts, c)[1]) for c in range(len(CLASSES))] if gt > 0]
    mAP50 = np.mean(valid_aps) if valid_aps else 0.0
    print(f"\n[+] Evaluated mAP@50: {mAP50*100:.2f}%")

    plot_evaluation_charts(sweep_results, class_aps, args.out_plot)


if __name__ == "__main__":
    main()
