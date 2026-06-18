"""
============================================================
src/evaluate.py
Model evaluation: accuracy, precision, recall, F1,
confusion matrix, ROC-AUC curve
============================================================
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve, average_precision_score,
)
from torch.utils.data import DataLoader

from src.utils import get_logger

logger = get_logger("evaluate")


# ------------------------------------------------------------------ #
# Collect predictions
# ------------------------------------------------------------------ #
@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model over the loader and collect:
      labels     : ground truth (N,)
      preds      : argmax predictions (N,)
      probs      : softmax probabilities (N, C)
    """
    model.eval()
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()
    all_preds  = all_probs.argmax(axis=1)

    return all_labels, all_preds, all_probs


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #
def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    Compute classification metrics.

    For binary classification:
      - AUC uses probability of positive class (index 1).
    For multi-class:
      - AUC uses OvR strategy.
    """
    n_classes = len(class_names)
    avg = "binary" if n_classes == 2 else "macro"

    metrics = {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average=avg, zero_division=0),
        "recall":    recall_score(labels, preds, average=avg, zero_division=0),
        "f1":        f1_score(labels, preds, average=avg, zero_division=0),
    }

    # Specificity (True Negative Rate) — important for medical screening
    if n_classes == 2:
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    try:
        if n_classes == 2:
            metrics["roc_auc"] = roc_auc_score(labels, probs[:, 1])
        else:
            metrics["roc_auc"] = roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            )
    except ValueError as e:
        logger.warning(f"ROC-AUC could not be computed: {e}")
        metrics["roc_auc"] = float("nan")

    # Pretty print
    logger.info("=" * 50)
    logger.info("Evaluation Metrics")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k:12s}: {v:.4f}")
    logger.info("\nClassification Report:")
    logger.info(
        "\n" + classification_report(labels, preds, target_names=class_names)
    )

    return metrics


# ------------------------------------------------------------------ #
# Confusion Matrix Plot
# ------------------------------------------------------------------ #
def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    save_dir: str = "outputs/plots",
):
    """Save a normalized + raw confusion matrix heatmap."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    cm     = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix", fontsize=15, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Raw Counts", "Normalized (%)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    path = f"{save_dir}/confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {path}")


# ------------------------------------------------------------------ #
# ROC Curve Plot
# ------------------------------------------------------------------ #
def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    save_dir: str = "outputs/plots",
):
    """Plot ROC curve (binary or per-class OvR for multi-class)."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    n_classes = len(class_names)

    plt.figure(figsize=(8, 6))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        auc = roc_auc_score(labels, probs[:, 1])
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    else:
        from sklearn.preprocessing import label_binarize
        lb = label_binarize(labels, classes=list(range(n_classes)))
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(lb[:, i], probs[:, i])
            auc = roc_auc_score(lb[:, i], probs[:, i])
            plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved → {path}")


# ------------------------------------------------------------------ #
# Precision-Recall Curve Plot
# ------------------------------------------------------------------ #
def plot_precision_recall_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    save_dir: str = "outputs/plots",
):
    """Plot Precision-Recall curve (important for imbalanced medical data)."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    n_classes = len(class_names)

    plt.figure(figsize=(8, 6))

    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
        ap = average_precision_score(labels, probs[:, 1])
        plt.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    else:
        from sklearn.preprocessing import label_binarize
        lb = label_binarize(labels, classes=list(range(n_classes)))
        for i, cls in enumerate(class_names):
            precision, recall, _ = precision_recall_curve(lb[:, i], probs[:, i])
            ap = average_precision_score(lb[:, i], probs[:, i])
            plt.plot(recall, precision, lw=2, label=f"{cls} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/precision_recall_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Precision-Recall curve saved → {path}")


# ------------------------------------------------------------------ #
# Full evaluate pipeline
# ------------------------------------------------------------------ #
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    split: str = "test",
) -> Dict[str, float]:
    """
    Run full evaluation:
      1. Collect predictions
      2. Compute metrics
      3. Plot confusion matrix
      4. Plot ROC curve
    """
    save_dir    = cfg["paths"]["plots_dir"]
    class_names = cfg["dataset"]["class_names"]

    logger.info(f"Evaluating on {split} set …")
    labels, preds, probs = collect_predictions(model, loader, device)

    metrics = compute_metrics(labels, preds, probs, class_names)

    # Save metrics to JSON
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = f"{save_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    # Save classification report to text file
    report_text = classification_report(labels, preds, target_names=class_names)
    report_path = f"{save_dir}/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Classification report saved → {report_path}")

    plot_confusion_matrix(labels, preds, class_names, save_dir)
    plot_roc_curve(labels, probs, class_names, save_dir)
    plot_precision_recall_curve(labels, probs, class_names, save_dir)

    return metrics
