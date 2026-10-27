"""
PneuSSLNet Evaluation, Metrics Suite, Calibration, and External Generalization Module.
"""
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score, brier_score_loss, roc_curve
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_comprehensive_metrics(
    labels: List[int],
    preds: List[int],
    probs: List[float],
    class_names: List[str] = ["NORMAL", "PNEUMONIA"],
    positive_class_idx: int = 1
) -> Dict[str, Any]:
    """
    Computes complete publication-grade metrics:
    Accuracy, Balanced Acc, Macro Prec, Macro Rec, Macro F1,
    Pneumonia Sensitivity, Specificity, AUROC, AUPRC, MCC, Brier Score, ECE,
    and Raw Confusion Matrix.
    """
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    probs_arr = np.array(probs)

    acc = accuracy_score(labels_arr, preds_arr)
    bal_acc = balanced_accuracy_score(labels_arr, preds_arr)
    macro_prec = precision_score(labels_arr, preds_arr, average="macro", zero_division=0)
    macro_rec = recall_score(labels_arr, preds_arr, average="macro", zero_division=0)
    macro_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    
    # Specificity & Sensitivity
    cm = confusion_matrix(labels_arr, preds_arr)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        sensitivity, specificity = float("nan"), float("nan")

    # AUROC & AUPRC
    try:
        auroc = roc_auc_score(labels_arr, probs_arr)
    except Exception:
        auroc = float("nan")

    try:
        auprc = average_precision_score(labels_arr, probs_arr)
    except Exception:
        auprc = float("nan")

    # MCC & Brier Score
    mcc = matthews_corrcoef(labels_arr, preds_arr)
    brier = brier_score_loss(labels_arr, probs_arr)

    # Expected Calibration Error (ECE) with 10 bins
    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(probs_arr, bins) - 1
    ece = 0.0
    for b in range(10):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(labels_arr[mask] == preds_arr[mask])
            bin_conf = np.mean(np.maximum(probs_arr[mask], 1.0 - probs_arr[mask]))
            ece += np.sum(mask) / len(probs_arr) * np.abs(bin_acc - bin_conf)

    # Per-class metrics
    per_class = {}
    for idx, cname in enumerate(class_names):
        c_prec = precision_score(labels_arr == idx, preds_arr == idx, zero_division=0)
        c_rec = recall_score(labels_arr == idx, preds_arr == idx, zero_division=0)
        c_f1 = f1_score(labels_arr == idx, preds_arr == idx, zero_division=0)
        c_supp = int(np.sum(labels_arr == idx))
        per_class[cname] = {
            "precision": round(c_prec * 100, 2),
            "recall": round(c_rec * 100, 2),
            "f1": round(c_f1 * 100, 2),
            "support": c_supp
        }

    return {
        "accuracy": round(acc * 100, 2),
        "balanced_accuracy": round(bal_acc * 100, 2),
        "macro_precision": round(macro_prec * 100, 2),
        "macro_recall": round(macro_rec * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "sensitivity": round(sensitivity * 100, 2),
        "specificity": round(specificity * 100, 2),
        "auroc": round(auroc * 100, 2),
        "auprc": round(auprc * 100, 2),
        "mcc": round(mcc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "per_class": per_class
    }


def evaluate_locked_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    decision_threshold: float = 0.5,
    enable_tta: bool = False
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Dedicated locked test evaluator.
    Called ONLY AFTER models are fully trained and checkpoints frozen.
    """
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            if enable_tta:
                probs_orig = F.softmax(model(x), dim=1)
                probs_flip = F.softmax(model(torch.flip(x, dims=[3])), dim=1)
                probs = (probs_orig + probs_flip) / 2.0
            else:
                probs = F.softmax(model(x), dim=1)

            pos_probs = probs[:, 1].cpu().numpy().tolist()
            preds = [1 if p >= decision_threshold else 0 for p in pos_probs]

            all_probs.extend(pos_probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy().tolist())

    metrics = compute_comprehensive_metrics(all_labels, all_preds, all_probs)
    
    df_preds = pd.DataFrame({
        "ground_truth": all_labels,
        "predicted_class": all_preds,
        "positive_probability": all_probs,
        "threshold_used": decision_threshold,
        "tta_enabled": enable_tta
    })

    return metrics, df_preds

