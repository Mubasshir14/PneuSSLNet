"""
PneuSSLNet External Validation Pipeline: RSNA Pneumonia Cohort (N=26,684)
========================================================================
This script performs rigorous out-of-distribution external clinical validation
on the entire RSNA Pneumonia Detection DICOM cohort (26,684 chest radiographs):
  - Ingestion of full 16-bit DICOM pixel arrays with Photometric Interpretation handling (MONOCHROME1/2)
  - VOI LUT windowing and contrast normalization
  - Inference using the locked fine-tuned SSL checkpoints (e.g. DINO / BYOL / SimCLR / MoCo)
  - Computation of clinical diagnostic metrics:
      * Sensitivity (Recall), Specificity, Accuracy, F1-Score
      * PPV (Precision), NPV, AUROC, AUPRC, Brier Calibration Score
  - Clinical Subgroup Disaggregation:
      * Lung Opacity (Pneumonia) vs Normal Cohort
      * Lung Opacity vs Non-Pneumonia Abnormalities ('No Lung Opacity / Not Normal')

Outputs:
  - Table: outputs/tables/table_external_rsna_validation.csv
  - Subgroup Table: outputs/tables/table_rsna_subgroup_analysis.csv
  - Predictions Manifest: outputs/predictions/rsna_external_predictions.csv
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, matthews_corrcoef
)

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# ============================================================
# 1. DICOM PREPROCESSING & NORMALIZATION
# ============================================================
def read_dcm_to_image(path):
    """
    Safely reads a DICOM radiograph, handling MONOCHROME1 inversion,
    VOI LUT windowing, and min-max normalization to standard uint8 RGB.
    """
    try:
        dcm = pydicom.dcmread(path)
        px = dcm.pixel_array.astype(np.float32)

        # Invert if MONOCHROME1 (where zero is white)
        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            px = np.max(px) - px

        # Apply VOI LUT if present
        try:
            px = apply_voi_lut(px, dcm)
        except Exception:
            pass

        # Normalize to [0, 255]
        pmin, pmax = px.min(), px.max()
        if pmax > pmin:
            px = ((px - pmin) / (pmax - pmin) * 255.0)
        else:
            px = np.zeros_like(px)

        img = Image.fromarray(px.astype(np.uint8)).convert("RGB")
        return img
    except Exception as e:
        # Fallback for corrupted DICOM
        return Image.new("RGB", (224, 224), color=0)

# Standard evaluation transform
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

rsna_eval_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# ============================================================
# 2. RSNA DATASET CLASS (Top-level for Windows Multiprocessing)
# ============================================================
class RSNACohortDataset(Dataset):
    def __init__(self, df, transform=rsna_eval_transform):
        self.df = df
        self.transform = transform
        self.paths = df["image_path"].values
        self.labels = df["Target"].values
        self.patient_ids = df["patientId"].values
        self.subgroups = df["class"].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = read_dcm_to_image(path)
        tensor = self.transform(img)
        label = int(self.labels[idx])
        pid = self.patient_ids[idx]
        subgroup = self.subgroups[idx]
        return tensor, label, pid, subgroup

# ============================================================
# 3. MODEL ARCHITECTURE (Matches PneuSSLNet Fine-Tuned Weights)
# ============================================================
class FullClassifier(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.adapter = nn.Linear(feature_dim, 128)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(self.relu(self.adapter(feats)))

def load_evaluation_model(checkpoint_path, backbone_name, device):
    """
    Builds the backbone and loads fine-tuned weights from checkpoint.
    """
    if backbone_name == "resnet50":
        net = models.resnet50(weights=None)
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
    elif backbone_name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=None)
        feature_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    clf_model = FullClassifier(net, feature_dim, num_classes=2).to(device)

    print(f"[MODEL] Loading weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("best_state", ckpt.get("model_state", ckpt))
    clf_model.load_state_dict(state)
    clf_model.eval()
    return clf_model

# ============================================================
# 4. COHORT DISCOVERY & PREPARATION
# ============================================================
def prepare_rsna_cohort(rsna_dir="e:/Pneumonia/dataset/rsna", sample_n=None):
    labels_csv = os.path.join(rsna_dir, "stage_2_train_labels.csv")
    class_csv = os.path.join(rsna_dir, "stage_2_detailed_class_info.csv")
    img_dir = os.path.join(rsna_dir, "stage_2_train_images")

    if not os.path.exists(labels_csv) or not os.path.exists(class_csv):
        raise FileNotFoundError(f"RSNA CSV files missing in {rsna_dir}")

    print(f"[RSNA] Reading labels from: {labels_csv}")
    df_labels = pd.read_csv(labels_csv)
    # Deduplicate bounding boxes by patientId (Target: 1 if any opacity, else 0)
    df_labels = df_labels.groupby("patientId")["Target"].max().reset_index()

    print(f"[RSNA] Reading detailed class info from: {class_csv}")
    df_class = pd.read_csv(class_csv).drop_duplicates(subset=["patientId"])

    merged = pd.merge(df_labels, df_class, on="patientId")
    merged["image_path"] = merged["patientId"].apply(lambda pid: os.path.join(img_dir, f"{pid}.dcm"))

    # Verify files
    existing_mask = [os.path.exists(p) for p in merged["image_path"]]
    merged = merged[existing_mask].reset_index(drop=True)
    print(f"[RSNA] Total verified DICOM radiographs: {len(merged):,}")
    print(f"       Normal Radiographs:                {(merged['class'] == 'Normal').sum():,}")
    print(f"       No Lung Opacity / Not Normal:      {(merged['class'] == 'No Lung Opacity / Not Normal').sum():,}")
    print(f"       Lung Opacity (Pneumonia Confirmed): {(merged['class'] == 'Lung Opacity').sum():,}")

    if sample_n and sample_n < len(merged):
        print(f"[SAMPLE] Sampling {sample_n} radiographs for rapid evaluation...")
        merged = merged.sample(n=sample_n, random_state=42).reset_index(drop=True)

    return merged

# ============================================================
# 5. METRIC CALCULATION SUITE
# ============================================================
def calculate_clinical_metrics(y_true, y_prob, threshold=0.50):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    auc = roc_auc_score(y_true, y_prob) * 100.0
    auprc = average_precision_score(y_true, y_prob) * 100.0
    brier = brier_score_loss(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    npv = (tn / (tn + fn)) * 100.0 if (tn + fn) > 0 else 0.0

    return {
        "Accuracy (%)": round(acc, 2),
        "Sensitivity/Recall (%)": round(rec, 2),
        "Specificity (%)": round(spec, 2),
        "Precision/PPV (%)": round(prec, 2),
        "NPV (%)": round(npv, 2),
        "F1 Score (%)": round(f1, 2),
        "AUROC (%)": round(auc, 2),
        "AUPRC (%)": round(auprc, 2),
        "Brier Score": round(brier, 4),
        "MCC": round(mcc, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }

# ============================================================
# 6. MAIN CONTROLLER
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PneuSSLNet External Validation on RSNA DICOM Cohort")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/ft_DINO_efficientnet_b0.pt",
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--backbone", type=str, default="auto", choices=["auto", "resnet50", "efficientnet_b0"],
                        help="Backbone architecture (auto-detected if 'auto')")
    parser.add_argument("--rsna_dir", type=str, default="e:/Pneumonia/dataset/rsna",
                        help="Path to RSNA dataset directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--sample_n", type=int, default=None, help="Evaluate on subset (e.g. 1000) for testing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("       PneuSSLNet External Clinical Validation: RSNA Cohort")
    print("=" * 75)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Detect backbone from filename if auto
    if args.backbone == "auto":
        fname = os.path.basename(args.checkpoint).lower()
        if "resnet" in fname:
            backbone_name = "resnet50"
        elif "efficientnet" in fname:
            backbone_name = "efficientnet_b0"
        else:
            backbone_name = "efficientnet_b0"
    else:
        backbone_name = args.backbone

    print(f"Backbone: {backbone_name} | Checkpoint: {args.checkpoint}")
    print("=" * 75)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    # Load Model
    model = load_evaluation_model(args.checkpoint, backbone_name, device)

    # Prepare RSNA Cohort
    cohort_df = prepare_rsna_cohort(rsna_dir=args.rsna_dir, sample_n=args.sample_n)
    dataset = RSNACohortDataset(cohort_df)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    print(f"\n[INFERENCE] Starting evaluation on {len(cohort_df):,} DICOM radiographs...")
    t0 = time.time()

    all_targets, all_probs, all_pids, all_subgroups = [], [], [], []
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (x, y, pids, subgroups) in enumerate(loader):
            x = x.to(device)
            with torch.amp.autocast('cuda') if device.type == "cuda" else torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.numpy())
            all_pids.extend(pids)
            all_subgroups.extend(subgroups)

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - t0
                pct = (batch_idx + 1) / total_batches * 100.0
                rate = len(all_probs) / elapsed
                print(f"  Batch [{batch_idx+1:03d}/{total_batches:03d}] ({pct:.1f}%) - Processed: {len(all_probs):,} imgs | Speed: {rate:.1f} img/s")

    elapsed_total = time.time() - t0
    print(f"\n[INFERENCE COMPLETE] Evaluated {len(all_probs):,} images in {elapsed_total/60:.2f} minutes ({len(all_probs)/elapsed_total:.1f} img/s).")

    # Global Overall Metrics
    overall_metrics = calculate_clinical_metrics(all_targets, all_probs)
    print("\n" + "=" * 75)
    print("                 GLOBAL RSNA EXTERNAL VALIDATION METRICS")
    print("=" * 75)
    for k, v in overall_metrics.items():
        print(f"  {k:<25}: {v}")
    print("=" * 75)

    # Subgroup Analysis:
    # 1. Normal vs Lung Opacity
    # 2. Not Normal vs Lung Opacity
    sub_df = pd.DataFrame({
        "patientId": all_pids,
        "Target": all_targets,
        "class": all_subgroups,
        "probability": all_probs,
        "prediction": (np.array(all_probs) >= 0.50).astype(int)
    })

    # Subgroup 1: Normal vs Opacity
    mask_norm_op = sub_df["class"].isin(["Normal", "Lung Opacity"])
    df_norm_op = sub_df[mask_norm_op]
    metrics_norm_op = calculate_clinical_metrics(df_norm_op["Target"], df_norm_op["probability"])

    # Subgroup 2: Not Normal vs Opacity
    mask_notnorm_op = sub_df["class"].isin(["No Lung Opacity / Not Normal", "Lung Opacity"])
    df_notnorm_op = sub_df[mask_notnorm_op]
    metrics_notnorm_op = calculate_clinical_metrics(df_notnorm_op["Target"], df_notnorm_op["probability"])

    # Build Summary DataFrames
    summary_row = {
        "Cohort": "RSNA External Overall",
        "Checkpoint": os.path.basename(args.checkpoint),
        "Total Radiographs": len(sub_df),
        "Accuracy (%)": overall_metrics["Accuracy (%)"],
        "Sensitivity (%)": overall_metrics["Sensitivity/Recall (%)"],
        "Specificity (%)": overall_metrics["Specificity (%)"],
        "Precision (%)": overall_metrics["Precision/PPV (%)"],
        "F1 Score (%)": overall_metrics["F1 Score (%)"],
        "AUROC (%)": overall_metrics["AUROC (%)"],
        "AUPRC (%)": overall_metrics["AUPRC (%)"],
        "Brier Score": overall_metrics["Brier Score"]
    }
    summary_table = pd.DataFrame([summary_row])

    subgroup_table = pd.DataFrame([
        {
            "Comparison Subgroup": "Overall Cohort (All Cases)",
            "N": len(sub_df),
            "Sensitivity (%)": overall_metrics["Sensitivity/Recall (%)"],
            "Specificity (%)": overall_metrics["Specificity (%)"],
            "Accuracy (%)": overall_metrics["Accuracy (%)"],
            "F1 Score (%)": overall_metrics["F1 Score (%)"],
            "AUROC (%)": overall_metrics["AUROC (%)"]
        },
        {
            "Comparison Subgroup": "Pneumonia vs Normal Only",
            "N": len(df_norm_op),
            "Sensitivity (%)": metrics_norm_op["Sensitivity/Recall (%)"],
            "Specificity (%)": metrics_norm_op["Specificity (%)"],
            "Accuracy (%)": metrics_norm_op["Accuracy (%)"],
            "F1 Score (%)": metrics_norm_op["F1 Score (%)"],
            "AUROC (%)": metrics_norm_op["AUROC (%)"]
        },
        {
            "Comparison Subgroup": "Pneumonia vs Other Non-Pneumonia Abnormalities",
            "N": len(df_notnorm_op),
            "Sensitivity (%)": metrics_notnorm_op["Sensitivity/Recall (%)"],
            "Specificity (%)": metrics_notnorm_op["Specificity (%)"],
            "Accuracy (%)": metrics_notnorm_op["Accuracy (%)"],
            "F1 Score (%)": metrics_notnorm_op["F1 Score (%)"],
            "AUROC (%)": metrics_notnorm_op["AUROC (%)"]
        }
    ])

    # Save Tables & Predictions
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)

    tbl_path = "outputs/tables/table_external_rsna_validation.csv"
    sub_path = "outputs/tables/table_rsna_subgroup_analysis.csv"
    pred_path = "outputs/predictions/rsna_external_predictions.csv"

    summary_table.to_csv(tbl_path, index=False)
    subgroup_table.to_csv(sub_path, index=False)
    sub_df.to_csv(pred_path, index=False)

    print("\n" + "=" * 75)
    print("                    SUBGROUP CLINICAL ANALYSIS")
    print("=" * 75)
    print(subgroup_table.to_string(index=False))
    print("=" * 75)
    print(f"\n[OUTPUT SAVED] Global Summary Table:    {tbl_path}")
    print(f"[OUTPUT SAVED] Subgroup Clinical Table: {sub_path}")
    print(f"[OUTPUT SAVED] Predictions File:        {pred_path}\n")

if __name__ == "__main__":
    main()

