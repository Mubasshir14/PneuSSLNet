"""
PneuSSLNet Revision Benchmark: Standalone Supervised Baselines
=============================================================
This script trains and evaluates purely supervised baseline models on the exact
same stratified 70/15/15 partition used for the Self-Supervised Learning (SSL) benchmark:
  1. Custom 4-Layer Medical CNN (trained from scratch)
  2. ResNet-50 (trained from scratch without SSL pretraining)
  3. EfficientNet-B0 (trained from scratch without SSL pretraining)

Outputs:
  - Checkpoints: outputs/checkpoints/baseline_*.pt
  - Comparative Table: outputs/tables/table_supervised_baselines.csv
"""

import os
import sys
import time
import argparse
import random
import copy
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
    roc_auc_score, confusion_matrix, brier_score_loss, matthews_corrcoef
)

# ============================================================
# 1. HARDWARE & DETERMINISM
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

# ============================================================
# 2. DATASET DEFINITION & TRANSFORMS
# ============================================================
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.15, contrast=0.15),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

eval_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

class PneumoniaSupervisedDataset(Dataset):
    def __init__(self, file_paths, labels, transform):
        self.file_paths = list(file_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with Image.open(path) as raw:
            img = raw.convert("RGB")
            tensor = self.transform(img)
        return tensor, self.labels[idx]

def load_split_data(manifest_path="outputs/tables/data_split_manifest.csv", data_dir="e:/Pneumonia/dataset/chest_xray/chest_xray"):
    if os.path.exists(manifest_path):
        print(f"[DATA] Loading verified stratified manifest from: {manifest_path}")
        df = pd.read_csv(manifest_path)
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
        return (
            (train_df["file_path"].values, train_df["label"].values),
            (val_df["file_path"].values, val_df["label"].values),
            (test_df["file_path"].values, test_df["label"].values)
        )
    else:
        print(f"[DATA] Manifest not found, discovering images from: {data_dir}")
        from sklearn.model_selection import train_test_split
        import glob
        all_files, all_labels = [], []
        for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
            imgs = glob.glob(os.path.join(data_dir, "**", cls, "*.jpeg"), recursive=True) + \
                   glob.glob(os.path.join(data_dir, "**", cls, "*.jpg"), recursive=True)
            imgs = sorted(list(set(imgs)))
            all_files.extend(imgs)
            all_labels.extend([label] * len(imgs))
        all_files, all_labels = np.array(all_files), np.array(all_labels)
        train_f, test_val_f, train_y, test_val_y = train_test_split(all_files, all_labels, test_size=0.30, random_state=42, stratify=all_labels)
        val_f, test_f, val_y, test_y = train_test_split(test_val_f, test_val_y, test_size=0.50, random_state=42, stratify=test_val_y)
        return (train_f, train_y), (val_f, val_y), (test_f, test_y)

# ============================================================
# 3. MODEL ARCHITECTURES
# ============================================================
class CustomCNN(nn.Module):
    """
    Standard 4-Block Convolutional Architecture for Chest Radiographs.
    Trained strictly from scratch.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)   # 14x14
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def build_scratch_resnet50(num_classes=2, pretrained=False):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.resnet50(weights=weights)
    in_feat = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(in_feat, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    return net

def build_scratch_efficientnet_b0(num_classes=2, pretrained=False):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    in_feat = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feat, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_classes)
    )
    return net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================
def evaluate_model(model, loader, device):
    model.eval()
    all_targets, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_targets, all_preds) * 100.0
    prec = precision_score(all_targets, all_preds, zero_division=0) * 100.0
    rec = recall_score(all_targets, all_preds, zero_division=0) * 100.0
    f1 = f1_score(all_targets, all_preds, zero_division=0) * 100.0
    auc = roc_auc_score(all_targets, all_probs) * 100.0
    brier = brier_score_loss(all_targets, all_probs)
    mcc = matthews_corrcoef(all_targets, all_preds)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    spec = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0

    return {
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "specificity": round(spec, 2),
        "f1": round(f1, 2),
        "auc": round(auc, 2),
        "brier": round(brier, 4),
        "mcc": round(mcc, 4)
    }

# ============================================================
# 5. TRAINING LOOP WITH AMP & SCHEDULER
# ============================================================
def train_single_baseline(model, name, train_loader, val_loader, test_loader, device, epochs=25, lr=1e-3):
    print("\n" + "=" * 70)
    print(f"TRAINING SUPERVISED BASELINE: {name.upper()}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Epochs: {epochs} | Initial LR: {lr} | Mixed Precision AMP: Enabled")
    print("=" * 70)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    best_val_f1 = 0.0
    best_state = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss, train_corr = 0.0, 0
        total_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_corr += (logits.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)

        scheduler.step()
        ep_train_loss = train_loss / total_samples
        ep_train_acc = (train_corr / total_samples) * 100.0

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        val_f1 = val_metrics["f1"]
        val_acc = val_metrics["accuracy"]

        history["train_loss"].append(ep_train_loss)
        history["val_f1"].append(val_f1)

        is_best = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            is_best = " [BEST MODEL]"

        print(f"Epoch [{ep:02d}/{epochs:02d}] Train Loss: {ep_train_loss:.4f} | Train Acc: {ep_train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}%{is_best}")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed/60:.2f} mins. Restoring Best Checkpoint from Epoch {best_epoch} (Val F1: {best_val_f1:.2f}%)...")
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final Test Set Evaluation
    print(f"[TEST EVALUATION] Running inference on unseen test cohort (N={len(test_loader.dataset)})...")
    test_metrics = evaluate_model(model, test_loader, device)
    test_metrics["model_name"] = name
    test_metrics["parameters"] = count_parameters(model)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["train_time_sec"] = round(elapsed, 1)

    print(f">> Final Results for {name}:")
    print(f"   Accuracy:    {test_metrics['accuracy']:.2f}%")
    print(f"   Precision:   {test_metrics['precision']:.2f}%")
    print(f"   Recall/Sens: {test_metrics['recall']:.2f}%")
    print(f"   Specificity: {test_metrics['specificity']:.2f}%")
    print(f"   F1 Score:    {test_metrics['f1']:.2f}%")
    print(f"   AUROC:       {test_metrics['auc']:.2f}%\n")

    # Save checkpoint
    os.makedirs("outputs/checkpoints", exist_ok=True)
    ckpt_path = os.path.join("outputs", "checkpoints", f"baseline_{name.lower().replace(' ', '_')}.pt")
    torch.save({
        "model_name": name,
        "model_state": model.state_dict(),
        "metrics": test_metrics,
        "history": history
    }, ckpt_path)
    print(f"[SAVED] Baseline checkpoint saved to: {ckpt_path}")

    return test_metrics

# ============================================================
# 6. MAIN CONTROLLER
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train Supervised Baselines for PneuSSLNet Benchmark")
    parser.add_argument("--models", type=str, default="all", choices=["all", "custom_cnn", "resnet50", "efficientnet_b0"],
                        help="Which baseline model(s) to train")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (0 safest for Windows)")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights instead of scratch")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("       PneuSSLNet Supervised Baselines Training Pipeline")
    print("=" * 70)
    print(f"Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")
    print(f"Target Models: {args.models} | Epochs: {args.epochs} | Scratch Mode: {not args.pretrained}")
    print("=" * 70)

    # Load verified data split
    (train_f, train_y), (val_f, val_y), (test_f, test_y) = load_split_data()
    print(f"Stratified Cohort Partitions -> Train: {len(train_f)} | Val: {len(val_f)} | Test: {len(test_f)}")

    train_ds = PneumoniaSupervisedDataset(train_f, train_y, train_transform)
    val_ds = PneumoniaSupervisedDataset(val_f, val_y, eval_transform)
    test_ds = PneumoniaSupervisedDataset(test_f, test_y, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    baseline_registry = []
    if args.models in ["all", "custom_cnn"]:
        baseline_registry.append(("Custom CNN", CustomCNN(num_classes=2), 1e-3))
    if args.models in ["all", "resnet50"]:
        mode_str = "Pretrained" if args.pretrained else "Scratch"
        baseline_registry.append((f"ResNet-50 ({mode_str})", build_scratch_resnet50(num_classes=2, pretrained=args.pretrained), 3e-4))
    if args.models in ["all", "efficientnet_b0"]:
        mode_str = "Pretrained" if args.pretrained else "Scratch"
        baseline_registry.append((f"EfficientNet-B0 ({mode_str})", build_scratch_efficientnet_b0(num_classes=2, pretrained=args.pretrained), 3e-4))

    all_results = []
    for model_name, model_inst, model_lr in baseline_registry:
        res = train_single_baseline(
            model=model_inst,
            name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=model_lr
        )
        all_results.append(res)

    # Save summary table
    summary_df = pd.DataFrame(all_results)
    table_cols = ["model_name", "parameters", "accuracy", "precision", "recall", "specificity", "f1", "auc", "brier", "mcc", "best_epoch", "train_time_sec"]
    summary_df = summary_df[table_cols]
    summary_df.columns = ["Model Architecture", "Parameters", "Accuracy (%)", "Precision (%)", "Recall (%)", "Specificity (%)", "F1 Score (%)", "AUROC (%)", "Brier Score", "MCC", "Best Epoch", "Training Time (s)"]

    os.makedirs("outputs/tables", exist_ok=True)
    out_table_path = "outputs/tables/table_supervised_baselines.csv"
    summary_df.to_csv(out_table_path, index=False)

    print("\n" + "=" * 80)
    print("                    SUPERVISED BASELINES BENCHMARK SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"\n[DONE] Comparative Table successfully saved to: {out_table_path}\n")

if __name__ == "__main__":
    main()

