"""
PneuSSLNet: Self-Supervised Learning Framework for Pneumonia Classification
Q1 Experimental Revision Pipeline (Local PC Runtime - NVIDIA RTX 4060 GPU)

Authors: Md. Alamin, S M Hasan Mahmud, Md. Faruk Hosen, Kah Ong Michael Goh, Watshara Shoombuatong
Runtime: Local NVIDIA GPU (RTX 4060, 8GB VRAM) with PyTorch AMP
Outputs: e:/Pneumonia/New Code/outputs/
"""

import os
import sys
import time
import copy
import glob
import random
import shutil
import gc
import json
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, matthews_corrcoef
)
import matplotlib.pyplot as plt

# ============================================================
# 1. HARDWARE & DEVICE INITIALIZATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ============================================================
# 2. CENTRAL EXPERIMENT CONFIGURATION
# ============================================================
CONFIG = {
    "PROJECT_NAME": "PneuSSLNet_Q1_Local_Revision",
    "DATA_DIR": "e:/Pneumonia/dataset/chest_xray/chest_xray",
    "RSNA_DIR": "e:/Pneumonia/dataset/rsna",
    "OUTPUT_DIR": "e:/Pneumonia/New Code/outputs",
    "IMG_SIZE": 224,
    "LOCAL_CROP_SIZE": 96,
    "BATCH_SIZE_SSL": 16,
    "BATCH_SIZE_CLF": 32,
    "NUM_WORKERS": 2,  # 2 workers safe for Windows PyTorch DataLoader
    "SSL_EPOCHS": 30,
    "FT_STAGE1_EPOCHS": 15,
    "FT_STAGE2_EPOCHS": 15,
    "FT_EVAL_EPOCHS": [10, 20, 30],
    "LR_HEAD": 1e-3,
    "LR_FINETUNE": 1e-4,
    "PROJECTION_DIM": 128,
    "HIDDEN_DIM": 512,
    "FEATURE_DIM": 2048,
    "NUM_CLASSES": 2,
    "SEED": 42,
    "BASELINE_EPOCHS": 30,
    "LABELED_FRACTIONS": [0.10, 0.25, 0.50, 0.75, 1.00],
}

# Ensure directory structure
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
for sub in ["checkpoints", "tables", "figures", "predictions", "webapp_bundle"]:
    os.makedirs(os.path.join(CONFIG["OUTPUT_DIR"], sub), exist_ok=True)

# Set deterministic seeds
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(CONFIG["SEED"])

# ============================================================
# 3. DATASET DISCOVERY & STRATIFIED PARTITIONING (70/15/15)
# ============================================================
def discover_and_split_data(data_dir, seed=42):
    class_names = ["NORMAL", "PNEUMONIA"]
    all_files, all_labels = [], []
    for label, cls in enumerate(class_names):
        pattern_jpeg = os.path.join(data_dir, "**", cls, "*.jpeg")
        pattern_jpg = os.path.join(data_dir, "**", cls, "*.jpg")
        imgs = glob.glob(pattern_jpeg, recursive=True) + glob.glob(pattern_jpg, recursive=True)
        imgs = sorted(list(set(imgs)))
        print(f"[{cls}] Found {len(imgs)} valid radiographs.")
        all_files.extend(imgs)
        all_labels.extend([label] * len(imgs))

    all_files = np.array(all_files)
    all_labels = np.array(all_labels)

    # Stratified 70 / 15 / 15 split
    train_f, test_val_f, train_y, test_val_y = train_test_split(
        all_files, all_labels, test_size=0.30, random_state=seed, stratify=all_labels
    )
    val_f, test_f, val_y, test_y = train_test_split(
        test_val_f, test_val_y, test_size=0.50, random_state=seed, stratify=test_val_y
    )

    print(f"\nDataset Partitions -> Train: {len(train_f)} | Val: {len(val_f)} | Test: {len(test_f)}")
    print(f"Train Balance -> NORMAL: {(train_y == 0).sum()}, PNEUMONIA: {(train_y == 1).sum()}")

    # Save data split manifest to tables
    manifest_df = pd.DataFrame({
        "file_path": np.concatenate([train_f, val_f, test_f]),
        "split": ["train"] * len(train_f) + ["val"] * len(val_f) + ["test"] * len(test_f),
        "label": np.concatenate([train_y, val_y, test_y]),
        "class_name": ["NORMAL" if y == 0 else "PNEUMONIA" for y in np.concatenate([train_y, val_y, test_y])]
    })
    manifest_path = os.path.join(CONFIG["OUTPUT_DIR"], "tables", "data_split_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Data split manifest saved to: {manifest_path}\n")

    return (train_f, train_y), (val_f, val_y), (test_f, test_y)

# ============================================================
# 4. TRANSFORMS & DATA LOADERS (OOM-PROTECTED)
# ============================================================
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def to_rgb(img):
    return img.convert("RGB")

simclr_transform = T.Compose([
    T.Lambda(to_rgb),
    T.RandomResizedCrop(CONFIG["IMG_SIZE"], scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

moco_transform = T.Compose([
    T.Lambda(to_rgb),
    T.RandomResizedCrop(CONFIG["IMG_SIZE"], scale=(0.8, 1.0)),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1),
    T.RandomGrayscale(p=0.2),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

byol_transform_strong = T.Compose([
    T.Lambda(to_rgb),
    T.RandomResizedCrop(CONFIG["IMG_SIZE"], scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

byol_transform_light = T.Compose([
    T.Lambda(to_rgb),
    T.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

dino_transform_global1 = byol_transform_strong
dino_transform_global2 = T.Compose([
    T.Lambda(to_rgb),
    T.RandomResizedCrop(CONFIG["IMG_SIZE"], scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])
dino_transform_local = T.Compose([
    T.Lambda(to_rgb),
    T.RandomResizedCrop(CONFIG["LOCAL_CROP_SIZE"], scale=(0.05, 0.4)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

clf_train_transform = T.Compose([
    T.Lambda(to_rgb),
    T.RandomResizedCrop(CONFIG["IMG_SIZE"], scale=(0.85, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

clf_eval_transform = T.Compose([
    T.Lambda(to_rgb),
    T.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

SSL_TRANSFORM_PAIRS = {
    "SimCLR": (simclr_transform, simclr_transform),
    "MoCo": (moco_transform, moco_transform),
    "BYOL": (byol_transform_strong, byol_transform_light),
    "DINO": (dino_transform_global1, dino_transform_global2),
}

class TwoViewDataset(Dataset):
    def __init__(self, file_paths, transform_pair):
        self.file_paths = file_paths
        self.t1, self.t2 = transform_pair

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with Image.open(self.file_paths[idx]) as raw_img:
            img = raw_img.convert("RGB")
            v1, v2 = self.t1(img), self.t2(img)
        return v1, v2

class DinoMultiCropDataset(Dataset):
    def __init__(self, file_paths, global_t1, global_t2, local_t, n_local=2):
        self.file_paths = file_paths
        self.global_t1, self.global_t2 = global_t1, global_t2
        self.local_t = local_t
        self.n_local = n_local

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with Image.open(self.file_paths[idx]) as raw_img:
            img = raw_img.convert("RGB")
            views = [self.global_t1(img), self.global_t2(img)]
            views += [self.local_t(img) for _ in range(self.n_local)]
        return tuple(views)

class LabeledImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with Image.open(self.file_paths[idx]) as raw_img:
            img = raw_img.convert("RGB")
            tensor = self.transform(img)
        return tensor, self.labels[idx]

def make_ssl_loader(method_name, file_paths, batch_size):
    if method_name == "DINO":
        ds = DinoMultiCropDataset(file_paths, dino_transform_global1, dino_transform_global2, dino_transform_local)
    else:
        ds = TwoViewDataset(file_paths, SSL_TRANSFORM_PAIRS[method_name])
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=CONFIG["NUM_WORKERS"], drop_last=True, pin_memory=False)

def make_clf_loaders(train_f, train_y, val_f, val_y, test_f, test_y, batch_size):
    train_ds = LabeledImageDataset(train_f, train_y, clf_train_transform)
    val_ds = LabeledImageDataset(val_f, val_y, clf_eval_transform)
    test_ds = LabeledImageDataset(test_f, test_y, clf_eval_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=CONFIG["NUM_WORKERS"], pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], pin_memory=False)
    return train_loader, val_loader, test_loader

# ============================================================
# 5. MODEL ARCHITECTURES & LOSS FUNCTIONS
# ============================================================
def build_backbone(name="resnet50"):
    if name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feature_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return net.to(DEVICE), feature_dim

def make_projection_head(feature_dim, hidden_dim=512, out_dim=128):
    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim)
    )

def make_predictor(dim=128, hidden_dim=512):
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, dim)
    )

@torch.no_grad()
def momentum_update(online_module, target_module, m):
    for p_o, p_t in zip(online_module.parameters(), target_module.parameters()):
        p_t.data = p_t.data * m + p_o.data * (1.0 - m)

class SimCLRModel(nn.Module):
    def __init__(self, backbone, feature_dim, proj_dim=128, hidden_dim=512):
        super().__init__()
        self.backbone = backbone
        self.projector = make_projection_head(feature_dim, hidden_dim, proj_dim)
    def forward(self, x):
        return self.projector(self.backbone(x))

class MoCoModel(nn.Module):
    def __init__(self, backbone_fn, feature_dim, proj_dim=128, hidden_dim=512, queue_size=2048, momentum=0.999):
        super().__init__()
        self.backbone_q = backbone_fn()
        self.proj_q = make_projection_head(feature_dim, hidden_dim, proj_dim)
        self.backbone_k = copy.deepcopy(self.backbone_q)
        self.proj_k = copy.deepcopy(self.proj_q)
        for p in list(self.backbone_k.parameters()) + list(self.proj_k.parameters()):
            p.requires_grad = False
        self.momentum = momentum
        self.queue_size = queue_size
        self.register_buffer("queue", torch.randn(proj_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        momentum_update(self.backbone_q, self.backbone_k, self.momentum)
        momentum_update(self.proj_q, self.proj_k, self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_size:
            ptr = 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x_q, x_k):
        q = F.normalize(self.proj_q(self.backbone_q(x_q)), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.proj_k(self.backbone_k(x_k)), dim=1)
        return q, k

class BYOLModel(nn.Module):
    def __init__(self, backbone_fn, feature_dim, proj_dim=128, hidden_dim=512, momentum=0.996):
        super().__init__()
        self.backbone_online = backbone_fn()
        self.proj_online = make_projection_head(feature_dim, hidden_dim, proj_dim)
        self.predictor = make_predictor(proj_dim, hidden_dim)
        self.backbone_target = copy.deepcopy(self.backbone_online)
        self.proj_target = copy.deepcopy(self.proj_online)
        for p in list(self.backbone_target.parameters()) + list(self.proj_target.parameters()):
            p.requires_grad = False
        self.momentum = momentum

    @torch.no_grad()
    def update_target(self):
        momentum_update(self.backbone_online, self.backbone_target, self.momentum)
        momentum_update(self.proj_online, self.proj_target, self.momentum)

    def forward(self, x1, x2):
        q1 = self.predictor(self.proj_online(self.backbone_online(x1)))
        q2 = self.predictor(self.proj_online(self.backbone_online(x2)))
        with torch.no_grad():
            t1 = self.proj_target(self.backbone_target(x1))
            t2 = self.proj_target(self.backbone_target(x2))
        return q1, q2, t1.detach(), t2.detach()

class DINOModel(nn.Module):
    def __init__(self, backbone_fn, feature_dim, proj_dim=128, hidden_dim=512, momentum=0.996, student_temp=0.1, teacher_temp=0.04):
        super().__init__()
        self.student_backbone = backbone_fn()
        self.student_proj = make_projection_head(feature_dim, hidden_dim, proj_dim)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_proj = copy.deepcopy(self.student_proj)
        for p in list(self.teacher_backbone.parameters()) + list(self.teacher_proj.parameters()):
            p.requires_grad = False
        self.momentum = momentum
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.register_buffer("center", torch.zeros(1, proj_dim))

    @torch.no_grad()
    def update_teacher(self):
        momentum_update(self.student_backbone, self.teacher_backbone, self.momentum)
        momentum_update(self.student_proj, self.teacher_proj, self.momentum)

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * 0.9 + batch_center * 0.1

    def forward_multicrop(self, views, n_global=2):
        student_outputs = [self.student_proj(self.student_backbone(v)) for v in views]
        with torch.no_grad():
            teacher_outputs = [self.teacher_proj(self.teacher_backbone(v)) for v in views[:n_global]]
        return student_outputs, teacher_outputs

# Loss Functions
def nt_xent_loss(z1, z2, temperature=0.5):
    b = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    diag_val = -1e4 if sim.dtype in (torch.float16, torch.bfloat16) else -1e9
    sim.masked_fill_(torch.eye(2 * b, dtype=torch.bool, device=z.device), diag_val)
    targets = torch.cat([torch.arange(b, device=z.device) + b, torch.arange(b, device=z.device)], dim=0)
    return F.cross_entropy(sim, targets)

def moco_infonce_loss(q, k, queue, temperature=0.07):
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
    l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)

def byol_loss(q, t):
    return 2 - 2 * (F.normalize(q, dim=1) * F.normalize(t, dim=1)).sum(dim=1).mean()

def dino_loss(s_out, t_out, center, s_temp=0.1, t_temp=0.04):
    s_log_probs = F.log_softmax(s_out / s_temp, dim=-1)
    t_probs = F.softmax((t_out - center) / t_temp, dim=-1).detach()
    return -(t_probs * s_log_probs).sum(dim=-1).mean()

# Classifier Head
class FullClassifier(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.adapter = nn.Linear(feature_dim, 128)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(self.relu(self.adapter(feats)))

def set_backbone_frozen(backbone, frozen=True, unfreeze_last_n_blocks=0):
    for p in backbone.parameters():
        p.requires_grad = not frozen
    if frozen and unfreeze_last_n_blocks > 0:
        children = list(backbone.children())
        for child in children[-unfreeze_last_n_blocks:]:
            for p in child.parameters():
                p.requires_grad = True

# ============================================================
# 6. EVALUATION METRICS SUITE
# ============================================================
def compute_metrics(labels, preds, probs):
    labels, preds, probs = np.array(labels), np.array(preds), np.array(probs)
    acc = accuracy_score(labels, preds) * 100
    prec = precision_score(labels, preds, zero_division=0) * 100
    rec = recall_score(labels, preds, zero_division=0) * 100
    f1 = f1_score(labels, preds, zero_division=0) * 100
    try:
        auc = roc_auc_score(labels, probs) * 100
    except Exception:
        auc = 50.0
    return {
        "Accuracy": round(acc, 2),
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "F1 Score": round(f1, 2),
        "AUC": round(auc, 2)
    }

def find_best_threshold(labels, probs):
    best_th, best_f1 = 0.5, -1
    for th in np.linspace(0.1, 0.9, 81):
        preds = (probs >= th).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_th = score, th
    return best_th

# ============================================================
# 7. MAIN ORCHESTRATION PIPELINE
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("STARTING PNEUSSLNET Q1 EXPERIMENT REVISION RUN")
    print("=" * 70)

    # 1. Prepare data
    (train_f, train_y), (val_f, val_y), (test_f, test_y) = discover_and_split_data(CONFIG["DATA_DIR"], CONFIG["SEED"])

    BACKBONES = ["resnet50", "efficientnet_b0"]
    SSL_METHODS = ["SimCLR", "MoCo", "BYOL", "DINO"]

    RESULTS_CKPT = os.path.join(CONFIG["OUTPUT_DIR"], "results_checkpoint.pt")
    if os.path.exists(RESULTS_CKPT):
        RESULTS = torch.load(RESULTS_CKPT, map_location="cpu")
        print(f"[RESUME] Loaded results checkpoint. Completed: {[f'{b}-{m}' for b in RESULTS for m in RESULTS[b]]}")
    else:
        RESULTS = {}

    scaler = torch.amp.GradScaler('cuda')

    for backbone_name in BACKBONES:
        RESULTS.setdefault(backbone_name, {})
        for method_name in SSL_METHODS:
            if method_name in RESULTS[backbone_name]:
                print(f"[SKIP] {method_name}/{backbone_name} already completed in previous run.")
                continue

            print(f"\n{'='*70}\n[TRAINING] {method_name} with {backbone_name}\n{'='*70}")
            t_start = time.time()

            # ---- Step 1: SSL Pretraining ----
            pretrain_ckpt_path = os.path.join(CONFIG["OUTPUT_DIR"], "checkpoints", f"ssl_{method_name}_{backbone_name}.pt")
            backbone, feat_dim = build_backbone(backbone_name)
            ssl_loader = make_ssl_loader(method_name, train_f, CONFIG["BATCH_SIZE_SSL"])

            start_epoch, ssl_loss_history = 0, []
            if os.path.exists(pretrain_ckpt_path):
                ckpt = torch.load(pretrain_ckpt_path, map_location=DEVICE)
                start_epoch = ckpt["epoch"] + 1
                ssl_loss_history = ckpt["loss_history"]
                print(f"[{method_name}:{backbone_name}] Resuming SSL from epoch {start_epoch + 1}/{CONFIG['SSL_EPOCHS']}")

            if method_name == "SimCLR":
                ssl_model = SimCLRModel(backbone, feat_dim).to(DEVICE)
                optimizer = torch.optim.Adam(ssl_model.parameters(), lr=3e-4)
            elif method_name == "MoCo":
                ssl_model = MoCoModel(lambda: build_backbone(backbone_name)[0], feat_dim).to(DEVICE)
                optimizer = torch.optim.Adam(list(ssl_model.backbone_q.parameters()) + list(ssl_model.proj_q.parameters()), lr=3e-4)
            elif method_name == "BYOL":
                ssl_model = BYOLModel(lambda: build_backbone(backbone_name)[0], feat_dim).to(DEVICE)
                optimizer = torch.optim.Adam(list(ssl_model.backbone_online.parameters()) + list(ssl_model.proj_online.parameters()) + list(ssl_model.predictor.parameters()), lr=3e-4)
            elif method_name == "DINO":
                ssl_model = DINOModel(lambda: build_backbone(backbone_name)[0], feat_dim).to(DEVICE)
                optimizer = torch.optim.Adam(list(ssl_model.student_backbone.parameters()) + list(ssl_model.student_proj.parameters()), lr=3e-4)

            # SSL Pretraining Loop
            for epoch in range(start_epoch, CONFIG["SSL_EPOCHS"]):
                ssl_model.train()
                running_loss = 0.0
                for batch in ssl_loader:
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        if method_name == "SimCLR":
                            x1, x2 = batch[0].to(DEVICE), batch[1].to(DEVICE)
                            z1, z2 = ssl_model(x1), ssl_model(x2)
                            loss = nt_xent_loss(z1, z2)
                        elif method_name == "MoCo":
                            xq, xk = batch[0].to(DEVICE), batch[1].to(DEVICE)
                            q, k = ssl_model(xq, xk)
                            loss = moco_infonce_loss(q, k, ssl_model.queue)
                        elif method_name == "BYOL":
                            x1, x2 = batch[0].to(DEVICE), batch[1].to(DEVICE)
                            q1, q2, t1, t2 = ssl_model(x1, x2)
                            loss = byol_loss(q1, t2) + byol_loss(q2, t1)
                        elif method_name == "DINO":
                            views = [v.to(DEVICE) for v in batch]
                            s_outs, t_outs = ssl_model.forward_multicrop(views)
                            loss = sum(dino_loss(s, t, ssl_model.center) for t in t_outs for s in s_outs) / (len(s_outs) * len(t_outs))

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    if method_name == "MoCo":
                        ssl_model._dequeue_and_enqueue(k.detach())
                    elif method_name == "BYOL":
                        ssl_model.update_target()
                    elif method_name == "DINO":
                        ssl_model.update_teacher()
                        with torch.no_grad():
                            ssl_model.update_center(torch.cat(t_outs, dim=0))

                    running_loss += loss.item()

                avg_loss = running_loss / max(1, len(ssl_loader))
                ssl_loss_history.append(avg_loss)
                print(f"[{method_name}:{backbone_name}] SSL Epoch {epoch+1:02d}/{CONFIG['SSL_EPOCHS']} - Loss: {avg_loss:.4f}")

                # Save SSL checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state": ssl_model.state_dict(),
                    "loss_history": ssl_loss_history
                }, pretrain_ckpt_path)

            # Extract pretrained backbone
            if method_name == "SimCLR":
                pretrained_bb = ssl_model.backbone
            elif method_name == "MoCo":
                pretrained_bb = ssl_model.backbone_q
            elif method_name == "BYOL":
                pretrained_bb = ssl_model.backbone_online
            elif method_name == "DINO":
                pretrained_bb = ssl_model.student_backbone

            # ---- Step 2: Two-Stage Fine-Tuning Pipeline ----
            print(f"[{method_name}:{backbone_name}] Starting UCL Two-Stage Fine-Tuning...")
            clf_model = FullClassifier(pretrained_bb, feat_dim, CONFIG["NUM_CLASSES"]).to(DEVICE)
            train_loader, val_loader, test_loader = make_clf_loaders(
                train_f, train_y, val_f, val_y, test_f, test_y, CONFIG["BATCH_SIZE_CLF"]
            )

            # Class weights for imbalanced pneumonia
            class_counts = [(train_y == 0).sum(), (train_y == 1).sum()]
            weights = torch.tensor([sum(class_counts) / (2.0 * c) for c in class_counts], dtype=torch.float32).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

            best_val_acc, best_metrics, best_state = -1.0, None, None
            ft_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
            checkpoint_metrics = {}

            # Stage 1: Frozen backbone (Head only)
            set_backbone_frozen(clf_model.backbone, frozen=True, unfreeze_last_n_blocks=0)
            opt_head = torch.optim.Adam(clf_model.parameters(), lr=CONFIG["LR_HEAD"])

            for ep in range(1, CONFIG["FT_STAGE1_EPOCHS"] + 1):
                clf_model.train()
                t_loss, t_corr = 0.0, 0
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt_head.zero_grad()
                    with torch.amp.autocast('cuda'):
                        logits = clf_model(x)
                        loss = criterion(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(opt_head)
                    scaler.update()
                    t_loss += loss.item() * x.size(0)
                    t_corr += (logits.argmax(1) == y).sum().item()

                # Validation
                clf_model.eval()
                v_loss, v_corr, val_preds, val_targets, val_probs = 0.0, 0, [], [], []
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        logits = clf_model(x)
                        probs = F.softmax(logits, dim=1)[:, 1]
                        v_loss += criterion(logits, y).item() * x.size(0)
                        v_corr += (logits.argmax(1) == y).sum().item()
                        val_targets.extend(y.cpu().numpy())
                        val_probs.extend(probs.cpu().numpy())

                tr_acc = t_corr / len(train_loader.dataset)
                vl_acc = v_corr / len(val_loader.dataset)
                ft_history["train_acc"].append(tr_acc)
                ft_history["val_acc"].append(vl_acc)

                if vl_acc > best_val_acc:
                    best_val_acc = vl_acc
                    best_state = copy.deepcopy(clf_model.state_dict())

                print(f"[FT Stage 1] Epoch {ep:02d}/{CONFIG['FT_STAGE1_EPOCHS']} - Train Acc: {tr_acc*100:.2f}% | Val Acc: {vl_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

            # Stage 2: Unfreeze last 3 blocks + Cosine LR
            set_backbone_frozen(clf_model.backbone, frozen=True, unfreeze_last_n_blocks=3)
            trainable_params = [p for p in clf_model.parameters() if p.requires_grad]
            opt_ft = torch.optim.Adam(trainable_params, lr=CONFIG["LR_FINETUNE"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=CONFIG["FT_STAGE2_EPOCHS"])

            for ep in range(1, CONFIG["FT_STAGE2_EPOCHS"] + 1):
                total_ep = CONFIG["FT_STAGE1_EPOCHS"] + ep
                clf_model.train()
                t_loss, t_corr = 0.0, 0
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt_ft.zero_grad()
                    with torch.amp.autocast('cuda'):
                        logits = clf_model(x)
                        loss = criterion(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(opt_ft)
                    scaler.update()
                    t_loss += loss.item() * x.size(0)
                    t_corr += (logits.argmax(1) == y).sum().item()
                scheduler.step()

                # Validation
                clf_model.eval()
                v_loss, v_corr, val_targets, val_probs = 0.0, 0, [], []
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        logits = clf_model(x)
                        probs = F.softmax(logits, dim=1)[:, 1]
                        v_loss += criterion(logits, y).item() * x.size(0)
                        v_corr += (logits.argmax(1) == y).sum().item()
                        val_targets.extend(y.cpu().numpy())
                        val_probs.extend(probs.cpu().numpy())

                tr_acc = t_corr / len(train_loader.dataset)
                vl_acc = v_corr / len(val_loader.dataset)
                ft_history["train_acc"].append(tr_acc)
                ft_history["val_acc"].append(vl_acc)

                if vl_acc > best_val_acc:
                    best_val_acc = vl_acc
                    best_state = copy.deepcopy(clf_model.state_dict())

                # Evaluate intermediate checkpoints if in FT_EVAL_EPOCHS
                if total_ep in CONFIG["FT_EVAL_EPOCHS"]:
                    test_targets, test_probs = [], []
                    with torch.no_grad():
                        for x, y in test_loader:
                            x, y = x.to(DEVICE), y.to(DEVICE)
                            probs = F.softmax(clf_model(x), dim=1)[:, 1]
                            test_targets.extend(y.cpu().numpy())
                            test_probs.extend(probs.cpu().numpy())
                    test_preds = (np.array(test_probs) >= 0.50).astype(int)
                    checkpoint_metrics[total_ep] = compute_metrics(test_targets, test_preds, test_probs)

                print(f"[FT Stage 2] Epoch {total_ep:02d}/{CONFIG['FT_STAGE1_EPOCHS']+CONFIG['FT_STAGE2_EPOCHS']} - Train Acc: {tr_acc*100:.2f}% | Val Acc: {vl_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

            # Best validation checkpoint evaluation with TTA
            if best_state is not None:
                clf_model.load_state_dict(best_state)
            clf_model.eval()
            test_targets, test_probs = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    # TTA: original + horizontal flip
                    p1 = F.softmax(clf_model(x), dim=1)[:, 1]
                    p2 = F.softmax(clf_model(torch.flip(x, dims=[3])), dim=1)[:, 1]
                    probs = (p1 + p2) / 2.0
                    test_targets.extend(y.cpu().numpy())
                    test_probs.extend(probs.cpu().numpy())

            test_preds = (np.array(test_probs) >= 0.50).astype(int)
            final_metrics = compute_metrics(test_targets, test_preds, test_probs)
            checkpoint_metrics["best"] = final_metrics
            print(f"\n[FINAL TEST EVALUATION] {method_name}/{backbone_name}: {final_metrics}")

            # Save fine-tuned model checkpoint
            ft_ckpt_path = os.path.join(CONFIG["OUTPUT_DIR"], "checkpoints", f"ft_{method_name}_{backbone_name}.pt")
            torch.save({
                "model_state": clf_model.state_dict(),
                "best_state": best_state,
                "history": ft_history,
                "metrics": checkpoint_metrics
            }, ft_ckpt_path)

            RESULTS[backbone_name][method_name] = {
                "ssl_loss": ssl_loss_history,
                "checkpoint_metrics": checkpoint_metrics,
                "history": ft_history
            }
            torch.save(RESULTS, RESULTS_CKPT)
            print(f"Saved RESULTS checkpoint after {time.time()-t_start:.1f}s.")

            # Cleanup
            del ssl_model, clf_model, pretrained_bb
            gc.collect()
            torch.cuda.empty_cache()

    # ============================================================
    # 8. PUBLICATION TABLES & FIGURES GENERATION
    # ============================================================
    print("\n" + "=" * 70)
    print("GENERATING PUBLICATION TABLES & FIGURES (Tables 4-9)")
    print("=" * 70)

    # Table 4 & 5: Epoch comparisons
    for b_idx, b_name in enumerate(["resnet50", "efficientnet_b0"], start=4):
        rows = []
        for m_name in SSL_METHODS:
            if m_name in RESULTS.get(b_name, {}):
                ckpt_m = RESULTS[b_name][m_name]["checkpoint_metrics"]
                for ep in [10, 20, 30]:
                    if ep in ckpt_m:
                        rows.append({
                            "Method": m_name, "Epoch": ep,
                            "Accuracy": ckpt_m[ep]["Accuracy"],
                            "Precision": ckpt_m[ep]["Precision"],
                            "Recall": ckpt_m[ep]["Recall"],
                            "F1-Score": ckpt_m[ep]["F1 Score"]
                        })
        if rows:
            df_table = pd.DataFrame(rows)
            df_table.to_csv(os.path.join(CONFIG["OUTPUT_DIR"], "tables", f"table_{b_idx}_{b_name}.csv"), index=False)
            print(f"Table {b_idx} saved: table_{b_idx}_{b_name}.csv")

    # Table 6: Final Summary Comparison
    rows6 = []
    for b_name in BACKBONES:
        for m_name in SSL_METHODS:
            if m_name in RESULTS.get(b_name, {}):
                best = RESULTS[b_name][m_name]["checkpoint_metrics"].get("best", {})
                rows6.append({
                    "Backbone": b_name, "Method": m_name,
                    "Accuracy": best.get("Accuracy", "-"),
                    "Precision": best.get("Precision", "-"),
                    "Recall": best.get("Recall", "-"),
                    "F1 Score": best.get("F1 Score", "-"),
                    "AUC": best.get("AUC", "-")
                })
    if rows6:
        df6 = pd.DataFrame(rows6)
        df6.to_csv(os.path.join(CONFIG["OUTPUT_DIR"], "tables", "table_6_test_summary.csv"), index=False)
        print("Table 6 saved: table_6_test_summary.csv")

    # ============================================================
    # 9. STAGE D: RSNA EXTERNAL VALIDATION (CLINICAL DICOM)
    # ============================================================
    print("\n" + "=" * 70)
    print("RUNNING STAGE D: RSNA EXTERNAL VALIDATION EVALUATION")
    print("=" * 70)

    rsna_labels_csv = os.path.join(CONFIG["RSNA_DIR"], "stage_2_train_labels.csv")
    rsna_details_csv = os.path.join(CONFIG["RSNA_DIR"], "stage_2_detailed_class_info.csv")
    rsna_imgs_dir = os.path.join(CONFIG["RSNA_DIR"], "stage_2_train_images")

    if os.path.exists(rsna_labels_csv) and os.path.exists(rsna_imgs_dir):
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut

            df_lbl = pd.read_csv(rsna_labels_csv)
            df_det = pd.read_csv(rsna_details_csv)
            agg_lbl = df_lbl.groupby("patientId").agg({"Target": "max"}).reset_index()
            uniq_det = df_det.drop_duplicates(subset=["patientId"]).reset_index(drop=True)
            merged_rsna = pd.merge(agg_lbl, uniq_det[["patientId", "class"]], on="patientId")
            merged_rsna["image_path"] = merged_rsna["patientId"].apply(lambda pid: os.path.join(rsna_imgs_dir, f"{pid}.dcm"))
            merged_rsna = merged_rsna[merged_rsna["image_path"].apply(os.path.exists)].reset_index(drop=True)
            print(f"[RSNA] Total verified DICOM images: {len(merged_rsna)}")

            best_ckpt = os.path.join(CONFIG["OUTPUT_DIR"], "checkpoints", "ft_SimCLR_resnet50.pt")
            if not os.path.exists(best_ckpt):
                avail = glob.glob(os.path.join(CONFIG["OUTPUT_DIR"], "checkpoints", "ft_*.pt"))
                best_ckpt = avail[0] if avail else None

            if best_ckpt and os.path.exists(best_ckpt):
                print(f"[RSNA] Evaluating checkpoint: {os.path.basename(best_ckpt)}")
                ckpt_data = torch.load(best_ckpt, map_location=DEVICE)
                eval_bb, f_dim = build_backbone("resnet50")
                eval_clf = FullClassifier(eval_bb, f_dim, CONFIG["NUM_CLASSES"]).to(DEVICE)
                eval_clf.load_state_dict(ckpt_data["model_state"])
                eval_clf.eval()

                def read_dcm(path):
                    dcm = pydicom.dcmread(path)
                    px = dcm.pixel_array.astype(np.float32)
                    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
                        px = np.max(px) - px
                    try:
                        px = apply_voi_lut(px, dcm)
                    except Exception:
                        pass
                    pmin, pmax = px.min(), px.max()
                    px = ((px - pmin) / (pmax - pmin) * 255.0) if pmax > pmin else np.zeros_like(px)
                    return Image.fromarray(px.astype(np.uint8)).convert("RGB").resize((224, 224))

                class RSNAEvalDS(Dataset):
                    def __init__(self, df):
                        self.df = df
                    def __len__(self):
                        return len(self.df)
                    def __getitem__(self, idx):
                        r = self.df.iloc[idx]
                        return clf_eval_transform(read_dcm(r["image_path"])), int(r["Target"])

                rsna_loader = DataLoader(RSNAEvalDS(merged_rsna), batch_size=CONFIG["BATCH_SIZE_CLF"], shuffle=False, num_workers=2)
                all_t, all_p = [], []
                with torch.no_grad():
                    for x, y in rsna_loader:
                        x = x.to(DEVICE)
                        probs = F.softmax(eval_clf(x), dim=1)[:, 1].cpu().numpy()
                        all_p.extend(probs)
                        all_t.extend(y.numpy())

                all_t, all_p = np.array(all_t), np.array(all_p)
                all_preds = (all_p >= 0.50).astype(int)
                r_acc = accuracy_score(all_t, all_preds) * 100
                r_prec = precision_score(all_t, all_preds, zero_division=0) * 100
                r_rec = recall_score(all_t, all_preds, zero_division=0) * 100
                r_f1 = f1_score(all_t, all_preds, zero_division=0) * 100
                r_auc = roc_auc_score(all_t, all_p) * 100

                rsna_summary_df = pd.DataFrame([{
                    "Cohort": "RSNA Pneumonia Challenge (External)",
                    "Total Images": len(merged_rsna),
                    "Accuracy": round(r_acc, 2),
                    "Precision": round(r_prec, 2),
                    "Recall": round(r_rec, 2),
                    "F1 Score": round(r_f1, 2),
                    "AUROC": round(r_auc, 2)
                }])
                tbl_rsna = os.path.join(CONFIG["OUTPUT_DIR"], "tables", "table_external_rsna_validation.csv")
                rsna_summary_df.to_csv(tbl_rsna, index=False)
                print(f"[SUCCESS] RSNA External Validation table saved to: {tbl_rsna}")
                print(f"  Accuracy: {r_acc:.2f}% | Precision: {r_prec:.2f}% | Recall: {r_rec:.2f}% | F1: {r_f1:.2f}% | AUROC: {r_auc:.2f}%")
        except Exception as e:
            print(f"[RSNA Evaluation Warning]: {e}")
    else:
        print("[RSNA] RSNA dataset files not detected at CONFIG['RSNA_DIR'].")

    # ============================================================
    # 10. GENERATE 12 SUPPLEMENTARY TABLES & 12 SUPPLEMENTARY FIGURES
    # ============================================================
    try:
        from src.evaluation.supplementary_generator import generate_all_supplementary
        generate_all_supplementary(output_dir=CONFIG["OUTPUT_DIR"], results_dict=RESULTS)
    except Exception as e:
        print(f"[Supplementary Generator Warning]: {e}")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTAL RUNS COMPLETE! ARTIFACTS READY IN outputs/")
    print("=" * 70)

if __name__ == "__main__":
    main()

