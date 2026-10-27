"""
PneuSSLNet Training & Loss Functions:
- SSL Pretraining Loops (SimCLR, MoCo, BYOL, DINO)
- Two-Stage Downstream Fine-Tuning with strict Test-Set Firewall
- Supervised Full Training for Baselines & Ablations
"""
import time
import copy
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.models.ssl_models import (
    build_backbone, SimCLRModel, MoCoModel, BYOLModel, DINOModel,
    FullClassifier, set_backbone_unfreezing
)
from src.utils.common import CheckpointManager


# --- SSL Losses ---

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    batch_size = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    diag_val = -1e4 if sim.dtype in (torch.float16, torch.bfloat16) else -1e9
    sim.masked_fill_(mask, diag_val)

    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    return F.cross_entropy(sim, targets)


def moco_infonce_loss(q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
    l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


def byol_loss(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=1)
    t = F.normalize(t, dim=1)
    return 2.0 - 2.0 * (q * t).sum(dim=1).mean()


def dino_loss(
    student_out: List[torch.Tensor],
    teacher_out: List[torch.Tensor],
    center: torch.Tensor,
    student_temp: float = 0.1,
    teacher_temp: float = 0.04
) -> torch.Tensor:
    total_loss = 0.0
    n_loss_terms = 0
    for s_idx, s in enumerate(student_out):
        for t_idx, t in enumerate(teacher_out):
            if s_idx == t_idx and s_idx < len(teacher_out):
                continue
            s_softmax = F.log_softmax(s / student_temp, dim=-1)
            t_softmax = F.softmax((t - center) / teacher_temp, dim=-1).detach()
            total_loss += -(t_softmax * s_softmax).sum(dim=-1).mean()
            n_loss_terms += 1
    return total_loss / max(1, n_loss_terms)


# --- Core Training Loops ---

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    device: torch.device = torch.device("cpu")
) -> Tuple[float, List[int], List[int], List[float]]:
    """
    Executes a single epoch over a DataLoader.
    If optimizer is None, runs in eval mode without gradients.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item() * x.size(0)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, all_labels, all_preds, all_probs


def train_downstream_classifier(
    backbone: nn.Module,
    backbone_name: str,
    feature_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stage1_epochs: int,
    stage2_epochs: int,
    lr_head: float,
    lr_finetune: float,
    class_weights: Optional[torch.Tensor],
    label_smoothing: float,
    checkpoint_mgr: CheckpointManager,
    experiment_id: str,
    method_name: str,
    seed: int,
    device: torch.device
) -> Tuple[FullClassifier, Dict[str, List[float]]]:
    """
    Strict Two-Stage Fine-Tuning.
    TEST_LOADER IS NOT ACCEPTED HERE - Strictly Enforcing Information Firewall!
    """
    model = FullClassifier(backbone, feature_dim, num_classes=2).to(device)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    
    # Check for resume
    ckpt_s2 = checkpoint_mgr.load_checkpoint(experiment_id, seed, "stage-2", device)
    ckpt_s1 = checkpoint_mgr.load_checkpoint(experiment_id, seed, "stage-1", device)
    
    start_stage = 1
    start_epoch = 0
    best_val_f1 = -1.0
    
    if ckpt_s2 is not None:
        model.load_state_dict(ckpt_s2["model_state"])
        history = ckpt_s2["history"]
        start_stage = 2
        start_epoch = ckpt_s2["epoch"] + 1
        print(f"[{experiment_id} seed {seed}] Resuming Stage 2 at epoch {start_epoch + 1}/{stage2_epochs}")
    elif ckpt_s1 is not None:
        model.load_state_dict(ckpt_s1["model_state"])
        history = ckpt_s1["history"]
        if ckpt_s1["epoch"] + 1 >= stage1_epochs:
            start_stage = 2
            start_epoch = 0
            print(f"[{experiment_id} seed {seed}] Stage 1 already complete. Starting Stage 2.")
        else:
            start_stage = 1
            start_epoch = ckpt_s1["epoch"] + 1
            print(f"[{experiment_id} seed {seed}] Resuming Stage 1 at epoch {start_epoch + 1}/{stage1_epochs}")

    # --- STAGE 1: Frozen Feature Extractor (Head Only) ---
    if start_stage == 1:
        set_backbone_unfreezing(model.backbone, backbone_name, stage=1)
        optimizer = torch.optim.Adam(model.head.parameters(), lr=lr_head, weight_decay=1e-4)
        
        for ep in range(start_epoch, stage1_epochs):
            tr_loss, tr_y, tr_p, _ = run_epoch(
                model, train_loader, optimizer, class_weights=class_weights,
                label_smoothing=label_smoothing, device=device
            )
            va_loss, va_y, va_p, _ = run_epoch(
                model, val_loader, optimizer=None, class_weights=class_weights,
                label_smoothing=label_smoothing, device=device
            )
            
            tr_acc = accuracy_score(tr_y, tr_p)
            va_acc = accuracy_score(va_y, va_p)
            va_f1 = f1_score(va_y, va_p, average="macro", zero_division=0)
            
            history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)
            history["val_f1"].append(va_f1)
            
            print(f"[Stage 1: Ep {ep+1:02d}/{stage1_epochs:02d}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} F1: {va_f1:.4f}")
            
            checkpoint_mgr.save_checkpoint(
                experiment_id=experiment_id, method=method_name, backbone=backbone_name,
                seed=seed, stage="stage-1", epoch=ep, global_step=(ep+1)*len(train_loader),
                model=model, optimizer=optimizer, history=history, is_final=(ep+1 == stage1_epochs)
            )
            
        start_stage = 2
        start_epoch = 0

    # --- STAGE 2: Partial Unfreezing + Fine-Tuning ---
    if start_stage == 2:
        set_backbone_unfreezing(model.backbone, backbone_name, stage=2)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr_finetune, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, stage2_epochs))
        
        for ep in range(start_epoch, stage2_epochs):
            tr_loss, tr_y, tr_p, _ = run_epoch(
                model, train_loader, optimizer, class_weights=class_weights,
                label_smoothing=label_smoothing, device=device
            )
            va_loss, va_y, va_p, _ = run_epoch(
                model, val_loader, optimizer=None, class_weights=class_weights,
                label_smoothing=label_smoothing, device=device
            )
            scheduler.step()
            
            tr_acc = accuracy_score(tr_y, tr_p)
            va_acc = accuracy_score(va_y, va_p)
            va_f1 = f1_score(va_y, va_p, average="macro", zero_division=0)
            
            history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)
            history["val_f1"].append(va_f1)
            
            is_best = va_f1 > best_val_f1
            if is_best:
                best_val_f1 = va_f1
                
            print(f"[Stage 2: Ep {ep+1:02d}/{stage2_epochs:02d}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} F1: {va_f1:.4f} (best {best_val_f1:.4f})")
            
            checkpoint_mgr.save_checkpoint(
                experiment_id=experiment_id, method=method_name, backbone=backbone_name,
                seed=seed, stage="stage-2", epoch=ep, global_step=(stage1_epochs + ep + 1)*len(train_loader),
                model=model, optimizer=optimizer, scheduler=scheduler, history=history,
                best_metric=best_val_f1, is_best=is_best, is_final=(ep+1 == stage2_epochs)
            )

    return model, history

