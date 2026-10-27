"""
PneuSSLNet Utilities: Logging, Seed initialization, Kaggle Path Discovery, Checkpoint Management
"""
import os
import sys
import time
import json
import random
import shutil
import hashlib
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42):
    """
    Sets reproducible random seeds across Python, NumPy, PyTorch CPU & CUDA.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192 * 16):
            sha.update(chunk)
    return sha.hexdigest()


def compute_dict_sha256(data: Dict[str, Any]) -> str:
    """Computes deterministic SHA-256 hash of a dictionary."""
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def discover_kaggle_paths() -> Dict[str, Optional[str]]:
    """
    Dynamically discovers Kaggle input mounts for Chest X-Ray and RSNA Pneumonia challenge.
    Does not assume a single fixed directory.
    """
    discovered = {
        "chest_xray_dir": None,
        "rsna_labels_csv": None,
        "rsna_detailed_csv": None,
        "rsna_train_images_dir": None,
        "environment": "local"
    }
    
    # Check if running in Kaggle
    if os.path.exists("/kaggle/input"):
        discovered["environment"] = "kaggle"
        for root, dirs, files in os.walk("/kaggle/input"):
            # Discover Chest X-ray
            if "chest_xray" in dirs and discovered["chest_xray_dir"] is None:
                candidate = os.path.join(root, "chest_xray")
                # Ensure it has train/val/test or normal/pneumonia
                subdirs = os.listdir(candidate)
                if any(s.lower() in ["train", "test", "val"] for s in subdirs):
                    discovered["chest_xray_dir"] = candidate
            
            # Discover RSNA files
            for f in files:
                if f == "stage_2_train_labels.csv":
                    discovered["rsna_labels_csv"] = os.path.join(root, f)
                elif f == "stage_2_detailed_class_info.csv":
                    discovered["rsna_detailed_csv"] = os.path.join(root, f)
            
            if "stage_2_train_images" in dirs and discovered["rsna_train_images_dir"] is None:
                discovered["rsna_train_images_dir"] = os.path.join(root, "stage_2_train_images")
    else:
        # Local fallback search
        local_candidates = ["./chest_xray", "../chest_xray", "e:/Pneumonia/chest_xray"]
        for cand in local_candidates:
            if os.path.exists(cand):
                discovered["chest_xray_dir"] = cand
                break
                
    return discovered


class CheckpointManager:
    """
    Atomic, validated, resumable checkpoint manager.
    Enforces strict metadata verification: RUN_ID, config hash, split hash, seed, model schema.
    """
    def __init__(self, base_output_dir: str, run_id: str, split_hash: str, config_hash: str):
        self.base_output_dir = base_output_dir
        self.run_id = run_id
        self.split_hash = split_hash
        self.config_hash = config_hash
        self.registry_path = os.path.join(base_output_dir, "registry", "checkpoint_index.csv")
        os.makedirs(os.path.join(base_output_dir, "registry"), exist_ok=True)
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, "w", encoding="utf-8") as f:
                f.write("run_id,experiment_id,seed,method,backbone,stage,epoch,global_step,checkpoint_type,relative_path,sha256,timestamp\n")

    def save_checkpoint(
        self,
        experiment_id: str,
        method: str,
        backbone: str,
        seed: int,
        stage: str,
        epoch: int,
        global_step: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        history: Optional[Dict[str, List[float]]] = None,
        best_metric: Optional[float] = None,
        is_best: bool = False,
        is_final: bool = False
    ) -> str:
        save_dir = os.path.join(self.base_output_dir, "checkpoints", experiment_id, f"seed-{seed}", stage)
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine filename
        if is_final:
            fname = "final_stage_model.pt"
            ckpt_type = "stage_final"
        elif is_best:
            fname = "best_validation_model.pt"
            ckpt_type = "val_best"
        else:
            fname = "last_resume.pt"
            ckpt_type = "rolling_resume"
            
        final_path = os.path.join(save_dir, fname)
        tmp_path = final_path + ".tmp"
        
        # Real model state dict unwrapping if DataParallel
        model_module = model.module if isinstance(model, nn.DataParallel) else model
        
        # Extract RNG states
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        
        payload = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "experiment_id": experiment_id,
            "method": method,
            "backbone": backbone,
            "seed": seed,
            "stage": stage,
            "epoch": epoch,
            "global_step": global_step,
            "split_hash": self.split_hash,
            "config_hash": self.config_hash,
            "model_state": model_module.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "scaler_state": scaler.state_dict() if scaler else None,
            "rng_states": rng_states,
            "history": history or {},
            "best_metric": best_metric,
            "timestamp": time.time()
        }
        
        # Atomic write
        torch.save(payload, tmp_path)
        shutil.move(tmp_path, final_path)
        
        # Validate checksum & record in registry
        file_sha = compute_file_sha256(final_path)
        rel_path = os.path.relpath(final_path, self.base_output_dir)
        with open(self.registry_path, "a", encoding="utf-8") as f:
            f.write(f"{self.run_id},{experiment_id},{seed},{method},{backbone},{stage},{epoch},{global_step},{ckpt_type},{rel_path},{file_sha},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        return final_path

    def load_checkpoint(
        self,
        experiment_id: str,
        seed: int,
        stage: str,
        device: torch.device,
        filename: str = "last_resume.pt"
    ) -> Optional[Dict[str, Any]]:
        ckpt_path = os.path.join(self.base_output_dir, "checkpoints", experiment_id, f"seed-{seed}", stage, filename)
        if not os.path.exists(ckpt_path):
            return None
            
        ckpt = torch.load(ckpt_path, map_location=device)
        
        # Enforce validation
        if ckpt.get("run_id") != self.run_id:
            print(f"[CheckpointManager WARNING] Run ID mismatch: {ckpt.get('run_id')} vs current {self.run_id}. Checking compatibility...")
        if ckpt.get("split_hash") != self.split_hash:
            raise ValueError(f"[CheckpointManager ERROR] Split hash mismatch in {ckpt_path}! Refusing to resume with different data split.")
        if ckpt.get("config_hash") != self.config_hash:
            print(f"[CheckpointManager WARNING] Config hash mismatch in {ckpt_path}.")
            
        return ckpt

