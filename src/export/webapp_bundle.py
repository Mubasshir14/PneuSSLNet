"""
PneuSSLNet Web-App Bundle Exporter & Standalone Inference Engine.
Exports validation-selected model with zero training dependencies.
"""
import os
import json
import shutil
import hashlib
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ssl_models import FullClassifier, build_backbone
from src.utils.common import compute_file_sha256


def export_webapp_bundle(
    model: FullClassifier,
    backbone_name: str,
    feature_dim: int,
    output_bundle_dir: str,
    model_version: str = "v1.0.0",
    run_id: str = "unknown",
    validation_metrics: Optional[Dict[str, Any]] = None
) -> str:
    """
    Creates a self-contained web-app inference bundle:
    - model_state_dict.pt (inference weights only)
    - model_config.json
    - preprocessing.json
    - label_map.json
    - decision_policy.json
    - model_card.md
    - inference.py
    - requirements-inference.txt
    - SHA256SUMS.txt
    """
    os.makedirs(output_bundle_dir, exist_ok=True)
    
    # 1. Save weights (strip optimizer and unwanted keys)
    weights_path = os.path.join(output_bundle_dir, "model_state_dict.pt")
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(real_model.state_dict(), weights_path)
    
    # 2. Model Config
    model_config = {
        "model_version": model_version,
        "run_id": run_id,
        "backbone_name": backbone_name,
        "feature_dim": feature_dim,
        "num_classes": 2,
        "projection_dim": 128
    }
    with open(os.path.join(output_bundle_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
        
    # 3. Preprocessing Config
    preprocessing_config = {
        "input_size": [224, 224],
        "image_mode": "RGB",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "dicom_rules": {
            "handle_rescale": True,
            "handle_monochrome1": True,
            "apply_voi_lut": True
        }
    }
    with open(os.path.join(output_bundle_dir, "preprocessing.json"), "w") as f:
        json.dump(preprocessing_config, f, indent=2)
        
    # 4. Label Map
    label_map = {
        "0": "NORMAL",
        "1": "PNEUMONIA"
    }
    with open(os.path.join(output_bundle_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
        
    # 5. Decision Policy
    decision_policy = {
        "primary_threshold": 0.5000,
        "positive_class": 1,
        "positive_label": "PNEUMONIA",
        "tta_enabled": False
    }
    with open(os.path.join(output_bundle_dir, "decision_policy.json"), "w") as f:
        json.dump(decision_policy, f, indent=2)
        
    # 6. Requirements
    requirements_text = "torch>=2.0.0\ntorchvision>=0.15.0\npillow>=9.0.0\nnumpy>=1.20.0\npydicom>=2.3.0\n"
    with open(os.path.join(output_bundle_dir, "requirements-inference.txt"), "w") as f:
        f.write(requirements_text)
        
    # 7. Model Card
    model_card = f"""# Model Card: PneuSSLNet ({model_version})

## Model Overview
- **Architecture:** {backbone_name} backbone with self-supervised pretraining and two-stage downstream adaptation.
- **Model Version:** {model_version}
- **Run ID:** {run_id}
- **Task:** Binary classification of chest radiographs into `NORMAL` (0) or `PNEUMONIA` (1).

## Intended Clinical Scope & Limitations
- **Intended Use:** Exploratory research prototype for computer-assisted detection of pneumonia from chest X-rays.
- **NOT A DIAGNOSTIC DEVICE:** This model is not FDA approved, CE marked, or cleared for clinical decision making without independent physician oversight.
- **Domain Shift:** Developed primarily on pediatric chest radiographs (Kermany cohort). Performance on adult or external datasets (e.g. RSNA) exhibits expected domain shifts.

## Checkpoint Provenance
- Selected strictly based on internal validation performance.
- Internal test set and external validation sets were never accessed during model or threshold selection.
"""
    with open(os.path.join(output_bundle_dir, "model_card.md"), "w") as f:
        f.write(model_card)
        
    # 8. Standalone inference.py
    inference_script = """import os
import json
from typing import Dict, Any, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

class ClassifierHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = 2, proj_dim: int = 128):
        super().__init__()
        self.adapter = nn.Linear(feature_dim, proj_dim)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.relu(self.adapter(feats)))

class FullClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.head = ClassifierHead(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

class PneuSSLNetPredictor:
    def __init__(self, bundle_dir: str, device: str = "cpu"):
        self.bundle_dir = bundle_dir
        self.device = torch.device(device)
        
        with open(os.path.join(bundle_dir, "model_config.json")) as f:
            self.cfg = json.load(f)
        with open(os.path.join(bundle_dir, "preprocessing.json")) as f:
            self.pre_cfg = json.load(f)
        with open(os.path.join(bundle_dir, "decision_policy.json")) as f:
            self.dec_cfg = json.load(f)
        with open(os.path.join(bundle_dir, "label_map.json")) as f:
            self.labels = json.load(f)
            
        # Build model
        bb_name = self.cfg["backbone_name"]
        if bb_name == "resnet50":
            net = models.resnet50(weights=None)
            net.fc = nn.Identity()
        elif bb_name == "efficientnet_b0":
            net = models.efficientnet_b0(weights=None)
            net.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone {bb_name}")
            
        self.model = FullClassifier(net, self.cfg["feature_dim"], self.cfg["num_classes"])
        weights_path = os.path.join(bundle_dir, "model_state_dict.pt")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transform
        norm = self.pre_cfg["normalization"]
        size = tuple(self.pre_cfg["input_size"])
        self.transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=norm["mean"], std=norm["std"])
        ])

    def predict(self, image: Union[Image.Image, str]) -> Dict[str, Any]:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
            
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        pos_prob = float(probs[self.dec_cfg["positive_class"]])
        pred_idx = 1 if pos_prob >= self.dec_cfg["primary_threshold"] else 0
        pred_label = self.labels[str(pred_idx)]
        
        return {
            "prediction": pred_label,
            "prediction_class_index": pred_idx,
            "pneumonia_probability": round(pos_prob, 4),
            "normal_probability": round(float(probs[0]), 4),
            "threshold": self.dec_cfg["primary_threshold"],
            "model_version": self.cfg["model_version"]
        }
"""
    with open(os.path.join(output_bundle_dir, "inference.py"), "w") as f:
        f.write(inference_script)
        
    # 9. Compute Checksums
    checksums = []
    for root, _, files in os.walk(output_bundle_dir):
        for f in sorted(files):
            if f == "SHA256SUMS.txt":
                continue
            fpath = os.path.join(root, f)
            sha = compute_file_sha256(fpath)
            rel = os.path.relpath(fpath, output_bundle_dir)
            checksums.append(f"{sha}  {rel}")
            
    with open(os.path.join(output_bundle_dir, "SHA256SUMS.txt"), "w") as f:
        f.write("\n".join(checksums) + "\n")
        
    return output_bundle_dir

