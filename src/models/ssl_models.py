"""
PneuSSLNet Models: Backbones, SSL Architectures (SimCLR, MoCo, BYOL, DINO),
and Downstream Classification Head with Proper Layer Unfreezing & BatchNorm Freezing.
"""
import copy
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def build_backbone(name: str = "resnet50", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Constructs ResNet-50 or EfficientNet-B0 backbone.
    Replaces final classification layer with nn.Identity().
    Returns (backbone_module, feature_dim).
    """
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        feature_dim = net.fc.in_features  # 2048
        net.fc = nn.Identity()
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        feature_dim = net.classifier[1].in_features  # 1280
        net.classifier = nn.Identity()
    elif name == "SimpleCNN":
        net = SimpleCNN()
        feature_dim = net.feature_dim
    else:
        raise ValueError(f"Unknown backbone name: {name}")

    return net, feature_dim


class SimpleCNN(nn.Module):
    """5-stage custom Conv-BN-ReLU-MaxPool CNN for scratch baseline."""
    def __init__(self):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feature_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


def set_backbone_unfreezing(backbone: nn.Module, backbone_name: str, stage: int):
    """
    Rigorous unfreezing and BatchNorm control:
    - Stage 1 (Head-only): Entire backbone frozen (requires_grad = False).
      Crucially, all BatchNorm2d layers are set to eval() mode to freeze running stats.
    - Stage 2 (Fine-tuning):
      For ResNet-50: Unfreezes 'layer4' specifically (the final 3 Bottleneck blocks).
      For EfficientNet-B0: Unfreezes 'features[7]' and 'features[8]' specifically.
    """
    real_backbone = backbone.module if isinstance(backbone, nn.DataParallel) else backbone

    if stage == 1:
        for p in real_backbone.parameters():
            p.requires_grad = False
        # Freeze BatchNorm running statistics during Stage 1
        for m in real_backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
    elif stage == 2:
        # First ensure all are frozen
        for p in real_backbone.parameters():
            p.requires_grad = False
            
        if backbone_name == "resnet50":
            # Unfreeze layer4 specifically
            if hasattr(real_backbone, "layer4"):
                for p in real_backbone.layer4.parameters():
                    p.requires_grad = True
        elif backbone_name == "efficientnet_b0":
            # In torchvision efficientnet_b0, features is an nn.Sequential of 9 stages (0..8)
            # features[7] is MBConv, features[8] is final 1x1 Conv2d+BN+SiLU
            if hasattr(real_backbone, "features"):
                for idx in [7, 8]:
                    if idx < len(real_backbone.features):
                        for p in real_backbone.features[idx].parameters():
                            p.requires_grad = True
        elif backbone_name == "SimpleCNN":
            # For scratch SimpleCNN, unfreeze all layers in stage 2
            for p in real_backbone.parameters():
                p.requires_grad = True
    else:
        raise ValueError(f"Invalid stage {stage}")


# --- Projection & Heads ---

def make_projection_head(feature_dim: int, hidden_dim: int = 512, out_dim: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


def make_predictor(dim: int = 128, hidden_dim: int = 512) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, dim),
    )


@torch.no_grad()
def momentum_update(online_module: nn.Module, target_module: nn.Module, m: float):
    for p_o, p_t in zip(online_module.parameters(), target_module.parameters()):
        p_t.data = p_t.data * m + p_o.data * (1.0 - m)


class SimCLRModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, proj_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.projector = make_projection_head(feature_dim, hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.projector(h)


class MoCoModel(nn.Module):
    def __init__(self, backbone_fn, feature_dim: int, proj_dim: int = 128, hidden_dim: int = 512,
                 queue_size: int = 2048, momentum: float = 0.999):
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
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_size:
            ptr = 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(self.proj_q(self.backbone_q(x_q)), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.proj_k(self.backbone_k(x_k)), dim=1)
        return q, k


class BYOLModel(nn.Module):
    def __init__(self, backbone_fn, feature_dim: int, proj_dim: int = 128, hidden_dim: int = 512, momentum: float = 0.996):
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

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        q1 = self.predictor(self.proj_online(self.backbone_online(x1)))
        q2 = self.predictor(self.proj_online(self.backbone_online(x2)))
        with torch.no_grad():
            t1 = self.proj_target(self.backbone_target(x1))
            t2 = self.proj_target(self.backbone_target(x2))
        return q1, q2, t1.detach(), t2.detach()


class DINOModel(nn.Module):
    """
    CNN-based DINO self-distillation model with multi-crop support,
    teacher EMA, centering buffer, and sharpening.
    """
    def __init__(self, backbone_fn, feature_dim: int, proj_dim: int = 128, hidden_dim: int = 512,
                 momentum: float = 0.996, center_momentum: float = 0.9,
                 student_temp: float = 0.1, teacher_temp: float = 0.04):
        super().__init__()
        self.student_backbone = backbone_fn()
        self.student_proj = make_projection_head(feature_dim, hidden_dim, proj_dim)

        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_proj = copy.deepcopy(self.student_proj)
        for p in list(self.teacher_backbone.parameters()) + list(self.teacher_proj.parameters()):
            p.requires_grad = False

        self.momentum = momentum
        self.center_momentum = center_momentum
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.register_buffer("center", torch.zeros(1, proj_dim))

    @torch.no_grad()
    def update_teacher(self):
        momentum_update(self.student_backbone, self.teacher_backbone, self.momentum)
        momentum_update(self.student_proj, self.teacher_proj, self.momentum)

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward_multicrop(self, views: List[torch.Tensor], n_global: int = 2):
        student_outputs = [self.student_proj(self.student_backbone(v)) for v in views]
        with torch.no_grad():
            teacher_outputs = [self.teacher_proj(self.teacher_backbone(v)) for v in views[:n_global]]
        return student_outputs, teacher_outputs


# --- Downstream Classifier ---

class ClassifierHead(nn.Module):
    """Adapter (feature_dim -> 128) + classification layer (128 -> num_classes)."""
    def __init__(self, feature_dim: int, num_classes: int = 2, proj_dim: int = 128):
        super().__init__()
        self.adapter = nn.Linear(feature_dim, proj_dim)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.relu(self.adapter(feats)))


class FullClassifier(nn.Module):
    """Integrated full classification model: Backbone + ClassifierHead."""
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.head = ClassifierHead(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

