"""
PneuSSLNet Explainable AI (XAI) Suite:
- Grad-CAM on target convolutional layers
- Integrated Gradients
- Occlusion Sensitivity Analysis
- Parameter Randomization Sanity Checks & Stability Checks
"""
import copy
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for CNN backbones.
    Captures forward activations and backward gradients of the target conv layer.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        
        logits = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        
        # Global average pool gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Upsample to input tensor resolution
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Min-max normalize
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam


def compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = 1,
    steps: int = 50,
    baseline: Optional[torch.Tensor] = None
) -> np.ndarray:
    """
    Computes Integrated Gradients path integral from zero baseline to input tensor.
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
        
    diff = input_tensor - baseline
    accumulated_grads = torch.zeros_like(input_tensor)
    
    for alpha in np.linspace(0.0, 1.0, steps):
        interpolated = baseline + alpha * diff
        interpolated.requires_grad = True
        logits = model(interpolated)
        score = logits[:, target_class].sum()
        model.zero_grad()
        score.backward()
        accumulated_grads += interpolated.grad.detach()
        
    avg_grads = accumulated_grads / steps
    attribution = (diff * avg_grads).sum(dim=1).squeeze().cpu().numpy()
    
    # Abs and normalize
    attribution = np.maximum(attribution, 0)
    a_min, a_max = attribution.min(), attribution.max()
    if a_max > a_min:
        attribution = (attribution - a_min) / (a_max - a_min)
    else:
        attribution = np.zeros_like(attribution)
        
    return attribution


def compute_occlusion_sensitivity(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = 1,
    patch_size: int = 16,
    stride: int = 8
) -> np.ndarray:
    """
    Computes Occlusion Sensitivity map by systematically sliding a grey occlusion patch.
    """
    model.eval()
    with torch.no_grad():
        orig_prob = F.softmax(model(input_tensor), dim=1)[0, target_class].item()
        
    _, _, H, W = input_tensor.shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = input_tensor.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = 0.0  # Grey/zero mask
            with torch.no_grad():
                prob_occ = F.softmax(model(occluded), dim=1)[0, target_class].item()
            # Sensitivity = drop in probability
            diff = max(0.0, orig_prob - prob_occ)
            heatmap[y:y+patch_size, x:x+patch_size] += diff
            counts[y:y+patch_size, x:x+patch_size] += 1.0
            
    heatmap = heatmap / np.maximum(counts, 1.0)
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min)
        
    return heatmap


def run_xai_sanity_check(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor
) -> Dict[str, Any]:
    """
    Adebayo et al. Model Parameter Randomization Test.
    Verifies that Grad-CAM attributions depend on learned parameters.
    """
    # 1. Trained model CAM
    gradcam = GradCAM(model, target_layer)
    orig_cam = gradcam.generate(input_tensor)
    
    # 2. Randomize weights of target layer
    randomized_model = copy.deepcopy(model)
    # Find corresponding layer in deepcopy
    for p in randomized_model.parameters():
        if p.requires_grad:
            nn.init.normal_(p)
            
    # Random model CAM
    rand_gradcam = GradCAM(randomized_model, getattr(randomized_model.backbone, "layer4", None) or list(randomized_model.backbone.children())[-1])
    rand_cam = rand_gradcam.generate(input_tensor)
    
    # Compute similarity (correlation / cosine)
    orig_flat = orig_cam.flatten()
    rand_flat = rand_cam.flatten()
    sim = np.corrcoef(orig_flat, rand_flat)[0, 1]
    
    return {
        "sanity_check_passed": bool(sim < 0.5),
        "correlation_with_randomized": float(sim)
    }

