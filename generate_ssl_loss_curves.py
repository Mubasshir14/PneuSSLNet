"""
PneuSSLNet Empirical SSL Pretraining Loss Visualizer & Exporter
==============================================================
Plots the exact 30-epoch empirical loss trajectories for all 4 SSL methods
(SimCLR, MoCo, BYOL, DINO) across both ResNet-50 and EfficientNet-B0.

Outputs:
  1. Main Publication Figure: outputs/figures/Figure_SSL_Pretraining_Loss_Curves_Main.png (.pdf)
  2. Supplementary Figure S1: outputs/supplementary/figures/Figure_S1_Pretraining_Loss_Trajectories.png (.pdf)
  3. Exact Numerical Table:   outputs/tables/table_ssl_pretraining_per_epoch_losses.csv
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Stylings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['grid.linestyle'] = '--'

# Load empirical loss history from checkpoint
ckpt_path = "outputs/results_checkpoint.pt"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

epochs = np.arange(1, 31)

# Extract loss arrays
r50_losses = {
    "SimCLR": data["resnet50"]["SimCLR"]["ssl_loss"],
    "MoCo": data["resnet50"]["MoCo"]["ssl_loss"],
    "BYOL": data["resnet50"]["BYOL"]["ssl_loss"],
    "DINO": data["resnet50"]["DINO"]["ssl_loss"]
}

eff_losses = {
    "SimCLR": data["efficientnet_b0"]["SimCLR"]["ssl_loss"],
    "MoCo": data["efficientnet_b0"]["MoCo"]["ssl_loss"],
    "BYOL": data["efficientnet_b0"]["BYOL"]["ssl_loss"],
    "DINO": data["efficientnet_b0"]["DINO"]["ssl_loss"]
}

# ============================================================
# 1. EXPORT EXACT PER-EPOCH NUMERICAL CSV TABLE
# ============================================================
os.makedirs("outputs/tables", exist_ok=True)
csv_rows = []
for ep_idx in range(30):
    ep = ep_idx + 1
    csv_rows.append({
        "Epoch": ep,
        "SimCLR (ResNet-50)": round(float(r50_losses["SimCLR"][ep_idx]), 4),
        "SimCLR (EfficientNet-B0)": round(float(eff_losses["SimCLR"][ep_idx]), 4),
        "MoCo (ResNet-50)": round(float(r50_losses["MoCo"][ep_idx]), 4),
        "MoCo (EfficientNet-B0)": round(float(eff_losses["MoCo"][ep_idx]), 4),
        "BYOL (ResNet-50)": round(float(r50_losses["BYOL"][ep_idx]), 4),
        "BYOL (EfficientNet-B0)": round(float(eff_losses["BYOL"][ep_idx]), 4),
        "DINO (ResNet-50)": round(float(r50_losses["DINO"][ep_idx]), 4),
        "DINO (EfficientNet-B0)": round(float(eff_losses["DINO"][ep_idx]), 4)
    })

df_loss = pd.DataFrame(csv_rows)
out_csv = "outputs/tables/table_ssl_pretraining_per_epoch_losses.csv"
df_loss.to_csv(out_csv, index=False)
print(f"[SAVED] Exact per-epoch numerical loss table: {out_csv}")

# ============================================================
# 2. GENERATE 4-PANEL PUBLICATION LOSS CURVES (300 DPI)
# ============================================================
def plot_ssl_loss_figure(out_png, out_pdf, is_supplementary=False):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    plt.subplots_adjust(wspace=0.18, hspace=0.25)

    configs = [
        ("SimCLR", "NT-Xent Contrastive Loss", axes[0, 0], "(A)", '#1f77b4', '#17becf'),
        ("MoCo", "Memory-Queue InfoNCE Loss", axes[0, 1], "(B)", '#2ca02c', '#8c564b'),
        ("BYOL", "Negative-Free Normalized Cosine Loss", axes[1, 0], "(C)", '#ff7f0e', '#d62728'),
        ("DINO", "Multi-Crop Self-Distillation Cross-Entropy", axes[1, 1], "(D)", '#9467bd', '#e377c2')
    ]

    for method, loss_title, ax, panel_tag, col_r50, col_eff in configs:
        l_r50 = r50_losses[method]
        l_eff = eff_losses[method]

        # ResNet-50 curve
        ax.plot(epochs, l_r50, marker='o', markersize=4.5, linewidth=2.0, color=col_r50,
                label=f'ResNet-50 (Final: {l_r50[-1]:.4f})')
        # EfficientNet-B0 curve
        ax.plot(epochs, l_eff, marker='s', markersize=4.5, linewidth=2.0, color=col_eff, linestyle='--',
                label=f'EfficientNet-B0 (Final: {l_eff[-1]:.4f})')

        ax.set_title(f"{panel_tag} {method}: {loss_title}", fontsize=11.5, fontweight='bold', pad=8)
        ax.set_xlabel("SSL Pretraining Epoch", fontsize=10.5)
        ax.set_ylabel(f"{method} Loss", fontsize=10.5)
        ax.set_xlim(0.5, 30.5)
        ax.set_xticks(range(5, 31, 5))
        ax.grid(True)
        ax.legend(fontsize=9.5, loc='upper right', framealpha=0.92)

        # Highlight minimum convergence
        min_r50 = min(l_r50)
        min_eff = min(l_eff)
        ax.axhline(y=min_r50, color=col_r50, linestyle=':', alpha=0.4)
        ax.axhline(y=min_eff, color=col_eff, linestyle=':', alpha=0.4)

    title_prefix = "Figure S1: Extended " if is_supplementary else "Figure: Empirical "
    fig.suptitle(f"{title_prefix}Self-Supervised Pretraining Loss Convergence Trajectories across 30 Epochs",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Figure saved: {out_png}")

# Generate Main Figure
os.makedirs("outputs/figures", exist_ok=True)
plot_ssl_loss_figure(
    out_png="outputs/figures/Figure_SSL_Pretraining_Loss_Curves_Main.png",
    out_pdf="outputs/figures/Figure_SSL_Pretraining_Loss_Curves_Main.pdf",
    is_supplementary=False
)

# Generate Supplementary Figure S1
os.makedirs("outputs/supplementary/figures", exist_ok=True)
plot_ssl_loss_figure(
    out_png="outputs/supplementary/figures/Figure_S1_Pretraining_Loss_Trajectories.png",
    out_pdf="outputs/supplementary/figures/Figure_S1_Pretraining_Loss_Trajectories.pdf",
    is_supplementary=True
)

print("\n" + "=" * 70)
print("ALL EMPIRICAL SSL LOSS CURVES & TABLES SUCCESSFULLY GENERATED!")
print("=" * 70)

