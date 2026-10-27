# PneuSSLNet Kaggle Execution & Run Guide

**Document Version:** 1.0  
**Target Platform:** Kaggle Notebooks  
**Hardware Accelerator:** GPU (NVIDIA Tesla P100 or T4 x2)  
**Main Revision Notebook:** [`PneuSSLNet_Q1_Revised_Kaggle.ipynb`](file:///e:/Pneumonia/New%20Code/PneuSSLNet_Q1_Revised_Kaggle.ipynb)  

---

## 1. Quick Start: Setting Up the Kaggle Environment

1. **Upload Notebook to Kaggle:**
   - Go to [Kaggle Kernels / Notebooks](https://www.kaggle.com/code).
   - Click **New Notebook** -> **File** -> **Upload Notebook**.
   - Select and upload `PneuSSLNet_Q1_Revised_Kaggle.ipynb`.

2. **Enable GPU Accelerator & Internet:**
   - In the notebook right-hand settings panel:
     - **Accelerator:** Select **GPU P100** (or **GPU T4 x2**).
     - **Internet:** Toggle to **On** (required to install `pydicom` for external RSNA validation).
     - **Environment:** Use Kaggle's default PyTorch image.

3. **Attach Required Datasets via "Add Input":**
   - Click **+ Add Input** (top right):
     1. Search for `paultimothymooney/chest-xray-pneumonia` and click **Add**.
     2. Search for the RSNA competition dataset: `rsna-pneumonia-detection-challenge` and click **Add** (ensure you have accepted the competition rules on Kaggle).
   - The notebook dynamically scans `/kaggle/input` and locates both datasets automatically.

---

## 2. Choosing Execution Mode (`RUN_MODE`)

In **Phase 2 (Cell 2)** of the notebook:

### Mode A: Smoke Test (`RUN_MODE = "smoke_test"`)
- **Purpose:** Verifies that all imports, data loaders, split creation, SSL losses, backbone freezing, test firewall, Grad-CAM, and web-app bundle exports execute without error.
- **Runtime:** ~5 to 8 minutes on GPU.
- **Budget:** 1 SSL pretrain epoch, 1 stage-1 epoch, 1 stage-2 epoch, single seed (`seed=42`).
- **Use Case:** Run this first to verify that both mounted datasets are detected and that the notebook runs end-to-end without crashing.

### Mode B: Full Journal Revision (`RUN_MODE = "full_experiment"`)
- **Purpose:** Executes the full experimental suite required for the journal revision.
- **Runtime:** ~6 to 8 hours per complete run on GPU P100.
- **Budget:**
  - 30 SSL pretraining epochs per method × backbone.
  - 15 frozen Stage-1 epochs + 15 fine-tuning Stage-2 epochs.
  - Multi-seed repetition: Seeds `[17, 42, 73]`.
  - 5 Supervised controls (Scratch CNN, Scratch ResNet-50, Scratch EfficientNet-B0, ImageNet ResNet-50, ImageNet EfficientNet-B0).
  - Multi-seed label-efficiency sweep (10%, 25%, 50%, 75%, 100%).
  - 5-condition ablation study.
  - Locked internal-test evaluation + RSNA dual-cohort evaluation.

---

## 3. Kaggle Runtime Continuity & Resuming Runs

Kaggle sessions have a 9-hour limit. If a session times out or disconnects:
1. **Atomic Rolling Checkpoints:** The notebook automatically saves `last_resume.pt` after every completed epoch and optimizer step interval under:
   `/kaggle/working/pneusslnet_runs/<RUN_ID>/checkpoints/`
2. **Resuming a Session:**
   - To resume from a previous Kaggle session, attach the saved output directory of the prior run as a Kaggle Input via **Add Input -> Your Work**.
   - Set the previous `RUN_ID` in `CONFIG["RUN_ID"]`.
   - The `CheckpointManager` verifies that the `split_hash` and `config_hash` match, then seamlessly resumes training from the last saved epoch without resetting optimizer, scheduler, or random states.
3. **Persisting Artifacts:**
   - Always click **Save Version** -> **Save & Run All** to commit the notebook and make the generated artifacts permanently downloadable.

---

## 4. Output Artifacts Manifest

All outputs are saved cleanly under `/kaggle/working/pneusslnet_runs/<RUN_ID>/`:

| Directory | Contents |
|:---|:---|
| `configs/` | `experiment_config.json`, SHA-256 configuration hashes |
| `manifests/` | `split_manifest.csv` (70/15/15 hashed split), `rsna_cohort_manifest.csv` |
| `registry/` | `checkpoint_index.csv`, append-only experiment status log |
| `checkpoints/` | Stage-final models, validation-best snapshots, rolling resumes |
| `predictions/` | Ground-truth labels and model probabilities for internal test and RSNA cohorts |
| `metrics/` | Seed-level and aggregated JSON/CSV performance tables |
| `figures/` | 300-DPI publication PNG and vector PDF figures (Figures 1–8) |
| `tables/` | Tables 1–7 formatted in CSV and publication-ready LaTeX |
| `explainability/` | Grad-CAM, Integrated Gradients, and Occlusion Sensitivity heatmaps & sanity reports |
| `webapp_bundle/` | Self-contained inference package for future web application deployment |

---

## 5. Standalone Web-App Bundle Verification

After the Kaggle run finishes, download the `webapp_bundle/` directory. You can test inference in an isolated Python environment using:

```bash
cd webapp_bundle/v1.0.0
python -c "
from inference import PneuSSLNetPredictor
predictor = PneuSSLNetPredictor('./')
res = predictor.predict('path/to/test_xray.jpeg')
print('Prediction Result:', res)
"
```
The web-app bundle requires no training code or dataset dependencies.

