# PneuSSLNet: Experimental Revision Protocol & Specification

**Document Version:** 1.0  
**Date:** 2026-10-17  
**Status:** FROZEN  
**Target Runtime:** Kaggle Notebooks (GPU Accelerator: NVIDIA Tesla P100 / T4)  
**Output Base Path:** `/kaggle/working/pneusslnet_runs/<RUN_ID>/`  

---

## 1. Protocol Overview & Scientific Principles

This document defines the frozen experimental protocol for the comprehensive revision of the **PneuSSLNet** study. All experiments, model evaluations, ablation sweeps, and explainability analyses must strictly adhere to the rules specified herein.

### Core Working Rules:
1. **Predeclared Epoch Rule:** The primary benchmark comparisons are evaluated strictly at the final predeclared epoch (Epoch 30 total: 15 Stage 1 + 15 Stage 2). Training curves and intermediate checkpoints are diagnostic only; no selective epoch reporting ("best epoch on test") is allowed.
2. **Default Decision Rule:** The primary decision threshold is strictly fixed at `0.5000` with Test-Time Augmentation (TTA) disabled (`TTA=False`). Secondary analyses involving validation-tuned thresholds (Youden's J) or TTA are evaluated separately and reported in supplementary materials with explicit notation.
3. **Strict Information Firewall:** No internal test or external validation images, labels, or predictions may be seen by training functions, optimizer loops, early stopping, threshold tuning, or hyperparameter selection.
4. **Reproducible Paired Seeds:** Model comparisons are evaluated across at least 3 distinct seeds: `SEEDS = [17, 42, 73]` (with seed 42 as the primary reference). All seeds use the exact same frozen data split manifest to ensure paired statistical testing.
5. **No Data Fabrication or Stale Reuse:** All reported numbers in Tables 1–7 and Figures 1–8 must be generated directly from raw prediction CSVs and JSON logs produced by verified runs.

---

## 2. Dataset Specifications & Data Governance

### 2.1 Main Development Cohort (Chest X-Ray Pneumonia)
- **Source:** Paul Mooney / Kermany et al. Pediatric CXR dataset (`paultimothymooney/chest-xray-pneumonia`).
- **Nominal Size:** 5,856 JPEG radiographs (`NORMAL`: 1,583; `PNEUMONIA`: 4,273).
- **Partitioning Protocol:**
  - All verified, non-corrupted images across original `train`, `val`, and `test` directories are pooled.
  - An exact SHA-256 hash is computed for every image to identify duplicate files. Perceptual image hashing is conducted to flag near-duplicates.
  - A single stratified 70% Train / 15% Validation / 15% Internal Test split is generated using a fixed split seed (`SPLIT_SEED = 2026`).
  - Nominal counts: ~4,099 Train, ~878 Validation, ~879 Internal Test.
  - Persisted in `manifests/split_manifest.csv` with fields: `image_path, class_name, class_idx, original_folder, new_split, sha256`.
  - The split manifest is hashed (`split_manifest_sha256`) and recorded in the run metadata.

### 2.2 External Validation Cohort (RSNA Pneumonia Detection Challenge)
- **Source:** RSNA Pneumonia Detection Challenge (`stage_2_train_labels.csv`, `stage_2_detailed_class_info.csv`, `stage_2_train_images/`).
- **Format:** DICOM radiographs (.dcm).
- **Label Aggregation:**
  - Repeated bounding box rows for each `patientId` are collapsed into exactly one image-level record.
  - `Target = 1` if one or more valid pneumonia-suspicious opacity boxes exist; `Target = 0` otherwise.
  - Joined one-to-one with detailed class info: `Normal`, `No Lung Opacity / Not Normal`, `Lung Opacity`.
- **Predeclared Evaluation Cohorts:**
  1. **Primary Challenge Cohort:** `Target = 1` (Pneumonia-suspicious opacity) vs. `Target = 0` (No definitive pneumonia opacity). Sample size: ~26,684 radiographs.
  2. **Strict Class-Matched Subgroup:** `Lung Opacity` (Class 1) vs. `Normal` (Class 0), excluding `No Lung Opacity / Not Normal`. Sample size: ~14,862 radiographs.
- **DICOM Preprocessing Pipeline:**
  - Read with `pydicom`. Apply Rescale Slope and Intercept (`pixel_array * slope + intercept`).
  - Handle `PhotometricInterpretation`: if `MONOCHROME1`, invert values (`max - pixel_array`) so high density (bone/opacity) is bright.
  - Apply VOI LUT (Window Width / Window Center) if present; otherwise min-max normalize to `[0, 255]`.
  - Convert 1-channel grayscale to 3-channel RGB. Resize to `224x224` using bilinear interpolation.
  - Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

---

## 3. Architecture & Augmentation Specification

### 3.1 Backbones & Weight Initialization
1. **ResNet-50:**
   - Pretrained source: `torchvision.models.ResNet50_Weights.IMAGENET1K_V2`.
   - Feature dimension after Global Average Pooling (GAP): `2048`.
2. **EfficientNet-B0:**
   - Pretrained source: `torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1`.
   - Feature dimension after GAP: `1280`.
3. **Scratch Controls:**
   - `SimpleCNN`: 5-stage custom Conv-BN-ReLU-MaxPool architecture (256-dim feature vector).
   - `ResNet-50 (Scratch)`: `weights=None`, random initialization.
   - `EfficientNet-B0 (Scratch)`: `weights=None`, random initialization.
4. **Supervised Transfer Controls:**
   - `ResNet-50 (ImageNet)`: `IMAGENET1K_V2` initialized, supervised on CXR training split.
   - `EfficientNet-B0 (ImageNet)`: `IMAGENET1K_V1` initialized, supervised on CXR training split.

### 3.2 Projection & Classification Heads
- **SSL Projection Head (SimCLR, MoCo, BYOL, DINO):**
  - MLP: `Linear(feature_dim, 512) -> ReLU -> Linear(512, 128)`.
  - Normalization: L2 normalization along feature dimension where required by loss.
- **BYOL Predictor Head (Online branch only):**
  - MLP: `Linear(128, 512) -> ReLU -> Linear(512, 128)`.
- **Downstream Classification Head (UCL Block):**
  - Bottleneck Adapter: `Linear(feature_dim, 128) -> ReLU -> Linear(128, 2)`.

### 3.3 Backbone Freezing & Staged Fine-Tuning Rules
- **Stage 1 (Frozen Feature Extractor, Epochs 1–15):**
  - Backbone parameters: `requires_grad = False`.
  - BatchNorm behavior: All `nn.BatchNorm2d` layers inside the backbone are explicitly forced into `.eval()` mode so running mean and running variance are frozen.
  - Head parameters: `requires_grad = True`.
  - Optimizer: Adam, Learning Rate `1e-3`, Weight Decay `1e-4`.
- **Stage 2 (Partial Unfreezing, Epochs 16–30):**
  - **ResNet-50 Unfreezing Target:** `layer4` (the final residual stage comprising 3 Bottleneck blocks) plus `head`. Stages `conv1`, `bn1`, `layer1`, `layer2`, `layer3` remain frozen.
  - **EfficientNet-B0 Unfreezing Target:** `features[7]` and `features[8]` (the final MBConv stages and 1x1 conv) plus `head`. Stages `features[0]` through `features[6]` remain frozen.
  - Optimizer: Adam, Learning Rate `1e-4`, Cosine Annealing learning rate schedule (`T_max = 15`).

### 3.4 Data Augmentation Pipelines (NISE Block)
- **Input Image Size:** `224x224` pixels.
- **SimCLR Pipeline:** Symmetric strong views. RandomResizedCrop(224, scale=(0.8, 1.0)), RandomHorizontalFlip(p=0.5), RandomRotation(degrees=10), ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05).
- **MoCo Pipeline:** RandomResizedCrop(224, scale=(0.8, 1.0)), GaussianBlur(kernel_size=3, p=0.1), RandomGrayscale(p=0.2), RandomHorizontalFlip(p=0.5), ColorJitter(0.1, 0.1, 0.1, 0.05).
- **BYOL Pipeline:** Asymmetric views. View 1 (strong): RandomResizedCrop(224, scale=(0.5, 1.0)), RandomHorizontalFlip(p=0.5), ColorJitter(0.4, 0.4, 0.4, 0.1), GaussianBlur(kernel_size=3, p=0.5). View 2 (light): Resize(224), RandomHorizontalFlip(p=0.5), ColorJitter(0.1, 0.1, 0.0, 0.0).
- **DINO Multi-Crop Pipeline:**
  - 2 Global Crops: `224x224`, scale=(0.5, 1.0).
  - 2 Local Crops: `96x96`, scale=(0.05, 0.40).
  - Student processes all 4 crops; Teacher processes only the 2 global crops.
  - Teacher centering momentum = `0.9`, Teacher sharpening temperature = `0.04`, Student temperature = `0.1`.
- **Evaluation / Test Transform:** Deterministic `Resize(224, 224)` followed by ToTensor and ImageNet normalization. Zero stochastic operations.

---

## 4. Experimental Matrix & Execution Plan

### Experiment A: Main Controlled SSL Benchmark
- **Configurations:** 4 SSL Methods (SimCLR, MoCo v2, BYOL, DINO) × 2 Backbones (ResNet-50, EfficientNet-B0) = 8 SSL configurations.
- **Controls:** 5 Supervised baselines (CustomCNN Scratch, ResNet50 Scratch, EfficientNet-B0 Scratch, ResNet50 ImageNet, EfficientNet-B0 ImageNet).
- **Total Primary Configurations:** 13 models.
- **Seeds:** 3 seeds (`17, 42, 73`) = 39 complete training trajectories.
- **Training Budget:**
  - SSL Pretraining: 30 epochs (Batch size: 16 per GPU, Adam lr=3e-4).
  - Downstream Fine-tuning: Stage 1 (15 epochs, lr=1e-3) + Stage 2 (15 epochs, lr=1e-4, Cosine schedule).
- **Metrics Logged per Epoch:** Train Loss, Val Loss, Train Acc, Val Acc, Val Macro-F1, Val AUROC.
- **Reporting:** Mean ± SD across seeds evaluated at Epoch 30 on locked internal test set.

### Experiment B: Label-Efficiency Analysis
- **Fractions of Training Split:** `[0.10, 0.25, 0.50, 0.75, 1.00]`.
- **Protocol:**
  - Unlabeled pretraining performed once on 100% of the training split.
  - Stratified label subsets sampled deterministically using seed control; identical subsets used across all competing methods.
  - Competing Models: SimCLR, MoCo, BYOL, DINO on ResNet-50 + Supervised Controls (ResNet-50 Scratch, ResNet-50 ImageNet).
  - Fine-tuning executed for 30 epochs on the subsampled labeled data.
  - Evaluated on the locked internal test set.
- **Reporting:** Accuracy vs. Label Fraction line chart with seed variability bands.

### Experiment C: Isolated Component Ablation
- **Backbone:** ResNet-50. Repeated across 3 seeds.
- **5 Predeclared Conditions:**
  1. `A1`: Random initialization (Scratch) + No downstream augmentation.
  2. `A2`: ImageNet-pretrained supervised init (No in-domain SSL) + No downstream augmentation.
  3. `A3`: ImageNet init + DINO in-domain SSL + No downstream augmentation.
  4. `A4`: ImageNet init + DINO in-domain SSL + Light downstream augmentation (Resize + Flip).
  5. `A5`: ImageNet init + DINO in-domain SSL + Strong downstream augmentation (NISE full pipeline).
- **Reporting:** Table 5 reporting Mean ± SD across seeds on locked internal test set.

### Experiment D: External RSNA Generalization
- Evaluated on the winning model configuration (selected solely via internal validation) plus key reference models.
- Evaluated on both:
  1. Challenge Cohort (`Target=1` vs `0`).
  2. Strict Matched Subgroup (`Lung Opacity` vs `Normal`).
- Metrics: AUROC, AUPRC, Sensitivity, Specificity, Macro-F1, Accuracy, Brier score, with 95% stratified bootstrap confidence intervals (2,000 resamples).

### Experiment E: Statistical & Calibration Analysis
- **Metrics Suite:** Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro-F1, Pneumonia Sensitivity, Pneumonia Specificity, AUROC, AUPRC, Matthews Correlation Coefficient (MCC), Brier Score, Expected Calibration Error (ECE, 10 equal-frequency bins).
- **Uncertainty & Significance:**
  - Seed variance: Reported as standard deviation across runs.
  - Sample confidence intervals: Stratified 2,000-sample bootstrap for test metrics.
  - Paired Hypothesis Testing: Paired bootstrap test and McNemar's test comparing winning SSL model against ImageNet-supervised baseline, with Holm-Bonferroni correction.

### Experiment F: Explainable AI (XAI) Suite
- **Primary Method:** Grad-CAM on `layer4` (ResNet-50) and `features[8]` (EfficientNet-B0).
- **Secondary Methods:** Integrated Gradients (50 steps, zero black baseline) and Occlusion Sensitivity (patch size 16x16, stride 8).
- **Sampling Protocol:** Deterministic stratified sample of 16 test cases covering True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
- **Scientific Sanity Checks:**
  1. Parameter Randomization Test: Cascading layer-by-layer randomization of model weights to verify attribution maps collapse into noise.
  2. Perturbation Stability Test: Measure cosine similarity of attribution maps under small Gaussian noise perturbations ($\sigma = 0.02$).
  3. Deletion / Insertion Curve: Occlude top-attributed regions in 5% increments and plot probability decay.
  4. RSNA Pointing Game: For opacity-positive cases with radiologist bounding boxes, measure the proportion of max attribution points falling inside annotated pathology boxes.

---

## 5. Checkpoint & Artifact Architecture

### 5.1 Storage Directory Structure
```text
/kaggle/working/pneusslnet_runs/<RUN_ID>/
├── configs/
│   ├── experiment_config.yaml
│   └── hashes.json
├── manifests/
│   ├── split_manifest.csv
│   └── rsna_cohort_manifest.csv
├── audit/
│   ├── environment_info.json
│   ├── duplicate_audit.json
│   └── test_firewall_verification.log
├── registry/
│   ├── experiment_registry.csv
│   └── checkpoint_index.csv
├── checkpoints/
│   ├── ssl_pretrain/<method>/<backbone>/seed-<seed>/
│   ├── downstream/<method>/<backbone>/seed-<seed>/
│   ├── baselines/<name>/seed-<seed>/
│   ├── label_efficiency/<method>/fraction-<frac>/seed-<seed>/
│   └── ablation/<exp_id>/seed-<seed>/
├── histories/
├── predictions/
│   ├── internal_test/
│   └── rsna_external/
├── metrics/
├── figures/
│   ├── main/
│   └── supplementary/
├── explainability/
├── tables/
└── webapp_bundle/
    └── <model_version>/
```

### 5.2 Resumable Checkpoint Contract
Every `last_resume.pt` checkpoint contains:
- `model_state_dict`: Full model weights.
- `optimizer_state_dict`: Full Adam optimizer states.
- `scheduler_state_dict`: Cosine scheduler states.
- `epoch`: Current completed epoch index.
- `global_step`: Total optimizer steps.
- `rng_states`: Dict of Python `random.getstate()`, NumPy `np.random.get_state()`, and PyTorch `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`.
- `metadata`: Dict containing `RUN_ID`, `experiment_id`, `method`, `backbone`, `seed`, `stage`, `split_manifest_sha256`, `config_sha256`, `code_version`, `timestamp`.

### 5.3 Web-App Deployment Bundle (`webapp_bundle/v1.0.0/`)
- `model_state_dict.pt`: Stripped weights (no optimizer, no grad).
- `model_config.json`: Architecture parameters, constructor arguments, version.
- `preprocessing.json`: Input resolution (224), mean, std, grayscale/RGB conversion.
- `label_map.json`: `{"0": "NORMAL", "1": "PNEUMONIA"}`.
- `decision_policy.json`: Primary threshold (0.5000), positive class (1).
- `inference.py`: Zero-training standalone inference module.
- `requirements-inference.txt`: Minimal dependencies (`torch`, `torchvision`, `pillow`, `numpy`).
- `model_card.md`: Clinical scope, intended use, limitations, performance bounds.
- `SHA256SUMS.txt`: Cryptographic integrity checksums.

---
**Protocol Approved:** Frozen for implementation in `PneuSSLNet_Q1_Revised_Kaggle.ipynb`.

