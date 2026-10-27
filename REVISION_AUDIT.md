# Stage A Audit: PneuSSLNet Research Revision Audit Report

**Date:** 2026-10-17  
**Target Runtime:** Kaggle Notebooks (GPU Accelerator)  
**Primary Codebase:** `PNEUMONIA SSL SL FINAL(20261017-164315).ipynb` / `PNEUMONIA SSL SL FINAL.ipynb`  
**Manuscripts Audited:** `Pneumonia.pdf` (21 pages / 23 pages total manuscript), `pneumonia paper update guidelines.pdf`  
**Master Instructions:** `PneuSSLNet_Q1_Experimental_Revision_Master_Instructions_KAGGLE_XAI.md`  

---

## 1. Executive Summary & Audit Overview

This audit document fulfills **Stage A (Audit before coding)** of the PneuSSLNet research revision pipeline. A rigorous, line-by-line inspection of the original Jupyter notebook (`PNEUMONIA SSL SL FINAL.ipynb`), the existing submission manuscript (`Pneumonia_Total_Page_23.pdf`), and the editorial revision guidelines (`pneumonia paper update guidelines.pdf`) was performed.

Multiple critical scientific, statistical, architectural, and procedural flaws were uncovered in the existing implementation. These include:
1. **Severe Test-Set Leakage:** During downstream classifier training, the internal test set was evaluated every 10 epochs. Furthermore, Test-Time Augmentation (TTA) and optimal decision thresholds were computed across test sets, and supervised baselines, ablations, and label-efficiency sweeps directly evaluated test data inside training loops.
2. **Stale Checkpoint & Results Caching:** Historical checkpoint dictionaries (`results_checkpoint.pt`, `baseline_results.pt`, `label_fraction_results.pt`, `ablation_results.pt`) were copied into outputs and used to skip actual experiment execution without verifying split hashes, code commit, or random seed integrity.
3. **Split Protocol Discrepancies & Flawed Validation Split:** While the original manuscript claimed a tiny 16-image validation set (8 Normal, 8 Pneumonia), the existing notebook pooled all images and generated a stratified 70/15/15 split (~4,099 train, 878 val, 879 test). However, this split was never hashed, never locked against duplicate images, and was coupled to in-loop evaluation.
4. **Contradictory Headline Numbers & Single-Seed Variance:** Three conflicting "best" accuracy figures (97.38%, 98.07%, 98.18%) appear in the manuscript across different tables without variance or standard deviation, leaving small (<1%) inter-method differences scientifically unsupported.
5. **Architectural & Methodological Mismatches:** Discrepancies exist between the paper's claims and actual code in projection dimensions, DINO multi-crop implementations, classifier adapter layers, and backbone unfreezing logic. Crucially, the code used an unverified `.children()[-3:]` unfreezing heuristic which behaves completely differently on ResNet-50 vs. EfficientNet-B0 and fails to properly handle BatchNorm running statistics.
6. **Uncontrolled Supervised Controls:** The SSL backbones initialized from ImageNet-1K before chest X-ray self-supervision, but the supervised baselines were trained entirely from scratch (random initialization), creating an unfair and confounded comparison.
7. **Complete Absence of Mandated Modules:** The existing notebook contains zero code for external RSNA validation, DICOM loading, Explainable AI (Grad-CAM, Integrated Gradients, Occlusion Sensitivity), or exportable web-application inference bundles.

---

## 2. Comprehensive Issue Audit Matrix

| ID | Issue Identified | Evidence in Code / Manuscript | Risk & Impact | Required Correction | Verification Test |
|:---|:---|:---|:---|:---|:---|
| **ISSUE-01** | **Test-Set Leakage during Training** | In Cell 21 (`two_stage_finetune`), `test_loader` is passed directly; if `total_epoch_counter in eval_epoch_checkpoints`, `run_epoch(model, test_loader)` is called. In Cell 36 (`train_classifier_full`), `test_loader` is passed and evaluated with TTA and threshold tuning inside the training function. | Invalidates reported generalization; allows model selection and thresholding to overfit the test set; high risk of journal desk-rejection. | Completely strip `test_loader` from all training and validation routines (`two_stage_finetune`, `train_classifier_full`). Implement a strict Test-Set Firewall. Test evaluation occurs exactly once in a separate dedicated evaluation module after freezing model checkpoints. | Automated static test verifying that `test_loader` is not an argument to any training function. Run unit test ensuring test labels are never accessed during training. |
| **ISSUE-02** | **Stale Artifacts & Invalid Skip/Resume Behavior** | Cell 8 copies `results_checkpoint.pt`, `baseline_results.pt`, `label_fraction_results.pt`, and `ablation_results.pt` from Kaggle inputs. Cells 26, 37, 43, 46 check `if method_name in RESULTS: continue` and skip execution. Checkpoints contain no split hash, config hash, or RNG state. | Produces phantom runs where unverified historical weights from prior data partitions are reused. | Remove all hardcoded legacy file copying and naive skipping. Build a new `CheckpointManager` that requires matching `RUN_ID`, config hash, split hash, seed, and schema. Write append-only `experiment_registry.csv`. | Delete/omit legacy `.pt` files. Verify that the pipeline executes cleanly and refuses incompatible checkpoints. |
| **ISSUE-03** | **Validation Split Inconsistency & Duplicate Leakage** | Manuscript reports 16 validation images (Page 8, Table 2). Notebook Cell 6 pools `train`, `val`, `test` and performs `stratified_70_15_15`. No image hashing (SHA-256) or duplicate check is conducted. | Data leakage between splits via duplicate radiographs; discrepancy between manuscript text and actual experimental protocol. | Formally declare the revised 70/15/15 stratified partition. Perform SHA-256 hash checks and perceptual hashing across all images. Persist `split_manifest.csv` with SHA-256 and class labels. Assert zero overlap between train, val, and test partitions. | Programmatic assertion in notebook: `len(set(train_sha).intersection(val_sha)) == 0` and `len(set(train_sha).intersection(test_sha)) == 0`. |
| **ISSUE-04** | **Contradictory Headline Accuracy Claims** | Manuscript reports 98.07% in abstract, 97.38% in Table 6, 98.18% in Table 7. All numbers derived from single runs without confidence intervals or error bars. | Confuses reviewers; scientifically indefensible; undermines research integrity. | Re-run all main experiments over 3–5 seeds. Report primary results exclusively at the final predeclared epoch (epoch 30) with mean ± standard deviation. Primary evaluation uses standard threshold 0.5 without TTA. Report validation-tuned thresholds and TTA strictly as secondary sensitivity analyses. | Check that Table 3 and summary reports present mean ± SD from multi-seed runs, and all conflicting numbers are superseded. |
| **ISSUE-05** | **Confounded Supervised Baselines (Baseline Fairness)** | In Cell 35, `build_backbone_scratch` builds CNN, ResNet-50 (`weights=None`), and EfficientNet-B0 (`weights=None`). In Cell 13, SSL backbones are initialized with ImageNet weights (`ResNet50_Weights.IMAGENET1K_V2`, `EfficientNet_B0_Weights.IMAGENET1K_V1`). | Conflates the benefit of ImageNet pretraining with the benefit of chest X-ray SSL pretraining. SSL models appear artificially superior. | Add ImageNet-pretrained ResNet-50 and EfficientNet-B0 supervised transfer learning baselines alongside scratch baselines. Clearly distinguish scratch vs. ImageNet-pretrained vs. SSL-pretrained. | Verify baseline suite includes: `ResNet50_Scratch`, `ResNet50_ImageNet`, `EfficientNet_Scratch`, `EfficientNet_ImageNet`, and `CustomCNN_Scratch`. |
| **ISSUE-06** | **Flawed Backbone Freezing & BatchNorm Updating** | Cell 13 `set_backbone_frozen` unfreezes `children()[-unfreeze_last_n_blocks:]`. For ResNet-50, children[-3:] unfreezes `avgpool`, `fc`, and `layer4`. For EfficientNet-B0, children[-3:] unfreezes `features`, `avgpool`, `classifier` (which unintentionally unfreezes the ENTIRE feature extractor!). Moreover, during Stage 1 `model.train()` updates BatchNorm `running_mean` and `running_var` despite frozen weights. | EfficientNet-B0 was completely unfrozen during Stage 2, destroying the controlled two-stage comparison. Stage 1 BatchNorm statistics drift on small batches. | Explicitly target named architectural stages: for ResNet-50 unfreeze `layer4`; for EfficientNet-B0 unfreeze `features[7]` and `features[8]`. Set BatchNorm layers explicitly to `eval()` mode during Stage 1 so running stats are frozen. | Unit test that counts `requires_grad=True` parameters in Stage 1 and Stage 2 for both ResNet-50 and EfficientNet-B0. Verify BatchNorm running statistics do not change during Stage 1 forward passes. |
| **ISSUE-07** | **Classifier Adapter / Architecture Mismatch** | Cell 21 defines `ClassifierHead` as `Linear(feat_dim, 128) -> ReLU -> Linear(128, 2)`. Manuscript Figure 1 and text depict direct projection or GAP feeding into classification without detailing the 128-dim bottleneck adapter. | Reviewers will detect discrepancies between architectural diagrams and implementation details. | Formally document `ClassifierHead` architecture in `EXPERIMENT_PROTOCOL.md`, Figure 1, and Table 2. Make projection dimension configurable. | Verify model architecture summary matches manuscript description and parameter counts. |
| **ISSUE-08** | **DINO Formulation Ambiguity** | Notebook implements a CNN-based multi-crop DINO (2 global 224px + 2 local 96px crops) with student-teacher cross-entropy, centering, and sharpening. Manuscript text lacks precise details on this CNN adaptation. | DINO is typically associated with Vision Transformers (ViT). Reviewers may assume ViT was used or question collapse prevention on CNNs. | Explicitly document in methods that this is a CNN-based DINO self-distillation implementation following Caron et al. with multi-crop strategy and centering/temperature sharpening. | Code inspection and verification that teacher centering buffer and momentum updates function stably without collapse. |
| **ISSUE-09** | **Label-Efficiency & Ablation Experimental Flaws** | Cell 43 (`label_fraction_results.pt`) ran on only a single seed and evaluated test data in the training loop. Cell 46 (`ablation_results.pt`) compared random init vs. DINO+ImageNet without an ImageNet-only control. | Unable to isolate the marginal gain of in-domain SSL at different label regimes or in ablations. | Run Experiment B (Label Efficiency: 10%, 25%, 50%, 75%, 100%) across multiple seeds with fixed stratified subsets. In Experiment C (Ablation), include: (1) Scratch + no aug, (2) ImageNet-supervised + no aug, (3) ImageNet+DINO + no aug, (4) ImageNet+DINO + light aug, (5) ImageNet+DINO + strong aug. | Multi-seed tabular and graphical summaries generated directly from experiment results. |
| **ISSUE-10** | **Complete Omission of External RSNA Validation** | Zero code exists in the original notebook for RSNA cohort evaluation, DICOM parsing, or bounding-box mapping. | Fails mandatory guideline requirement 11: "Add at least one external dataset for a generalization check." Paper remains single-center and clinically unvalidated. | Implement full RSNA validation module: deterministic DICOM reading (MONOCHROME1 inversion, rescale slope/intercept, VOI LUT), image-level label aggregation from `stage_2_train_labels.csv` and `stage_2_detailed_class_info.csv`. Evaluate both Primary Challenge Cohort (`Target=1` vs `Target=0`) and Strict Class-Matched Cohort (`Lung Opacity` vs `Normal`). | Unit test on dummy DICOM; programmatic evaluation pipeline that runs inference with frozen models on RSNA. |
| **ISSUE-11** | **Complete Omission of Explainable AI (XAI)** | Zero explainability code in original notebook. | Manuscript lacks interpretability, heatmap sanity checks, and localization evaluation required for Q1 medical imaging journals. | Implement Grad-CAM, Integrated Gradients, and Occlusion Sensitivity. Perform sanity check (model parameter randomization), attribution stability analysis, and exploratory RSNA bounding-box pointing game / hit rate. | Save raw attribution maps and overlay visualizations under `explainability/`. Output quantitative sanity and stability reports. |
| **ISSUE-12** | **Lack of Standalone Deployment Bundle** | Notebook outputs only training checkpoints. No web-app inference package. | Models cannot be ported to a web application or external inference service without training scripts and full dataset dependencies. | Build `webapp_bundle/<model_version>/` containing `model_state_dict.pt`, `model_config.json`, `preprocessing.json`, `label_map.json`, `decision_policy.json`, `inference.py`, `model_card.md`, and `requirements-inference.txt`. | Standalone Python validation test that loads bundle in an isolated process and reproduces exact class probabilities on test images. |
| **ISSUE-13** | **Editorial & Typographical Errors** | Manuscript title has spelling mistake ("neumonia"). Contributions list introduces "P4SSLNet" instead of "PneuSSLNet". Main text contains redundant near-duplicate Tables 4–9. | Diminishes paper credibility; signals lack of care to journal editors. | Standardize all nomenclature to `PneuSSLNet`. Correct title to "Pneumonia". Replace redundant tables with unified Tables 1–7 as prescribed. | Full text search confirming 0 occurrences of "P4SSLNet" and "neumonia". |

---

## 3. Detailed Root-Cause Analysis

### 3.1 Test-Set Leakage Mechanism
In the existing codebase:
```python
# Cell 21 in original notebook:
if total_epoch_counter in eval_epoch_checkpoints:
    test_loss, test_labels_, test_preds, test_probs = run_epoch(
        model, test_loader, optimizer=None, class_weights=class_weights, label_smoothing=label_smoothing
    )
    checkpoint_metrics[total_epoch_counter] = compute_metrics(test_labels_, test_preds, test_probs)
```
And in Cell 36:
```python
# Generic supervised training loop:
val_labels_best, val_probs_best = run_epoch_tta(model, val_loader)
best_threshold = find_best_threshold(val_labels_best, val_probs_best)
test_labels_best, test_probs_best = run_epoch_tta(model, test_loader)
test_preds_best = preds_from_threshold(test_probs_best, best_threshold)
metrics = compute_metrics(test_labels_best, test_preds_best, test_probs_best)
```
**Consequence:** The test set was actively loaded and evaluated multiple times during training. Furthermore, in the baseline and ablation loops, test data was evaluated under TTA with threshold tuning inside the training function itself. This represents direct information leakage. In the revised pipeline, the test loader is completely decoupled and inaccessible during all training, validation, and checkpoint-selection phases.

### 3.2 The Backbone Unfreezing Flaw
The existing code implemented:
```python
def set_backbone_frozen(backbone, frozen=True, unfreeze_last_n_blocks=0):
    real_backbone = backbone.module if isinstance(backbone, nn.DataParallel) else backbone
    for p in real_backbone.parameters():
        p.requires_grad = not frozen
    if frozen and unfreeze_last_n_blocks > 0:
        children = list(real_backbone.children())
        for child in children[-unfreeze_last_n_blocks:]:
            for p in child.parameters():
                p.requires_grad = True
```
When `unfreeze_last_n_blocks=3`:
- On `resnet50`: The top-level children are `[conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]`. Taking `children[-3:]` unfreezes `layer4`, `avgpool`, and `fc` (which is `nn.Identity()`). This accidentally worked as intended for ResNet-50.
- On `efficientnet_b0`: The top-level children are `[features, avgpool, classifier]`. Taking `children[-3:]` unfreezes **all three top-level children**, meaning the entire `features` container (all 8 MBConv stages) was completely unfrozen!
**Consequence:** EfficientNet-B0 was not partially fine-tuned; it was fully fine-tuned from epoch 16 to 30. The comparison between ResNet-50 and EfficientNet-B0 was completely compromised.

### 3.3 BatchNorm Running Statistics Drift
In PyTorch, calling `model.train()` sets all submodules, including `nn.BatchNorm2d`, into training mode. Even if `param.requires_grad = False`, a BatchNorm layer in training mode continues to update its `running_mean` and `running_var` using batch statistics during forward passes. In Stage 1 (linear evaluation / frozen feature extractor), feature representations were subtly corrupted by batch statistics of downstream batches. In the revision, BatchNorm layers must be explicitly frozen into `.eval()` mode during Stage 1.

---

## 4. Remediation Action Plan

1. **Implement `EXPERIMENT_PROTOCOL.md` and `experiment_config.yaml`:** Freeze all seeds (17, 42, 73), hyperparameters, layer unfreezing rules, evaluation transforms, and metrics before launching experiments.
2. **Build `PneuSSLNet_Q1_Revised_Kaggle.ipynb`:** A clean, modular, top-to-bottom Kaggle notebook structured into 18 logical execution phases.
3. **Data Integrity & Test Firewall:**
   - Detect Kaggle input mounts dynamically.
   - Run SHA-256 duplicate auditing.
   - Generate and verify `split_manifest.csv`.
   - Implement the Test Firewall: training functions accept only `train_loader` and `val_loader`.
4. **Controlled Multi-Seed Benchmark:**
   - 4 SSL methods (SimCLR, MoCo v2, BYOL, DINO multi-crop).
   - 2 Backbones (ResNet-50, EfficientNet-B0) with correct named block unfreezing.
   - 5 Supervised controls (Scratch CNN, Scratch ResNet-50, Scratch EfficientNet-B0, ImageNet-pretrained ResNet-50, ImageNet-pretrained EfficientNet-B0).
   - 3 seeds per combination (17, 42, 73).
5. **Multi-Seed Label Efficiency & Isolated Ablations:**
   - Label efficiency across 10%, 25%, 50%, 75%, 100% training labels with matched subsets.
   - 5-condition ablation isolating ImageNet pretraining, in-domain SSL, and augmentation strength.
6. **Robust Checkpoint Management:**
   - Resumable checkpoints (`last_resume.pt`) saved after every epoch and configurable step intervals.
   - Strict metadata validation (rejecting any checkpoint with mismatched split hash, config hash, or model schema).
   - Checkpoint manifest and append-only registry.
7. **External RSNA Validation & DICOM Pipeline:**
   - Deterministic DICOM windowing, MONOCHROME1 inversion, and RGB expansion.
   - Dual-cohort evaluation: Challenge Cohort (`Target=1` vs `0`) and Strict Matched Cohort (`Lung Opacity` vs `Normal`).
8. **Explainable AI (XAI) Suite:**
   - Grad-CAM on target convolutional layers (`layer4` for ResNet-50, `features[8]` for EfficientNet-B0).
   - Integrated Gradients and Occlusion Sensitivity.
   - Model parameter randomization sanity check and perturbation stability analysis.
9. **Deployable Web-App Bundle:**
   - Standalone inference package with weights, preprocessing config, prediction API, and model card.
   - Verification script demonstrating isolated execution.
10. **Publication Tables & Figures:**
    - Generate Tables 1–7 and Figures 1–8 strictly from saved JSON/CSV result files.

---
**Audit Approved for Implementation:** Stage A Complete. Proceeding to Stage B Protocol and Notebook Architecture.

