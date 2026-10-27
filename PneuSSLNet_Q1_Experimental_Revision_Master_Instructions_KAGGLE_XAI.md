# PneuSSLNet: Master Experimental Revision Pipeline

**Purpose:** Rebuild and strengthen the experimental pipeline for a rigorous, reproducible journal submission using the existing codebase, the current PneuSSLNet manuscript, and the revision-guidelines PDF. This is an implementation specification, not permission to invent results. A strong experimental design cannot guarantee acceptance in a Q1 journal; the work must be defensible, reproducible, clinically cautious, and aligned with the actual code and evidence.


## KAGGLE-FIRST EXECUTION CONTRACT — READ THIS FIRST

**The user will run the experiments in Kaggle Notebooks. Implement and organize the revised code specifically for Kaggle; do not assume local Windows/Linux paths, Google Colab, mounted Google Drive, an interactive terminal, or access to a persistent local disk.** The deliverable must be a Kaggle-ready notebook/source implementation that can be executed top-to-bottom in ordered stages and resumed safely after Kaggle session/runtime limits.

### Kaggle notebook requirements

1. **Inputs must be mounted through Kaggle.** In the notebook instructions, tell the user to click **Add Input** and attach the main `paultimothymooney/chest-xray-pneumonia` dataset. Attach the RSNA competition data as a separate input after accepting the competition rules/access requirements. Do not request or print Kaggle API tokens when the datasets are already mounted.
2. **Discover paths robustly.** At startup, search under `/kaggle/input` for the `chest_xray` directory and for RSNA files such as `stage_2_train_labels.csv`, `stage_2_detailed_class_info.csv`, and the DICOM image directories. Do not hardcode one guessed mount path. Resolve paths once, print the resolved locations and file counts, and fail early with a useful error plus discovered paths if an input is missing.
3. **Use Kaggle storage correctly.** Treat `/kaggle/input` as read-only. Write all generated content only under `/kaggle/working/pneusslnet_runs/<RUN_ID>/`. Never overwrite source inputs. Use a unique `RUN_ID` and create a complete artifact directory tree at startup.
4. **GPU and environment checks.** The first executable cell must report Python/PyTorch/torchvision/CUDA versions, `torch.cuda.is_available()`, GPU model and VRAM when available, available disk space, Kaggle input mounts, dependency versions, and estimated output/checkpoint storage. Prefer the Kaggle-provided environment. Install only genuinely missing dependencies, avoid unnecessary upgrades of PyTorch/CUDA, and fail clearly if a required package cannot be installed. Do not expose secrets in logs.
5. **Run modes must be explicit.** Provide configuration switches such as `RUN_MODE` (`smoke_test`, `experiment_phase`, `final_evaluation`, `export_only`) and `EXPERIMENT_GROUP` so the full study can be executed in stages within Kaggle's runtime limits. Every phase must be resumable and idempotent. Smoke-test artifacts must be stored in a different directory/run namespace and must never be included in paper tables or plots.
6. **No hidden environment assumption.** Do not require manual edits to dozens of cells. Put paths, seeds, phase switches, checkpoint intervals and output options in one central configuration cell plus a generated `experiment_config.yaml`. Use no interactive `input()` prompts in the main run.
7. **Kaggle runtime continuity.** Checkpoints and logs inside `/kaggle/working` protect work during the current session, but they are not a substitute for saving a Kaggle Notebook Version. The notebook must print a clear reminder to use **Save Version / Save & Run All** (or the available Kaggle equivalent) to persist outputs. For a new session, allow a prior saved output/checkpoint directory to be attached as a Kaggle input and copied into the new run's resume area only after metadata validation. Never claim that ephemeral `/kaggle/working` survives a reset.
8. **Staged execution and progress.** Write a `run_status.json` and an append-only `experiment_registry.csv` that records each experiment configuration/seed/stage as `pending`, `running`, `completed`, `failed`, or `resumable`, with timestamps, last epoch/step, checkpoint path, and error summary. Print concise progress and remaining planned tasks. Do not mark an experiment complete until expected checkpoints, logs, and metrics pass verification.
9. **Offline-safe data handling.** Once both datasets are mounted, data preparation and experiments must not depend on downloading image data at runtime. Any package installation/network limitation must be diagnosed before expensive training begins.
10. **User-facing run guide.** Generate `KAGGLE_RUN_GUIDE.md` with exact steps: enable GPU accelerator; attach both datasets; run the audit/bootstrap cells; run a smoke test; execute each full experiment phase; save a notebook version; resume from a saved checkpoint; and retrieve outputs/model bundles. The notebook must print the final output path and a short completion summary.

### Required Kaggle output layout

Use a structure equivalent to:

```text
/kaggle/working/pneusslnet_runs/<RUN_ID>/
  configs/                 # YAML/JSON plus hashes
  manifests/               # split, external cohorts, duplicates, label mapping
  audit/                   # code/data/environment audits
  logs/                    # console logs and run_status.json
  registry/                # experiment_registry.csv and checkpoint index
  checkpoints/
    ssl_pretrain/<method>/<backbone>/seed-<seed>/
    downstream/<method>/<backbone>/seed-<seed>/stage-1/
    downstream/<method>/<backbone>/seed-<seed>/stage-2/
    baselines/<baseline>/<backbone>/seed-<seed>/
    label_efficiency/<method>/<backbone>/fraction-<fraction>/seed-<seed>/
    ablation/<ablation_id>/seed-<seed>/
  histories/                # per-epoch JSONL/CSV
  predictions/internal_test/
  predictions/rsna_external/
  metrics/                  # per-seed and aggregate tables
  figures/main/
  figures/supplementary/
  explainability/           # Grad-CAM, IG, occlusion maps and audit results
  webapp_bundle/            # validation-selected inference-ready model package
  tables/
  reports/
```

## 0. Source of truth and working rules

Inspect these three local source files first and treat them as the primary project record:

1. `PNEUMONIA SSL SL FINAL(20261017-164315).ipynb` — the actual code, execution history, saved-output behavior, and current experiment implementation.
2. `Pneumonia.pdf` — the current 21-page manuscript, including its method, equations, tables, figures, results, limitations, and references.
3. `pneumonia paper update guidelines.pdf` — the required corrections and revised table plan.

Data sources:

- Main development dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- External validation dataset: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data
- Official challenge description and usage terms: https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018

**Never silently guess.** If code, manuscript, labels, checkpoints, or observed outputs disagree, log the discrepancy and resolve it explicitly. Do not fabricate metrics, sample counts, figures, confidence intervals, training completion, or statistical significance. Do not state that a run completed unless its logs and artifacts prove it. Do not alter the research question or replace the current DIR/method design without explaining the change. Preserve the original notebook and make a clean revision copy/branch.

## 1. Mandatory audit before implementation

Create `REVISION_AUDIT.md` before modifying the experiment. Include a table with columns: `Issue`, `Evidence in current code/manuscript`, `Risk`, `Required correction`, `Verification test`.

At minimum, audit these known problems:

1. **Stale artifacts and invalid resume behavior.** The current notebook copies and loads pre-existing `results_checkpoint.pt`, `baseline_results.pt`, `label_fraction_results.pt`, and `ablation_results.pt`; it skips work when these keys/files exist. SSL and fine-tuning checkpoints do not robustly prove compatibility with the current data split, seed, code, or configuration. Never reuse those results for the revised paper.
2. **Test-set leakage.** The current fine-tuning function evaluates the test loader at multiple epochs. The generic classifier loop also evaluates the test data for supervised baselines, label-fraction runs, and ablations, and applies test-time augmentation (TTA) plus a validation-tuned threshold. This repeatedly exposes test results and can bias model/configuration choices. Remove test evaluation from all training/validation functions.
3. **Tiny validation split and split ambiguity.** The manuscript reports only 16 validation images (8 per class). The notebook currently pools all images from the original `train`, `val`, and `test` folders and creates a new stratified 70/15/15 image split (observed output: 4,099 train / 878 validation / 879 test). These are not the same split/protocol as the manuscript's old counts. Declare the revised partition explicitly, regenerate all experiments from scratch, and make the test partition inaccessible to training, checkpoint selection, threshold selection, early stopping, and figure selection.
4. **Contradictory headline results.** The manuscript reports several “best” values (including 98.07%, 97.38%, and 98.18% accuracy in different contexts). Recompute the final results under the corrected protocol; do not carry any old result into the revised abstract, tables, or conclusion unless reproduced validly and labelled with its exact evaluation setting.
5. **One-run comparisons.** Existing SSL comparisons are not supported by run-to-run variability. Use at least 3 independent training seeds for every primary benchmark configuration; 5 seeds are preferred if compute allows.
6. **Baseline fairness.** The SSL backbones use ImageNet-pretrained initialization before chest-X-ray SSL pretraining, but the existing CNN/ResNet-50/EfficientNet-B0 supervised baselines are randomly initialized. Add supervised ImageNet-pretrained ResNet-50 and EfficientNet-B0 controls so the paper can distinguish gains from ImageNet initialization from gains associated with in-domain SSL. Keep the scratch baselines too if resources allow.
7. **Architecture/manuscript mismatch.** Audit the feature dimensions, classifier-head layers, projection dimensions, DINO crops/loss, normalization, frozen BatchNorm behavior, and the “last three blocks” unfreezing rule against the manuscript and Figure 1. Current code includes a `feature_dim -> 128 -> 2` classifier adapter, while the diagram/text may imply a different head. Do not silently call these identical: either align implementation with the documented method or revise the manuscript/figure to match the implemented model.
8. **DINO description mismatch.** The notebook contains descriptions of simplified 2-view DINO and 2-global-plus-2-local-crop DINO. Verify which exact code executes and report only that implementation. If this is a CNN-based DINO-style adaptation, describe it precisely; do not imply it is the original ViT-DINO implementation.
9. **Frozen-backbone behavior.** A frozen backbone can still update BatchNorm running statistics while the overall model is in training mode. Explicitly test and control whether BatchNorm statistics remain frozen during Stage 1; document the behavior.
10. **Label-efficiency and ablation confounds.** Existing label-efficiency runs use a single ResNet-50 SSL backbone mapping and cached results, and the ablation compares random initialization with an ImageNet-initialized-plus-DINO model. Add controls that isolate ImageNet initialization from the incremental effect of in-domain SSL; run repeated seeds.
11. **Publication inconsistencies.** Correct `neumonia` in the title, consistently use `PneuSSLNet` (remove `P4SSLNet` everywhere), remove near-duplicate main tables, make all values traceable to regenerated result files, and proofread the complete manuscript and reference list.

Do not change the code until this audit and the frozen experiment protocol are written.

## 2. Data governance and split lock

### 2.1 Main Kaggle dataset

The reported source dataset contains 5,856 JPEG radiographs: 1,583 `NORMAL` and 4,273 `PNEUMONIA`. Verify the actual files and class counts; do not assume they have not changed.

Because the supplied validation folder has only 16 images, create a documented internal 70/15/15 stratified train/validation/test split from the available main-dataset images, as required by the current revision plan. Make clear that this is a **newly defined internal split**, not the original Kaggle partition. Use one fixed, versioned split for all models. Seeds used for model training must not silently regenerate different data partitions.

Before splitting:

- inventory every file, extension, class, and original folder;
- verify that images can be opened and have valid dimensions;
- detect exact duplicates with SHA-256 and near-duplicates with a perceptual hash/image-similarity audit;
- keep duplicate clusters in one partition, or remove verified duplicate copies under a documented rule;
- use patient/study grouping if reliable identifiers exist; do not invent patient identifiers from filenames;
- report any limitation if only image-level grouping is possible.

Write `split_manifest.csv` with at least: `image_path`, `class_label`, `original_folder`, `new_split`, `sha256`, and, if available, `duplicate_group`/`patient_group`. Write a split summary with per-class counts and percentages. Assert programmatically that train/validation/test have no exact duplicate or duplicate-cluster overlap.

**Strict information boundaries:**

- SSL pretraining may use only images in the revised training split, with their diagnostic labels ignored.
- Supervised training and label-efficiency runs may use only training-split labels.
- Validation is used only for predeclared development tasks, such as observing learning curves and any explicitly planned model/threshold analysis.
- No internal test image, label, prediction, or test metric may be accessed by training, model selection, threshold selection, hyperparameter selection, early stopping, augmentation selection, or decision about which figure to show.
- Test evaluation must be executed by a separate final-evaluation function after the model/configuration protocol has been locked. Training functions should not accept a `test_loader` argument at all.

Because exploratory runs have existed, record that history in the audit. Never reuse their weights or scores. Make the revised run a clean experiment with a new `RUN_ID`, new output directory, no copied result files, and freshly initialized model/optimizer state.

### 2.2 External RSNA dataset: define the endpoint correctly

Use the **labelled training portion** of the RSNA challenge dataset for retrospective external validation, namely the image IDs and annotations associated with `stage_2_train_labels.csv` plus `stage_2_detailed_class_info.csv`. Do not use `stage_2_test_images` as validation data because the challenge test images do not provide the same public ground-truth labels.

The RSNA files are DICOM. The challenge labels include `Target` and may have multiple bounding-box rows per image ID. Before evaluating:

- create exactly one image-level row per `patientId`/image ID;
- aggregate repeated bounding-box rows without duplicating images (verify `Target` consistency; a positive target is positive if the image has a positive annotated box);
- join detailed class information using a validated one-to-one image-ID mapping;
- report class counts after every inclusion/exclusion rule;
- check exact/perceptual overlap between main and RSNA images and remove any confirmed overlap from external evaluation;
- do not use bounding boxes as inputs to the classifier; they are annotations, not features for this image-level task.

**Critical label caveat:** RSNA `Target=1` represents radiologist-annotated lung opacity considered suspicious for pneumonia; `Target=0` means no definitive pneumonia-associated opacity, not necessarily a healthy/normal radiograph. The detailed class file distinguishes `Normal`, `No Lung Opacity / Not Normal`, and `Lung Opacity`. Therefore report two predeclared external analyses where the records permit:

1. **Primary challenge-label analysis:** `Target=1` versus `Target=0`, described as `pneumonia-suspicious lung opacity` versus `no definitive pneumonia-associated opacity`, not “pneumonia versus normal.”
2. **Strict class-matched sensitivity analysis:** `Lung Opacity` versus `Normal`, excluding `No Lung Opacity / Not Normal`; state the exclusion and sample counts. This is a restricted cohort and must not replace the full-cohort result silently.

If class-file mappings or labels are inconsistent, stop and report the issue rather than guess. RSNA is an external-domain evaluation and must remain completely separate from training, SSL pretraining, threshold setting, model selection, and hyperparameter tuning.

### 2.3 DICOM preprocessing

Implement and unit-test a deterministic DICOM reader, preferably using `pydicom` and the correct display/intensity transforms. Check rescale slope/intercept and VOI LUT where applicable; handle `MONOCHROME1` polarity correctly; robustly convert the resulting grayscale image to three channels; then apply the same deterministic resize/normalization convention used by the trained model. Do not use random augmentations at evaluation time in the primary analysis. Document the exact DICOM-to-tensor steps and cache decoded pixels/images only with stable IDs and checksums. Compare main-dataset and RSNA examples visually to catch inverted, clipped, blank, or incorrectly windowed radiographs before inference.

## 3. Freeze the experimental protocol before running full experiments

Create a versioned `experiment_config.yaml` (or JSON) and a concise `EXPERIMENT_PROTOCOL.md`. Include:

- data split manifest hash;
- image size and deterministic evaluation transform;
- augmentation policy for each SSL method;
- backbone names and exact ImageNet weight source/version;
- SSL objective and all method-specific parameters;
- optimizer, learning rates, batch sizes, weight decay if used, scheduler, epoch counts, loss, class weights, label smoothing, and unfreezing policy;
- seed list, number of repetitions, checkpoint policy;
- primary metric and the primary decision threshold;
- exact model-selection rule;
- TTA policy;
- internal-test and external-evaluation policy;
- software/hardware versions and run ID.

Set `SEEDS` to at least three distinct values (for example 17, 42, 73); use five if feasible. Hold the dataset split fixed across model seeds to make model comparisons paired. Vary training randomness—initialization where applicable, augmentation, sampling, and data-loader randomness—under reproducible seed control. Seed Python, NumPy, PyTorch CPU/CUDA, and data-loader workers. Record any operations that remain nondeterministic on GPU.

Set the primary schedule before running: 30 SSL pretraining epochs, 15 frozen-backbone Stage-1 epochs and 15 Stage-2 fine-tuning epochs, with primary comparisons taken at the **final predeclared epoch only**. Do not report the best of 10/15/20/30 test results. Checkpoints may be saved for recovery, but recovery checkpoints must not become a model-selection mechanism.

### Cache/checkpoint policy

- Never import or copy legacy `results_checkpoint.pt`, `baseline_results.pt`, `label_fraction_results.pt`, `ablation_results.pt`, or legacy weight checkpoints into this run.
- Use a new, empty run directory, e.g. `outputs/pneusslnet_revision_<RUN_ID>/`.
- Every checkpoint must include `RUN_ID`, code version/commit, config hash, data-split hash, seed, method, backbone, stage, epoch, optimizer/scheduler state, and metrics history.
- Refuse to resume if any critical metadata differs. Use separate checkpoint names per seed × method × backbone × experiment.
- Add a clean reset option, but make destructive deletion explicit and scoped only to the new revision output directory.



### Required per-epoch/step checkpointing, safe resume, and naming

The checkpoint code is a required experiment component, not an optional convenience. It must work on Kaggle and must preserve enough state to resume a run and enough model metadata to create a web application later.

**Save policy:**

- At every completed epoch in every experiment configuration × seed × training stage, atomically write/refresh a full `last_resume.pt` checkpoint and append that epoch's losses, validation metrics, learning rate, elapsed time, and checkpoint path to a durable CSV/JSONL history.
- During long epochs, also write an atomic step-level recovery checkpoint every configurable `CHECKPOINT_EVERY_N_STEPS` (default 500 optimizer steps; if an epoch is shorter, the epoch-end save is mandatory). The step-level checkpoint must include `global_step` and the information needed by the implemented resume logic. Document any limitations to bit-exact mid-epoch resumption with multi-worker data loading; epoch-boundary resume must be reliable.
- At each stage boundary, save an immutable stage-final snapshot. For downstream stages, additionally preserve a `best_validation_<metric>.pt` snapshot only if that secondary checkpoint-selection rule was specified before training and the metric was computed on validation only. The primary benchmark remains the fixed final predeclared epoch; never replace it with the best internal-test checkpoint.
- Save a separate final `inference_weights.pt`/`model_state_dict.pt` without optimizer state for deployment. Do not depend on a training resume checkpoint inside the web application.
- To control Kaggle storage, overwrite the rolling `last_resume.pt` atomically rather than retaining every full optimizer state forever. Keep immutable final/stage snapshots, checkpoint indices, per-epoch history, and optionally every-epoch immutable snapshots for the small set of shortlisted models when space permits. If `SAVE_EVERY_EPOCH_SNAPSHOT=True`, estimate storage first and warn before enabling it for all seeds/configurations. Never silently run out of disk space.

**Checkpoint naming convention:**

Use names that identify the actual run and experiment, for example:

`<RUN_ID>__<stage>__<method>__<backbone>__seed-<SEED>__epoch-<EPOCH:03d>__step-<STEP:07d>__resume.pt`

Use method/backbone values such as `simclr`, `moco`, `byol`, `dino_style`, `resnet50`, `efficientnet_b0`, and explicitly named baseline/ablation/fraction IDs. Avoid ambiguous filenames such as `best.pt`, `checkpoint.pt`, or overwriting files from another run. Store each experiment in its own path as shown in the Kaggle output layout.

**Every resumable checkpoint must include at least:**

- model `state_dict` (encoder/backbone, projection head where relevant, classifier head, and teacher/target/online networks where relevant);
- optimizer state, scheduler state, AMP GradScaler state if used;
- epoch, stage, global optimizer step, batch index, best-so-far validation metric if the predeclared secondary rule uses one, and complete training/validation history or a stable history-file reference;
- Python, NumPy, PyTorch CPU/CUDA RNG states and sampler/generator state as supported by the implementation;
- `RUN_ID`, experiment ID, method, backbone, seed, data split-manifest hash, config hash, code/notebook hash or commit, package versions, dataset identifiers, class-to-index mapping, image size, normalization, augmentation configuration, optimizer/loss configuration, and timestamp;
- the checkpoint schema version and the exact path of the config and split manifest.

Save through a temporary file followed by an atomic rename. Validate the checkpoint after writing. Before resuming, compare run/config/split/model/schema metadata and refuse incompatible checkpoints with a clear diagnostic. Resume must not cause a completed run to be silently skipped if its artifact verification fails. Never load legacy checkpoints from the old notebook output or a different split.

**Checkpoint index:** append a row to `registry/checkpoint_index.csv` after each save with `RUN_ID`, experiment ID, seed, method, backbone, stage, epoch, global step, checkpoint kind, relative path, file size, SHA-256, and whether it was validated successfully. Write a separate `artifact_manifest.json` with hashes for every final model and inference bundle file.

## 4. Main controlled benchmark (mandatory)

### 4.1 Configurations

Run every combination of:

- SSL objectives: SimCLR, MoCo, BYOL, DINO-style self-distillation;
- backbones: ResNet-50, EfficientNet-B0;
- at least 3 seeds per configuration (5 preferred).

For fairness, use consistent training/validation/test partitions, evaluation preprocessing, downstream training budget, classification labels, metrics, and checkpoint policy. Report any method-specific augmentation/loss differences explicitly. Do not compare raw SSL loss magnitudes across methods as if the loss scales were equivalent; plot losses in separate method-specific panels.

Use only the revised training images for SSL pretraining. Initialize each SSL encoder using the same documented ImageNet-pretrained backbone source unless the protocol explicitly calls for another initialization. Ensure each method/seed starts from an independent clean backbone and optimizer state. Do not allow a model trained for one configuration to leak weights into another except for the intended SSL-to-downstream transfer.

### 4.2 Supervised controls

Keep the existing-from-scratch baselines if reproducible:

- custom CNN from random initialization;
- ResNet-50 from random initialization;
- EfficientNet-B0 from random initialization.

Add these essential transfer-learning controls:

- ImageNet-pretrained ResNet-50 supervised on the same labelled training split;
- ImageNet-pretrained EfficientNet-B0 supervised on the same labelled training split.

Use the same internal split, repeat with the same seed list, use validation-only development, and evaluate once on test. These controls are needed because an ImageNet-initialized SSL model should not be compared only with randomly initialized supervised models. Explain which baseline is a scratch baseline and which is a transfer-learning baseline.

### 4.3 Downstream training and model selection

- Stage 1: freeze the backbone weights and train the classification head for the fixed 15 epochs.
- Stage 2: unfreeze the specifically named final blocks/layers for the fixed 15 epochs and fine-tune under the predeclared scheduler. Verify the selected layers for both ResNet-50 and EfficientNet-B0; “last three blocks” must mean actual documented modules, not an accidental slice of parameters.
- During Stage 1, ensure frozen BatchNorm running statistics behave as intended; test this explicitly.
- Use class weights calculated from the training split only if retained. Do not estimate them from validation/test data. Fix label smoothing and any other loss settings in the protocol before main runs.
- Primary comparison uses the **final 30-epoch checkpoint** for every run. Validation curves are diagnostic; they must not cause selective reporting of test epochs. If an alternate validation-selected checkpoint is scientifically necessary, treat it as a separately predeclared secondary analysis and do not mix it into the final-epoch table.
- Primary evaluation uses **no TTA and a fixed probability threshold of 0.5**. Do not optimize the threshold on the test set. If a validation-chosen threshold or TTA analysis is retained as a secondary sensitivity analysis, define it in advance, select it using validation only, freeze it before test evaluation, and report it separately from the primary result.
- Train/validation curves must never contain test metrics.

## 5. Required additional experiments

### Experiment A — Repeated-seed main benchmark

For each of the 8 SSL × backbone configurations and each supervised control, run the same predeclared schedule with at least 3 seeds. Save per-epoch train/validation loss and metrics, the final checkpoint, run metadata, and per-image final probabilities. Summarize all prespecified final-epoch runs as mean ± standard deviation. Do not infer a winner from a single run or a sub-one-point difference without uncertainty analysis.

### Experiment B — Label efficiency

Use labelled fractions of 10%, 25%, 50%, 75%, and 100% of the **training partition only**. Generate stratified subsets reproducibly; use matched subsets across methods for each fraction and seed. For SSL, pretrain on all training-partition images with labels hidden, then use only the specified fraction of their labels for downstream classification. Explicitly state that this is label-efficient downstream adaptation with access to unlabeled training images—not a setting with only that fraction of all images available.

Run at least the four SSL methods with ResNet-50, all five labelled fractions, and all seeds, plus the supervised controls. If resources allow, repeat on EfficientNet-B0 and move these results to supplementary material if they do not fit the main narrative. Use the same validation and locked test set for every fraction; do not train on or sample labels from validation/test. Plot accuracy against labelled fraction as a line chart with mean and variability band/error bars. Add a second supplemental plot for AUROC or macro-F1 if useful. Replace the old label-efficiency main-text table with the prescribed figure.

### Experiment C — Ablation study

Retain the intended questions, but isolate the effects correctly. At minimum evaluate with repeated seeds on the prespecified ablation backbone (ResNet-50):

1. Scratch/random initialization + no stochastic downstream augmentation;
2. ImageNet-pretrained supervised initialization without in-domain SSL + no stochastic downstream augmentation;
3. ImageNet initialization + DINO-style in-domain SSL + no stochastic downstream augmentation;
4. ImageNet initialization + DINO-style SSL + light downstream augmentation;
5. ImageNet initialization + DINO-style SSL + strong downstream augmentation.

The ImageNet-only control is necessary to separate the benefit of ImageNet pretraining from the benefit of the in-domain SSL stage. Keep optimizer budget and splits comparable, clearly define what “none,” “light,” and “strong” mean, and use the same locked test policy. Do not modify the ablation after inspecting test results. Report all configurations and variability; do not select only the favorable rows.

### Experiment D — External RSNA validation

After all main models and decision rules are locked, run inference on the RSNA labelled cohort. Preferably evaluate all predeclared main configurations and supervised controls; at minimum evaluate the winner selected solely on internal validation plus the ImageNet-pretrained control and other predeclared comparison models. No RSNA fine-tuning, self-supervised pretraining, threshold tuning, normalization-statistic fitting with labels, or model selection is permitted.

Report both the full `Target` cohort and the strict `Normal` versus `Lung Opacity` subgroup as described in Section 2.2. Give class counts, label mapping, exclusion counts, DICOM preprocessing, exact model checkpoint/version, and all metrics. Quantify the internal-to-external performance change. State the population/domain and label-definition differences; do not claim RSNA proves clinical deployment readiness.

### Experiment E — Error, discrimination, and calibration analysis

For the final locked internal-test evaluation and the external evaluation, calculate, where mathematically defined:

- accuracy;
- macro precision, macro recall, and macro-F1;
- pneumonia sensitivity/recall and specificity;
- per-class precision, recall, F1, and support;
- AUROC and area under the precision-recall curve (AUPRC/PR-AUC);
- balanced accuracy and Matthews correlation coefficient (MCC);
- confusion matrix (raw counts and normalized percentages);
- Brier score and a calibration plot; optionally expected calibration error with its binning rule stated.

Use positive-class probability consistently and state which label is considered positive. Handle undefined values explicitly; never silently convert them to zero. Save per-image `image_id`, ground truth, probability, prediction, dataset/cohort, seed, method, and backbone for complete traceability.

Report mean ± standard deviation across model seeds in the main benchmark. Separately calculate uncertainty over evaluated images (for example, stratified bootstrap 95% confidence intervals with at least 2,000 resamples; use patient-level resampling if patient groups exist). Do not confuse seed variability with test-sample confidence intervals. For planned method comparisons, use paired comparisons because all models predict the same test cases; use an appropriate test/paired bootstrap, report effect sizes and confidence intervals, and adjust multiple pairwise comparisons (e.g. Holm). Treat unplanned comparisons as exploratory. If confidence intervals overlap and paired testing does not support a difference, do not claim a statistically superior method.



### Experiment F — Explainable AI (XAI): faithful, reproducible, clinically cautious

Add XAI as a planned interpretability analysis, provided the chosen CNN layers and saved model graph support it. XAI is an analysis of model behavior; it must not be presented as proof of causality, a validated diagnosis rationale, or evidence of clinical readiness.

**Models and methods:**

1. Primary method: Grad-CAM or Grad-CAM++ for the internal-validation-selected CNN classifier. Document the exact target layer for each backbone (for example, the final residual block for ResNet-50 and the final convolutional feature block for EfficientNet-B0); inspect actual model modules instead of assuming names.
2. Complementary method: Integrated Gradients with a declared baseline and integration-step count.
3. Complementary check: occlusion sensitivity with fixed, documented patch/window size and stride.
4. If two explanation methods disagree materially, report the disagreement instead of cherry-picking the more attractive heatmap.

**Sampling plan:**

- Select the model/configuration using internal validation only, before opening test/external results.
- Create a deterministic, stratified explanation sample from the locked internal test after the primary evaluation. Include NORMAL and PNEUMONIA, correct and incorrect predictions, and predeclared confidence ranges; show TP/TN/FP/FN examples when those categories exist. Use a fixed selection algorithm and record image IDs in a manifest, rather than hand-picking visually persuasive images.
- Run the same locked model on a predeclared sample of RSNA external cases, clearly separating the full challenge-label cohort from the strict `Normal` versus `Lung Opacity` cohort. External explanations are descriptive only and must not affect model selection.
- Ensure that no burned-in patient identifiers or DICOM metadata are visible in output images.

**XAI quality checks:**

- Run parameter-randomization/sanity checks where feasible to verify that explanations depend on learned model parameters.
- Measure stability of explanations under small, clinically plausible preprocessing perturbations, using a predeclared similarity metric.
- For at least the main explanation method, perform an occlusion/deletion sensitivity experiment and report how prediction scores change when high-attribution versus control regions are masked. State masking limitations and avoid overinterpreting artifacts.
- Use identical display scales/normalization rules across examples; save the raw heatmap arrays as well as overlays. Include the original radiograph, heatmap, and overlay side by side.
- Report failure cases and examples where heatmaps emphasize borders, text markers, devices, or non-lung regions. Do not remove such cases from aggregate reporting.

**Optional RSNA box-alignment analysis:**

Where RSNA bounding boxes and target labels are available, compare heatmaps with the annotated boxes using predeclared descriptive measures such as fraction of heatmap activation inside the box, pointing-game hit rate, or thresholded overlap. Treat this strictly as exploratory weak localization agreement: bounding boxes are not pixel-level lesion masks, and a classification model was not trained as a lesion-localization model. Include sample counts, aggregation details, baselines where feasible, and limitations; do not describe these results as validated lesion segmentation.

**Main/supplementary outputs:**

- Main paper: if the figure adds real interpretive value, include one compact XAI figure showing prespecified TP/TN/FP/FN examples from the selected model, across both internal and external domains where suitable. Include a false-positive and false-negative example when present; do not show only successful predictions.
- Supplement: Grad-CAM/Grad-CAM++ and Integrated Gradients comparison, occlusion maps, sanity checks, stability results, full example-selection manifest, external-domain explanation examples, and optional RSNA box-alignment analysis.
- Save all explanation outputs and metadata under `explainability/`, keyed by run/model/checkpoint hash, image ID, method, target class, and normalization settings. Generate figures from saved arrays/scripts; never hardcode heatmaps into the manuscript.

Do not describe saliency as a causal explanation or claim that the model “looks at pneumonia” solely from a few heatmaps. XAI results must be interpreted cautiously and evaluated with sanity/stability checks.
## 6. Test-set firewall and release gate

Implement the firewall in code, not only as a comment:

- training functions receive train and validation loaders only;
- threshold and model-selection utilities accept validation labels only;
- test and external data are loaded by a separate final-evaluation module after model/configuration hashes are frozen;
- every final evaluation writes a timestamp, code/config/checkpoint hashes and an immutable prediction CSV;
- the main-test evaluation is run once after all model choices are frozen;
- do not repeatedly inspect test scores and then change model, augmentations, threshold, epochs, or metrics;
- if a post-test change is made, declare the prior test exploratory and use a newly untouched test cohort for confirmation when possible.

Add automated checks that fail if test IDs occur in train/validation manifests, if any test-loader call occurs in a training loop, if a threshold was fit on test labels, if an old checkpoint is loaded, or if an output artifact is missing run/split/config metadata.

## 7. Publication tables (follow this structure)

Use one table to answer one question and regenerate all values directly from the final CSV/JSON result files. Remove the current overlapping/near-duplicate Tables 4–9 from the main text and do not include per-epoch rows in the main comparison table.

**Table 1 — Dataset and split.** Report main dataset source, total images, class counts, exact new train/validation/test counts and percentages, duplicate/group audit, split seed/hash, and the new validation-set procedure. Confirm that no internal test metric was used to select the model within the revised run. Add a separate row/section for the RSNA external cohort, with the two label mappings and cohort counts.

**Table 2 — Training and hyperparameter configuration.** Report architectures, exact ImageNet weights, SSL settings, augmentations/crops, optimizer, learning rates, batch sizes, epoch schedule, classifier-head dimensions, freeze/unfreeze modules, class weights/loss, seed count, hardware/software, checkpoint/model-selection protocol, primary threshold, and TTA policy. Primary: final epoch, TTA off, fixed 0.5 threshold. Explain any prespecified secondary threshold/TTA analysis separately.

**Table 3 — Main results.** One row per method × backbone (SimCLR/MoCo/BYOL/DINO × ResNet-50/EfficientNet-B0) plus the supervised controls. Report only final-epoch results as mean ± SD across 3–5 seeds for precision, recall, F1, accuracy, and AUROC. Include AUPRC, sensitivity/specificity or balanced accuracy in a supplementary table if table width is constrained. No 10/15/20/30-epoch rows here.

**Table 4 — Winning internal-test configuration: per-class performance.** Provide the raw confusion matrix and per-class precision, recall, F1, support for `NORMAL` and `PNEUMONIA`. State how the winning configuration was selected using validation only and whether this matrix comes from a predeclared single seed or a separately labelled seed ensemble. Add the full per-seed confusion matrices in supplementary material.

**Table 5 — Ablation study.** One row per predeclared ablation configuration, reporting metrics as mean ± SD across repeated seeds. Include the ImageNet-only control so the in-domain SSL effect is distinguishable.

**Figure — Label efficiency.** Replace the existing label-efficiency table in the main text with a line chart of accuracy versus labelled-data fraction, one line per SSL method (and clearly distinguished supervised controls), with variability bands/error bars across seeds. Explain unlabeled-image access during SSL pretraining in the caption.

**Table 6 — State-of-the-art comparison.** Retain the current comparison structure but insert the differences-in-split/protocol caveat immediately before the table. Clearly distinguish literature-reported results from results reproduced in this study. Never describe cross-paper scores as a controlled head-to-head comparison unless the data, split, preprocessing, and training protocol are genuinely matched.

**Table 7 — External RSNA validation.** Add a dedicated table for the locked external evaluation, separated by cohort/label mapping and model configuration, with sample counts, AUROC, AUPRC, sensitivity, specificity, F1, accuracy and confidence intervals. State that this is external-domain generalization, not clinical/prospective validation.

## 8. Publication figures and supplementary material

Prioritize figures that answer specific scientific questions; do not add decorative or redundant plots. All figures must be computed from logged results, have readable labels, appropriate uncertainty, consistent typography, and publication-grade exports (vector PDF/SVG where suitable plus at least 300-dpi raster output). Save the code and source data for every figure.

### Recommended main figures

1. **Figure 1 — Revised PneuSSLNet workflow.** Accurate architecture/data-flow diagram: training-only unlabeled SSL pretraining; SimCLR/MoCo/BYOL/DINO branches; ResNet-50/EfficientNet-B0 backbones; projection head; frozen and partial-unfreeze stages; validation-only development; locked internal test; independent RSNA external evaluation. Do not depict testing as a training-time feedback loop.
2. **Figure 2 — Data and preprocessing audit.** Representative NORMAL/PNEUMONIA main-dataset images plus example RSNA DICOM images after the documented preprocessing; optionally one compact example of each augmentation family. Do not show patient-identifiable metadata or invented examples.
3. **Figure 3 — Optimization and convergence.** Training/validation loss and metrics across epochs, aggregated across seeds with mean and variability bands. SSL-method losses should be in separate panels because their loss definitions/scales differ. Never plot test performance by epoch.
4. **Figure 4 — Label efficiency.** Accuracy versus labelled fraction with one line per predeclared method and seed variability.
5. **Figure 5 — Discrimination and external shift.** ROC and/or precision-recall curves for the selected prespecified models on internal test and RSNA external cohorts; identify label mapping and use confidence bands/intervals where robustly estimated.
6. **Figure 6 — External generalization.** Paired dot/forest plot showing internal-test and RSNA metrics and the change from internal to external domain. Include uncertainty; do not hide negative transfer or weak results.
7. **Figure 7 — Explainable-AI analysis (if it adds interpretive value).** Include prespecified Grad-CAM/Grad-CAM++ examples for the validation-selected model with representative TP/TN/FP/FN cases where available, across internal and external domains where appropriate. Include sanity/stability evidence in the supplement; never display only favorable examples.
8. **Figure 8, if space and results justify it — Error/calibration.** Calibration plots and/or normalized confusion matrix for the validation-selected winning configuration, with raw counts in Table 4.

### Recommended supplementary materials

- **Figure S1:** per-seed SSL pretraining loss trajectories for every method × backbone, plus the mean/variability plot;
- **Figure S2:** all train/validation loss, accuracy, AUROC or macro-F1 trajectories per seed; no test curves;
- **Figure S3:** full augmentation examples and exact DINO global/local crop illustration;
- **Figure S4:** label-efficiency results for secondary backbone(s), and supplementary metric curves for each labelled fraction;
- **Figure S5:** ablation comparison including ImageNet-only control;
- **Figure S6:** per-seed internal-test confusion matrices and precision-recall plots;
- **Figure S7:** external RSNA ROC/PR curves and confusion matrices for both predeclared label mappings;
- **Figure S8:** calibration and threshold-sensitivity analyses; if including threshold/TTA results, mark them as secondary and state how the threshold was fixed using validation only;
- **Figure S9:** DICOM preprocessing quality-control montage (normal, opacity, and no-opacity/not-normal categories); no fabricated labels or cherry-picked successes;
- **Figure S10:** XAI method comparison (Grad-CAM/Grad-CAM++, Integrated Gradients and occlusion sensitivity) with original radiographs, raw heatmaps, overlays, and prespecified TP/TN/FP/FN coverage;
- **Figure S11:** XAI sanity, stability and occlusion/deletion checks; optional RSNA bounding-box alignment with explicit exploratory-only interpretation;
- **Table S1:** all seed-level metrics and 95% confidence intervals;
- **Table S2:** all per-class metrics/support and confusion-matrix counts;
- **Table S3:** complete label-efficiency metrics across methods, fractions, and seeds;
- **Table S4:** full ablation metrics and configuration details;
- **Table S5:** external cohort counts and inclusion/exclusion/mapping rules;
- **Table S6:** environment, packages, GPU/CPU, training times, parameter counts, and checkpoint identifiers;
- **Table S7:** planned paired statistical comparisons with effect sizes and multiplicity correction.

Keep main-paper figures selective: use the most informative core set in the article and place detailed variants in the supplement. All visualizations must be reproducible from CSV/JSON/NPY data and plotting scripts; do not manually type metric values into plotting code.



## WEB-APPLICATION HANDOFF — REQUIRED DEPLOYMENT ARTIFACTS

The research notebook is run in Kaggle, but its final selected model must be portable to a separate web application. Build a model package after the architecture/preprocessing are frozen and a candidate is selected using internal validation only. External results may be attached to the model card after evaluation, but must not alter which checkpoint is exported.

### Deployment bundle

Write the selected validation-based model bundle to:

`/kaggle/working/pneusslnet_runs/<RUN_ID>/webapp_bundle/<model_version>/`

It must contain:

- `model_state_dict.pt` — inference-only model weights for the exact architecture; no optimizer state required.
- `model_config.json` — architecture, backbone, classifier/projection head as appropriate, feature dimensions, number of classes, model version, and required constructor arguments.
- `preprocessing.json` — input size, RGB/grayscale conversion, resize rule, scaling, normalization mean/std, DICOM preprocessing requirements if applicable, channel ordering, and deterministic transform definition.
- `label_map.json` — exact class index-to-name mapping (`NORMAL`, `PNEUMONIA`) and the explicitly selected positive class.
- `decision_policy.json` — primary threshold (0.5 unless a threshold was predeclared and selected on internal validation), probability output definition, TTA policy, and any calibration transform if actually fitted on validation.
- `validation_metrics.json` — validation-selected metrics and the rule used to select the model. Add internal-test/external metrics only after final evaluation, without changing the model selection.
- `model_card.md` — training dataset, intended use, limitations, known domain shift, evaluation cohorts, exact checkpoint ID, XAI caveats, and statement that it is a research prototype rather than a clinical device.
- `inference.py` — tested functions to load the bundle and run a single image or a batch of images with deterministic preprocessing and class probabilities.
- `requirements-inference.txt` — minimum tested Python/package requirements, avoiding pinned Kaggle-only paths.
- `SHA256SUMS.txt` and `artifact_manifest.json` — checksums and bundle metadata.

For convenience, optionally export TorchScript or another portable runtime format only after verifying numerical parity with the PyTorch reference on a fixed validation sample. Keep `state_dict` + reproducible Python architecture as the authoritative fallback. Do not assume export works without a test.

### Web-app interface contract

Implement and test a minimal inference API adapter (not a full product UI unless separately requested) that accepts an uploaded chest X-ray image, validates its type/size, applies the exact bundled preprocessing, returns class probabilities and the prediction, and includes the model version. Keep research metrics and explanation images separate from the clinical-looking prediction response. Return a clear error for unsupported or unreadable files. Do not return patient metadata.

If XAI is enabled in the inference bundle, make heatmap generation optional and clearly labelled “model attention/attribution visualization—not a diagnostic explanation.” Do not show Grad-CAM by default as if it confirms disease. Verify that XAI output generation does not modify model weights or prediction probabilities.

### Validation tests for the bundle

- Load the saved bundle in a fresh Python process with only the documented inference dependencies.
- Check exact model state loading and architecture compatibility.
- Run inference on a fixed non-identifying validation image set and verify finite probabilities, shape `(N, 2)`, normalized probabilities, and stable class mapping.
- Compare loaded-bundle probabilities with in-notebook reference probabilities within a documented numerical tolerance.
- Verify deterministic preprocessing and model version reporting.
- Test invalid/blank/corrupt file behavior.
- Create `webapp_bundle_validation_report.json` with results and environment information.
- Never package the dataset, full image files, hidden labels, optimizer histories, secrets, or unnecessary user/patient data into the bundle.

## 9. Files and reproducibility deliverables

Write all results to a fresh run-specific folder and keep the raw output unmodified. Required deliverables:

- `KAGGLE_RUN_GUIDE.md` with exact Kaggle setup/run/resume instructions
- `REVISION_AUDIT.md`
- `EXPERIMENT_PROTOCOL.md`
- `experiment_config.yaml`
- `split_manifest.csv` and a split audit summary
- `environment.txt` or `pip_freeze.txt`, GPU/device details, library versions, code commit/hash, run logs
- per-epoch train/validation logs and loss/metric histories
- per-epoch resumable checkpoints and per-step recovery checkpoints at the configured interval, plus immutable per-stage/final snapshots with run/config/split metadata and checkpoint index/checksums
- a validated `webapp_bundle/<model_version>/` with inference-only weights, preprocessing/label/decision configs, loader/predict function, model card, requirements and checksums
- XAI heatmaps, raw attribution arrays, image-selection manifests, sanity/stability/occlusion reports and optional RSNA box-alignment metrics
- per-image internal-test and external prediction CSVs
- seed-level and aggregated metrics CSV/JSON
- CSV/LaTeX versions of Tables 1–7 and supplementary tables
- plotting scripts and figures for main text and supplement in PDF/SVG + 300-dpi PNG/TIFF where appropriate
- `RESULTS_README.md` explaining how every table and figure is regenerated from raw predictions
- an updated notebook/source code with stale-results loading removed, test firewall implemented, and the new external RSNA evaluation module
- a concise final audit listing completed items, failed/unrun items, exact commands, artifact paths, and any unresolved limitations.

## 10. Manuscript revision rules

After—and only after—new experiments complete successfully:

- Correct the title spelling to “Pneumonia.”
- Use only `PneuSSLNet`; replace every `P4SSLNet` instance and verify with a document-wide search.
- Rewrite the abstract and conclusion around the controlled four-method SSL benchmark and label-efficiency analysis. Do not lead with the old headline accuracy until it is validly reproduced under the corrected protocol.
- Update the method section and Figure 1 to exactly match the implemented backbone initializations, projection/classification head, DINO views/loss, freeze/unfreeze modules, data split, and evaluation protocol.
- Explain the new 70/15/15 internal split, label counts, duplicate audit, seeds, uncertainty analysis, external RSNA label mapping, and locked evaluation procedure.
- Move the protocol caveat immediately above Table 6 (state-of-the-art comparison).
- Remove duplicated/overlapping main tables and follow the revised Table 1–7 plan.
- Update the limitations: single-source pediatric development data; RSNA domain/population and label-definition differences; image-level annotation limitations; no prospective or clinical deployment validation; possible site/acquisition shift; compute/reproducibility limitations.
- Do not claim clinical readiness, radiologist replacement, or clinical utility from retrospective classification metrics alone.
- Verify every reference, DOI, claim, table, caption, unit, and reported percentage. The abstract, main text, tables, figures, supplementary material, and conclusion must all agree exactly.

## 11. Acceptance criteria — do not call the revision complete until all applicable items pass

- [ ] No legacy result/checkpoint is silently reused.
- [ ] Split manifest and config hashes are saved and overlap/duplicate checks pass.
- [ ] Validation contains enough images for meaningful selection; exact class counts are reported.
- [ ] SSL pretraining uses training-split images only; no validation, internal test, or RSNA images enter training.
- [ ] Test data are absent from training loops and are evaluated only after the protocol is frozen.
- [ ] Primary metric uses final epoch, no TTA, fixed threshold 0.5; any secondary procedure is fully predeclared and separately reported.
- [ ] Every primary method × backbone has at least 3 seeds (5 preferred), with mean ± SD and retained seed-level metrics.
- [ ] ImageNet-supervised baselines are included or the manuscript explicitly narrows its claim and explains the missing control.
- [ ] Confusion matrix, per-class precision/recall/F1/support, AUROC/AUPRC, sensitivity/specificity and uncertainty are generated from final predictions.
- [ ] Label-efficiency and ablation studies use fixed splits, fair comparisons, repeated seeds and no test-driven choices.
- [ ] RSNA DICOM processing, unique-image aggregation, label mapping, cohort counts, and overlap checks pass; the negative class is not mislabelled as entirely healthy.
- [ ] Every published table/figure is generated from saved results and reproducible plotting code.
- [ ] The experiment executes in Kaggle from mounted inputs, writes only to `/kaggle/working`, reports the GPU/environment, and provides a reproducible run/resume guide.
- [ ] Each experiment/seed/stage saves validated full recovery checkpoints at every epoch and configured step intervals, with unique names, state/optimizer/RNG metadata, split/config hashes, and an index/checksum.
- [ ] The validation-selected web-app bundle loads in a fresh process and passes probability parity and invalid-input tests.
- [ ] XAI is included only with method/layer details, raw heatmaps, predeclared case selection, sanity/stability checks, false-positive/false-negative examples, and cautious claims.
- [ ] Old conflicting numbers are removed unless a valid new run independently reproduces them.
- [ ] Manuscript naming, title spelling, table structure, methods, captions, results and conclusion are consistent.
- [ ] The final report distinguishes completed experiments from blocked/unrun experiments and includes evidence/artifact paths.

**Execution order:** (1) set up and verify the Kaggle environment and mounted inputs, (2) audit all sources, (3) freeze protocol/config, (4) implement the split and test firewall, (5) unit/smoke tests in a separate quick-test directory, (6) full repeated-seed benchmark with epoch/step checkpointing, (7) label-efficiency and ablations, (8) lock all choices, (9) one final internal test evaluation, (10) independent RSNA external evaluation, (11) XAI sanity/stability analyses, (12) statistical analysis and figures, (13) build and validate the web-app bundle, (14) regenerate tables and supplementary files, (15) revise/proofread the manuscript, (16) run the acceptance checklist and report honestly what remains incomplete.
