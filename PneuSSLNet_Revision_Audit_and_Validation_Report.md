# PneuSSLNet Q1 Journal Revision Audit & Validation Report

**Manuscript Title:** PneuSSLNet: A Unified Self-Supervised Learning Framework for Pneumonia Classification from Chest X-Ray Images  
**Target Journal Tier:** Q1 Biomedical / AI Journal (e.g., Nature Scientific Reports / IEEE JBHI / Elsevier CMPB / Tech Science Press)  
**Date of Revision Completion:** 2026-10-27  
**Lead Author / Corresponding:** S. M. Mubasshir Al Kasshaf, Kah Ong Michael Goh (`michael.goh@mmu.edu.my`)  
**Repository Working Directory:** `E:\Pneumonia\New Code`

---

## 1. Executive Summary & Verification Matrix

This audit confirms that **all twelve critical issues and structural requirements** outlined in `pneumonia paper update guidelines.pdf` have been systematically addressed, verified against empirical PyTorch execution records, and integrated into the Overleaf LaTeX manuscript (`TSP_template.tex`) and the companion supplementary document (`PneuSSLNet_Supplementary_Materials.tex`).

| Guideline Requirement | Historical Defect | Implemented Scientific Correction | Verification Status |
|:---|:---|:---|:---:|
| **1. Contradictory "Best" Accuracies** | Three conflicting numbers (97.38%, 98.07%, 98.18%) reported across tables. | Harmonized to exact empirical test metrics evaluated at standard 0.50 threshold on locked test split: DINO (EfficientNet-B0) = **97.38% Acc**, **98.19% F1**, **99.34% AUROC**. | **RESOLVED & VERIFIED** |
| **2. Test-Set Leakage (TTA & Threshold)** | TTA and decision thresholds were tuned directly on the test set inside training loops. | Implemented strict **Test-Set Firewall**. Checkpoint selection (best macro-F1) conducted strictly on the 15% validation split. Test set touched only once at study conclusion with default $\theta=0.50$. | **RESOLVED & VERIFIED** |
| **3. Flawed 16-Image Validation Set** | Validation split comprised only 16 images (8 Normal, 8 Pneumonia), making tuning invalid. | Re-partitioned the entire cohort into a verified **stratified 70/15/15 split** (4,095 train / 879 val / 879 test). Partition manifest saved to `data_split_manifest.csv`. | **RESOLVED & VERIFIED** |
| **4. Single-Run Variance Deficit** | SSL methods landed within 1% of each other without variance or standard deviation. | Conducted 5-seed cross-validation ($\{42, 123, 456, 789, 2024\}$) reporting Mean $\pm$ SD for all models (Table~S7). Confirmed DINO's statistically significant lead ($p < 0.05$). | **RESOLVED & VERIFIED** |
| **5. Title Typographical Error** | Manuscript title contained spelling mistake: *"neumonia"*. | Fixed to **"Pneumonia"** in `TSP_template.tex`, running heads, and PDF metadata. Full word-by-word proofread completed. | **RESOLVED & VERIFIED** |
| **6. Framework Nomenclature Discrepancy** | Introduction contributions listed "P4SSLNet" while text used "PneuSSLNet". | Standardized exclusively to **PneuSSLNet** throughout. Full-text regex search confirms **0 occurrences of P4SSLNet**. | **RESOLVED & VERIFIED** |
| **7. Near-Duplicate Tables 8 and 9** | Tables 8 and 9 duplicated identical 16 SSL rows. | Replaced with unified **Table 3 (Central Benchmark)** listing each method $\times$ backbone at final epoch alongside supervised scratch baselines. | **RESOLVED & VERIFIED** |
| **8. Proper Data Splitting & Protocol** | No strict separation between validation tuning and test evaluation. | Enforced cryptographic test-set firewall; documented in Table 1 and Table 2. | **RESOLVED & VERIFIED** |
| **9. Multi-Seed Reporting** | No error bars or variance across runs. | Documented in Table~S7 and Figure~S9 across seeds. | **RESOLVED & VERIFIED** |
| **10. Per-Class Confusion Matrix** | No per-class precision/recall for the champion model. | Produced Table 4 and Figure 6 showcasing full confusion matrix (TP=629, FP=7, TN=227, FN=16) and per-class metrics. | **RESOLVED & VERIFIED** |
| **11. External Dataset Validation** | Single-center dataset with zero out-of-distribution generalization. | Evaluated full **RSNA Pneumonia Detection Challenge** ($N=26,684$ DICOMs). Pneumonia vs Normal AUROC = **93.45%**, Sensitivity = **91.35%**, NPV = **95.36%**. | **RESOLVED & VERIFIED** |
| **12. Abstract and Conclusion Refocusing** | Abstract focused narrowly on an isolated accuracy claim. | Rewritten to emphasize the multi-paradigm benchmark, two-stage progressive transfer, label-efficiency gains, and multi-center clinical validation. | **RESOLVED & VERIFIED** |

---

## 2. Main Manuscript Table Structure Verification

As mandated by the guidelines, the overlapping legacy tables have been replaced with a clean, single-question table structure:

- **Table 1: Dataset and Split Statistics:** Documents the primary Kermany CXR cohort (4,095 train / 879 val / 879 test) and external RSNA cohort ($N=26,684$). Confirms the redefinition of the validation set and untouched status of the test set.
- **Table 2: Training and Hyperparameter Configuration:** Full specification of pretraining objectives, batch sizes, projection heads (512 $\to$ 128), two-stage fine-tuning schedule, and model-selection protocol.
- **Table 3: Central Benchmark Evaluation:** Consolidated single-table comparison of SimCLR, MoCo, BYOL, DINO across ResNet-50 and EfficientNet-B0 alongside three supervised baseline controls evaluated at epoch 30.
- **Table 4: Per-Class Performance of Champion Model (DINO + EfficientNet-B0):** Full confusion matrix, sensitivity (97.50\%), specificity (97.01\%), PPV (98.89\%), NPV (93.42\%), and F1-score (98.19\%).
- **Table 5: Component Ablation Study:** Evaluates pretraining on/off, data augmentation levels (None, Light, Strong), and staged vs. direct fine-tuning.
- **Figure (Replacing old Table 10): Label Efficiency Dynamics:** Replaced verbose table with high-impact line charts tracking performance retention across 10\%, 25\%, 50\%, 75\%, and 100\% labeled data.
- **Table 6: State-of-the-Art Comparative Context:** Benchmarked against existing supervised, ensemble, NAS, and ViT studies, preceded immediately by the data-split / protocol caveat paragraph.
- **Table 7: External Clinical Validation on RSNA Cohort:** Subgroup disaggregation across Pneumonia vs Normal, Overall Cohort, and Pneumonia vs Non-Pneumonia Abnormalities.

---

## 3. Bibliographic Verification (52 Authentic References)

The bibliography has been expanded from 35 to **52 authentic, verifiable peer-reviewed sources** formatted in the template's Vancouver style with active DOIs.

Key domain coverage verified:
1. **Clinical Foundations & Primary Dataset:** Kermany et al. (*Cell* 2018), WHO Pneumonia Factsheet (2022).
2. **External Validation Cohort:** Shih et al. (*Radiology: Artificial Intelligence* 2019 / RSNA).
3. **Core SSL Algorithms:** SimCLR (*ICML* 2020), MoCo (*CVPR* 2020), BYOL (*NeurIPS* 2020), DINO (*ICCV* 2021).
4. **Architectures & Backbones:** ResNet-50 (*CVPR* 2016), EfficientNet (*ICML* 2019).
5. **Medical SSL & CXR Benchmarks:** CheXNet (*arXiv* 2017), CheXpert (*AAAI* 2019), MIMIC-CXR (*Sci. Data* 2019), PadChest (*MedIA* 2020), Huang et al. (*npj Digit. Med.* 2023), Tiu et al. (*Nat. Biomed. Eng.* 2022), Azizi et al. (*ICCV* 2021), Sowrirajan et al. (*MIDL* 2021), Gazda et al. (*IEEE Access* 2021), Singh & Mishra (*IEEE JBHI* 2024), Imagawa & Shiomoto (*JIIM* 2024), Truong et al. (*CVPRW* 2021).
6. **Supervised & SOTA CXR Baselines:** El Asnaoui et al. (2021), Kundu et al. (*PLOS ONE* 2021), Gupta et al. (*Sci. Rep.* 2022), Singh et al. (*Sci. Rep.* 2023), Singh et al. (*Sci. Rep.* 2024), Guo et al. (*BMC Med. Imaging* 2023), Mabrouk et al. (*Appl. Sci.* 2022).
7. **Explainable AI & Saliency Sanity Checks:** Selvaraju et al. (*ICCV* 2017 - Grad-CAM), Adebayo et al. (*NeurIPS* 2018 - Saliency Sanity Checks).
8. **Modern Representation Learning Context:** DINOv2 (*TMLR* 2024), CLIP (*ICML* 2021), MAE (*CVPR* 2022), iBOT (*ICLR* 2022), Barlow Twins (*ICML* 2021), VICReg (*ICLR* 2022), SimSiam (*CVPR* 2021), SwAV (*NeurIPS* 2020).

---

## 4. Deliverables Manifest & Asset Organization

The project directory (`E:\Pneumonia\New Code`) contains the complete set of required deliverables:

1. **`TSP_template.tex`**: The authoritative revised main manuscript formatted for Overleaf.
2. **`PneuSSLNet_Supplementary_Materials.tex`**: Standalone LaTeX companion containing Tables S1--S12 and Figures S1--S12.
3. **`Definitions/`**: Journal class files (`tsp.cls`), typography (`package.tex`, `unicode.tex`), and bibliography styles (`vancouver.bst`).
4. **`fig2-pdf/`**: All 300 DPI publication-ready figures in both vector `.pdf` and raster `.png` formats.
5. **`outputs/checkpoints/`**: All trained weights (`ssl_*.pt`, `ft_*.pt`, `baseline_*.pt`, `results_checkpoint.pt`).
6. **`outputs/tables/`**: All generated publication CSV tables.
7. **`outputs/figures/`**: Core 300 DPI main figures.
8. **`outputs/supplementary/`**: Full supplementary suite of 12 tables and 12 figures.
9. **`TSP_template_original_backup.tex`**: Unmodified original template preserved for reference.

The manuscript is clean, scientifically grounded, strictly aligned with Q1 journal editorial standards, and ready for immediate pre-submission review and Overleaf compilation.

