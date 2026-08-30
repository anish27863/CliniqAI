# CliniqAI Project Log

HEAD
**Project:** Unified Multi-Modal Medical Imaging Diagnostic System  
**Team:** Anish, Roneet, [3rd member]  
**Status:** Active Development  
**Last Updated:** July 25, 2026

---

## Overview

CliniqAI is a graded project exhibition course deliverable implementing an adaptive multi-modal medical imaging diagnostic system using Mixture of Experts (MoE) with hard routing. The system routes medical images to specialized diagnostic models (specialists) based on modality detection, with Grad-CAM explainability for clinical interpretability.

**Architecture:**
- **Router Model:** Classifies medical imaging modality (MRI Brain, CT Kidney, Fundus Retinal)
- **Specialist Models:** Task-specific classifiers for each modality
- **Explainability:** Grad-CAM visualization for model predictions
- **Deployment:** FastAPI backend + Next.js frontend

---

## Week 1 (Jul 15 - Jul 28)

### Project Conception & Planning
- Initial brainstorming and requirements gathering
- Reviewed existing brain tumor MRI classifier (EfficientNet-B0, FastAPI + Next.js, deployed on HF Spaces + Vercel)
- Identified OOD detector (v2) with hard negatives training for production safety
- Designed multi-modal routing architecture (Mixture of Experts with hard routing)
- Selected three medical imaging modalities:
  1. **Brain Tumor MRI** — Existing classifier foundation
  2. **Retinal Fundus** — Diabetic Retinopathy (EyePACS dataset)
  3. **Kidney CT** — Stone/Cyst/Tumor/Normal detection (Kaggle dataset)

### Infrastructure Setup
- Repository created: [anish27863/CliniqAI](https://github.com/anish27863/CliniqAI)
- Git branching strategy established:
  - `main` — Production/stable releases
  - `develop` — Integration branch
  - `feature/*` — Feature branches (e.g., `feature/anish-specialists`)
- Project directory structure created:
  ```
  E:\programming\CliniqAI/
  ├── data/
  │   ├── raw/
  │   │   ├── router_dataset/
  │   │   ├── diabetic_retinopathy/
  │   │   └── kidney_stone/
  │   └── processed/
  ├── models/
  │   ├── router/
  │   ├── specialists/
  │   └── checkpoints/
  ├── scripts/
  ├── notebooks/
  ├── logs/
  ├── frontend/ (Next.js)
  └── backend/ (FastAPI)
  ```
- Google Drive folder structure for team collaboration:
  - `medical-ai-models/trained_models/` — Model checkpoints
  - `medical-ai-models/evaluation_metrics/` — Performance results
  - `medical-ai-models/gradcam_outputs/` — Explainability visualizations
  - `medical-ai-models/datasets/` — Data references
  - `medical-ai-models/deployment_versions/` — Production releases

### Dataset Preparation
- **Router Dataset:** 900 images (3 classes)
  - Brain MRI: 300 images
  - CT Kidney: 300 images
  - Fundus Retinal: 300 images
- **Diabetic Retinopathy:** 3,662 images (5 severity classes)
  - Train: 2,930 | Val: 366 | Test: 366
  - Classes: No DR, Mild, Moderate, Severe, Proliferative
- **Kidney Stone CT:** 12,446 images (4 classes)
  - Cyst: 3,709 | Normal: 5,077 | Stone: 1,377 | Tumor: 2,283
  - Classes: Cyst, Normal, Stone, Tumor

---

## Week 2 (Jul 29 - Aug 4)

### Anish's Work

#### Phase 1: Router Model Training ✅
- **Objective:** Train MoE router to classify medical imaging modality
- **Architecture:** EfficientNet-B0 with 3-class output
- **Dataset:** 900 images (300 per class) with stratified train/val splits
- **Results:**
  - Training completed with **100% validation accuracy**
  - Checkpoint: `models/router/router_v1.pt`
  - Git branch: `feature/router` → merged to `develop`
  - Uploaded to Drive: `medical-ai-models/trained_models/router_v1.pt`

#### Phase 2: Retinal Specialist Training ✅
- **Objective:** Train specialist for diabetic retinopathy classification
- **Architecture:** EfficientNet-B0, 5-class classifier
- **Dataset:** 3,662 fundus images (stratified splits)
  - Train: 2,930 | Val: 366 | Test: 366
- **Training Configuration:**
  - Batch size: 32 (RTX 4050 optimization)
  - Learning rate: 1e-3, weight decay: 1e-4
  - Early stopping: patience=10 epochs
  - Image size: 224×224
- **Results:**
  - Best Val Accuracy: **81.15%**
  - Test Accuracy: **79.51%**
  - Test Precision: 79.07%
  - Test Recall: 79.51%
  - Test F1 Score: 0.7610
  - Checkpoint: `models/specialists/retinal_specialist_v1.pt`
  - Git branch: `feature/anish-specialists`
  - Uploaded to Drive: `medical-ai-models/trained_models/retinal_specialist_v1.pt`

#### Phase 3: Kidney Stone Specialist Training ✅
- **Objective:** Train specialist for kidney CT classification
- **Architecture:** EfficientNet-B0, 4-class classifier
- **Dataset:** 12,446 kidney CT images (stratified splits)
  - Train: 8,960 | Val: 996 | Test: 2,490
  - Classes: Cyst, Normal, Stone, Tumor
- **Training Configuration:**
  - Batch size: 32 (RTX 4050 optimization)
  - Learning rate: 1e-3, weight decay: 1e-4
  - Early stopping: patience=10 epochs
  - Image size: 224×224
- **Results:**
  - Best Val Accuracy: **98.80%**
  - Test Accuracy: **98.31%**
  - Test Precision: 98.39%
  - Test Recall: 98.31%
  - Test F1 Score: 0.9833
  - Checkpoint: `models/specialists/kidney_specialist_v1.pt`
  - Git branch: `feature/anish-specialists`
  - Uploaded to Drive: `medical-ai-models/trained_models/kidney_specialist_v1.pt`

#### Technical Artifacts Created
- `scripts/build_retinal_dataset.py` — Retinal dataset loader with nested folder handling
- `scripts/build_kidney_dataset.py` — Kidney CT dataset loader with stratified splits
- `scripts/train_retinal_specialist.py` — Retinal specialist trainer (EfficientNet-B0)
- `scripts/train_kidney_specialist.py` — Kidney specialist trainer (EfficientNet-B0)
- `notebooks/router_training.ipynb` — Router training notebook
- `notebooks/retinal_specialist_training.ipynb` — Retinal specialist training notebook
- `notebooks/kidney_specialist_training.ipynb` — Kidney specialist training notebook

#### Git & Version Control
- Created branches:
  - `feature/router` → merged to `develop` (now deleted)
  - `feature/anish-specialists` → active, contains all specialist work
- Commits:
  - "feat: router training pipeline (100% val accuracy)"
  - "feat: retinal specialist training pipeline (79.5% test accuracy)"
  - "feat: kidney stone specialist training pipeline (98.3% test accuracy)"

### Roneet's Work

#### Environment Setup ✅
- Machine specs: Core i7-14th gen, RTX 5050 (8GB VRAM), 16GB DDR5 RAM, 512GB SSD
- Cloned repository: `https://github.com/anish27863/CliniqAI.git`
- Created Conda environment from `environment.yml`:
  - PyTorch 2.5.0 + CUDA 12.4
  - Key packages: numpy 1.26.4, pandas 2.2.0, scikit-learn 1.3.2, xgboost 2.0.3
- Configured `.env` file with credentials and paths
- Verified connections: MongoDB, Google Drive API, PyTorch/CUDA

#### Dataset Preparation - In Progress 🔄
- Downloading datasets from Kaggle:
  - Brain Tumor MRI — queued
  - Additional datasets for enhanced training — TBD
- Expected completion: End of Week 2

---

## Week 3 (Aug 5 - Aug 11) - Planned

### Remaining Tasks

#### Anish
- [ ] **Brain Tumor Specialist Training**
  - Build dataset loader for brain MRI tumor classification
  - Train EfficientNet-B0 specialist (4 classes: No Tumor, Glioma, Meningioma, Pituitary)
  - Expected test accuracy: 85-90% (existing classifier baseline: 95%+)
- [ ] **Router Integration & Testing**
  - Implement hard routing logic (router → specialist selection)
  - Test end-to-end inference pipeline
  - Validate routing accuracy across all three modalities
- [ ] **Grad-CAM Explainability**
  - Implement Grad-CAM for all specialist models
  - Add visualization endpoint to FastAPI backend
  - Create interactive frontend for explainability display

#### Roneet
- [ ] **Dataset Download & Validation**
  - Complete Kaggle dataset downloads
  - Verify image integrity and format consistency
  - Organize datasets into standard structure
- [ ] **Parallel Specialist Training**
  - Train Brain Tumor specialist (if dataset ready)
  - Benchmark on RTX 5050 (8GB VRAM)
  - Compare performance with Anish's RTX 4050 results

#### Both
- [ ] **Deployment Preparation**
  - Update FastAPI backend with router + specialist inference
  - Integrate Grad-CAM visualization endpoints
  - Test frontend-backend communication
- [ ] **Performance Optimization**
  - Model quantization (if needed for latency)
  - Batch inference optimization
  - Memory profiling on deployment targets

---

## Performance Summary

| Model | Task | Test Accuracy | F1 Score | Status |
|-------|------|---------------|----------|--------|
| Router | Modality Classification | 100.00% | — | ✅ Complete |
| Retinal Specialist | Diabetic Retinopathy | 79.51% | 0.7610 | ✅ Complete |
| Kidney Specialist | CT Classification | 98.31% | 0.9833 | ✅ Complete |
| Brain Tumor Specialist | MRI Classification | 89.81% | 0.8952 | ✅ Complete |
---

## Hardware & Environment

**Anish's Machine:**
- CPU: Intel i5-13450HX
- GPU: NVIDIA RTX 4050 Laptop (6GB VRAM)
- RAM: 16GB DDR5
- OS: Windows
- Conda env: `CliniqAI`

**Roneet's Machine:**
- CPU: Core i7-14th gen
- GPU: NVIDIA RTX 5050 (8GB VRAM)
- RAM: 16GB DDR5
- SSD: 512GB
- OS: Windows
- Conda env: `CliniqAI`

---

## Key Technologies & Libraries

- **Deep Learning:** PyTorch 2.5.0, torchvision
- **Model Architecture:** EfficientNet-B0 (transfer learning)
- **Training:** Adam optimizer, ReduceLROnPlateau scheduler, early stopping
- **Evaluation:** scikit-learn (accuracy, precision, recall, F1, confusion matrix)
- **Explainability:** Grad-CAM (visual explanations)
- **Backend:** FastAPI, Hugging Face Spaces
- **Frontend:** Next.js, Vercel
- **Data Management:** Google Drive API, Kaggle API
- **Version Control:** Git, GitHub

---

## Grading Rubric Alignment (Project Exhibition)

### Planned Reviews
- **Review 1 (Aug 5-10):** Router + 2 Specialists (Retinal, Kidney) ✅
  - Expected: Positive feedback on specialist performance & architecture
- **Review 2 (Aug 19-24):** Brain Tumor Specialist + Router Integration
  - Expected: Integrated system with hard routing working end-to-end
- **Final Demo (Early October):** Complete system with explainability
  - Expected: Clinical interpretability + deployment readiness

### Portfolio-Worthy Elements
1. ✅ Novel architecture (Mixture of Experts with hard routing)
2. ✅ Multi-modal medical imaging (3 modalities, 4 specialist models)
3. ✅ High-performance specialists (98.31% accuracy on kidney CT)
4. ✅ Explainability layer (Grad-CAM for clinical transparency)
5. ⏳ Production deployment (FastAPI + Next.js integration)
6. ⏳ End-to-end inference pipeline (router → specialist → explanation)

---

## Notes

- All model checkpoints backed up to Google Drive for team collaboration
- Dataset builders handle nested folder structures automatically
- Early stopping prevents overfitting; patience=10 allows sufficient training
- Batch size optimized for 6GB VRAM (RTX 4050); Roneet's RTX 5050 (8GB) can handle larger batches if needed
- Next focus: Brain Tumor specialist + integration testing

---

**Team Contact:** Anish, Roneet  
**Repository:** https://github.com/anish27863/CliniqAI  
**Drive Folder:** My Drive > medical-ai-models  
**Review Schedule:** [TBD by course coordinator]
=======
## Week 1 (Jul 23-28)

### Anish
- [x] GitHub repo setup
- [x] Environment setup (CliniqAI)
- [x] MongoDB connection verified
- [x] Google Drive API verified
- [x] Data directory structure created
- [x] Kaggle CLI configured
- [x] Download EyePACS (8 GB) ✅
- [x] Download CT Kidney (0.6 GB) ✅
- [ ] Download CIFAR-10 (hard negatives) — IN PROGRESS
- [ ] Verify all data downloaded
- [ ] Create data preprocessing scripts
- [ ] Organize splits (train/val/test)

### Roneet
- [ ] Laptop setup complete
- [ ] Environment created
- [ ] Connections verified
- [ ] Clone CliniqAI repo

### Progress
- Downloaded: 8.6 GB / ~15 GB (Week 1 target)
- Status: On track

### Next Steps
- Complete CIFAR-10 download
- Create data preprocessing pipeline
- Generate train/val/test splits
- Start router model training (Week 2)
 develop
# CliniqAI — Handoff Addendum (v4 → v5)

**Purpose:** append to `CliniqAI — Project Handoff v4`. Everything in v4 still stands except
where noted below. This session did not touch the gatekeeper bug, kidney bug, or Stage 2
confusion bug directly — it opened a new, higher-priority thread: **replacing the
breast_histology specialist's training data entirely.**

---

## New this session: breast_histology dataset replacement

**Decision made:** the IDC dataset (50×50px patches upscaled 4.5x) that both the
`breast_histology` specialist and the gatekeeper's `breast_histology` positive class were
trained on is being retired — not patched further with augmentation. Quantity was fine (large
dataset), quality was the actual flaw: tiny, blocky patches with no real magnification variety.

**New dataset selected: BreakHis** (Spanhol et al., IEEE TBME 2016). Kaggle mirror
`ambarish/breakhis` (select `BreaKHis_v1`). 7,909 real histopathology images, 700×460px RGB,
four magnification levels (40X/100X/200X/400X), 82 patients, benign/malignant across 8
histological subtypes (Adenosis, Fibroadenoma, Tubular Adenoma, Phyllodes Tumor / Ductal
Carcinoma, Lobular Carcinoma, Mucinous Carcinoma, Papillary Carcinoma). Free for non-commercial
research/educational use — cite the original paper if used in the report.

**Class semantics change — action required before deploy:** BreakHis's native labels are
`benign`/`malignant` (tumor-level diagnosis), not `idc_negative`/`idc_positive` (invasive ductal
carcinoma presence in a patch — a different clinical question). The new notebook defaults to
binary `benign`/`malignant`. An 8-subtype multiclass mode is built in
(`MULTICLASS_SUBTYPES = True` in Step 1) but not the default. **Whichever is used,
`backend/app/schemas.py` and any frontend label-display logic need updating** before this
checkpoint is pointed at from `config.py` — the old IDC-style output format doesn't apply.

**Deliverable produced:** `breast_histology_v2_pipeline.ipynb` — a single, run-top-to-bottom,
fully automated notebook covering the entire pipeline end to end:

1. Kaggle download (pinned `kaggle==1.6.17`, per the established MemoryError-on-large-datasets
   fix)
2. Filename-based parsing of label / subtype / magnification / patient_id via regex on the
   official `SOB_{B|M}_{SUBTYPE}-{patient_id}-{mag}-{seq}.png` convention, with directory-based
   fallback for any mismatched files
3. Corrupted-file pre-validation pass (established project convention)
4. **Patient-grouped** train/val/test split (~80/10/10), stratified by label — BreakHis has
   multiple images per patient across magnifications, so an image-level split would leak
   near-duplicate tissue across sets and inflate eval numbers
5. Standalone `scripts/breast_histology_dataset_v2.py` Dataset module, written to disk by the
   notebook itself (Windows `num_workers>0` multiprocessing requires this to live outside the
   notebook)
6. Standard project training recipe: EfficientNet-B0, **simple head**
   (`Dropout→Linear(1280,num_classes)` — matches the existing checkpoint group so
   `model_loader.py`'s `_build_matching_head()` needs no changes), `WeightedRandomSampler` +
   weighted `CrossEntropyLoss` for the ~2.2x benign/malignant imbalance, non-deprecated AMP
   (`torch.amp.autocast`/`GradScaler`), `channels_last`, `num_workers=4` /
   `pin_memory=True` / `persistent_workers=True` / `prefetch_factor=4`,
   `torch.backends.cudnn.benchmark=True`
7. Per-epoch checkpointing to `.checkpoint_inprogress.pt` on val_loss improvement, with a resume
   cell for kernel-restart recovery
8. Final save to both `models/specialists/breast_histology_specialist_v2.pt` and the
   `backend/models/specialists/` copy, plus a companion `.history.json`
9. Test-set evaluation broken down **overall, per-magnification, and per-subtype** — not a
   single aggregate number
10. Real-world sanity check against whatever's in `data/raw/manual_sanity_check/` (the same
    folder used for gatekeeper sanity testing), since in-distribution accuracy alone has already
    been shown (on the gatekeeper) not to guarantee real-world generalization

**Status: written, not yet run.** No checkpoint exists yet from this pipeline. Nothing has been
committed — the notebook and `breast_histology_dataset_v2.py` should go on a new branch (e.g.
`feature/breast-histology-v2`) once training succeeds and the sanity check looks right, following
the same discipline as the gatekeeper v3 checkpoints (don't point `config.py` at it until it
passes real-world sanity, not just in-distribution accuracy).

**Bonus angle noted, not yet acted on:** BreakHis's four native magnification levels are real
zoom-scale variety — exactly what the gatekeeper's old 50×50-patch-only positive class was
missing. Once this specialist is trained and validated, swapping BreakHis in as the gatekeeper's
`breast_histology` positive-class source is a plausible fix for the zoom/crop-scale hypothesis
from v4 — likely worth trying **before** a third resolution-only fine-tune attempt on the
gatekeeper. This is a separate task, not part of this notebook.

---

## Everything else from v4 — unchanged, still open

- **Gatekeeper breast_histology rejection bug** — still open. Two fine-tune attempts (resolution
  augmentation only) both failed on the real-world sanity image. The crop-scale diagnostic test
  from v4 (crop the sanity image to patch-scale, re-test) was never run this session — still the
  next actionable step *if* the BreakHis-swap approach above isn't tried first.
- **Kidney specialist label-order bug** — still fully open, untouched.
  `Select-String -Path "E:\programming\CliniqAI\scripts\build_kidney_dataset.py" -Pattern "class_to_idx" -Context 3,3`
  still hasn't been run.
- **Stage 2 kidney/brain_mri confusion bug** — still fully open, untouched. Lowest priority.
- **Open PRs** (`feature/router`, `feature/backend-api` into `develop`) — still unconfirmed.
- **Google Drive sync** of `brain_tumor_specialist_v2.pt` / gatekeeper v2 checkpoint — still
  unconfirmed.
- **Uncommitted from v4**, still uncommitted: the v3 gatekeeper fine-tune notebook, the updated
  `gatekeeper_dataset_v2.py` with `RandomResolutionDegradation`, the crop-test cell, and now also
  this session's `breast_histology_v2_pipeline.ipynb` + `breast_histology_dataset_v2.py`.
- `bugs.txt` — still untracked, still unclear if it's scratch or meant to be tracked.

## Priority order going into next session

1. Run `breast_histology_v2_pipeline.ipynb` — download, train, evaluate, sanity-check. This is
   the only item with a ready-to-run deliverable sitting untouched.
2. Kidney label-order check — one command, cheap, still not run since v4.
3. Decide gatekeeper fix path: BreakHis-swap (new, likely better) vs. the crop-scale diagnostic
   from v4 (older, narrower). Don't run both blind — pick one based on what's learned from step 1.
4. Stage 2 kidney/brain_mri confusion retrain — lowest priority, unchanged.
5. Commit sweep: everything listed under "Uncommitted from v4" above, once it's not mid-experiment.