# Unified Multi-Modal Medical Imaging Diagnostic System

## Overview

A comprehensive AI system that accepts any medical image (MRI, X-ray, retinal fundus, CT, etc.), automatically classifies its modality, and routes it to a specialized deep learning model for diagnosis with explainable AI (Grad-CAM).

## Architecture

- **Router Model:** Classifies input image modality
- **8 Specialist Models:** One for each medical imaging task
- **Explainability Layer:** Grad-CAM heatmaps for every prediction
- **Robustness:** Multi-layer OOD detection and confidence thresholding

## Team

- Anish (VIT Bhopal)
- Roneet (VIT Bhopal)
- [Additional members]

## Datasets

- Brain Tumor MRI: BraTS 2024 (4,500 scans)
- Diabetic Retinopathy: EyePACS (88,702 images)
- Pneumonia: MIMIC-CXR (377,110 images)
- Bone Fracture: MURA (40,895 images)
- Skin Lesion: ISIC 2024 (400,000 images)
- Kidney Stone: CT Kidney (12,446 images)
- Breast Histology: TCGA-BRCA (10,000+ images)
- COVID-19: VinDr-CXR (18,000+ images)

## Setup

See `SETUP_INSTRUCTIONS.md`

## Training

See `TRAINING_GUIDE.md` in docs/

## Deployment

Deployed on Hugging Face Spaces. See `DEPLOYMENT_GUIDE.md`.

## Review Dates

- Review 0: Jul 13-15
- Review 1: Aug 19-21
- Review 2: Sep 15-17
- Final Review: Oct 5-7
- Report Due: Oct 9

## License

MIT