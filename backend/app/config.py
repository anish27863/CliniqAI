"""
Central configuration: model paths and the router-to-specialist mapping.

This is the ONE place that knows which checkpoint file corresponds to which
router-predicted modality. Adding an 8th specialist later means adding one
line here -- nothing else in the pipeline needs to change.
"""

from pathlib import Path

# Root of the models folder. In local dev this is your existing
# E:\programming\CliniqAI\models structure. When deploying to Hugging Face
# Spaces, this whole models/ folder gets committed into the Space repo via
# git-lfs (see DEPLOY.md), so this relative path still resolves correctly.
MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"

ROUTER_STAGE1_PATH = MODELS_ROOT / "router" / "router_stage1_v5_efficientnet_b0.pt"
ROUTER_STAGE2_PATH = MODELS_ROOT / "router" / "router_stage2_v2_efficientnet_b0.pt"

SPECIALISTS_DIR = MODELS_ROOT / "specialists"

# Router Stage 2 class name -> specialist checkpoint filename.
# Router class names and specialist filenames don't match 1:1 (e.g. router
# says "brain_mri", the checkpoint is "brain_tumor_specialist_v1.pt") --
# that's expected, this mapping is exactly what bridges that gap.
SPECIALIST_REGISTRY = {
    "brain_mri": "brain_tumor_specialist_v2.pt",
    "kidney_ct": "kidney_specialist_v1.pt",
    "retinal": "retinal_specialist_v1.pt",
    "pneumonia": "pneumonia_specialist_v1.pt",
    "skin_lesion": "skin_lesion_specialist_v1.pt",
    "breast_histology": "breast_histology_specialist_v2.pt",
    "bone_fracture": "bone_fracture_specialist_v1.pt",
}

# Below this confidence, flag the prediction as low-confidence rather than
# presenting it to the frontend as a clean, trustworthy result.
CONFIDENCE_THRESHOLD = 0.90

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
