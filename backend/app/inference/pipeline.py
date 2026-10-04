"""
Orchestrates the full inference pipeline: Stage 1 gatekeeper -> Stage 2
modality classifier -> routed specialist -> final prediction.
"""

import torch
import torch.nn.functional as F

from app.config import (
    ROUTER_STAGE1_PATH,
    ROUTER_STAGE2_PATH,
    SPECIALISTS_DIR,
    SPECIALIST_REGISTRY,
    CONFIDENCE_THRESHOLD,
)
from app.inference.model_loader import load_efficientnet_checkpoint, IncompatibleCheckpointError
from app.inference.preprocess import inference_transform
from app.gradcam import explain as gradcam_explain


class ModelRegistry:
    """Holds every loaded model in memory for the lifetime of the app.
    Populated once at startup -- requests just do dict lookups, no disk I/O."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stage1_model = None
        self.stage1_classes = None
        self.stage2_model = None
        self.stage2_classes = None
        self.specialists = {}  # router_class_name -> (model, class_names)

    def load_all(self):
        print(f"Loading models on device: {self.device}")

        self.stage1_model, self.stage1_classes = load_efficientnet_checkpoint(
            ROUTER_STAGE1_PATH, self.device, model_name="router_stage1"
        )
        self.stage2_model, self.stage2_classes = load_efficientnet_checkpoint(
            ROUTER_STAGE2_PATH, self.device, model_name="router_stage2"
        )

        for modality, filename in SPECIALIST_REGISTRY.items():
            checkpoint_path = SPECIALISTS_DIR / filename
            if not checkpoint_path.exists():
                print(f"[warning] Specialist checkpoint missing for '{modality}': "
                      f"{checkpoint_path} -- this modality will fail at request time "
                      f"until the file is present.")
                continue
            try:
                model, class_names = load_efficientnet_checkpoint(
                    checkpoint_path, self.device, model_name=f"specialist:{modality}"
                )
                self.specialists[modality] = (model, class_names)
            except IncompatibleCheckpointError as e:
                print(f"[SKIPPED] {e}\n")
                # This modality stays unavailable -- run_inference() already
                # handles "modality not in registry.specialists" gracefully,
                # returning a clean message instead of crashing the request.

        print(f"Startup complete. {len(self.specialists)}/{len(SPECIALIST_REGISTRY)} "
              f"specialists loaded.")


def _predict(model, class_names, image_tensor, device):
    """Runs one forward pass, returns (predicted_label, confidence, all_class_probs)."""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        all_probs = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}
    return class_names[pred_idx], confidence, all_probs


def run_inference(pil_image, registry: ModelRegistry, explain: bool = False, explain_method: str = "gradcam"):
    """
    Full pipeline: gatekeeper -> modality classifier -> specialist.
    Returns a plain dict matching the AnalyzeResponse schema.
    """
    image_tensor = inference_transform(pil_image.convert("RGB"))

    # --- Stage 1: medical vs non-medical ---
    stage1_label, stage1_confidence, _ = _predict(
        registry.stage1_model, registry.stage1_classes, image_tensor, registry.device
    )
    is_medical = stage1_label == "medical"

    if not is_medical:
        return {
            "is_medical": False,
            "gatekeeper_confidence": stage1_confidence,
            "modality": None,
            "modality_confidence": None,
            "prediction": None,
            "prediction_confidence": None,
            "low_confidence_warning": False,
            "message": "Image was not identified as a medical scan. No diagnosis performed.",
        }

    # --- Stage 2: which modality ---
    modality, modality_confidence, _ = _predict(
        registry.stage2_model, registry.stage2_classes, image_tensor, registry.device
    )

    if modality not in registry.specialists:
        return {
            "is_medical": True,
            "gatekeeper_confidence": stage1_confidence,
            "modality": modality,
            "modality_confidence": modality_confidence,
            "prediction": None,
            "prediction_confidence": None,
            "low_confidence_warning": False,
            "message": f"Modality '{modality}' was identified, but its specialist model "
                       f"isn't loaded (missing checkpoint). No diagnosis performed.",
        }

    # --- Specialist: the actual diagnosis ---
    specialist_model, specialist_classes = registry.specialists[modality]
    prediction, prediction_confidence, all_probs = _predict(
        specialist_model, specialist_classes, image_tensor, registry.device
    )

    low_confidence = (
        modality_confidence < CONFIDENCE_THRESHOLD
        or prediction_confidence < CONFIDENCE_THRESHOLD
    )

    result = {
        "is_medical": True,
        "gatekeeper_confidence": stage1_confidence,
        "modality": modality,
        "modality_confidence": modality_confidence,
        "prediction": prediction,
        "prediction_confidence": prediction_confidence,
        "all_class_probabilities": all_probs,
        "low_confidence_warning": low_confidence,
        "message": "OK",
    }

    if explain:
        method = "gradcam++" if explain_method in ("gradcampp", "gradcam++") else "gradcam"
        try:
            result["explanation"] = gradcam_explain(
                specialist_model,
                pil_image.convert("RGB"),
                preprocess=inference_transform,      # identical preprocessing to the prediction above
                class_names=specialist_classes,
                method=method,
            )
        except Exception as e:  # an explanation failure must never break the diagnosis itself
            print(f"[warning] Grad-CAM failed for '{modality}': {e}")
            result["explanation_error"] = str(e)

    return result
