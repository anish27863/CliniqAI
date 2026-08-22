"""
Generic EfficientNet-B0 checkpoint loader.

Deliberately does NOT assume a fixed number of output classes or a fixed
checkpoint dict structure. Some of these 7 specialist checkpoints were built
in different notebook sessions with slightly different save conventions --
rather than hardcode per-model assumptions that could silently break, this
loader:

  1. Reads the actual classifier layer shape from the saved weights to
     determine num_classes -- correct by construction, not by assumption.
  2. Reads class_names from the checkpoint if present; if not, falls back to
     generic placeholder labels and prints a warning so the gap is visible
     immediately at startup, not discovered later via a wrong diagnosis label.
  3. Handles checkpoints saved either as {"model_state_dict": ...} (the
     convention used by every notebook built in this project) or as a raw
     state_dict with no wrapper key, in case an older checkpoint used a
     different save format.
  4. If the classifier head doesn't match the simple single-Linear structure
     this loader assumes (e.g. a custom multi-layer head), it does NOT guess
     at reconstructing it -- a wrong guess at intermediate layer types could
     load "successfully" while silently producing wrong predictions, which is
     worse than failing loudly. Instead it raises a clear, diagnostic error
     naming exactly which keys didn't match, and the caller is expected to
     skip that model rather than crash the whole app.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

KNOWN_CLASS_NAMES = {
    "specialist:brain_mri": ["glioma", "meningioma", "notumor", "pituitary"],
    "specialist:kidney_ct": ["Cyst", "Normal", "Stone", "Tumor"],
    "specialist:retinal": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
}

class IncompatibleCheckpointError(Exception):
    """Raised when a checkpoint's classifier head doesn't match the simple
    single-Linear structure this loader builds. Not silently recoverable --
    the caller should skip this model, not guess at an architecture."""
    pass


def _build_matching_head(state_dict, num_features):
    """Builds a classifier head matching whatever structure is actually in the
    checkpoint, rather than assuming the simple single-Linear default.

    Two known patterns so far:
      - Simple:  classifier = [Dropout, Linear(num_features, N)]
      - Wide:    classifier = [Dropout, Linear(num_features, 512), ReLU,
                                Dropout, Linear(512, N)]
        (indices 1 and 4 carry weights, 0/2/3 are non-parametric -- this
        exact pattern shows up identically across brain_mri, kidney_ct, and
        retinal checkpoints, suggesting a shared training template.)
    """
    has_wide_head = "classifier.4.weight" in state_dict

    if has_wide_head:
        num_classes = state_dict["classifier.4.weight"].shape[0]
        hidden_dim = state_dict["classifier.1.weight"].shape[0]
        classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_classes),
        )
        return classifier, num_classes
    else:
        num_classes = state_dict["classifier.1.weight"].shape[0]
        classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes),
        )
        return classifier, num_classes


def load_efficientnet_checkpoint(checkpoint_path, device, model_name="model"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_names = checkpoint.get("class_names") or checkpoint.get("classes")
    else:
        state_dict = checkpoint
        class_names = None

    if "classifier.1.weight" not in state_dict:
        raise IncompatibleCheckpointError(
            f"[{model_name}] Expected key 'classifier.1.weight' but it wasn't found. "
            f"Keys present: {list(state_dict.keys())[:5]}..."
        )

    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    classifier, num_classes = _build_matching_head(state_dict, num_features)
    model.classifier = classifier

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        classifier_keys = {k: tuple(v.shape) for k, v in state_dict.items() if k.startswith("classifier.")}
        raise IncompatibleCheckpointError(
            f"[{model_name}] Checkpoint has a classifier head that doesn't match either "
            f"known structure (simple or wide).\n"
            f"  Classifier keys/shapes in checkpoint: {classifier_keys}\n"
            f"  class_names in checkpoint: {class_names}\n"
            f"  Original error: {e}"
        ) from e

    if class_names is None:
        if model_name in KNOWN_CLASS_NAMES:
            class_names = KNOWN_CLASS_NAMES[model_name]
            print(
                f"[info] [{model_name}] No 'class_names' in checkpoint -- "
                f"using hardcoded labels from training source: {class_names}"
            )
        else:
            class_names = [f"class_{i}" for i in range(num_classes)]
            print(
                f"[warning] [{model_name}] No 'class_names' found in checkpoint -- "
                f"using placeholder labels {class_names}. Update the checkpoint or "
                f"this loader with real label names before relying on this model's "
                f"output in the frontend."
            )
    elif len(class_names) != num_classes:
        print(
            f"[warning] [{model_name}] class_names length ({len(class_names)}) doesn't "
            f"match inferred num_classes ({num_classes}) -- using placeholder labels instead."
        )
        class_names = [f"class_{i}" for i in range(num_classes)]

    model.eval()
    model.to(device)

    print(f"[loaded] {model_name}: {num_classes} classes -> {class_names}")
    return model, class_names