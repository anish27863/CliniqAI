"""
gradcam.py -- Grad-CAM / Grad-CAM++ explanations for CliniqAI classifiers (torchvision EfficientNet-B0 family).

Drop into  backend/app/gradcam.py.  Standalone: it only needs the loaded nn.Module, a PIL image and the SAME
preprocessing the pipeline uses at inference.

    from app.gradcam import explain
    out = explain(model, pil_image, preprocess=my_eval_tf, class_names=names)   # explains the predicted class
    out["overlay_png_b64"], out["heatmap_png_b64"], out["class_name"], out["confidence"], ...

Notes
  * Target layer = last conv block (`model.features[-1]`, 1280 ch, 7x7 at 224 input). Falls back to the last Conv2d.
  * Uses a forward hook + torch.autograd.grad (no backward hooks), so EfficientNet's in-place SiLU is not a problem.
  * Works even if the caller is inside torch.no_grad(); it does NOT work inside torch.inference_mode().
  * Heatmap is mapped back onto the original image by resizing the 224x224 map to the original size. That is only
    correct if inference preprocessing is a plain Resize((224,224)). If the pipeline center-crops, the map must be
    cropped/padded to match -- check pipeline.py.
"""
import base64
import io
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_LOCK = threading.Lock()   # the hook stores activations on the shared model: one explanation at a time

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def default_preprocess():
    """Only a fallback -- pass the pipeline's own transform so CAM and prediction see identical inputs."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def find_target_layer(model):
    feats = getattr(model, "features", None)
    if feats is not None and len(feats) > 0:
        return feats[-1]
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("No conv layer found for Grad-CAM; pass target_layer explicitly.")
    return last


class GradCAM:
    def __init__(self, model, target_layer=None, method="gradcam"):
        if method not in ("gradcam", "gradcam++"):
            raise ValueError("method must be 'gradcam' or 'gradcam++'")
        self.model, self.method = model, method
        self.layer = target_layer if target_layer is not None else find_target_layer(model)
        self._acts = None
        self._handle = self.layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self._acts = output

    def close(self):
        self._handle.remove()
        self._acts = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __call__(self, x, class_idx=None):
        """x: preprocessed (1,3,H,W). Returns dict(cam=(H,W) float32 in [0,1], probs, class_idx, degenerate)."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Grad-CAM cannot run under torch.inference_mode(); use no_grad or nothing.")
        was_training = self.model.training
        self.model.eval()
        dev = next(self.model.parameters()).device
        x = x.to(dev).float().requires_grad_(True)      # guarantees a graph even if params are frozen
        try:
            with torch.enable_grad():
                self._acts = None
                logits = self.model(x)
                probs = torch.softmax(logits.detach(), 1)[0].cpu().numpy()
                if class_idx is None:
                    class_idx = int(logits.argmax(1)[0])
                acts = self._acts
                if acts is None:
                    raise RuntimeError("Target layer did not fire - wrong target_layer?")
                grads = torch.autograd.grad(logits[0, class_idx], acts)[0]
            acts, grads = acts.detach(), grads.detach()

            if self.method == "gradcam":
                w = grads.mean(dim=(2, 3), keepdim=True)
            else:                                        # Grad-CAM++
                g2, g3 = grads ** 2, grads ** 3
                denom = 2 * g2 + acts.sum(dim=(2, 3), keepdim=True) * g3
                denom = torch.where(denom != 0, denom, torch.ones_like(denom))
                w = ((g2 / denom) * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

            cam = F.relu((w * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            cam = cam - cam.min()
            mx = cam.max()
            degenerate = bool(mx < 1e-8)
            cam = torch.zeros_like(cam) if degenerate else cam / mx
            return dict(cam=cam.cpu().numpy().astype(np.float32), probs=probs,
                        class_idx=int(class_idx), degenerate=degenerate)
        finally:
            self._acts = None
            self.model.train(was_training)


# ----------------------------------------------------------------------------- rendering helpers (numpy/PIL only)
_STOPS = np.array([0.0, 0.35, 0.6, 0.8, 1.0])
_COLORS = np.array([[0, 200, 255], [0, 255, 120], [255, 240, 0], [255, 120, 0], [255, 0, 60]], dtype=np.float32)


def jet(v):
    """Name kept so nothing else changes. Bright ramp that stays visible on dark and red images."""
    v = np.clip(v, 0, 1)
    out = np.stack([np.interp(v, _STOPS, _COLORS[:, c]) for c in range(3)], axis=-1)
    return out.astype(np.uint8)


def resize_cam(cam, size_wh):
    im = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize(size_wh, Image.BILINEAR), dtype=np.float32) / 255.0


def make_overlay(image, cam, alpha=0.75, cutoff=0.2):
    """Blend heat onto the image. The image is darkened under the heat so the colours stay visible."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    t = np.clip((cam - cutoff) / (1 - cutoff), 0, 1)
    a = (alpha * t ** 0.7)[..., None]
    dim = 1 - 0.5 * t[..., None]
    out = base * dim * (1 - a) + jet(cam).astype(np.float32) * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def make_heat_rgba(cam, cutoff=0.2):
    t = np.clip((cam - cutoff) / (1 - cutoff), 0, 1)
    a = (255 * t ** 0.7).astype(np.uint8)
    return Image.fromarray(np.dstack([jet(cam), a]), "RGBA")

def to_b64_png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----------------------------------------------------------------------------- one-call API
def explain(model, image, preprocess=None, class_names=None, class_idx=None, method="gradcam",
            alpha=0.75, max_side=768, target_layer=None):
    """
    Explain `model`'s prediction (or `class_idx`) for a PIL image.
    Returns: class_idx, class_name, confidence (prob of the explained class), top3, method, degenerate,
             original_png_b64 (plain image), overlay_png_b64 (image + heat), heatmap_png_b64 (RGBA heat only), size [w, h] of both PNGs.
    """
    image = image.convert("RGB")
    preprocess = preprocess or default_preprocess()
    x = preprocess(image).unsqueeze(0)
    with _LOCK:
        with GradCAM(model, target_layer, method) as engine:
            res = engine(x, class_idx)

    disp = image.copy()
    disp.thumbnail((max_side, max_side))                 # keep the payload small for big fundus/CT images
    cam = resize_cam(res["cam"], disp.size)
    names = list(class_names) if class_names is not None else [str(i) for i in range(len(res["probs"]))]
    order = np.argsort(res["probs"])[::-1][:3]
    return dict(
        class_idx=res["class_idx"],
        class_name=names[res["class_idx"]],
        confidence=float(res["probs"][res["class_idx"]]),
        top3=[{"class_name": names[i], "prob": float(res["probs"][i])} for i in order],
        method=method,
        degenerate=res["degenerate"],
        size=[disp.size[0], disp.size[1]],
        original_png_b64=to_b64_png(disp),                # plain image, same size as the overlay (lines up 1:1)
        overlay_png_b64=to_b64_png(make_overlay(disp, cam, alpha)),
        heatmap_png_b64=to_b64_png(make_heat_rgba(cam)),
    )
