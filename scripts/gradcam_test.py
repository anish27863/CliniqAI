"""
gradcam_test.py -- run Grad-CAM on real images with any CliniqAI checkpoint and save side-by-side PNGs.

From the project root (E:\\programming\\CliniqAI), e.g.:
    python scripts/gradcam_test.py --ckpt models/kidney_ct_specialist.pt --images data/raw/manual_sanity_check_stage2/kidney_ct --out gradcam_out
    python scripts/gradcam_test.py --ckpt models/xyz.pt --images path/to/one.jpg --method gradcam++ --class-idx 2

The model is rebuilt from the checkpoint's own tensor shapes (standard torchvision head, or the wide
Dropout->Linear(1280,512)->ReLU->Dropout->Linear head). If your checkpoint has a different head, this stops
with a clear error instead of guessing.

How to READ the output (this is an audit tool, not proof): heat on the lesion/organ = plausible; heat on borders,
corners, burned-in text/markers, scanner edges = the model is using a shortcut -- exactly the failure mode that
sank bone_fracture v1/v2.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.gradcam import explain, to_b64_png  # noqa: E402,F401
import base64, io  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build_from_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    names = list(ck.get("class_names") or ck.get("classes") or [])
    idxs = sorted({int(m.group(1)) for k in sd for m in [re.match(r"classifier\.(\d+)\.weight$", k)] if m})
    m = models.efficientnet_b0(weights=None)
    if idxs == [1]:
        n_out = sd["classifier.1.weight"].shape[0]
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, n_out))
    elif idxs == [1, 4]:
        mid, n_out = sd["classifier.1.weight"].shape[0], sd["classifier.4.weight"].shape[0]
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, mid), nn.ReLU(),
                                     nn.Dropout(0.3), nn.Linear(mid, n_out))
    else:
        raise SystemExit(f"Unrecognised classifier layout (weight indices {idxs}). Paste backend/app/model_loader.py "
                         f"and I'll match it.")
    m.load_state_dict(sd)
    if not names:
        names = [str(i) for i in range(n_out)]
    return m.eval(), names


def collect(p):
    p = Path(p)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)


def b64_to_img(b):
    return Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True, help="image file or folder")
    ap.add_argument("--out", default="gradcam_out")
    ap.add_argument("--method", default="gradcam", choices=["gradcam", "gradcam++"])
    ap.add_argument("--class-idx", type=int, default=None, help="explain this class instead of the predicted one")
    ap.add_argument("--max", type=int, default=24, help="max images when a folder is given")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, names = build_from_checkpoint(a.ckpt)
    model.to(dev)
    print(f"classes ({len(names)}): {names}")
    # MUST equal the pipeline's inference preprocessing:
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    files = collect(a.images)[: a.max]
    if not files:
        raise SystemExit(f"No images found at {a.images}")
    tally = {}
    for f in files:
        img = Image.open(f).convert("RGB")
        r = explain(model, img, preprocess=tf, class_names=names, class_idx=a.class_idx, method=a.method)
        over = b64_to_img(r["overlay_png_b64"])
        orig = img.copy()
        orig.thumbnail(over.size)
        canvas = Image.new("RGB", (orig.width + over.width, max(orig.height, over.height)), "white")
        canvas.paste(orig, (0, 0))
        canvas.paste(over, (orig.width, 0))
        ImageDraw.Draw(canvas).text((6, 6), f"{r['class_name']} {r['confidence']:.1%}", fill=(255, 255, 0))
        dst = out / f"{f.stem}_{a.method}_{r['class_name']}.png"
        canvas.save(dst)
        tally[r['class_name']] = tally.get(r['class_name'], 0) + 1
        top = ", ".join(f"{t['class_name']} {t['prob']:.2f}" for t in r["top3"])
        print(f"{f.name[:40]:40s} -> {top}{'  [DEGENERATE: no positive evidence]' if r['degenerate'] else ''}  saved {dst.name}")

    print(f"\nPrediction tally over {len(files)} images (predicted class -> count): {dict(sorted(tally.items()))}")


if __name__ == "__main__":
    main()
