"""
explain_client_test.py -- calls the running API with explain=true and saves the heatmaps as real PNG files.

1) start the server:   cd E:\\programming\\CliniqAI\\backend ; uvicorn app.main:app --port 8000
2) in another terminal (project root):
   python scripts/explain_client_test.py --image "path\\to\\scan.jpg" [--method gradcampp]
Writes explain_out/<name>_original.png, _overlay.png, _heatmap.png and a side-by-side _compare.png and prints the JSON without the big base64 fields.
"""
import argparse
import base64
import json
import mimetypes
from pathlib import Path

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--url", default="http://localhost:8000/analyze")
ap.add_argument("--method", default="gradcam", choices=["gradcam", "gradcampp"])
ap.add_argument("--out", default="explain_out")
a = ap.parse_args()

img = Path(a.image)
mime = mimetypes.guess_type(img.name)[0] or "image/jpeg"
with open(img, "rb") as f:
    r = requests.post(a.url, params={"explain": "true", "method": a.method}, files={"file": (img.name, f, mime)})
r.raise_for_status()
data = r.json()
expl = data.get("explanation")
out = Path(a.out)
out.mkdir(exist_ok=True)
if expl:
    for key, tag in (("original_png_b64", "original"), ("overlay_png_b64", "overlay"), ("heatmap_png_b64", "heatmap")):
        if expl.get(key):
            (out / f"{img.stem}_{tag}.png").write_bytes(base64.b64decode(expl[key]))
            expl[key] = f"<{len(expl[key])} chars, saved as {img.stem}_{tag}.png>"
    o, v = out / f"{img.stem}_original.png", out / f"{img.stem}_overlay.png"
    if o.exists() and v.exists():                      # side-by-side: normal image | Grad-CAM overlay
        from PIL import Image, ImageDraw
        a_, b_ = Image.open(o).convert("RGB"), Image.open(v).convert("RGB")
        c = Image.new("RGB", (a_.width + b_.width + 10, a_.height), "white")
        c.paste(a_, (0, 0)); c.paste(b_, (a_.width + 10, 0))
        ImageDraw.Draw(c).text((6, 6), "original", fill=(0, 0, 0))
        ImageDraw.Draw(c).text((a_.width + 16, 6), f"{expl['method']}: {expl['class_name']}", fill=(0, 0, 0))
        c.save(out / f"{img.stem}_compare.png")
        print(f"Saved {img.stem}_original.png, _overlay.png, _heatmap.png and _compare.png in {out}/")
print(json.dumps(data, indent=2))
if not expl:
    print("\nNo explanation in the response (the image may not have reached a specialist, or see explanation_error).")
