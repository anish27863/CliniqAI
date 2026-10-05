"""
apply_gradcam_patch.py -- wires Grad-CAM into the CliniqAI backend.

Run from the project root:   python scripts/apply_gradcam_patch.py
Edits (with .bak backups next to each file):
  backend/app/schemas.py              + Explanation model, `explanation` / `explanation_error` fields
  backend/app/inference/pipeline.py   + `explain` / `explain_method` args, runs Grad-CAM on the chosen specialist
  backend/app/main.py                 + /analyze?explain=true&method=gradcam|gradcampp
Every edit is an exact-text replacement that must match exactly once; if anything does not match, NOTHING is
written for that file and the script says which anchor failed. Safe to re-run (already-patched files are skipped).
Review afterwards with:  git diff
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "backend" / "app"

SCHEMAS_EXPLANATION = '''class Explanation(BaseModel):
    method: str                 # "gradcam" or "gradcam++"
    class_idx: int
    class_name: str             # the class the heatmap explains (= the specialist's prediction)
    confidence: float
    degenerate: bool            # True -> no positive evidence for that class; heatmap is blank
    size: List[int]             # [width, height] of both PNGs
    overlay_png_b64: str        # original image + heat, PNG, base64
    heatmap_png_b64: str        # transparent RGBA heat only, PNG, base64


'''

PIPELINE_TAIL = '''        "message": "OK",
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
'''

PATCHES = {
    APP / "schemas.py": [
        ("from typing import Optional, Dict\n", "from typing import Optional, Dict, List\n"),
        ("class AnalyzeResponse(BaseModel):", SCHEMAS_EXPLANATION + "class AnalyzeResponse(BaseModel):"),
        ("    message: str", "    message: str\n    explanation: Optional[Explanation] = None\n    explanation_error: Optional[str] = None"),
    ],
    APP / "inference" / "pipeline.py": [
        ("from app.inference.preprocess import inference_transform\n",
         "from app.inference.preprocess import inference_transform\nfrom app.gradcam import explain as gradcam_explain\n"),
        ("def run_inference(pil_image, registry: ModelRegistry):",
         'def run_inference(pil_image, registry: ModelRegistry, explain: bool = False, explain_method: str = "gradcam"):'),
        ('    return {\n        "is_medical": True,\n        "gatekeeper_confidence": stage1_confidence,\n'
         '        "modality": modality,\n        "modality_confidence": modality_confidence,\n'
         '        "prediction": prediction,\n',
         '    result = {\n        "is_medical": True,\n        "gatekeeper_confidence": stage1_confidence,\n'
         '        "modality": modality,\n        "modality_confidence": modality_confidence,\n'
         '        "prediction": prediction,\n'),
        ('        "message": "OK",\n    }\n', PIPELINE_TAIL),
    ],
    APP / "main.py": [
        ("from io import BytesIO\n", "from io import BytesIO\nfrom typing import Literal\n"),
        ("async def analyze(file: UploadFile = File(...)):",
         'async def analyze(\n    file: UploadFile = File(...),\n    explain: bool = False,\n'
         '    method: Literal["gradcam", "gradcampp"] = "gradcam",\n):'),
        ("    result = run_inference(pil_image, registry)",
         "    result = run_inference(pil_image, registry, explain=explain, explain_method=method)"),
    ],
}

MARKERS = {"schemas.py": "class Explanation(", "pipeline.py": "gradcam_explain", "main.py": "explain_method=method"}


def main():
    ok = True
    for path, edits in PATCHES.items():
        if not path.exists():
            print(f"[MISSING] {path}")
            ok = False
            continue
        text = path.read_text(encoding="utf-8")
        if MARKERS[path.name] in text:
            print(f"[skip] {path.name}: already patched")
            continue
        new = text
        for i, (old, rep) in enumerate(edits, 1):
            n = new.count(old)
            if n != 1:
                print(f"[FAILED] {path.name}: edit {i} anchor matched {n} times (need exactly 1):\n---\n{old}\n---")
                ok = False
                break
            new = new.replace(old, rep, 1)
        else:
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(path, bak)
            path.write_text(new, encoding="utf-8")
            print(f"[patched] {path.relative_to(ROOT)}  ({len(edits)} edits, backup: {bak.name})")
    if not (APP / "gradcam.py").exists():
        print("[MISSING] backend/app/gradcam.py - copy the latest gradcam.py there first")
        ok = False
    print("\nDone." if ok else "\nFinished with problems - nothing was half-written for failed files; see above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
