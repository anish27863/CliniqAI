from contextlib import asynccontextmanager
from io import BytesIO
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.inference.pipeline import ModelRegistry, run_inference
from app.schemas import AnalyzeResponse

registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loads all 9 models (2 router stages + 7 specialists) once, at startup.
    # Requests after this just do dict lookups -- no per-request disk I/O.
    registry.load_all()
    yield


app = FastAPI(title="CliniqAI API", version="0.1.0", lifespan=lifespan)

# NOTE: wide open for local dev / early integration testing. Restrict this to
# your actual frontend's origin before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "specialists_loaded": list(registry.specialists.keys()),
        "device": str(registry.device),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    explain: bool = False,
    method: Literal["gradcam", "gradcampp"] = "gradcam",
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    try:
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
        pil_image.load() # forces full decode -- catches truncated/corrupted uploads early
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read this file as an image.")
    except OSError:
        raise HTTPException(status_code=400, detail="Image file appears corrupted or truncated.")

    result = run_inference(pil_image, registry, explain=explain, explain_method=method)
    return AnalyzeResponse(**result)
