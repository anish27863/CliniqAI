from typing import Optional, Dict, List
from pydantic import BaseModel


class Explanation(BaseModel):
    method: str                 # "gradcam" or "gradcam++"
    class_idx: int
    class_name: str             # the class the heatmap explains (= the specialist's prediction)
    confidence: float
    degenerate: bool            # True -> no positive evidence for that class; heatmap is blank
    size: List[int]             # [width, height] of both PNGs
    overlay_png_b64: str        # original image + heat, PNG, base64
    original_png_b64: Optional[str] = None   # plain image, same size as the overlay
    heatmap_png_b64: str        # transparent RGBA heat only, PNG, base64


class AnalyzeResponse(BaseModel):
    is_medical: bool
    gatekeeper_confidence: float
    modality: Optional[str] = None
    modality_confidence: Optional[float] = None
    prediction: Optional[str] = None
    prediction_confidence: Optional[float] = None
    all_class_probabilities: Optional[Dict[str, float]] = None
    low_confidence_warning: bool = False
    message: str
    explanation: Optional[Explanation] = None
    explanation_error: Optional[str] = None
