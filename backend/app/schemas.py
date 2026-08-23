from typing import Optional, Dict
from pydantic import BaseModel


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
