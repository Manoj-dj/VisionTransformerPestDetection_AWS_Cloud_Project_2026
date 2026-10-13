"""Pydantic response/request models for the AgriVision PestGuard API.

Keeping these in one module makes the JSON contract explicit and easy to
keep JSON-safe for a future API Gateway / Lambda integration.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    num_classes: int
    device: str


class MetadataResponse(BaseModel):
    model_name: str
    model_version: str
    num_classes: int
    input_width: int
    input_height: int
    normalization_mean: list[float]
    normalization_std: list[float]
    gradcam_enabled: bool
    top_k: int
    app_version: str


class ClassPrediction(BaseModel):
    class_index: int = Field(..., description="Zero-based index as produced by the model output layer")
    class_label: int = Field(..., description="One-based label as written in classes.txt")
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class ImageInfo(BaseModel):
    width: int
    height: int
    color_mode: str


class ExplanationSuccess(BaseModel):
    enabled: bool = True
    method: str = "Grad-CAM"
    heatmap_base64: str
    overlay_base64: str


class ExplanationFailure(BaseModel):
    enabled: bool = False
    error: str


class PredictionResponse(BaseModel):
    request_id: str
    predicted_class: ClassPrediction
    top_k_predictions: list[ClassPrediction]
    explanation: Optional[dict] = None
    image: ImageInfo
    latency_ms: float
    model_version: str


class ClassificationOnlyResponse(BaseModel):
    request_id: str
    predicted_class: ClassPrediction
    top_k_predictions: list[ClassPrediction]
    image: ImageInfo
    latency_ms: float
    model_version: str


class ErrorResponse(BaseModel):
    error: str
    request_id: Optional[str] = None


# --- Gemini advisory chatbot ---------------------------------------------------


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class ChatInsightsRequest(BaseModel):
    predicted_class: ClassPrediction
    top_k_predictions: list[ClassPrediction]


class ChatInsightsResponse(BaseModel):
    message: str


class ChatMessageRequest(BaseModel):
    predicted_class: ClassPrediction
    top_k_predictions: list[ClassPrediction]
    history: list[ChatMessage] = Field(default_factory=list, max_length=30)
    message: str = Field(..., min_length=1, max_length=2000)


class ChatMessageResponse(BaseModel):
    message: str


class TranscribeResponse(BaseModel):
    text: str
