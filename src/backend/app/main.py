"""FastAPI application entrypoint for AgriVision PestGuard.

Routes are kept thin: they validate the request shape, delegate to
app.image_utils / app.inference / app.explainability for business logic,
and shape the JSON response. This keeps the inference pipeline reusable
outside of FastAPI (e.g. inside a SageMaker container) and makes the API
straightforward to front with API Gateway later.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.explainability import ExplainabilityError, generate_gradcam
from app.image_utils import (
    ImageValidationError,
    decode_image,
    preprocess_for_model,
    resized_rgb_array,
    validate_upload,
)
from app.inference import classify
from app.logging_config import setup_logging
from app.model_loader import ClassListError, ModelBundle, load_model_bundle
from app.schemas import (
    ClassPrediction,
    HealthResponse,
    ImageInfo,
    MetadataResponse,
)

setup_logging()
logger = logging.getLogger("agrivision.api")

# The single, read-only model bundle populated once at startup.
_model_bundle: ModelBundle | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_bundle
    logger.info("Starting AgriVision PestGuard API - loading model checkpoint...")
    try:
        _model_bundle = load_model_bundle()
    except (FileNotFoundError, ClassListError) as exc:
        logger.error("Startup failed: %s", exc)
        raise
    logger.info("Model ready on device=%s, num_classes=%d", _model_bundle.device, len(_model_bundle.class_entries))
    yield
    logger.info("Shutting down AgriVision PestGuard API")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_model() -> ModelBundle:
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return _model_bundle


@app.middleware("http")
async def add_timing_and_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    request.state.request_id = request_id
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.exception("Unhandled error for request_id=%s", request_id)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error.", "request_id": request_id},
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Fast liveness/readiness check. Does not reload the model."""

    return HealthResponse(
        status="ok",
        model_loaded=_model_bundle is not None,
        model_name=settings.model_name,
        num_classes=settings.num_classes,
        device=_model_bundle.device if _model_bundle else "unknown",
    )


@app.get("/metadata", response_model=MetadataResponse)
async def metadata() -> MetadataResponse:
    return MetadataResponse(
        model_name=settings.model_name,
        model_version=settings.model_version,
        num_classes=settings.num_classes,
        input_width=settings.input_size,
        input_height=settings.input_size,
        normalization_mean=list(settings.imagenet_mean),
        normalization_std=list(settings.imagenet_std),
        gradcam_enabled=settings.gradcam_enabled,
        top_k=settings.top_k,
        app_version=settings.app_version,
    )


async def _read_and_validate_upload(file: UploadFile) -> bytes:
    raw_bytes = await file.read()
    try:
        validate_upload(file.filename, file.content_type, len(raw_bytes))
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return raw_bytes


def _ranked_to_schema(prediction) -> ClassPrediction:
    return ClassPrediction(
        class_index=prediction.class_index,
        class_label=prediction.class_label,
        class_name=prediction.class_name,
        confidence=round(prediction.confidence, 6),
    )


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """Classify an uploaded pest/crop image and explain the prediction with Grad-CAM."""

    bundle = _require_model()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start = time.perf_counter()

    raw_bytes = await _read_and_validate_upload(file)

    try:
        decoded = decode_image(raw_bytes)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    input_tensor = preprocess_for_model(decoded.pil_image)
    result = classify(bundle, input_tensor)

    explanation: dict
    gradcam_ok = False
    if settings.gradcam_enabled:
        try:
            rgb_array = resized_rgb_array(decoded.pil_image)
            heatmap_b64, overlay_b64 = generate_gradcam(
                bundle=bundle,
                input_tensor=input_tensor,
                rgb_float_image=rgb_array,
                target_class_index=result.predicted.class_index,
            )
            explanation = {
                "enabled": True,
                "method": "Grad-CAM",
                "heatmap_base64": heatmap_b64,
                "overlay_base64": overlay_b64,
            }
            gradcam_ok = True
        except ExplainabilityError as exc:
            explanation = {"enabled": False, "error": str(exc)}
    else:
        explanation = {"enabled": False, "error": "Grad-CAM is disabled by configuration."}

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "request_id=%s file_size=%d content_type=%s latency_ms=%.2f predicted_class=%s "
        "confidence=%.4f gradcam_ok=%s model_version=%s",
        request_id,
        len(raw_bytes),
        file.content_type,
        latency_ms,
        result.predicted.class_name,
        result.predicted.confidence,
        gradcam_ok,
        settings.model_version,
    )

    return {
        "request_id": request_id,
        "predicted_class": _ranked_to_schema(result.predicted).model_dump(),
        "top_k_predictions": [_ranked_to_schema(p).model_dump() for p in result.top_k],
        "explanation": explanation,
        "image": ImageInfo(
            width=settings.input_size, height=settings.input_size, color_mode="RGB"
        ).model_dump(),
        "latency_ms": round(latency_ms, 2),
        "model_version": settings.model_version,
    }


@app.post("/predict/classification-only")
async def predict_classification_only(request: Request, file: UploadFile = File(...)):
    """Classify an uploaded image without generating Grad-CAM, for lower latency."""

    bundle = _require_model()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start = time.perf_counter()

    raw_bytes = await _read_and_validate_upload(file)

    try:
        decoded = decode_image(raw_bytes)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    input_tensor = preprocess_for_model(decoded.pil_image)
    result = classify(bundle, input_tensor)

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "request_id=%s file_size=%d content_type=%s latency_ms=%.2f predicted_class=%s "
        "confidence=%.4f gradcam_ok=n/a model_version=%s",
        request_id,
        len(raw_bytes),
        file.content_type,
        latency_ms,
        result.predicted.class_name,
        result.predicted.confidence,
        settings.model_version,
    )

    return {
        "request_id": request_id,
        "predicted_class": _ranked_to_schema(result.predicted).model_dump(),
        "top_k_predictions": [_ranked_to_schema(p).model_dump() for p in result.top_k],
        "image": ImageInfo(
            width=settings.input_size, height=settings.input_size, color_mode="RGB"
        ).model_dump(),
        "latency_ms": round(latency_ms, 2),
        "model_version": settings.model_version,
    }
