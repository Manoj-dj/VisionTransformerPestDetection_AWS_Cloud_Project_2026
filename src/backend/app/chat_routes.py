"""Thin FastAPI routes for the Gemini pest-advisory chatbot and voice input.

Routes validate the request and delegate to app.chat_service /
app.gemini_client for all business logic, matching the separation of
concerns used for the prediction endpoints in app.main.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.chat_service import generate_chat_response, generate_initial_insights
from app.config import settings
from app.gemini_client import GeminiError, transcribe_audio
from app.schemas import (
    ChatInsightsRequest,
    ChatInsightsResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    TranscribeResponse,
)

logger = logging.getLogger("agrivision.chat")

router = APIRouter(prefix="/chat", tags=["chat"])


def _require_gemini() -> None:
    if not settings.gemini_enabled:
        raise HTTPException(
            status_code=503,
            detail="The AI assistant is not configured on the server (missing GEMINI_API_KEY).",
        )


@router.post("/insights", response_model=ChatInsightsResponse)
async def chat_insights(payload: ChatInsightsRequest) -> ChatInsightsResponse:
    """Generate the automatic recommendation message shown right after a prediction."""

    _require_gemini()
    try:
        message = await generate_initial_insights(payload.predicted_class, payload.top_k_predictions)
    except GeminiError as exc:
        logger.warning("Gemini insights generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ChatInsightsResponse(message=message)


@router.post("/message", response_model=ChatMessageResponse)
async def chat_message(payload: ChatMessageRequest) -> ChatMessageResponse:
    """Continue the chat conversation about the detected pest."""

    _require_gemini()
    try:
        message = await generate_chat_response(
            payload.predicted_class, payload.top_k_predictions, payload.history, payload.message
        )
    except GeminiError as exc:
        logger.warning("Gemini chat reply failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ChatMessageResponse(message=message)


@router.post("/transcribe", response_model=TranscribeResponse)
async def chat_transcribe(audio: UploadFile = File(...)) -> TranscribeResponse:
    """Transcribe a recorded voice note to text using the Gemini STT model."""

    _require_gemini()

    raw_bytes = await audio.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")
    if len(raw_bytes) > settings.max_audio_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file is too large. Maximum allowed size is {settings.max_audio_mb:.0f} MB.",
        )

    content_type = (audio.content_type or "").lower()
    if content_type and not content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an audio recording.")

    mime_type = content_type if content_type.startswith("audio/") else "audio/webm"

    try:
        text = await transcribe_audio(raw_bytes, mime_type)
    except GeminiError as exc:
        logger.warning("Gemini transcription failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return TranscribeResponse(text=text)
