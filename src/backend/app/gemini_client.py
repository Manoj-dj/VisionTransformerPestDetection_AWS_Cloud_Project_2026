"""Thin REST client for the Gemini Generative Language API.

Deliberately framework-independent (no FastAPI imports) and holds no
mutable state, so it can be reused outside the web layer. The API key is
read once from app.config.settings and is never exposed to the frontend;
the frontend only ever talks to our own backend endpoints.
"""

from __future__ import annotations

import logging

import httpx

from app.config import settings

logger = logging.getLogger("agrivision.gemini")


class GeminiError(RuntimeError):
    """Raised for any Gemini API failure. Messages must stay safe to show users."""


def _endpoint(model: str) -> str:
    return f"{settings.gemini_api_base}/models/{model}:generateContent"


async def _call_generate_content(model: str, contents: list[dict], system_instruction: str | None = None) -> str:
    if not settings.gemini_enabled:
        raise GeminiError("Gemini API key is not configured on the server.")

    payload: dict = {"contents": contents}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    try:
        async with httpx.AsyncClient(timeout=settings.gemini_timeout_seconds) as client:
            response = await client.post(
                _endpoint(model),
                params={"key": settings.gemini_api_key},
                json=payload,
            )
    except httpx.TimeoutException as exc:
        logger.warning(
            "Gemini request to model '%s' timed out after %.0fs (%s: %s)",
            model,
            settings.gemini_timeout_seconds,
            type(exc).__name__,
            exc,
        )
        raise GeminiError("The AI assistant timed out. Please try again.") from exc
    except httpx.HTTPError as exc:
        logger.warning(
            "Gemini request to model '%s' failed before a response was received (%s: %s)",
            model,
            type(exc).__name__,
            exc,
        )
        raise GeminiError("Could not reach the AI assistant service.") from exc

    if response.status_code != 200:
        logger.warning("Gemini API returned status %d: %s", response.status_code, response.text[:500])
        raise GeminiError(f"The AI assistant service returned an error (status {response.status_code}).")

    try:
        data = response.json()
        candidates = data["candidates"]
        parts = candidates[0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts).strip()
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("Unexpected Gemini response shape: %s", str(exc))
        raise GeminiError("The AI assistant returned an unexpected response.") from exc

    if not text:
        raise GeminiError("The AI assistant returned an empty response.")

    return text


async def generate_chat_reply(contents: list[dict], system_instruction: str) -> str:
    """Generate a text reply from the configured chat model (GEMINI_MODEL)."""

    return await _call_generate_content(settings.gemini_model, contents, system_instruction)


async def transcribe_audio(audio_bytes: bytes, mime_type: str) -> str:
    """Transcribe spoken audio to plain text using the configured STT model.

    Uses the Generative Language API's multimodal `generateContent` call with
    an inline audio part, targeting GEMINI_TRANSCRIBE_MODEL.
    """

    import base64

    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    contents = [
        {
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": mime_type, "data": audio_b64}},
                {"text": "Transcribe the spoken audio to plain text. Return only the transcript, with no commentary, labels, or quotation marks."},
            ],
        }
    ]
    return await _call_generate_content(settings.gemini_transcribe_model, contents, system_instruction=None)
