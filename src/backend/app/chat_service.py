"""Business logic for the Gemini-powered pest-advisory chatbot.

Kept independent of FastAPI route code (see app.gemini_client for the same
separation). Builds grounded prompts from the ViT prediction result so the
assistant's advice is always anchored to what the model actually predicted,
and never invents a different pest.
"""

from __future__ import annotations

from app.gemini_client import generate_chat_reply
from app.schemas import ChatMessage, ClassPrediction

_SYSTEM_PROMPT_TEMPLATE = """You are the AgriVision PestGuard advisory assistant, embedded in a crop \
pest-detection web application. A Vision Transformer model (ViT-B/16, trained on the IP102 dataset) \
has just classified an uploaded image. Ground every answer in the prediction below — do not contradict \
it or invent a different pest.

Prediction context:
- Top predicted pest: {top_name} ({top_confidence:.1f}% confidence)
- Other candidates considered: {other_candidates}

Your responsibilities:
1. Briefly explain what this pest is and the typical damage it causes to crops.
2. Recommend practical integrated pest management (IPM) actions: monitoring, cultural/mechanical \
controls, biological controls, and (only if relevant) general classes of pesticide that are commonly \
used against it — without prescribing exact brand names, doses, or application rates, since those \
depend on local regulations and label instructions.
3. Ask the farmer 2-3 short clarifying questions that would help you tailor the advice, such as the \
crop growth stage, the type of land/field and soil, the current month/season, and the region or \
climate, so you can refine your recommendation as they answer.
4. Always include a brief safety reminder to follow local agricultural extension guidance and product \
labels before applying any pesticide.
5. Keep responses concise, practical, and formatted with short paragraphs or bullet points. Do not use \
markdown tables. Do not claim certainty beyond what the model's confidence score supports if confidence \
is low (below 50%).
6. If the confidence in the top prediction is low, mention that the result is uncertain and suggest \
reviewing the Grad-CAM heatmap and top-5 alternatives shown in the app, or re-uploading a clearer image.

Stay strictly on the topic of this pest, crop health, and pest management. If asked something unrelated, \
politely redirect the conversation back to pest management for this prediction."""


def _format_other_candidates(top_k: list[ClassPrediction]) -> str:
    if len(top_k) <= 1:
        return "none"
    others = top_k[1:]
    return ", ".join(f"{entry.class_name} ({entry.confidence * 100:.1f}%)" for entry in others)


def build_system_prompt(predicted_class: ClassPrediction, top_k_predictions: list[ClassPrediction]) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        top_name=predicted_class.class_name,
        top_confidence=predicted_class.confidence * 100,
        other_candidates=_format_other_candidates(top_k_predictions),
    )


def _history_to_contents(history: list[ChatMessage]) -> list[dict]:
    role_map = {"user": "user", "assistant": "model"}
    return [{"role": role_map[entry.role], "parts": [{"text": entry.content}]} for entry in history]


async def generate_initial_insights(
    predicted_class: ClassPrediction, top_k_predictions: list[ClassPrediction]
) -> str:
    """Produce the automatic post-prediction recommendation message."""

    system_prompt = build_system_prompt(predicted_class, top_k_predictions)
    kickoff_contents = [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "A new image was just analyzed. Give your initial assessment and "
                        "recommendations now, following your instructions, including your "
                        "clarifying questions."
                    )
                }
            ],
        }
    ]
    return await generate_chat_reply(kickoff_contents, system_prompt)


async def generate_chat_response(
    predicted_class: ClassPrediction,
    top_k_predictions: list[ClassPrediction],
    history: list[ChatMessage],
    message: str,
) -> str:
    """Continue the conversation, grounded in the same prediction context."""

    system_prompt = build_system_prompt(predicted_class, top_k_predictions)
    contents = _history_to_contents(history)
    contents.append({"role": "user", "parts": [{"text": message}]})
    return await generate_chat_reply(contents, system_prompt)
