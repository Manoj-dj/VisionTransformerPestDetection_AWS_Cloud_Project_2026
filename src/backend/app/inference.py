"""Core classification inference logic, independent of FastAPI route code.

Kept separate from app/main.py so this exact function can be reused inside
a SageMaker inference container, a Lambda handler, or a batch script
without dragging in any web-framework dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from app.config import settings
from app.model_loader import ClassEntry, ModelBundle


@dataclass(frozen=True)
class RankedPrediction:
    class_index: int
    class_label: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class ClassificationResult:
    predicted: RankedPrediction
    top_k: list[RankedPrediction]
    logits: torch.Tensor  # kept for the caller to optionally drive Grad-CAM


def _to_ranked(entry: ClassEntry, confidence: float) -> RankedPrediction:
    return RankedPrediction(
        class_index=entry.model_index,
        class_label=entry.class_label,
        class_name=entry.class_name,
        confidence=confidence,
    )


@torch.no_grad()
def classify(bundle: ModelBundle, input_tensor: torch.Tensor) -> ClassificationResult:
    """Run the model in eval mode on a preprocessed (1, 3, H, W) tensor.

    Returns the top prediction plus the top-K predictions sorted strictly
    descending by confidence.
    """

    input_tensor = input_tensor.to(bundle.device)
    logits = bundle.model(input_tensor)
    probabilities = F.softmax(logits, dim=1).squeeze(0)

    k = min(settings.top_k, probabilities.shape[0])
    top_values, top_indices = torch.topk(probabilities, k=k)

    top_k_predictions = [
        _to_ranked(bundle.class_entry_for_index(int(idx)), float(val))
        for val, idx in zip(top_values.tolist(), top_indices.tolist())
    ]

    return ClassificationResult(predicted=top_k_predictions[0], top_k=top_k_predictions, logits=logits)
