"""Image validation, decoding, and preprocessing shared by every endpoint.

Kept independent of FastAPI route code so the same logic can be reused
verbatim inside a SageMaker inference container or a Lambda handler.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.config import settings


class ImageValidationError(ValueError):
    """Raised for any user-facing, HTTP-400-worthy image problem."""


@dataclass(frozen=True)
class DecodedImage:
    pil_image: Image.Image  # RGB, original resolution
    width: int
    height: int


def validate_upload(filename: str | None, content_type: str | None, size_bytes: int) -> None:
    """Validate filename extension, declared MIME type, and file size.

    Raises ImageValidationError with a human-readable message on any
    violation. Does not touch file bytes.
    """

    if size_bytes <= 0:
        raise ImageValidationError("Uploaded file is empty.")

    if size_bytes > settings.max_upload_bytes:
        raise ImageValidationError(
            f"Uploaded file is too large ({size_bytes / (1024 * 1024):.2f} MB). "
            f"Maximum allowed size is {settings.max_upload_mb:.0f} MB."
        )

    if not filename:
        raise ImageValidationError("Uploaded file has no filename.")

    lower_name = filename.lower()
    if not lower_name.endswith(settings.allowed_extensions):
        raise ImageValidationError(
            f"Unsupported file extension for '{filename}'. Allowed types: "
            f"{', '.join(settings.allowed_extensions)}."
        )

    if content_type and content_type.lower() not in settings.allowed_content_types:
        raise ImageValidationError(
            f"Unsupported content type '{content_type}'. Allowed types: "
            f"{', '.join(settings.allowed_content_types)}."
        )


def decode_image(raw_bytes: bytes) -> DecodedImage:
    """Safely decode raw bytes into an RGB PIL image.

    Raises ImageValidationError if the bytes are not a valid, decodable
    image (e.g. corrupted or truncated upload).
    """

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("Uploaded file could not be decoded as a valid image.") from exc

    rgb_image = image.convert("RGB")
    return DecodedImage(pil_image=rgb_image, width=rgb_image.width, height=rgb_image.height)


def build_preprocess_transform() -> transforms.Compose:
    """The exact inference-time preprocessing pipeline: resize, tensor, normalize.

    Deliberately excludes any training-time augmentation (no random crop,
    rotation, brightness jitter, or flips) since this path only serves
    user-uploaded images at inference time.
    """

    size = settings.input_size
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(settings.imagenet_mean), std=list(settings.imagenet_std)),
        ]
    )


_PREPROCESS = build_preprocess_transform()


def preprocess_for_model(pil_image: Image.Image) -> torch.Tensor:
    """Resize/normalize an RGB PIL image into a (1, 3, H, W) model input tensor."""

    tensor = _PREPROCESS(pil_image)
    return tensor.unsqueeze(0)


def resized_rgb_array(pil_image: Image.Image) -> np.ndarray:
    """Return the image resized to the model input size as a float32 [0, 1] HWC array.

    Used as the visual base for the Grad-CAM overlay so the heatmap aligns
    pixel-for-pixel with what the model actually saw.
    """

    size = settings.input_size
    resized = pil_image.resize((size, size))
    array = np.asarray(resized).astype(np.float32) / 255.0
    return array
