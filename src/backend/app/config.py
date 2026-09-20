"""Centralized configuration for the AgriVision PestGuard backend.

All configurable values are read from environment variables (optionally
loaded from a local .env file) so that the same code can run unchanged
on a laptop, an EC2 instance, or inside a SageMaker/ECS container. No
absolute local paths are hardcoded; defaults are derived relative to the
repository layout so a fresh checkout works out of the box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load a .env file if present (does nothing if it doesn't exist). This must
# happen before we read any os.getenv() calls below.
load_dotenv()

# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------
# app/config.py -> app/ -> backend/ -> src/
_APP_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _APP_DIR.parent
_SRC_DIR = _BACKEND_DIR.parent

_DEFAULT_MODEL_PATH = _SRC_DIR / "ai_model" / "ViT_best.pth"
_DEFAULT_CLASSES_PATH = _SRC_DIR / "ai_model" / "classes.txt"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Immutable application settings, populated once at import time."""

    # Model / checkpoint
    model_path: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_PATH", str(_DEFAULT_MODEL_PATH)))
    )
    classes_path: Path = field(
        default_factory=lambda: Path(os.getenv("CLASSES_PATH", str(_DEFAULT_CLASSES_PATH)))
    )
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "vit_base_patch16_224"))
    num_classes: int = field(default_factory=lambda: int(os.getenv("NUM_CLASSES", "102")))
    model_version: str = field(default_factory=lambda: os.getenv("MODEL_VERSION", "vit-ip102-v1"))
    device_preference: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))

    # Optional future S3 model artifact location (documented for AWS migration,
    # not used for local inference).
    s3_model_uri: str = field(default_factory=lambda: os.getenv("S3_MODEL_URI", ""))

    # Inference / preprocessing
    input_size: int = field(default_factory=lambda: int(os.getenv("INPUT_SIZE", "224")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    gradcam_enabled: bool = field(default_factory=lambda: _env_bool("GRADCAM_ENABLED", True))

    # Upload validation
    max_upload_mb: float = field(default_factory=lambda: float(os.getenv("MAX_UPLOAD_MB", "10")))
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    allowed_content_types: tuple[str, ...] = ("image/jpeg", "image/jpg", "image/png")

    # CORS
    allowed_origins: list[str] = field(
        default_factory=lambda: _env_list(
            "ALLOWED_ORIGINS",
            [
                "http://127.0.0.1:5500",
                "http://localhost:5500",
                "http://127.0.0.1:3000",
                "http://localhost:3000",
            ],
        )
    )

    # Metadata
    app_version: str = field(default_factory=lambda: os.getenv("APP_VERSION", "1.0.0"))
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "AgriVision PestGuard API"))

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def imagenet_mean(self) -> tuple[float, float, float]:
        return (0.485, 0.456, 0.406)

    @property
    def imagenet_std(self) -> tuple[float, float, float]:
        return (0.229, 0.224, 0.225)


settings = Settings()
