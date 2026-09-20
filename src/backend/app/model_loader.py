"""Model and class-label loading for the AgriVision PestGuard backend.

The model and the ordered class-name list are loaded exactly once, at
application startup, and cached in this module. Nothing here reloads the
checkpoint per-request. This is the only intentional module-level mutable
state in the application, and it is treated as read-only after startup.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import timm
import torch

from app.config import settings

logger = logging.getLogger("agrivision.model_loader")

# Grad-CAM backward passes are not guaranteed to be thread-safe against
# concurrent forward/backward calls on the same module. A single global lock
# serializes any model execution that computes gradients.
gradcam_lock = threading.Lock()


class ClassListError(RuntimeError):
    """Raised when classes.txt is missing, malformed, or the wrong size."""


@dataclass(frozen=True)
class ClassEntry:
    """One ordered class entry.

    model_index: zero-based index as produced by the model's output layer.
    class_label: one-based label exactly as written in classes.txt.
    class_name: human-readable class name.
    """

    model_index: int
    class_label: int
    class_name: str


def _write_placeholder_classes_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PLACEHOLDER classes.txt - DO NOT USE FOR REAL INFERENCE.",
        "# Replace this file with the real, ordered IP102 class list.",
        "# Format: one class per line, one-based label followed by the class name, e.g.:",
        "# 1 rice leaf roller",
        "# 2 rice leaf caterpillar",
        "# ...",
        "# 102 Cicadellidae",
        "#",
        "# The application will refuse to start until 102 correctly ordered",
        "# labels are present in this file.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_class_entries(classes_path: Path, expected_count: int) -> list[ClassEntry]:
    """Load and validate the ordered IP102 class list.

    Raises ClassListError with a human-readable, actionable message if the
    file is missing, malformed, or does not contain exactly `expected_count`
    sequentially-ordered one-based labels. This function never invents or
    reorders labels.
    """

    if not classes_path.exists():
        _write_placeholder_classes_file(classes_path)
        raise ClassListError(
            f"classes.txt was not found at '{classes_path}'. A placeholder file has been "
            "created there with formatting instructions. You must replace it with the real, "
            "ordered IP102 class list (102 lines, 'one_based_label class_name' per line) "
            "before the API can serve predictions."
        )

    entries: list[ClassEntry] = []
    with classes_path.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ClassListError(
                    f"classes.txt line {line_number} is malformed: '{raw_line.rstrip()}'. "
                    "Expected format: '<one_based_label> <class_name>'."
                )
            label_str, name = parts
            try:
                class_label = int(label_str)
            except ValueError as exc:
                raise ClassListError(
                    f"classes.txt line {line_number} has a non-integer label '{label_str}'."
                ) from exc

            model_index = len(entries)
            expected_label = model_index + 1
            if class_label != expected_label:
                raise ClassListError(
                    f"classes.txt is out of order at line {line_number}: expected one-based "
                    f"label {expected_label} but found {class_label}. Labels must appear in "
                    "strictly increasing order starting at 1, with no gaps or duplicates."
                )

            entries.append(
                ClassEntry(model_index=model_index, class_label=class_label, class_name=name.strip())
            )

    if len(entries) != expected_count:
        raise ClassListError(
            f"classes.txt must contain exactly {expected_count} classes, but {len(entries)} "
            f"were found in '{classes_path}'. Verify the real IP102 classes.txt was copied in full."
        )

    return entries


def resolve_device(preference: str) -> str:
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not torch.cuda.is_available():
            logger.warning("DEVICE=cuda requested but CUDA is not available; falling back to cpu.")
            return "cpu"
        return "cuda"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


class ModelBundle:
    """Holds the loaded ViT model, its device, and the ordered class list."""

    def __init__(self, model: torch.nn.Module, device: str, class_entries: list[ClassEntry]):
        self.model = model
        self.device = device
        self.class_entries = class_entries

    def class_entry_for_index(self, model_index: int) -> ClassEntry:
        return self.class_entries[model_index]


def load_model_bundle() -> ModelBundle:
    """Instantiate the ViT architecture and load the supplied checkpoint.

    This must be called exactly once, at application startup. Uses
    pretrained=False because the checkpoint already contains the trained
    weights; downloading ImageNet-pretrained weights is neither needed nor
    desired here.
    """

    model_path = settings.model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. Ensure ViT_best.pth is present at "
            "src/ai_model/ViT_best.pth, or set MODEL_PATH to its location."
        )

    class_entries = load_class_entries(settings.classes_path, settings.num_classes)

    device = resolve_device(settings.device_preference)

    logger.info("Loading model architecture '%s' (num_classes=%d)", settings.model_name, settings.num_classes)
    model = timm.create_model(settings.model_name, pretrained=False, num_classes=settings.num_classes)

    logger.info("Loading checkpoint weights from '%s'", model_path)
    state_dict = torch.load(str(model_path), map_location="cpu")

    # Some checkpoints wrap the actual weights under a "state_dict" key.
    if isinstance(state_dict, dict) and "state_dict" in state_dict and not any(
        key.startswith(("cls_token", "pos_embed", "patch_embed", "blocks.", "model."))
        for key in state_dict.keys()
    ):
        state_dict = state_dict["state_dict"]

    # Defensively strip a "module." prefix in case the checkpoint was ever
    # saved from a DataParallel-wrapped model.
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    # The supplied checkpoint was saved from a training-time wrapper class
    # (e.g. `self.model = timm.create_model(...)`), so every key is prefixed
    # with "model.". Strip it so the keys match the raw timm architecture
    # instantiated above. This only renames keys; it does not alter weights.
    if any(key.startswith("model.") for key in state_dict.keys()):
        state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully on device '%s'", device)
    return ModelBundle(model=model, device=device, class_entries=class_entries)
