"""Grad-CAM explainability for the ViT-B/16 IP102 classifier.

Uses pytorch-grad-cam against the final transformer block, with the
reshape transform required to turn ViT's (batch, tokens, channels)
activations back into a spatial (batch, channels, height, width) map.

This is an explanation heatmap over the 224x224 input the model actually
saw, NOT an object-detection bounding box. It highlights image regions
that most influenced the predicted class.
"""

from __future__ import annotations

import base64
import io
import logging

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.model_loader import ModelBundle, gradcam_lock

logger = logging.getLogger("agrivision.explainability")


class ExplainabilityError(RuntimeError):
    """Raised when Grad-CAM generation fails; must never crash /predict."""


def reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Reshape ViT token activations (batch, 1 + N, C) into (batch, C, H, W).

    Drops the leading class token, then reshapes the remaining N=H*W patch
    tokens into a spatial grid pytorch-grad-cam can treat like a CNN
    feature map.
    """

    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def _encode_png_base64(rgb_uint8: np.ndarray) -> str:
    image = Image.fromarray(rgb_uint8)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def generate_gradcam(
    bundle: ModelBundle,
    input_tensor: torch.Tensor,
    rgb_float_image: np.ndarray,
    target_class_index: int,
) -> tuple[str, str]:
    """Generate a Grad-CAM heatmap and overlay for the given predicted class.

    Parameters
    ----------
    bundle: the loaded model bundle (model + device).
    input_tensor: the (1, 3, H, W) normalized model input (gradients required).
    rgb_float_image: the (H, W, 3) float32 [0, 1] RGB array the heatmap is
        overlaid on (should be the resized, un-normalized image).
    target_class_index: zero-based model class index to explain (the
        predicted class by default).

    Returns
    -------
    (heatmap_base64_png, overlay_base64_png)

    Raises
    ------
    ExplainabilityError if Grad-CAM computation fails for any reason. The
    caller is expected to catch this and degrade gracefully rather than
    fail the whole prediction request.
    """

    model = bundle.model
    target_layer = model.blocks[-1].norm1

    input_tensor = input_tensor.to(bundle.device).clone()
    input_tensor.requires_grad_(True)

    try:
        with gradcam_lock:
            was_training = model.training
            model.eval()
            try:
                cam = GradCAM(
                    model=model,
                    target_layers=[target_layer],
                    reshape_transform=reshape_transform,
                )
                targets = [ClassifierOutputTarget(target_class_index)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            finally:
                if was_training:
                    model.train()

        grayscale_cam = grayscale_cam[0, :]  # (H, W) in [0, 1]

        overlay = show_cam_on_image(rgb_float_image, grayscale_cam, use_rgb=True)

        heatmap_color = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        heatmap_base64 = _encode_png_base64(heatmap_color)
        overlay_base64 = _encode_png_base64(overlay)

        return heatmap_base64, overlay_base64
    except Exception as exc:  # noqa: BLE001 - must never propagate to /predict
        logger.exception("Grad-CAM generation failed")
        raise ExplainabilityError(f"Grad-CAM generation failed: {exc}") from exc
    finally:
        input_tensor.requires_grad_(False)
        del input_tensor
