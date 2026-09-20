"""Shared pytest fixtures for the AgriVision PestGuard backend tests.

These tests exercise the real checkpoint at src/ai_model/ViT_best.pth via
the FastAPI TestClient's lifespan startup - there is no mocked model.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _make_image_bytes(fmt: str, size=(300, 200), color=(34, 139, 34)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    return _make_image_bytes("JPEG")


@pytest.fixture
def valid_png_bytes() -> bytes:
    return _make_image_bytes("PNG")


@pytest.fixture
def corrupted_image_bytes() -> bytes:
    # Looks like a JPEG by extension but is not decodable image data.
    return b"this-is-not-a-real-jpeg-file" * 10
