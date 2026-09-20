"""Tests for /health and /metadata."""

from __future__ import annotations


def test_health_returns_200_and_expected_shape(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "vit_base_patch16_224"
    assert body["num_classes"] == 102
    assert body["device"] in {"cpu", "cuda"}


def test_metadata_reports_102_classes_and_224_input(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["num_classes"] == 102
    assert body["input_width"] == 224
    assert body["input_height"] == 224
    assert body["normalization_mean"] == [0.485, 0.456, 0.406]
    assert body["normalization_std"] == [0.229, 0.224, 0.225]
    assert isinstance(body["gradcam_enabled"], bool)
