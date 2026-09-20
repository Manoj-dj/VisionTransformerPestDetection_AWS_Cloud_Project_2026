"""Tests for /predict and /predict/classification-only using the real checkpoint."""

from __future__ import annotations


def test_rejects_invalid_file_extension(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400


def test_rejects_empty_file(client):
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400


def test_rejects_corrupted_image(client, corrupted_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("fake.jpg", corrupted_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 400


def test_valid_jpeg_returns_prediction_with_confidence_in_range(client, valid_jpeg_bytes):
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", valid_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()

    predicted = body["predicted_class"]
    assert 0.0 <= predicted["confidence"] <= 1.0
    assert 0 <= predicted["class_index"] <= 101
    assert predicted["class_label"] == predicted["class_index"] + 1
    assert isinstance(predicted["class_name"], str) and predicted["class_name"]

    assert "request_id" in body
    assert body["latency_ms"] >= 0
    assert body["model_version"] == "vit-ip102-v1"
    assert body["image"] == {"width": 224, "height": 224, "color_mode": "RGB"}


def test_valid_png_accepted(client, valid_png_bytes):
    response = client.post(
        "/predict",
        files={"file": ("sample.png", valid_png_bytes, "image/png")},
    )
    assert response.status_code == 200


def test_top_k_predictions_sorted_descending(client, valid_jpeg_bytes):
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", valid_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    top_k = response.json()["top_k_predictions"]
    confidences = [entry["confidence"] for entry in top_k]
    assert confidences == sorted(confidences, reverse=True)
    assert len(top_k) <= 5


def test_gradcam_present_or_structured_failure(client, valid_jpeg_bytes):
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", valid_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    explanation = response.json()["explanation"]
    assert "enabled" in explanation
    if explanation["enabled"]:
        assert explanation["method"] == "Grad-CAM"
        assert explanation["heatmap_base64"]
        assert explanation["overlay_base64"]
    else:
        assert "error" in explanation


def test_classification_only_has_no_explanation_field(client, valid_jpeg_bytes):
    response = client.post(
        "/predict/classification-only",
        files={"file": ("sample.jpg", valid_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "explanation" not in body
    assert 0.0 <= body["predicted_class"]["confidence"] <= 1.0


def test_model_checkpoint_loaded_at_startup(client):
    # If the checkpoint failed to load, the TestClient's lifespan startup
    # would have raised during fixture setup, so reaching this point at all
    # already proves the checkpoint loaded. /health confirms it explicitly.
    response = client.get("/health")
    assert response.json()["model_loaded"] is True
