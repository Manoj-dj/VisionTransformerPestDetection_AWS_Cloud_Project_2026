# AgriVision PestGuard — Backend

FastAPI service that serves the trained ViT-B/16 IP102 pest classifier
(`src/ai_model/ViT_best.pth`) plus Grad-CAM explanations. Runs locally on
CPU or GPU and is structured to migrate cleanly to AWS (S3 + SageMaker/EC2 +
API Gateway + CloudWatch + Cognito + SNS) without rewriting the inference
logic.

## 1. Model summary

| Property | Value |
|---|---|
| Architecture | `timm` Vision Transformer |
| Model name | `vit_base_patch16_224` |
| Input size | 224 x 224 RGB |
| Patch size | 16 |
| Classes | 102 (IP102) |
| Checkpoint | `src/ai_model/ViT_best.pth` (raw `state_dict()`, no `module.` prefixes) |

Inference preprocessing (must match training/validation exactly, **no
augmentation**):

```python
Resize((224, 224))
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Original training hyperparameters (documented for reproducibility only — not used at inference time)

```
BASE_LR         = 3e-5   # backbone learning rate (AdamW)
HEAD_LR         = 1e-4   # classification head learning rate (AdamW)
WEIGHT_DECAY    = 0.05
WARMUP_EPOCHS   = 2
MAX_EPOCHS      = 20
LABEL_SMOOTHING = 0.1
GRAD_CLIP_NORM  = 1.0
```

Training also used a weighted sampler, class-weighted loss, and cosine LR
scheduling with early stopping. None of this affects inference — the
checkpoint already contains the trained weights and is loaded with
`pretrained=False`.

## 2. Class labels

Class names are loaded from `src/ai_model/classes.txt` (one class per
line: `<one_based_label> <class_name>`). The file is one-based; the model
output is zero-based, converted as `model_index = class_file_label - 1`.
Every API response includes both `class_index` (zero-based) and
`class_label` (one-based, matching `classes.txt`).

If `classes.txt` is missing, the app writes a placeholder file with format
instructions to that path and refuses to start until you replace it with
the real, ordered IP102 list — it never invents or reorders labels.

## 3. Project layout

```
src/backend/
├── app/
│   ├── main.py              # FastAPI app, routes, middleware, exception handlers
│   ├── config.py            # Environment-driven settings (paths, CORS, limits)
│   ├── schemas.py           # Pydantic request/response models
│   ├── model_loader.py      # Checkpoint + classes.txt loading (once, at startup)
│   ├── inference.py         # Pure classification logic (framework-independent)
│   ├── explainability.py    # Grad-CAM generation
│   ├── image_utils.py       # Upload validation, decoding, preprocessing
│   └── logging_config.py    # Structured stdout logging
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   └── test_prediction.py
├── requirements.txt
├── .env.example
└── pytest.ini
```

## 4. Setup and run (Windows)

```powershell
cd src/backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Using `uv` instead of `venv`/`pip` (Windows)

```powershell
cd src/backend
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## 5. Setup and run (Linux/macOS)

```bash
cd src/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Verify it's up:

```bash
curl http://127.0.0.1:8000/health
```

## 6. API

### `GET /health`
Fast liveness/readiness check. Never reloads the model.

### `GET /metadata`
Model name, class count, input size, normalization stats, Grad-CAM flag,
app version.

### `POST /predict`
Multipart upload, field name `file`. JPEG/JPG/PNG only, 10 MB max
(configurable via `MAX_UPLOAD_MB`). Returns top-5 predictions, image info,
latency, and a Grad-CAM explanation with Base64-encoded PNGs.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/test_image.jpg"
```

### `POST /predict/classification-only`
Same validation and classification as `/predict`, without Grad-CAM, for
lower latency.

## 7. Tests

```bash
pytest -q
```

Tests load the real checkpoint via FastAPI's startup lifecycle — there is
no mocked model. They cover `/health`, `/metadata`, invalid/corrupted
uploads, a real prediction's confidence range, descending top-k ordering,
and graceful Grad-CAM failure handling.

## 8. Configuration (environment variables)

See `.env.example`. Key variables: `MODEL_PATH`, `CLASSES_PATH`, `DEVICE`
(`auto`/`cpu`/`cuda`), `MAX_UPLOAD_MB`, `ALLOWED_ORIGINS`,
`GRADCAM_ENABLED`, `TOP_K`.

CORS defaults to the local frontend dev origins only
(`http://127.0.0.1:5500`, `http://localhost:5500`,
`http://127.0.0.1:3000`, `http://localhost:3000`) — `allow_origins=["*"]`
is never used by default.

## 9. AWS migration map

| Local component | Future AWS service |
|---|---|
| `src/ai_model/ViT_best.pth` on disk | S3 model artifact (`S3_MODEL_URI`) |
| FastAPI process (`uvicorn`) | SageMaker inference endpoint, or ECS/EC2 behind an ALB |
| `POST /predict` | API Gateway route → Lambda or container backend |
| `logging_config.py` stdout logs | CloudWatch Logs |
| (none locally) | Cognito for authenticated API access |
| Prediction result / high-confidence pest alert | SNS notification |

The inference pipeline (`image_utils.py`, `inference.py`,
`explainability.py`) has no FastAPI or filesystem-path dependencies beyond
reading the checkpoint and class list, so it can be called directly from a
SageMaker inference script or a Lambda handler without modification.

## 10. Security notes

- Uploaded files are validated by extension and declared MIME type, size-limited, and never written to disk.
- Internal exception details are never returned to clients; a generic 500 with a request ID is returned instead, with the full trace only in server logs.
- No image bytes are logged.
