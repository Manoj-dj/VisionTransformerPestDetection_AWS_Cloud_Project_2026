# Deployment Notes — Local → AWS

This maps each local component to the AWS service it would become in
production. These are planning notes, not executable infrastructure code.

## 1. Model artifact: local file → S3

- Upload `src/ai_model/ViT_best.pth` and `src/ai_model/classes.txt` to an S3
  bucket, e.g. `s3://agrivision-pestguard-models/v1/`.
- Set `S3_MODEL_URI` (already present in `.env.example`) to that location.
- At container/instance startup, download the artifact from S3 to a local
  path before calling `load_model_bundle()` — `model_loader.py` already
  reads the checkpoint path from `MODEL_PATH`, so only the startup script
  changes, not the inference code.

## 2. Compute: FastAPI process → SageMaker or EC2/ECS

Two realistic paths:

- **SageMaker real-time inference endpoint**: wrap `model_loader.py` /
  `inference.py` / `explainability.py` in a SageMaker inference script
  (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) inside a custom
  container (e.g. based on the PyTorch inference DLC). Deploy behind a
  SageMaker endpoint; invoke via `InvokeEndpoint`.
- **EC2 / ECS with the existing FastAPI app**: run the same `uvicorn
  app.main:app` process inside a container on ECS Fargate, or directly on
  an EC2 instance behind an Application Load Balancer. This requires the
  least code change since the FastAPI app is used as-is.

Either way, keep `DEVICE=auto` (or pin to the instance's actual hardware)
and size the instance based on ViT-B/16 CPU/GPU latency requirements.

## 3. API routing: `/predict` → API Gateway

- If using SageMaker: API Gateway → Lambda (thin proxy that base64-decodes
  the multipart body, calls `InvokeEndpoint`, and reshapes the SageMaker
  response into the same JSON contract the frontend already expects).
- If using EC2/ECS: API Gateway HTTP API with a VPC Link to the ALB, or
  expose the ALB directly with its own custom domain + TLS certificate.
- The response body is already JSON with Base64-encoded images, which is
  compatible with API Gateway's Lambda proxy integration payload format.

## 4. Logs: stdout → CloudWatch

- On EC2/ECS: attach the CloudWatch Logs agent or the `awslogs` log driver
  to the container/service definition; no application code changes needed
  since `logging_config.py` already logs structured lines to stdout.
- On Lambda: CloudWatch Logs is automatic for anything printed via the
  standard `logging` module.
- Consider a CloudWatch metric filter on `confidence=` and `latency_ms=`
  log fields to build a prediction-quality/latency dashboard.

## 5. Auth: none locally → Cognito

- Local dev intentionally has no authentication (CORS is restricted to
  known local origins instead).
- In production, put a Cognito User Pool in front of API Gateway (Cognito
  authorizer) so `/predict` requires a valid JWT. The frontend would add an
  `Authorization: Bearer <token>` header to its `fetch()` call; `app.js`'s
  single `API_BASE_URL` constant plus a small auth module would be the only
  frontend change needed.

## 6. Alerts: none locally → SNS

- When a prediction's confidence for a known high-risk pest class exceeds a
  threshold, publish a message to an SNS topic (e.g.
  `agrivision-pest-alerts`) so subscribed agronomists/field staff get an
  email/SMS notification.
- This would be a small addition inside the `/predict` route (or the
  Lambda proxy in the API Gateway path) after classification completes —
  it does not require changing the model or preprocessing pipeline.

## 7. Frontend: local static files → S3 + CloudFront

- `src/frontend/` has no build step, so it can be uploaded to an S3 bucket
  configured for static website hosting and served through CloudFront with
  HTTPS.
- Update `API_BASE_URL` in `js/app.js` to the deployed API Gateway/ALB
  domain before publishing.
