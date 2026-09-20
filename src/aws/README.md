# AWS Migration Notes

This directory intentionally contains **documentation only** — no
CloudFormation/Terraform/CDK stubs that would pretend to deploy
successfully without real configuration (VPC, IAM roles, model artifact
locations, domain names, etc. all vary per AWS account).

See `deployment_notes.md` for the component-by-component migration plan
from the local FastAPI + static-frontend stack to a production AWS
architecture.

## Why the local app is already AWS-shaped

- **No hardcoded local paths**: `MODEL_PATH` / `CLASSES_PATH` are environment
  variables (see `src/backend/app/config.py`), defaulting to a path relative
  to the repo. In AWS these would point at a path where the model artifact
  was downloaded from S3 (e.g. by a SageMaker container's `model_fn`, or an
  EC2 startup script).
- **Stateless requests**: `/predict` reads the upload into memory, never
  writes it to disk, and returns a JSON-safe response (Base64-encoded PNGs,
  no local file paths) — compatible with API Gateway's request/response
  handling and with SageMaker's synchronous invocation contract.
- **Framework-independent inference core**: `image_utils.py`,
  `inference.py`, and `explainability.py` do not import FastAPI. The same
  functions can be called from a SageMaker inference script
  (`model_fn`/`input_fn`/`predict_fn`/`output_fn`) or a Lambda handler.
- **Structured logging**: `logging_config.py` logs to stdout in a plain
  text format; this is exactly what the CloudWatch Logs agent / awslogs
  driver captures with zero code changes.
- **Configurable CORS and limits**: origins, upload size limits, and device
  selection are all environment variables, matching how ECS/EC2 task
  definitions and SageMaker environment variables are typically configured.
