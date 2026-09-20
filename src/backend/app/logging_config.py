"""Logging setup for the AgriVision PestGuard backend.

Uses the standard library logging module with a plain formatter so logs
can later be shipped to CloudWatch (e.g. via the CloudWatch agent or the
awslogs driver) without any change to how the application logs. Never log
raw image bytes or other sensitive payload data here.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Already configured (e.g. re-imported under a test runner).
        root_logger.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # Keep third-party libraries reasonably quiet.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("timm").setLevel(logging.WARNING)
