"""Agent adapters. Add a platform by subclassing Target and registering it here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from ..models import ConfigError
from .a2a import A2ATarget
from .base import Target, TargetError
from .http import HttpTarget
from .openai_compat import OpenAITarget
from .python_fn import PythonTarget

TARGETS: dict[str, type[Target]] = {
    "http": HttpTarget,
    "openai": OpenAITarget,
    "a2a": A2ATarget,
    "python": PythonTarget,
}


def create_target(config: dict[str, Any], base_dir: Path, transport: httpx.AsyncBaseTransport | None = None) -> Target:
    kind = config.get("type")
    if kind not in TARGETS:
        raise ConfigError(f"Unknown target type '{kind}'; available: {sorted(TARGETS)}")
    return TARGETS[kind](config, base_dir, transport)


__all__ = ["TARGETS", "Target", "TargetError", "create_target"]
