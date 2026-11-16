"""OpenAI-compatible chat completions target (OpenAI, Azure/Foundry v1 endpoints, Ollama, LM Studio, vLLM...)."""

from __future__ import annotations

import os
from typing import Any

from ..models import AgentResponse
from .base import Target, TargetError, normalize_tool_calls


def auth_headers(api_key_env: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = dict(extra or {})
    key = os.environ.get(api_key_env) if api_key_env else None
    if key and "Authorization" not in headers and "api-key" not in headers:
        headers["Authorization"] = f"Bearer {key}"
    return headers


class OpenAITarget(Target):
    async def send(self, messages: list[dict[str, Any]], session: dict[str, Any]) -> AgentResponse:
        cfg = self.config
        if "model" not in cfg:
            raise TargetError("openai target needs a 'model'")
        base = str(cfg.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        system = [{"role": "system", "content": cfg["system"]}] if cfg.get("system") else []
        body = {"model": cfg["model"], "messages": system + messages, **cfg.get("body", {})}
        data = await self.request_json(
            "POST", f"{base}/chat/completions", json=body,
            headers=auth_headers(cfg.get("api_key_env", "OPENAI_API_KEY"), cfg.get("headers")),
        )
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise TargetError(f"Unexpected chat completions response: {str(data)[:300]}") from e
        return AgentResponse(
            output=message.get("content") or "",
            tool_calls=normalize_tool_calls(message.get("tool_calls")),
            usage=data.get("usage"),
        )
