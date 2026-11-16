"""Generic HTTP target: any REST endpoint, described with a request template and response paths."""

from __future__ import annotations

import re
import uuid
from typing import Any

from ..models import AgentResponse
from .base import Target, TargetError, get_path, normalize_tool_calls

_WHOLE = re.compile(r"\s*\{\{\s*(\w+)\s*\}\}\s*")
_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_TEXT_KEYS = ("output", "response", "reply", "answer", "text", "message", "content", "result")


def render(template: Any, variables: dict[str, Any]) -> Any:
    """Fill {{input}}, {{messages}} and {{session_id}}. A string that is only a placeholder keeps the
    variable's type, so `messages: "{{messages}}"` becomes a real list."""
    if isinstance(template, str):
        whole = _WHOLE.fullmatch(template)
        if whole and whole.group(1) in variables:
            return variables[whole.group(1)]
        return _VAR.sub(lambda m: str(variables.get(m.group(1), m.group(0))), template)
    if isinstance(template, dict):
        return {k: render(v, variables) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, variables) for v in template]
    return template


class HttpTarget(Target):
    async def send(self, messages: list[dict[str, Any]], session: dict[str, Any]) -> AgentResponse:
        cfg = self.config
        if "url" not in cfg:
            raise TargetError("http target needs a 'url'")
        variables = {
            "input": messages[-1]["content"],
            "messages": messages,
            "session_id": session.setdefault("session_id", uuid.uuid4().hex),
        }
        method = str(cfg.get("method", "POST")).upper()
        kwargs: dict[str, Any] = {"headers": render(cfg.get("headers", {}), variables)}
        if "params" in cfg:
            kwargs["params"] = render(cfg["params"], variables)
        if method != "GET":
            kwargs["json"] = render(cfg.get("body", {"input": "{{input}}"}), variables)
        data = await self.request_json(method, render(cfg["url"], variables), **kwargs)

        if cfg.get("response_path"):
            output = get_path(data, cfg["response_path"])
        elif isinstance(data, dict):
            output = next((data[k] for k in _TEXT_KEYS if isinstance(data.get(k), str)), None)
            if output is None:
                raise TargetError(
                    f"Could not find the reply text in the response (keys: {sorted(data)}). "
                    f"Set 'response_path' on the target."
                )
        else:
            output = data
        tool_calls = get_path(data, cfg["tool_calls_path"]) if cfg.get("tool_calls_path") else (
            data.get("tool_calls") if isinstance(data, dict) else None
        )
        return AgentResponse(output=output if isinstance(output, str) else str(output),
                             tool_calls=normalize_tool_calls(tool_calls))
