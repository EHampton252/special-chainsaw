"""Target interface and helpers shared by the agent adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from ..models import AgentResponse, ToolCall


class TargetError(Exception):
    """The agent could not be reached or returned something unusable."""


class Target:
    """An agent under test. `messages` is the whole conversation so far, OpenAI-style, ending with
    the new user message. `session` is a per-case dict the target may use to keep conversation state."""

    def __init__(self, config: dict[str, Any], base_dir: Path, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.base_dir = base_dir
        self.client = httpx.AsyncClient(
            timeout=float(config.get("timeout", 120)), transport=transport, follow_redirects=True
        )

    async def send(self, messages: list[dict[str, Any]], session: dict[str, Any]) -> AgentResponse:
        raise NotImplementedError

    async def aclose(self) -> None:
        await self.client.aclose()

    async def request_json(self, method: str, url: str, **kwargs: Any) -> Any:
        try:
            r = await self.client.request(method, url, **kwargs)
        except httpx.HTTPError as e:
            raise TargetError(f"{method} {url} failed: {type(e).__name__}: {e}") from e
        if r.status_code >= 400:
            raise TargetError(f"{method} {url} returned HTTP {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError:
            return r.text


def get_path(obj: Any, path: str) -> Any:
    """Follow a dotted path such as 'choices.0.message.content'."""
    for part in path.split("."):
        try:
            obj = obj[int(part)] if isinstance(obj, list) else obj[part]
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise TargetError(f"Path '{path}' not found in the agent response (failed at '{part}')") from e
    return obj


def normalize_tool_calls(items: Any) -> list[ToolCall]:
    """Accept tool calls as names, {name, args} mappings, or OpenAI-style {function: {name, arguments}}."""
    calls = []
    for item in items or []:
        if isinstance(item, str):
            calls.append(ToolCall(name=item))
            continue
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name") or fn.get("tool") or fn.get("tool_name")
        args = next((fn[k] for k in ("args", "arguments", "input", "parameters") if fn.get(k) is not None), {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if name:
            calls.append(ToolCall(name=str(name), args=args if isinstance(args, dict) else {"_value": args}))
    return calls
