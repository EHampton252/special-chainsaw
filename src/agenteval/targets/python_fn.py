"""In-process target: call a Python function directly. Good for local agents and for trying agenteval out."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
from typing import Any

from ..models import AgentResponse
from .base import Target, TargetError, normalize_tool_calls


class PythonTarget(Target):
    """`callable` is 'path/to/file.py:function' (relative to the suite) or 'package.module:function'.

    The function receives (input, history), where history is the earlier messages of the conversation,
    and returns either the reply text or {"output": str, "tool_calls": [...]}. It may be async."""

    _fn: Any = None

    def _load(self) -> Any:
        ref = str(self.config.get("callable", ""))
        location, _, name = ref.rpartition(":")
        if not location or not name:
            raise TargetError("python target needs 'callable' as 'file.py:function' or 'module:function'")
        try:
            if location.endswith(".py"):
                path = self.base_dir / location
                spec = importlib.util.spec_from_file_location(path.stem, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot load {path}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(location)
            return getattr(module, name)
        except (ImportError, AttributeError, FileNotFoundError) as e:
            raise TargetError(f"Could not load '{ref}': {e}") from e

    async def send(self, messages: list[dict[str, Any]], session: dict[str, Any]) -> AgentResponse:
        if self._fn is None:
            self._fn = self._load()
        text, history = messages[-1]["content"], messages[:-1]
        if inspect.iscoroutinefunction(self._fn):
            result = await self._fn(text, history)
        else:
            result = await asyncio.to_thread(self._fn, text, history)
        if isinstance(result, dict):
            return AgentResponse(output=str(result.get("output", "")),
                                 tool_calls=normalize_tool_calls(result.get("tool_calls")))
        return AgentResponse(output=str(result))
