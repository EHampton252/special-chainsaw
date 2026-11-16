"""Agent2Agent (A2A) target. Speaks protocol 1.0 (SendMessage / HTTP+JSON) and legacy 0.3 (message/send)."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from ..models import AgentResponse
from .base import Target, TargetError

_RUNNING = {"submitted", "working"}
_FAILED = {"failed", "canceled", "rejected"}


def _state(task: dict[str, Any]) -> str:
    return str((task.get("status") or {}).get("state", "")).lower().removeprefix("task_state_")


def _text(parts: Any) -> str:
    return "\n".join(p["text"] for p in parts or [] if isinstance(p, dict) and isinstance(p.get("text"), str))


class A2ATarget(Target):
    endpoint: str | None = None
    binding = "JSONRPC"
    version = "1.0"
    legacy = False
    _ready = False

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._lock = asyncio.Lock()
        self.headers: dict[str, str] = {}

    async def _discover(self) -> None:
        cfg = self.config
        if "url" not in cfg:
            raise TargetError("a2a target needs a 'url' (the agent's base URL or its agent card URL)")
        url = str(cfg["url"]).rstrip("/")
        self.endpoint, self.headers = url, dict(cfg.get("headers", {}))
        version = cfg.get("protocol_version")
        is_card_url = url.endswith(".json")
        if cfg.get("discover", True):
            card_url = url if is_card_url else f"{url}/.well-known/agent-card.json"
            try:
                card = await self.request_json("GET", card_url, headers=self.headers)
            except TargetError:
                if is_card_url:
                    raise
                card = None  # no card published: treat the URL as the JSON-RPC endpoint
            if isinstance(card, dict):
                interfaces = card.get("supportedInterfaces") or []
                pick = next((i for i in interfaces if i.get("protocolBinding") == "JSONRPC"), None) or next(
                    (i for i in interfaces if i.get("protocolBinding") == "HTTP+JSON"), None)
                if pick:
                    self.endpoint, self.binding = pick["url"].rstrip("/"), pick["protocolBinding"]
                    version = version or pick.get("protocolVersion")
                elif card.get("url"):  # 0.3-style card
                    legacy = [{"url": card["url"], "transport": card.get("preferredTransport", "JSONRPC")}]
                    legacy += card.get("additionalInterfaces") or []
                    rpc = next((i for i in legacy if i.get("transport") == "JSONRPC"), None)
                    if not rpc:
                        raise TargetError("The agent card offers no JSONRPC or HTTP+JSON interface")
                    self.endpoint = rpc["url"].rstrip("/")
                    version = version or card.get("protocolVersion") or "0.3"
                else:
                    raise TargetError("The agent card has neither 'supportedInterfaces' nor 'url'")
        self.version = str(version or "1.0")
        self.legacy = self.version.startswith("0.")
        if not self.legacy:
            self.headers.setdefault("A2A-Version", self.version)

    async def _call(self, method: str, legacy_method: str, params: dict[str, Any], rest: tuple[str, str]) -> dict[str, Any]:
        assert self.endpoint
        if self.binding == "HTTP+JSON":
            verb, path = rest
            data = await self.request_json(verb, self.endpoint + path, headers=self.headers,
                                           **({"json": params} if verb == "POST" else {}))
        else:
            payload = {"jsonrpc": "2.0", "id": uuid.uuid4().hex,
                       "method": legacy_method if self.legacy else method, "params": params}
            data = await self.request_json("POST", self.endpoint, json=payload, headers=self.headers)
            if isinstance(data, dict) and data.get("error"):
                err = data["error"]
                raise TargetError(f"A2A error {err.get('code')}: {err.get('message')}")
            data = data.get("result") if isinstance(data, dict) else None
        if not isinstance(data, dict):
            raise TargetError(f"Unexpected A2A response: {str(data)[:300]}")
        return data

    async def send(self, messages: list[dict[str, Any]], session: dict[str, Any]) -> AgentResponse:
        async with self._lock:  # cases run concurrently; discover the agent card once
            if not self._ready:
                await self._discover()
                self._ready = True
        text = messages[-1]["content"]
        if self.legacy:
            message: dict[str, Any] = {"kind": "message", "role": "user", "parts": [{"kind": "text", "text": text}]}
        else:
            message = {"role": "ROLE_USER", "parts": [{"text": text}]}
        message["messageId"] = uuid.uuid4().hex
        if session.get("context_id"):
            message["contextId"] = session["context_id"]

        result = await self._call("SendMessage", "message/send", {"message": message}, ("POST", "/message:send"))
        # 1.0 wraps the result as {"task": ...} or {"message": ...}; 0.3 returns the object directly.
        task = result.get("task") or (result if "status" in result else None)
        reply = result.get("message") or (result if task is None else None)
        if task is None:
            session["context_id"] = reply.get("contextId") or session.get("context_id")
            return AgentResponse(output=_text(reply.get("parts")))

        deadline = time.monotonic() + float(self.config.get("timeout", 120))
        while _state(task) in _RUNNING:
            if time.monotonic() > deadline:
                raise TargetError(f"A2A task {task.get('id')} did not finish in time (state: {_state(task)})")
            await asyncio.sleep(float(self.config.get("poll_interval", 1)))
            got = await self._call("GetTask", "tasks/get", {"id": task["id"]}, ("GET", f"/tasks/{task['id']}"))
            task = got.get("task") or got
        session["context_id"] = task.get("contextId") or session.get("context_id")
        status_text = _text(((task.get("status") or {}).get("message") or {}).get("parts"))
        if _state(task) in _FAILED:
            raise TargetError(f"A2A task ended as '{_state(task)}': {status_text or 'no detail given'}")
        output = "\n".join(t for t in (_text(a.get("parts")) for a in task.get("artifacts") or []) if t)
        return AgentResponse(output=output or status_text)
