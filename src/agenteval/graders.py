"""Deterministic graders. They cost nothing to run, so the runner evaluates them before any LLM judge."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import GradeResult, GraderSpec, ToolCall

JUDGE_TYPES = {"rubric", "correctness"}
NEEDS_EXPECTED = {"correctness", "similarity", "equals_expected"}


@dataclass
class GradeContext:
    input: str
    output: str
    expected: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0


def _values(p: dict[str, Any]) -> list[str]:
    v = p.get("value", p.get("values", []))
    return [str(x) for x in (v if isinstance(v, list) else [v])]


def _norm(text: str, p: dict[str, Any]) -> str:
    return text.lower() if p.get("ignore_case", True) else text


def _contains(ctx: GradeContext, p: dict[str, Any]) -> tuple[bool, str]:
    missing = [v for v in _values(p) if _norm(v, p) not in _norm(ctx.output, p)]
    return not missing, f"missing: {missing}" if missing else "all values found"


def _contains_any(ctx: GradeContext, p: dict[str, Any]) -> tuple[bool, str]:
    found = [v for v in _values(p) if _norm(v, p) in _norm(ctx.output, p)]
    return bool(found), f"found: {found}" if found else f"none of {_values(p)} found"


def _not_contains(ctx: GradeContext, p: dict[str, Any]) -> tuple[bool, str]:
    found = [v for v in _values(p) if _norm(v, p) in _norm(ctx.output, p)]
    return not found, f"forbidden value(s) present: {found}" if found else "no forbidden values"


def _equals(ctx: GradeContext, p: dict[str, Any]) -> tuple[bool, str]:
    want = str(p.get("value", ""))
    p = {"ignore_case": False, **p}
    ok = _norm(ctx.output.strip(), p) == _norm(want.strip(), p)
    return ok, "exact match" if ok else f"expected exactly {want!r}"


def _equals_expected(ctx: GradeContext, p: dict[str, Any]) -> tuple[bool, str]:
    return _equals(ctx, {**p, "value": ctx.expected or ""})


def _regex(ctx: GradeContext, p: dict[str, Any]) -> tuple[bool, str]:
    pattern = str(p.get("pattern", p.get("value", "")))
    ok = re.search(pattern, ctx.output, re.IGNORECASE if p.get("ignore_case") else 0) is not None
    return ok, f"pattern {pattern!r} " + ("matched" if ok else "did not match")


def _not_empty(ctx: GradeContext, p: dict[str, Any]) -> tuple[bool, str]:
    ok = bool(ctx.output.strip())
    return ok, "response is not empty" if ok else "response is empty"


def _is_json(ctx: GradeContext, p: dict[str, Any]) -> tuple[bool, str]:
    try:
        json.loads(ctx.output)
        return True, "valid JSON"
    except json.JSONDecodeError as e:
        return False, f"not valid JSON: {e.msg}"


def _similarity(ctx: GradeContext, p: dict[str, Any]) -> tuple[bool, str]:
    threshold = float(p.get("threshold", p.get("value", 0.8)))
    ratio = difflib.SequenceMatcher(None, ctx.output.lower().split(), (ctx.expected or "").lower().split()).ratio()
    return ratio >= threshold, f"word similarity {ratio:.2f} (threshold {threshold:.2f})"


def _latency(ctx: GradeContext, p: dict[str, Any]) -> tuple[bool, str]:
    limit = float(p.get("max_ms", p.get("value", 0)))
    return ctx.latency_ms <= limit, f"{ctx.latency_ms:.0f} ms (limit {limit:.0f} ms)"


def _args_match(expected: dict[str, Any], actual: dict[str, Any], mode: str) -> bool:
    if mode == "ignore":
        return True
    if mode == "exact":
        return expected == actual
    return all(k in actual and actual[k] == v for k, v in expected.items())  # subset


def _tool_calls(ctx: GradeContext, p: dict[str, Any]) -> tuple[bool, str]:
    """Compare the agent's tool-call trajectory to a reference.

    mode: strict (same calls, same order) | unordered (same calls, any order)
        | superset (all expected calls made, extras allowed; default) | subset (no calls beyond expected)
    args: subset (expected args must be present; default) | exact | ignore
    """
    raw = p.get("expected", p.get("value", []))
    expected = [ToolCall(name=e) if isinstance(e, str) else ToolCall(**e) for e in raw]
    mode, args_mode = p.get("mode", "superset"), p.get("args", "subset")
    if mode not in ("strict", "unordered", "superset", "subset"):
        return False, f"unknown tool_calls mode '{mode}'"
    actual = ctx.tool_calls

    def same(e: ToolCall, a: ToolCall) -> bool:
        return e.name == a.name and _args_match(e.args, a.args, args_mode)

    called = [f"{a.name}({json.dumps(a.args, default=str)})" for a in actual]
    if mode == "strict":
        ok = len(expected) == len(actual) and all(same(e, a) for e, a in zip(expected, actual))
        return ok, f"called in order: {called}"
    remaining = list(actual)
    unmatched = []
    for e in expected:
        hit = next((a for a in remaining if same(e, a)), None)
        if hit is None:
            unmatched.append(e.name)
        else:
            remaining.remove(hit)
    extra = [a.name for a in remaining]
    if mode == "superset":
        return not unmatched, f"expected call(s) not made: {unmatched}; called: {called}" if unmatched else f"called: {called}"
    if mode == "subset":
        return not extra, f"unexpected call(s): {extra}" if extra else f"called: {called}"
    ok = not unmatched and not extra
    return ok, f"called: {called}" if ok else f"not made: {unmatched}; unexpected: {extra}"


DETERMINISTIC: dict[str, Callable[[GradeContext, dict[str, Any]], tuple[bool, str]]] = {
    "contains": _contains,
    "contains_any": _contains_any,
    "not_contains": _not_contains,
    "equals": _equals,
    "equals_expected": _equals_expected,
    "regex": _regex,
    "not_empty": _not_empty,
    "is_json": _is_json,
    "similarity": _similarity,
    "latency": _latency,
    "tool_calls": _tool_calls,
}
GRADER_TYPES = set(DETERMINISTIC) | JUDGE_TYPES


def label(spec: GraderSpec) -> str:
    p = spec.params
    detail = p.get("name") or p.get("criteria") or p.get("pattern") or p.get("value") or p.get("expected") or p.get("max_ms") or ""
    text = detail if isinstance(detail, str) else json.dumps(detail, default=str)
    text = " ".join(text.split())
    return f"{spec.type}: {text[:77] + '...' if len(text) > 80 else text}" if text else spec.type


def grade_deterministic(spec: GraderSpec, ctx: GradeContext) -> GradeResult:
    try:
        passed, reason = DETERMINISTIC[spec.type](ctx, spec.params)
    except Exception as e:  # a malformed grader config should fail the check, not the whole run
        passed, reason = False, f"grader error: {e}"
    return GradeResult(type=spec.type, label=label(spec), passed=passed, reason=reason)
