"""Run a suite: talk to the agent, grade each case, collect results."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from .graders import JUDGE_TYPES, GradeContext, grade_deterministic, label
from .judge import Judge
from .models import Case, CaseResult, GradeResult, GraderSpec, JudgeUsage, RunResult, Suite, ToolCall, TurnResult
from .targets import Target, create_target


async def _grade(specs: list[GraderSpec], ctx: GradeContext, judge: Judge | None,
                 already_failed: bool, always_judge: bool) -> list[GradeResult]:
    """Free deterministic checks run first. Judge calls cost tokens, so they are skipped once the
    case has already failed, unless the caller asked to always judge."""
    results: dict[int, GradeResult] = {}
    for i, spec in enumerate(specs):
        if spec.type not in JUDGE_TYPES:
            results[i] = grade_deterministic(spec, ctx)
    failed = not always_judge and (already_failed or any(r.passed is False for r in results.values()))
    for i, spec in enumerate(specs):
        if spec.type in JUDGE_TYPES:
            if failed or judge is None:
                results[i] = GradeResult(type=spec.type, label=label(spec), passed=None,
                                         reason="skipped to save judge tokens: the case had already failed")
            else:
                results[i] = await judge.grade(spec, ctx)
    return [results[i] for i in range(len(specs))]


async def run_case(case: Case, target: Target, judge: Judge | None, always_judge: bool) -> CaseResult:
    started = time.perf_counter()
    result = CaseResult(id=case.id, tags=case.tags, status="pass")
    messages: list[dict[str, str]] = []
    session: dict = {}
    all_calls: list[ToolCall] = []
    failed = False
    try:
        for turn in case.turns:
            messages.append({"role": "user", "content": turn.input})
            t0 = time.perf_counter()
            response = await target.send(messages, session)
            latency = (time.perf_counter() - t0) * 1000
            messages.append({"role": "assistant", "content": response.output})
            all_calls.extend(response.tool_calls)
            ctx = GradeContext(turn.input, response.output, turn.expected, response.tool_calls, latency)
            grades = await _grade(turn.graders, ctx, judge, failed, always_judge)
            failed = failed or any(g.passed is False for g in grades)
            result.turns.append(TurnResult(
                input=turn.input, expected=turn.expected, output=response.output,
                tool_calls=response.tool_calls, latency_ms=latency, grades=grades,
            ))
        last = result.turns[-1]
        ctx = GradeContext(last.input, last.output, last.expected, all_calls, sum(t.latency_ms for t in result.turns))
        result.grades = await _grade(case.graders, ctx, judge, failed, always_judge)
        if any(g.passed is False for g in result.all_grades()):
            result.status = "fail"
    except Exception as e:  # one unreachable agent call must not abort the rest of the suite
        result.status = "error"
        result.error = f"{type(e).__name__}: {e}"
    result.duration_ms = (time.perf_counter() - started) * 1000
    return result


async def run_suite(
    suite: Suite,
    *,
    concurrency: int = 4,
    always_judge: bool = False,
    cache_dir: Path | None = None,
    only: list[str] | None = None,
    tags: list[str] | None = None,
    on_case: Callable[[CaseResult], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    judge_transport: httpx.AsyncBaseTransport | None = None,
) -> RunResult:
    cases = [c for c in suite.cases
             if (not only or c.id in only) and (not tags or set(tags) & set(c.tags))]
    target = create_target(suite.target, suite.base_dir, transport)
    judge = Judge(suite.judge, cache_dir, judge_transport) if suite.judge else None
    semaphore = asyncio.Semaphore(max(1, concurrency))
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.perf_counter()

    async def one(case: Case) -> CaseResult:
        async with semaphore:
            result = await run_case(case, target, judge, always_judge)
        if on_case:
            on_case(result)
        return result

    try:
        results = await asyncio.gather(*(one(c) for c in cases))
    finally:
        await target.aclose()
        if judge:
            await judge.aclose()
    return RunResult(
        suite=suite.name, target_type=str(suite.target.get("type")),
        judge_model=suite.judge.model if suite.judge else None,
        started_at=started_at, duration_s=time.perf_counter() - started,
        cases=list(results), judge_usage=judge.usage if judge else JudgeUsage(),
    )
