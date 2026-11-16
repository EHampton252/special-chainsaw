"""Data models shared by the loader, targets, graders, runner and reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConfigError(Exception):
    """The eval suite or its datasets are invalid."""


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class GraderSpec(BaseModel):
    """A grader type plus its free-form parameters."""

    model_config = ConfigDict(extra="allow")
    type: str

    @property
    def params(self) -> dict[str, Any]:
        return self.model_extra or {}


class Turn(BaseModel):
    input: str
    expected: str | None = None
    graders: list[GraderSpec] = Field(default_factory=list)


class Case(BaseModel):
    id: str
    turns: list[Turn] = Field(min_length=1)
    # Case-level graders run against the final response and every tool call made in the case.
    graders: list[GraderSpec] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class JudgeConfig(BaseModel):
    """An OpenAI-compatible chat completions endpoint used as the LLM judge."""

    model_config = ConfigDict(extra="forbid")
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str | None = "OPENAI_API_KEY"
    temperature: float | None = 0
    timeout: float = 120
    headers: dict[str, str] = Field(default_factory=dict)


class Suite(BaseModel):
    name: str
    description: str | None = None
    target: dict[str, Any]
    judge: JudgeConfig | None = None
    cases: list[Case]
    base_dir: Path


class AgentResponse(BaseModel):
    output: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, Any] | None = None


class GradeResult(BaseModel):
    type: str
    label: str
    passed: bool | None  # None means the grader was skipped
    reason: str = ""


class TurnResult(BaseModel):
    input: str
    expected: str | None = None
    output: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    latency_ms: float = 0
    grades: list[GradeResult] = Field(default_factory=list)


class CaseResult(BaseModel):
    id: str
    tags: list[str] = Field(default_factory=list)
    status: str  # "pass" | "fail" | "error"
    error: str | None = None
    turns: list[TurnResult] = Field(default_factory=list)
    grades: list[GradeResult] = Field(default_factory=list)
    duration_ms: float = 0

    def all_grades(self) -> list[GradeResult]:
        return [g for t in self.turns for g in t.grades] + self.grades


class JudgeUsage(BaseModel):
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RunResult(BaseModel):
    suite: str
    target_type: str
    judge_model: str | None = None
    started_at: str
    duration_s: float
    cases: list[CaseResult]
    judge_usage: JudgeUsage = Field(default_factory=JudgeUsage)

    def count(self, status: str) -> int:
        return sum(1 for c in self.cases if c.status == status)
