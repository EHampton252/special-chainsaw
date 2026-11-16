"""Load eval suites and datasets from YAML, JSON, JSONL, CSV and Markdown front matter."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .graders import GRADER_TYPES, JUDGE_TYPES, NEEDS_EXPECTED
from .models import Case, ConfigError, JudgeConfig, Suite

INPUT_KEYS = ("input", "query", "question", "prompt", "user")
EXPECTED_KEYS = (
    "expected", "expected_output", "expected_response", "response",
    "ground_truth", "answer", "reference",
)
SUITE_KEYS = {"name", "description", "target", "judge", "defaults", "cases", "dataset", "datasets"}
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_suite(path: str | Path, resolve_env: bool = True) -> Suite:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Suite file not found: {path}")
    if path.suffix.lower() not in (".yaml", ".yml", ".json"):
        raise ConfigError(f"A suite must be a .yaml, .yml or .json file, got: {path.name}")
    raw = _read_structured(path)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: a suite must be a mapping with 'target' and 'cases'")
    unknown = set(raw) - SUITE_KEYS
    if unknown:
        raise ConfigError(f"{path.name}: unknown key(s) {sorted(unknown)}; allowed: {sorted(SUITE_KEYS)}")
    target = raw.get("target")
    if not isinstance(target, dict) or "type" not in target:
        raise ConfigError(f"{path.name}: 'target' must be a mapping with a 'type'")

    base_dir = path.resolve().parent
    judge_raw = raw.get("judge")
    if resolve_env:
        target = _resolve_env(target)
        judge_raw = _resolve_env(judge_raw)
    try:
        judge = JudgeConfig(**judge_raw) if judge_raw else None
    except (ValidationError, TypeError) as e:
        raise ConfigError(f"{path.name}: invalid 'judge' config: {e}") from e

    raw_cases = list(raw.get("cases") or [])
    datasets = raw.get("datasets") or raw.get("dataset") or []
    for pattern in [datasets] if isinstance(datasets, str) else datasets:
        files = sorted(base_dir.glob(pattern))
        if not files:
            raise ConfigError(f"{path.name}: dataset '{pattern}' matched no files")
        for f in files:
            raw_cases.extend(load_dataset(f))
    if not raw_cases:
        raise ConfigError(f"{path.name}: no cases found (add 'cases' or a 'dataset')")

    defaults = [_grader(g, "defaults") for g in (raw.get("defaults") or {}).get("graders", [])]
    cases = [_case(c, i, defaults, judge is not None) for i, c in enumerate(raw_cases)]
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise ConfigError(f"Duplicate case id: '{c.id}'")
        seen.add(c.id)
    return Suite(
        name=str(raw.get("name") or path.stem), description=raw.get("description"),
        target=target, judge=judge, cases=cases, base_dir=base_dir,
    )


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Return raw case dicts from a dataset file; the format is chosen by file extension."""
    ext = path.suffix.lower()
    if ext == ".jsonl":
        rows = []
        for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ConfigError(f"{path.name} line {n}: invalid JSON ({e.msg})") from e
        return rows
    if ext == ".csv":
        return _read_csv(path)
    if ext in (".md", ".markdown"):
        return [_read_markdown(path)]
    if ext in (".yaml", ".yml", ".json"):
        data = _read_structured(path)
        if isinstance(data, dict):
            data = data.get("cases")
        if not isinstance(data, list):
            raise ConfigError(f"{path.name}: expected a list of cases or a mapping with 'cases'")
        return data
    raise ConfigError(f"Unsupported dataset format: {path.name}")


def _read_structured(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ConfigError(f"{path.name}: could not parse file: {e}") from e


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items() if k and v and v.strip()} for r in csv.DictReader(f)]
    conv_key = next((k for r in rows for k in r if k.lower() == "conversationnumber"), None)
    if conv_key:
        # Copilot Studio test-set layout: conversationNumber,question,response
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(r.get(conv_key, ""), []).append(r)
        return [
            {"id": f"conversation-{num}", "turns": [{k: v for k, v in r.items() if k != conv_key} for r in turns]}
            for num, turns in groups.items()
        ]
    for r in rows:
        for key in ("graders", "turns"):
            if key in r:
                try:
                    r[key] = json.loads(r[key])
                except json.JSONDecodeError as e:
                    raise ConfigError(f"{path.name}: column '{key}' must contain JSON ({e.msg})") from e
    return rows


def _read_markdown(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        raise ConfigError(f"{path.name}: Markdown cases need YAML front matter between '---' lines")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name}: invalid front matter: {e}") from e
    if not isinstance(meta, dict):
        raise ConfigError(f"{path.name}: front matter must be a mapping")
    meta.setdefault("id", path.stem)
    if not any(k in meta for k in (*INPUT_KEYS, "turns")):
        meta["input"] = m.group(2).strip()
    return meta


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match[str]) -> str:
            if m.group(1) not in os.environ:
                raise ConfigError(f"Environment variable '{m.group(1)}' is referenced in the suite but not set")
            return os.environ[m.group(1)]
        return _ENV.sub(sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _grader(raw: Any, where: str) -> dict[str, Any]:
    """Accept 'is_json', {contains: 'x'}, {rubric: {criteria: ...}} or {type: ..., ...}."""
    if isinstance(raw, str):
        spec: dict[str, Any] = {"type": raw}
    elif isinstance(raw, dict) and "type" in raw:
        spec = dict(raw)
    elif isinstance(raw, dict) and len(raw) == 1:
        (key, val), = raw.items()
        spec = {"type": key, **val} if isinstance(val, dict) else {"type": key, "value": val}
    else:
        raise ConfigError(f"{where}: invalid grader {raw!r}")
    if spec["type"] not in GRADER_TYPES:
        raise ConfigError(f"{where}: unknown grader '{spec['type']}'; available: {sorted(GRADER_TYPES)}")
    return spec


def _first(raw: dict[str, Any], keys: tuple[str, ...]) -> Any:
    return next((raw[k] for k in keys if raw.get(k) not in (None, "")), None)


def _tools(value: Any) -> list[Any]:
    if isinstance(value, str):
        value = json.loads(value) if value.lstrip().startswith("[") else re.split(r"[,;]", value)
    return [v.strip() if isinstance(v, str) else v for v in value if v]


def _turn(raw: Any, where: str) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"input": raw}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: a turn must be a string or a mapping")
    text = _first(raw, INPUT_KEYS)
    if text is None:
        raise ConfigError(f"{where}: turn has no input (use one of {list(INPUT_KEYS)})")
    expected = _first(raw, EXPECTED_KEYS)
    return {
        "input": str(text),
        "expected": None if expected is None else str(expected),
        "graders": [_grader(g, where) for g in raw.get("graders") or []],
    }


def _case(raw: Any, index: int, defaults: list[dict[str, Any]], has_judge: bool) -> Case:
    if not isinstance(raw, dict):
        raise ConfigError(f"Case {index + 1}: must be a mapping, got {type(raw).__name__}")
    case_id = str(raw.get("id") or raw.get("name") or f"case-{index + 1}")
    where = f"Case '{case_id}'"
    if raw.get("turns"):
        turns = [_turn(t, where) for t in raw["turns"]]
    else:
        turns = [_turn({k: raw.get(k) for k in (*INPUT_KEYS, *EXPECTED_KEYS)}, where)]
    graders = defaults + [_grader(g, where) for g in raw.get("graders") or []]
    if raw.get("expected_tools"):
        graders.append({"type": "tool_calls", "expected": _tools(raw["expected_tools"])})

    if not graders and not any(t["graders"] for t in turns):
        # Datasets that only carry question/expected-answer pairs get a correctness judge.
        if not any(t["expected"] for t in turns):
            raise ConfigError(f"{where}: nothing to check; add graders or an expected answer")
        if not has_judge:
            raise ConfigError(
                f"{where}: has an expected answer but no graders. Add graders (for example "
                f"'similarity' or 'contains') or configure a 'judge' to grade correctness."
            )
        for t in turns:
            if t["expected"]:
                t["graders"].append({"type": "correctness"})

    last_expected = turns[-1]["expected"]
    for scope, specs, expected in [(where, graders, last_expected)] + [(where, t["graders"], t["expected"]) for t in turns]:
        for g in specs:
            if g["type"] in JUDGE_TYPES and not has_judge:
                raise ConfigError(f"{scope}: grader '{g['type']}' needs a 'judge' section in the suite")
            if g["type"] in NEEDS_EXPECTED and not expected:
                raise ConfigError(f"{scope}: grader '{g['type']}' needs an expected answer")

    tags = raw.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in re.split(r"[,;]", tags) if t.strip()]
    try:
        return Case(id=case_id, turns=turns, graders=graders, tags=tags)
    except ValidationError as e:
        raise ConfigError(f"{where}: {e}") from e
