import json
from pathlib import Path

import pytest

from agenteval.graders import GradeContext, grade_deterministic
from agenteval.loader import load_suite
from agenteval.models import ConfigError, GraderSpec, ToolCall

TARGET = "target: {type: python, callable: 'a.py:f'}\n"


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def check(spec: dict, output: str = "", **ctx) -> bool:
    return grade_deterministic(GraderSpec(**spec), GradeContext(input="q", output=output, **ctx)).passed


def test_yaml_suite_with_grader_shorthands(tmp_path):
    suite = load_suite(write(tmp_path, "s.yaml", TARGET + """
defaults: {graders: [not_empty]}
cases:
  - input: hi
    graders: [{contains: hello}, {type: regex, pattern: "h.+"}, {latency: {max_ms: 5}}]
"""))
    case = suite.cases[0]
    assert case.id == "case-1"
    assert [g.type for g in case.graders] == ["not_empty", "contains", "regex", "latency"]
    assert case.graders[1].params == {"value": "hello"}
    assert case.graders[3].params == {"max_ms": 5}


def test_json_suite_and_all_dataset_formats(tmp_path):
    write(tmp_path, "a.jsonl", '{"query": "q1", "expected_tools": ["search"]}\n\n{"question": "q2", "graders": ["not_empty"]}\n')
    write(tmp_path, "b.csv", "id,input,graders,tags\nrow,q3,\"[\"\"not_empty\"\"]\",\"x, y\"\n")
    write(tmp_path, "c.md", "---\ngraders: [not_empty]\n---\nBody is the input\n")
    write(tmp_path, "d.yaml", "- {id: y1, prompt: q4, graders: [is_json]}\n")
    suite = load_suite(write(tmp_path, "s.json", json.dumps({
        "target": {"type": "http", "url": "http://x"},
        "datasets": ["a.jsonl", "b.csv", "*.md", "d.yaml"],
    })))
    by_id = {c.id: c for c in suite.cases}
    assert list(by_id) == ["case-1", "case-2", "row", "c", "y1"]
    assert by_id["case-1"].graders[0].params == {"expected": ["search"]}
    assert by_id["row"].tags == ["x", "y"]
    assert by_id["c"].turns[0].input == "Body is the input"


def test_copilot_studio_csv_becomes_multi_turn_correctness_cases(tmp_path):
    write(tmp_path, "t.csv", "conversationNumber,question,response\n1,a,A\n2,b,B\n2,c,C\n")
    suite = load_suite(write(tmp_path, "s.yaml", TARGET + "judge: {model: m}\ndataset: t.csv\n"))
    assert [c.id for c in suite.cases] == ["conversation-1", "conversation-2"]
    second = suite.cases[1]
    assert [(t.input, t.expected) for t in second.turns] == [("b", "B"), ("c", "C")]
    assert all(t.graders[0].type == "correctness" for t in second.turns)


@pytest.mark.parametrize("body, message", [
    ("cases: [{input: hi}]", "nothing to check"),
    ("cases: [{input: hi, expected: yo}]", "configure a 'judge'"),
    ("cases: [{input: hi, graders: [{rubric: nice}]}]", "needs a 'judge'"),
    ("cases: [{input: hi, graders: [similarity]}]", "needs an expected answer"),
    ("cases: [{input: hi, graders: [contians]}]", "unknown grader"),
    ("cases: [{graders: [not_empty]}]", "no input"),
    ("cases: [{id: a, input: x, graders: [not_empty]}, {id: a, input: y, graders: [not_empty]}]", "Duplicate"),
    ("casez: []", "unknown key"),
    ("dataset: missing.csv", "matched no files"),
])
def test_invalid_suites_are_rejected_with_a_clear_message(tmp_path, body, message):
    with pytest.raises(ConfigError, match=message):
        load_suite(write(tmp_path, "s.yaml", TARGET + body + "\n"))


def test_env_vars_resolve_or_fail(tmp_path, monkeypatch):
    path = write(tmp_path, "s.yaml", "target: {type: http, url: 'http://x/${AE_TEST_VAR}'}\ncases: [{input: q, graders: [not_empty]}]\n")
    monkeypatch.delenv("AE_TEST_VAR", raising=False)
    with pytest.raises(ConfigError, match="AE_TEST_VAR"):
        load_suite(path)
    assert load_suite(path, resolve_env=False).target["url"].endswith("${AE_TEST_VAR}")
    monkeypatch.setenv("AE_TEST_VAR", "chat")
    assert load_suite(path).target["url"] == "http://x/chat"


def test_text_graders():
    assert check({"type": "contains", "value": ["30 Days", "refund"]}, "Refund within 30 days")
    assert not check({"type": "contains", "value": "30 Days", "ignore_case": False}, "30 days")
    assert check({"type": "contains_any", "value": ["a", "zzz"]}, "banana")
    assert not check({"type": "not_contains", "value": ["error"]}, "An Error occurred")
    assert check({"type": "equals", "value": "yes"}, " yes\n") and not check({"type": "equals", "value": "yes"}, "Yes")
    assert check({"type": "equals_expected"}, "42", expected="42")
    assert check({"type": "regex", "value": r"\d{3}-\d{4}"}, "call 555-1234")
    assert check({"type": "is_json"}, '{"a": 1}') and not check({"type": "is_json"}, "nope")
    assert not check({"type": "not_empty"}, "  ")
    assert check({"type": "similarity", "value": 0.7}, "returns accepted within 30 days", expected="Returns are accepted within 30 days")
    assert not check({"type": "similarity"}, "no idea", expected="Returns are accepted within 30 days")
    assert check({"type": "latency", "max_ms": 100}, "x", latency_ms=50) and not check({"type": "latency", "value": 10}, "x", latency_ms=50)


def test_tool_call_trajectory_modes():
    calls = [ToolCall(name="search", args={"q": "cats", "limit": 5}), ToolCall(name="summarize")]

    def tc(expected, **params):
        return check({"type": "tool_calls", "expected": expected, **params}, "x", tool_calls=calls)

    assert tc(["search"])                                   # superset: extras allowed
    assert not tc(["search", "email"])
    assert tc(["summarize", "search"], mode="unordered")
    assert not tc(["search"], mode="unordered")             # summarize is unexpected
    assert tc(["search", "summarize"], mode="strict")
    assert not tc(["summarize", "search"], mode="strict")
    assert tc(["search", "summarize", "email"], mode="subset")
    assert not tc(["search"], mode="subset")
    assert tc([{"name": "search", "args": {"q": "cats"}}])  # args: subset by default
    assert not tc([{"name": "search", "args": {"q": "cats"}}], args="exact")
    assert not tc([{"name": "search", "args": {"q": "dogs"}}])
    assert tc([{"name": "search", "args": {"q": "dogs"}}], args="ignore")
    assert not tc(["search"], mode="bogus")
    assert check({"type": "tool_calls", "expected": [], "mode": "strict"}, "x", tool_calls=[])
