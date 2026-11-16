import asyncio
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

from agenteval.cli import main
from agenteval.loader import load_suite
from agenteval.report import write_reports
from agenteval.runner import run_suite
from agenteval.targets import create_target

ROOT = Path(__file__).resolve().parent.parent


def suite_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def send(target, text="hi", session=None, history=()):
    async def go():
        try:
            return await target.send([*history, {"role": "user", "content": text}], {} if session is None else session)
        finally:
            await target.aclose()
    return asyncio.run(go())


def judge_transport(verdicts: list[bool], seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        verdict = verdicts[min(len(seen), len(verdicts)) - 1]
        content = 'Sure:\n{"reasoning": "because", "pass": %s}' % str(verdict).lower()
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    return httpx.MockTransport(handler)


def test_quickstart_example_passes_and_writes_reports(tmp_path):
    result = asyncio.run(run_suite(load_suite(ROOT / "examples/quickstart/suite.yaml")))
    assert [c.status for c in result.cases] == ["pass"] * 5
    multi = next(c for c in result.cases if c.id == "follow-up-remembers-order")
    assert "order 1002" in multi.turns[1].output
    paths = write_reports(result, tmp_path / "out")
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["suite"] == "quickstart-support-agent"
    assert ET.parse(paths["junit"]).getroot().get("failures") == "0"
    assert "follow-up-remembers-order" in paths["html"].read_text(encoding="utf-8")


def test_http_target_templates_request_and_reads_response():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": {"reply": "hello"}, "calls": [
            {"function": {"name": "search", "arguments": '{"q": "x"}'}}]})

    target = create_target({
        "type": "http", "url": "http://agent/chat", "headers": {"X-Session": "{{session_id}}"},
        "body": {"msg": "User said: {{input}}", "history": "{{messages}}"},
        "response_path": "data.reply", "tool_calls_path": "calls",
    }, Path("."), httpx.MockTransport(handler))
    response = send(target, "hi", {"session_id": "abc"})
    body = json.loads(seen[0].content)
    assert body == {"msg": "User said: hi", "history": [{"role": "user", "content": "hi"}]}
    assert seen[0].headers["X-Session"] == "abc"
    assert response.output == "hello"
    assert response.tool_calls[0].model_dump() == {"name": "search", "args": {"q": "x"}}


def test_openai_target_sends_history_and_auth(monkeypatch):
    monkeypatch.setenv("AE_KEY", "secret")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}]}}]})

    target = create_target({"type": "openai", "base_url": "http://llm/v1/", "model": "m", "system": "Be brief",
                            "api_key_env": "AE_KEY"}, Path("."), httpx.MockTransport(handler))
    response = send(target, "weather?")
    assert str(seen[0].url) == "http://llm/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer secret"
    assert [m["role"] for m in json.loads(seen[0].content)["messages"]] == ["system", "user"]
    assert response.output == "" and response.tool_calls[0].args == {"city": "Oslo"}


def test_a2a_v1_discovers_card_polls_task_and_keeps_context():
    rpc = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert request.url.path == "/.well-known/agent-card.json"
            return httpx.Response(200, json={"supportedInterfaces": [
                {"url": "http://agent/grpc", "protocolBinding": "GRPC", "protocolVersion": "1.0"},
                {"url": "http://agent/rpc", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}]})
        call = json.loads(request.content)
        rpc.append((str(request.url), request.headers.get("A2A-Version"), call))
        if call["method"] == "SendMessage":
            task = {"id": "t1", "contextId": "ctx", "status": {"state": "TASK_STATE_WORKING"}}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": call["id"], "result": {"task": task}})
        task = {"id": "t1", "contextId": "ctx", "status": {"state": "TASK_STATE_COMPLETED"},
                "artifacts": [{"artifactId": "a", "parts": [{"text": "done"}, {"text": "and dusted"}]}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": call["id"], "result": task})

    target = create_target({"type": "a2a", "url": "http://agent/", "poll_interval": 0}, Path("."),
                           httpx.MockTransport(handler))
    session = {"context_id": "earlier"}
    response = send(target, "go", session)
    assert response.output == "done\nand dusted"
    assert session["context_id"] == "ctx"
    url, version, first = rpc[0]
    assert (url, version) == ("http://agent/rpc", "1.0")
    message = first["params"]["message"]
    assert message["role"] == "ROLE_USER" and message["parts"] == [{"text": "go"}] and message["contextId"] == "earlier"
    assert rpc[1][2]["method"] == "GetTask" and rpc[1][2]["params"] == {"id": "t1"}


def test_a2a_legacy_card_uses_0_3_wire_format():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"url": "http://agent/a2a", "protocolVersion": "0.3.0"})
        calls.append((request.headers.get("A2A-Version"), json.loads(request.content)))
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": {
            "kind": "message", "role": "agent", "contextId": "c9", "parts": [{"kind": "text", "text": "hi there"}]}})

    target = create_target({"type": "a2a", "url": "http://agent"}, Path("."), httpx.MockTransport(handler))
    assert send(target).output == "hi there"
    version, call = calls[0]
    assert version is None and call["method"] == "message/send"
    assert call["params"]["message"]["parts"] == [{"kind": "text", "text": "hi"}]


def test_a2a_http_json_binding_and_failed_task_is_an_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"supportedInterfaces": [
                {"url": "http://agent/v1", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}]})
        assert request.url.path == "/v1/message:send"
        return httpx.Response(200, json={"task": {"id": "t", "status": {
            "state": "TASK_STATE_FAILED", "message": {"parts": [{"text": "boom"}]}}}})

    suite = load_suite(suite_file(tmp_path, "target: {type: a2a, url: 'http://agent'}\ncases: [{input: q, graders: [not_empty]}]\n"))
    result = asyncio.run(run_suite(suite, transport=httpx.MockTransport(handler)))
    assert result.cases[0].status == "error" and "boom" in result.cases[0].error


JUDGED = """
target: {type: python, callable: '%s:respond'}
judge: {model: judge-model, base_url: 'http://judge/v1', api_key_env: null}
cases:
  - id: passes-free-check
    input: What is your refund policy?
    expected: Returns within 30 days.
    graders: [{contains: "30 days"}, correctness, {rubric: Is polite}]
  - id: fails-free-check
    input: What is your refund policy?
    graders: [{contains: "90 days"}, {rubric: Is polite}]
""" % (ROOT / "examples/quickstart/agent.py").as_posix()


def test_judge_is_skipped_after_free_failure_and_verdicts_are_cached(tmp_path):
    suite = load_suite(suite_file(tmp_path, JUDGED))
    seen: list = []
    run = lambda **kw: asyncio.run(run_suite(suite, cache_dir=tmp_path / "cache",
                                             judge_transport=judge_transport([True, False], seen), **kw))
    result = run()
    ok, bad = result.cases
    assert ok.status == "fail"  # the second judge verdict (rubric) was a fail
    assert [g.passed for g in ok.grades] == [True, True, False]
    assert ok.grades[2].reason == "because"
    assert bad.status == "fail" and [g.passed for g in bad.grades] == [False, None]
    assert len(seen) == 2 and result.judge_usage.calls == 2 and result.judge_usage.prompt_tokens == 20
    prompt = seen[0]["messages"][1]["content"]
    assert "<expected_answer>\nReturns within 30 days." in prompt and "<agent_response>" in prompt

    again = run()
    assert len(seen) == 2 and again.judge_usage.cached == 2 and again.judge_usage.calls == 0

    forced = run(always_judge=True, only=["fails-free-check"])
    assert len(forced.cases) == 1 and forced.cases[0].grades[1].passed is not None


def test_unparseable_judge_output_fails_the_check(tmp_path):
    suite = load_suite(suite_file(tmp_path, JUDGED))
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "LGTM"}}]}))
    result = asyncio.run(run_suite(suite, judge_transport=transport, only=["passes-free-check"]))
    grade = result.cases[0].grades[1]
    assert grade.passed is False and "could not parse" in grade.reason


def test_cli_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    quickstart = str(ROOT / "examples/quickstart/suite.yaml")
    assert main(["run", quickstart, "-o", "out", "--tag", "tools"]) == 0
    assert (tmp_path / "out" / "report.html").is_file()
    assert "2/2 passed" in capsys.readouterr().out

    failing = suite_file(tmp_path, "target: {type: python, callable: '%s:respond'}\ncases: [{input: hi, graders: [{contains: nope}]}]\n"
                         % (ROOT / "examples/quickstart/agent.py").as_posix())
    assert main(["run", str(failing), "-o", "out"]) == 1
    assert main(["validate", str(failing)]) == 0
    assert main(["run", str(tmp_path / "missing.yaml")]) == 2
    assert main(["init", "new.yaml"]) == 0 and main(["validate", "new.yaml"]) == 0
    assert main(["init", "new.yaml"]) == 2
