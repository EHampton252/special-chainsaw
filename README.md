# agenteval

A free, open source evaluator for AI agents, wherever they run. Describe your test cases in YAML,
JSON, JSONL, CSV or Markdown, point agenteval at a local or cloud-hosted agent, and get a pass/fail
report you can read in a browser or gate a CI build on.

The only thing a run can cost is the model calls themselves, and agenteval keeps those low:

- **Free checks first.** Text, regex, JSON, latency and tool-call checks need no model at all.
- **The judge is skipped when a case has already failed** a free check (override with `--always-judge`).
- **Judge verdicts are cached on disk**, so re-running an unchanged case costs nothing.
- **The judge can be a local model** (Ollama, LM Studio, vLLM), which makes a whole run free.

> Status: early MVP. The suite format may still change.

## Quickstart

```bash
pip install -e .
```

```bash
agenteval run examples/quickstart/suite.yaml
```

The quickstart runs a toy in-process agent offline, with no API keys. It prints a summary and
writes `agenteval-results/report.html`, `results.json` and `junit.xml`. The exit code is `0` when
every case passes, `1` when any case fails or errors, and `2` for a configuration problem.

Start your own suite with `agenteval init`, and check it without calling the agent with
`agenteval validate suite.yaml`.

## A suite

```yaml
name: support-agent

target:                       # the agent under test
  type: http
  url: http://localhost:8000/chat
  body: { message: "{{input}}", session_id: "{{session_id}}" }
  response_path: reply

judge:                        # optional; only needed for rubric / correctness graders
  model: llama3.1
  base_url: http://localhost:11434/v1
  api_key_env: null

defaults:
  graders: [not_empty]        # applied to every case

dataset: [cases.jsonl, cases/*.md]   # optional extra cases from other files

cases:
  - id: order-status
    input: Where is order 1001?
    tags: [orders]
    graders:
      - contains: shipped
      - tool_calls: { expected: [lookup_order], mode: strict }
      - rubric: The reply is polite and does not ask for information it already has.

  - id: follow-up
    turns:                    # multi-turn conversation
      - input: Where is order 1002?
      - input: Can I still return it?
        expected: Yes, within 30 days of delivery.
        graders: [correctness]
```

`${NAME}` anywhere in `target` or `judge` is replaced with that environment variable at run time,
so secrets stay out of the file.

## Targets

| Type | Reaches | Key settings |
| --- | --- | --- |
| `http` | Any REST endpoint: a local FastAPI, LangGraph or CrewAI server, or a hosted agent | `url`, `method`, `headers`, `body` (template), `response_path`, `tool_calls_path` |
| `openai` | Any OpenAI-compatible chat completions API: OpenAI, Azure/Foundry v1 endpoints, Ollama, LM Studio, vLLM | `base_url`, `model`, `api_key_env`, `system`, `body` |
| `a2a` | Any agent that speaks the Agent2Agent protocol (1.0 and 0.3), on any platform | `url` (agent base URL or agent card URL), `headers`, `protocol_version` |
| `python` | A Python function in your own code | `callable: file.py:function` |

In an `http` body, `{{input}}` is the new user message, `{{messages}}` is the whole conversation
(OpenAI-style) and `{{session_id}}` is stable across the turns of one case. See
[examples/](examples/) for a suite per target type.

## Graders

| Grader | Passes when | Cost |
| --- | --- | --- |
| `contains`, `contains_any`, `not_contains` | The reply has all / any / none of the values | Free |
| `equals`, `equals_expected` | The reply equals a value / the case's `expected` | Free |
| `regex` | The pattern matches | Free |
| `not_empty`, `is_json` | The reply is non-empty / valid JSON | Free |
| `similarity` | Word overlap with `expected` reaches the threshold (default 0.8) | Free |
| `latency` | The reply arrived within `max_ms` | Free |
| `tool_calls` | The agent's tool calls match a reference trajectory | Free |
| `rubric` | An LLM judge decides the reply meets your `criteria` | Judge call |
| `correctness` | An LLM judge decides the reply agrees with `expected` | Judge call |

`tool_calls` takes `mode`: `superset` (all expected calls were made, extras allowed; the default),
`strict` (same calls in the same order), `unordered` (same calls in any order) or `subset` (no
calls beyond the expected ones); and `args`: `subset` (the default), `exact` or `ignore`.

Graders can be written as a name (`not_empty`), a one-key shorthand (`contains: "30 days"`) or in
full (`{type: regex, pattern: "\\d+", ignore_case: true}`). Judge verdicts are binary pass/fail
with a short written reason, which is easier to check against your own judgment than a 1-5 score.

## Dataset formats

| Format | Shape |
| --- | --- |
| YAML / JSON | A list of cases, or a mapping with `cases` |
| JSONL | One case per line, e.g. `{"query": "...", "ground_truth": "...", "expected_tools": ["search"]}` |
| CSV | One case per row; `graders` and `turns` columns hold JSON |
| CSV (Copilot Studio test set) | `conversationNumber,question,response`; rows sharing a number become one multi-turn case |
| Markdown | YAML front matter holds the case fields; the body is the input |

The input may be called `input`, `query`, `question`, `prompt` or `user`. The expected answer may be
called `expected`, `expected_output`, `expected_response`, `response`, `ground_truth`, `answer` or
`reference`, so datasets exported from other tools usually load unchanged. A case that has only an
expected answer and no graders is graded for `correctness` by the judge.

## How it relates to other tools

- **promptfoo** inspired the declarative YAML suite and assertion shorthands.
- **LangChain agentevals** and **Google ADK** inspired the tool-call trajectory match modes.
- **Microsoft Foundry** and **DeepEval** informed the judge graders and the query / response /
  tool-call data shape; **Copilot Studio** test-set CSVs load directly.
- **Inspect AI** informed the dataset / target / grader split.

What agenteval adds is a single tool that is platform-neutral, needs no account, and treats judge
spend as something to minimise.

## Not built yet

- Native Copilot Studio and Foundry agent targets (use `http`, `openai` or `a2a` today)
- Repeating cases to measure flakiness, and comparing two runs
- Simulated-user conversations and generated test cases
- Judge calibration against human labels
- Judge providers that are not OpenAI-compatible

## Development

```bash
pip install -e ".[dev]"
```

```bash
pytest
```
