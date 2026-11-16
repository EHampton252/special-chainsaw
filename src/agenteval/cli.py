"""Command line interface: agenteval run | validate | init."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__
from .loader import load_suite
from .models import CaseResult, ConfigError
from .report import write_reports
from .runner import run_suite

STARTER = """\
# A starter eval suite. Run it with: agenteval run agenteval.yaml
name: my-agent

target:
  # Any HTTP endpoint. Other types: openai, a2a, python (see the README).
  type: http
  url: http://localhost:8000/chat
  body: { message: "{{input}}" }
  response_path: reply

# Optional LLM judge for 'rubric' and 'correctness' graders. Any OpenAI-compatible endpoint works;
# pointing it at a local model (Ollama, LM Studio) makes judging free.
# judge:
#   model: llama3.1
#   base_url: http://localhost:11434/v1
#   api_key_env: null

cases:
  - id: greets-user
    input: Hello!
    graders:
      - not_empty
      - not_contains: ["error", "exception"]

  - id: knows-refund-window
    input: How long do I have to return an item?
    graders:
      - contains: "30 days"
      # - rubric: The reply is polite and explains how to start a return.
"""


def _print_case(result: CaseResult) -> None:
    print(f"  {result.status.upper():<5} {result.id}  ({result.duration_ms / 1000:.1f}s)")
    if result.error:
        print(f"        {result.error}")
    for g in result.all_grades():
        if g.passed is False:
            print(f"        - {g.label}: {g.reason}")


def _run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    print(f"Running '{suite.name}': {len(suite.cases)} case(s) against a '{suite.target['type']}' target")
    result = asyncio.run(run_suite(
        suite, concurrency=args.concurrency, always_judge=args.always_judge,
        cache_dir=None if args.no_cache else Path(".agenteval") / "judge-cache",
        only=args.case, tags=args.tag, on_case=_print_case,
    ))
    if not result.cases:
        print("No cases matched the --case/--tag filters.")
        return 2
    paths = write_reports(result, Path(args.output))
    passed, total = result.count("pass"), len(result.cases)
    print(f"\n{passed}/{total} passed, {result.count('fail')} failed, {result.count('error')} errored "
          f"in {result.duration_s:.1f}s")
    if result.judge_model:
        u = result.judge_usage
        print(f"Judge ({result.judge_model}): {u.calls} call(s), {u.cached} served from cache, "
              f"{u.prompt_tokens + u.completion_tokens:,} tokens")
    print(f"Report: {paths['html']}")
    return 0 if passed == total else 1


def _validate(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite, resolve_env=False)
    turns = sum(len(c.turns) for c in suite.cases)
    print(f"OK: '{suite.name}' has {len(suite.cases)} case(s), {turns} turn(s), target type '{suite.target['type']}'")
    return 0


def _init(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists():
        print(f"{path} already exists; not overwriting it.", file=sys.stderr)
        return 2
    path.write_text(STARTER, encoding="utf-8")
    print(f"Wrote {path}. Edit the target, then run: agenteval run {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agenteval", description="Evaluate local and cloud-hosted AI agents.")
    parser.add_argument("--version", action="version", version=f"agenteval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an eval suite")
    run.add_argument("suite", help="path to a suite file (.yaml or .json)")
    run.add_argument("-o", "--output", default="agenteval-results", help="directory for reports (default: %(default)s)")
    run.add_argument("-c", "--concurrency", type=int, default=4, help="cases to run in parallel (default: %(default)s)")
    run.add_argument("--case", action="append", help="run only this case id (repeatable)")
    run.add_argument("--tag", action="append", help="run only cases with this tag (repeatable)")
    run.add_argument("--no-cache", action="store_true", help="do not reuse cached judge verdicts")
    run.add_argument("--always-judge", action="store_true",
                     help="call the LLM judge even when a free deterministic check already failed")
    run.set_defaults(func=_run)

    validate = sub.add_parser("validate", help="check a suite file without calling the agent")
    validate.add_argument("suite")
    validate.set_defaults(func=_validate)

    init = sub.add_parser("init", help="write a starter suite")
    init.add_argument("path", nargs="?", default="agenteval.yaml")
    init.set_defaults(func=_init)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
