"""Write results as JSON, JUnit XML and a self-contained HTML report."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from xml.etree import ElementTree as ET

from .models import CaseResult, GradeResult, RunResult


def write_reports(result: RunResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / "results.json", "junit": out_dir / "junit.xml", "html": out_dir / "report.html"}
    paths["json"].write_text(result.model_dump_json(indent=2), encoding="utf-8")
    paths["junit"].write_bytes(junit(result))
    paths["html"].write_text(html(result), encoding="utf-8")
    return paths


def _failure_text(case: CaseResult) -> str:
    lines = [f"[{g.label}] {g.reason}" for g in case.all_grades() if g.passed is False]
    if case.turns:
        lines.append(f"Final response: {case.turns[-1].output}")
    return "\n".join(lines)


def junit(result: RunResult) -> bytes:
    suite = ET.Element("testsuite", name=result.suite, tests=str(len(result.cases)),
                       failures=str(result.count("fail")), errors=str(result.count("error")),
                       time=f"{result.duration_s:.3f}", timestamp=result.started_at)
    for case in result.cases:
        el = ET.SubElement(suite, "testcase", classname=result.suite, name=case.id,
                           time=f"{case.duration_ms / 1000:.3f}")
        if case.status == "error":
            ET.SubElement(el, "error", message=case.error or "error")
        elif case.status == "fail":
            ET.SubElement(el, "failure", message="one or more graders failed").text = _failure_text(case)
    return ET.tostring(suite, encoding="utf-8", xml_declaration=True)


_CSS = """
:root{--bg:#f7f7f5;--card:#fff;--text:#1c1c1a;--muted:#6b6b66;--line:#e2e2dc;--pass:#1a7f4b;--fail:#c0392b;--error:#b26a00;--skip:#77777a;--code:#f0f0ec}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--text:#ecece8;--muted:#a0a09a;--line:#34342f;--pass:#4cc38a;--fail:#ff7b6b;--error:#f0b24a;--skip:#9a9a9e;--code:#2a2a27}}
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}
main{max-width:960px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}.meta{color:var(--muted);margin-bottom:16px}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px}.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 16px;min-width:110px}
.stat b{display:block;font-size:22px}.stat span{color:var(--muted);font-size:13px}
label.filter{display:inline-block;margin-bottom:12px;color:var(--muted)}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin-bottom:8px}
summary{cursor:pointer;padding:10px 14px;display:flex;gap:10px;align-items:center}summary .id{font-weight:600;flex:1;overflow-wrap:anywhere}
summary .dur{color:var(--muted);font-size:13px}.body{padding:0 14px 14px}
.badge{font-size:12px;font-weight:700;padding:2px 8px;border-radius:999px;color:#fff;text-transform:uppercase}
.pass{background:var(--pass)}.fail{background:var(--fail)}.error{background:var(--error)}.skip{background:var(--skip)}
h3{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin:14px 0 4px}
pre{background:var(--code);border-radius:6px;padding:10px;margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.45 ui-monospace,Consolas,monospace}
ul.grades{list-style:none;padding:0;margin:0}ul.grades li{display:flex;gap:8px;padding:4px 0;border-top:1px solid var(--line);align-items:baseline}
ul.grades li:first-child{border-top:0}.reason{color:var(--muted);overflow-wrap:anywhere}.glabel{overflow-wrap:anywhere}
body.failed-only details[data-status=pass]{display:none}
"""


def _grades(grades: list[GradeResult]) -> str:
    if not grades:
        return ""
    items = []
    for g in grades:
        cls, word = {True: ("pass", "pass"), False: ("fail", "fail"), None: ("skip", "skip")}[g.passed]
        items.append(f'<li><span class="badge {cls}">{word}</span><span><span class="glabel">{escape(g.label)}</span>'
                     f'<br><span class="reason">{escape(g.reason)}</span></span></li>')
    return f'<h3>Checks</h3><ul class="grades">{"".join(items)}</ul>'


def _case(case: CaseResult) -> str:
    parts = []
    if case.error:
        parts.append(f"<h3>Error</h3><pre>{escape(case.error)}</pre>")
    multi = len(case.turns) > 1
    for i, t in enumerate(case.turns, 1):
        prefix = f"Turn {i}: " if multi else ""
        parts.append(f"<h3>{prefix}Input</h3><pre>{escape(t.input)}</pre>")
        parts.append(f"<h3>{prefix}Response ({t.latency_ms:.0f} ms)</h3><pre>{escape(t.output)}</pre>")
        if t.expected:
            parts.append(f"<h3>{prefix}Expected</h3><pre>{escape(t.expected)}</pre>")
        if t.tool_calls:
            calls = "\n".join(f"{c.name}({json.dumps(c.args, default=str)})" for c in t.tool_calls)
            parts.append(f"<h3>{prefix}Tool calls</h3><pre>{escape(calls)}</pre>")
        parts.append(_grades(t.grades))
    if case.grades:
        parts.append(_grades(case.grades))
    tags = f' <span class="dur">{escape(", ".join(case.tags))}</span>' if case.tags else ""
    open_attr = "" if case.status == "pass" else " open"
    return (f'<details data-status="{case.status}"{open_attr}><summary><span class="badge {case.status}">{case.status}</span>'
            f'<span class="id">{escape(case.id)}</span>{tags}<span class="dur">{case.duration_ms / 1000:.1f}s</span></summary>'
            f'<div class="body">{"".join(parts)}</div></details>')


def html(result: RunResult) -> str:
    total = len(result.cases)
    passed = result.count("pass")
    rate = f"{100 * passed / total:.0f}%" if total else "n/a"
    u = result.judge_usage
    stats = [(rate, "pass rate"), (str(passed), "passed"), (str(result.count("fail")), "failed"),
             (str(result.count("error")), "errors"), (f"{result.duration_s:.1f}s", "duration")]
    if result.judge_model:
        stats += [(str(u.calls), "judge calls"), (str(u.cached), "judge calls saved by cache"),
                  (f"{u.prompt_tokens + u.completion_tokens:,}", "judge tokens")]
    stat_html = "".join(f'<div class="stat"><b>{escape(v)}</b><span>{escape(k)}</span></div>' for v, k in stats)
    judge = f" · judge: {escape(result.judge_model)}" if result.judge_model else " · no LLM judge"
    order = {"error": 0, "fail": 1, "pass": 2}
    cases = "".join(_case(c) for c in sorted(result.cases, key=lambda c: order.get(c.status, 3)))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(result.suite)} eval report</title>'
            f"<style>{_CSS}</style></head><body><main><h1>{escape(result.suite)}</h1>"
            f'<div class="meta">{escape(result.started_at)} · target: {escape(result.target_type)}{judge}</div>'
            f'<div class="stats">{stat_html}</div>'
            f'<label class="filter"><input type="checkbox" onchange="document.body.classList.toggle(\'failed-only\',this.checked)"> '
            f"Show only failures and errors</label>{cases}</main></body></html>")
