"""Convert OpenTelemetry (OTLP JSON) traces into AgentLens's trace schema.

langfuse_adapter.py covers one store. Most agent stacks don't use Langfuse;
they emit OpenTelemetry spans to a Collector, which can write them to a file
as OTLP JSON. This adapter reads that file directly, so anything that follows
the OpenTelemetry GenAI semantic conventions becomes an AgentLens input.

Span mapping (by gen_ai.operation.name):
  chat / generate_content / text_completion -> `reasoning` (the model's output)
  execute_tool                              -> `tool_call` + `tool_result`
  invoke_agent (root)                       -> supplies `task` + `final_answer`
  invoke_agent (nested)                     -> not a step; labels the steps under it
  anything else (HTTP, DB, ...)             -> skipped

Two things a real OTel trace often lacks, and this adapter refuses to paper
over: tool arguments/results and message contents are Opt-In in the spec, and
values get cut at OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT. recording_gaps() reports
both, because a judge reading a trace with no tool results can't find a
contradiction that was never recorded -- and that must not read as a clean run.

The mapping is a pure function over span dicts (spans_to_trace), separate from
file loading, so it can be tested against fixtures -- see test_otel_adapter.py.

Usage:
    python otel_adapter.py traces.jsonl real_traces/imported.json [--trace-id <hex>]
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from langfuse_adapter import _as_text, _maybe_json, trace_to_dict
from schema import Step, Trace

LLM_OPERATIONS = {"chat", "generate_content", "text_completion"}
STATUS_ERROR = (2, "STATUS_CODE_ERROR")

# Placeholders for Opt-In attributes that weren't recorded. tool_args must not
# be empty: rules.py flags an empty tool_args as a malformed call, so leaving
# it empty would invent a finding for a call that was simply not captured.
ARGS_NOT_RECORDED = {"_not_recorded": "gen_ai.tool.call.arguments is opt-in and was not captured"}
RESULT_NOT_RECORDED = "(not recorded: gen_ai.tool.call.result is opt-in and was not captured)"


class OtelTraceError(RuntimeError):
    """Raised when the file holds no usable trace.

    Deliberately distinct from "returned an empty trace", for the same reason
    langfuse_adapter separates LangfuseUnavailableError: an import that found
    nothing, scored as an empty trace, would look like a clean run.
    """


# ── OTLP JSON decoding ─────────────────────────────────────────


def _any_value(value: dict[str, Any]) -> Any:
    """Decode one OTLP AnyValue. intValue arrives as a string in OTLP JSON."""
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_any_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {kv["key"]: _any_value(kv.get("value", {})) for kv in value["kvlistValue"].get("values", [])}
    if "bytesValue" in value:
        return value["bytesValue"]
    return None


def attributes(span: dict[str, Any]) -> dict[str, Any]:
    return {a["key"]: _any_value(a.get("value", {})) for a in span.get("attributes", [])}


def extract_spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one ExportTraceServiceRequest (resourceSpans -> scopeSpans -> spans)."""
    spans: list[dict[str, Any]] = []
    for rs in payload.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            spans.extend(ss.get("spans", []))
    return spans


def load_otlp_file(path: str) -> list[dict[str, Any]]:
    """Read either one OTLP JSON document or the Collector file exporter's
    JSON-lines output (one export batch per line)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise OtelTraceError(f"{path} is empty")
    try:
        payloads = [json.loads(text)]
    except json.JSONDecodeError:
        payloads = [json.loads(line) for line in text.splitlines() if line.strip()]
    spans = [s for p in payloads for s in extract_spans(p)]
    if not spans:
        raise OtelTraceError(f"{path} contains no spans")
    return spans


def select_trace(spans: list[dict[str, Any]], trace_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Keep one trace's spans. Several traces in one file are never merged --
    interleaving two runs would hand the judge a trajectory that never happened."""
    ids = sorted({s.get("traceId", "") for s in spans})
    if trace_id is not None:
        chosen = [s for s in spans if s.get("traceId") == trace_id]
        if not chosen:
            raise OtelTraceError(f"trace {trace_id} not in file (found: {', '.join(ids)})")
        return chosen
    if len(ids) > 1:
        raise OtelTraceError(f"file holds {len(ids)} traces; pass a trace id ({', '.join(ids)})")
    return spans


# ── Mapping ────────────────────────────────────────────────────


def _is_error(span: dict[str, Any]) -> bool:
    return span.get("status", {}).get("code") in STATUS_ERROR


def _error_text(span: dict[str, Any], attrs: dict[str, Any]) -> str:
    message = span.get("status", {}).get("message", "")
    kind = attrs.get("error.type")
    first_line = message.strip().splitlines()[0] if message.strip() else ""
    if kind and first_line.startswith(kind):
        return first_line
    return ": ".join(p for p in (kind, first_line) if p) or "error"


TRUNCATED_MARK = " …[truncated]"


def _salvage_truncated(value: str) -> str:
    """Recover readable text from a messages value cut mid-JSON.

    A long model output cut at OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT no longer
    parses, and handing the judge the raw escaped JSON fragment would bury
    the text it needs. Keep everything after the first "content" key,
    unescaped, and say it was cut.
    """
    start = value.find('"content": "')
    if start < 0:
        return value + TRUNCATED_MARK
    body = value[start + len('"content": "'):]
    # The cut can land mid-escape (a lone backslash, half a \\uXXXX), so trim
    # a few characters at most until the remainder decodes.
    for trim in range(0, 7):
        try:
            return json.loads('"' + body[: len(body) - trim] + '"') + TRUNCATED_MARK
        except json.JSONDecodeError:
            continue
    return body + TRUNCATED_MARK


def _messages_text(value: Any) -> str:
    """Render gen_ai.*.messages ([{role, parts: [...]}]) as plain text.

    Text parts are kept verbatim; tool_call parts are rendered inline so a
    model's decision to call a tool stays visible to the judge.
    """
    parsed = _maybe_json(value)
    if isinstance(value, str) and value[:1] == "[" and parsed is value:
        return _salvage_truncated(value)
    value = parsed
    if not isinstance(value, list):
        return _as_text(value)
    chunks: list[str] = []
    for message in value:
        if not isinstance(message, dict):
            chunks.append(_as_text(message))
            continue
        for part in message.get("parts", []):
            if part.get("type") == "text":
                chunks.append(_as_text(part.get("content")))
            elif part.get("type") == "tool_call":
                chunks.append(f"tool_call {part.get('name')}({_as_text(part.get('arguments'))})")
    return "\n".join(c for c in chunks if c)


def _sort_key(indexed: tuple[int, dict[str, Any]]) -> tuple[int, int]:
    position, span = indexed
    return (int(span.get("startTimeUnixNano", 0)), position)


def _agent_names(spans: list[dict[str, Any]]) -> dict[str, Optional[str]]:
    """span_id -> name of the nearest enclosing nested invoke_agent span."""
    by_id = {s.get("spanId"): s for s in spans}
    resolved: dict[str, Optional[str]] = {}

    def lookup(span_id: Optional[str], hops: int = 0) -> Optional[str]:
        if span_id is None or span_id not in by_id or hops > len(by_id):
            return None
        span = by_id[span_id]
        attrs = attributes(span)
        is_nested_agent = attrs.get("gen_ai.operation.name") == "invoke_agent" and span.get("parentSpanId")
        if is_nested_agent:
            return attrs.get("gen_ai.agent.name") or span.get("name")
        return lookup(span.get("parentSpanId") or None, hops + 1)

    for span in spans:
        resolved[span.get("spanId")] = lookup(span.get("parentSpanId") or None)
    return resolved


def _find_root(spans: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    ids = {s.get("spanId") for s in spans}
    roots = [s for s in spans if not s.get("parentSpanId") or s.get("parentSpanId") not in ids]
    return roots[0] if len(roots) == 1 else None


def spans_to_trace(spans: list[dict[str, Any]], task: Optional[str] = None) -> Trace:
    """Flatten one trace's spans into AgentLens's ordered step list."""
    if not spans:
        raise OtelTraceError("no spans to convert")
    ordered = [s for _, s in sorted(enumerate(spans), key=_sort_key)]
    root = _find_root(ordered)
    root_attrs = attributes(root) if root is not None else {}
    root_is_wrapper = root_attrs.get("gen_ai.operation.name") == "invoke_agent"
    agents = _agent_names(ordered)

    steps: list[Step] = []

    def add(step_type: str, **fields: Any) -> None:
        steps.append(Step(index=len(steps), type=step_type, **fields))

    for span in ordered:
        if span is root and root_is_wrapper:
            continue
        attrs = attributes(span)
        op = attrs.get("gen_ai.operation.name")
        agent = agents.get(span.get("spanId"))
        label = f"[{agent}] " if agent else ""

        if op in LLM_OPERATIONS:
            if _is_error(span):
                add("reasoning", content=f"{label}(model call failed: {_error_text(span, attrs)})", is_error=True)
            else:
                add("reasoning", content=f"{label}{_messages_text(attrs.get('gen_ai.output.messages'))}".strip())
        elif op == "execute_tool":
            name = attrs.get("gen_ai.tool.name") or span.get("name", "tool")
            raw_args = attrs.get("gen_ai.tool.call.arguments")
            parsed = _maybe_json(raw_args)
            if raw_args is None:
                args = ARGS_NOT_RECORDED
            elif isinstance(parsed, dict):
                args = parsed
            else:
                args = {"input": parsed}
            if _is_error(span):
                result: Any = {"error": _error_text(span, attrs)}
            elif "gen_ai.tool.call.result" in attrs:
                result = _maybe_json(attrs["gen_ai.tool.call.result"])
            else:
                result = RESULT_NOT_RECORDED
            add("tool_call", tool_name=name, tool_args=args)
            add("tool_result", tool_name=name, tool_result=result, is_error=_is_error(span))

    if not steps:
        raise OtelTraceError("trace has no GenAI spans (no chat/execute_tool operations to diagnose)")

    final_text = _messages_text(root_attrs.get("gen_ai.output.messages")) if root_is_wrapper else ""
    if final_text:
        add("final_answer", content=final_text)
    elif steps[-1].type == "reasoning" and not steps[-1].is_error and not (root and _is_error(root)):
        # No run-level output recorded: the last generation is the closest thing
        # to a final answer. A run that ended in an error has no final answer,
        # so nothing is relabelled for it.
        steps[-1].type = "final_answer"

    resolved_task = (
        task
        or _messages_text(root_attrs.get("gen_ai.input.messages"))
        or (root.get("name") if root else None)
        or "(no task recorded in OTel trace)"
    )
    return Trace(task=resolved_task, steps=steps)


def recording_gaps(spans: list[dict[str, Any]]) -> list[str]:
    """What the trace didn't capture, in words a diagnosis report can show.

    Empty means nothing obvious is missing -- not that the trace is complete.
    """
    tools = llms = no_args = no_result = no_output = truncated = 0
    for span in spans:
        attrs = attributes(span)
        op = attrs.get("gen_ai.operation.name")
        if op == "execute_tool":
            tools += 1
            no_args += "gen_ai.tool.call.arguments" not in attrs
            no_result += "gen_ai.tool.call.result" not in attrs and not _is_error(span)
        elif op in LLM_OPERATIONS:
            llms += 1
            no_output += "gen_ai.output.messages" not in attrs and not _is_error(span)
        for value in attrs.values():
            # A JSON-looking value that no longer parses was almost certainly
            # cut at OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT.
            if isinstance(value, str) and value[:1] in "{[" and _maybe_json(value) is value:
                truncated += 1

    gaps = []
    if no_result:
        gaps.append(f"{no_result}/{tools} tool calls have no recorded result -- contradictions against tool output can't be checked")
    if no_args:
        gaps.append(f"{no_args}/{tools} tool calls have no recorded arguments")
    if no_output:
        gaps.append(f"{no_output}/{llms} model calls have no recorded output")
    if truncated:
        gaps.append(f"{truncated} attribute values look truncated (OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT)")
    return gaps


def load_trace(path: str, trace_id: Optional[str] = None, task: Optional[str] = None) -> tuple[Trace, list[str]]:
    """File -> (AgentLens Trace, recording gaps) for one trace."""
    spans = select_trace(load_otlp_file(path), trace_id)
    return spans_to_trace(spans, task=task), recording_gaps(spans)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an OTLP JSON trace file into AgentLens's trace format")
    parser.add_argument("otlp_path", help="OTLP JSON, or the Collector file exporter's JSON-lines output")
    parser.add_argument("output_path")
    parser.add_argument("--trace-id", help="which trace to convert, if the file holds several")
    parser.add_argument("--task", help="override the task description")
    args = parser.parse_args()

    try:
        trace, gaps = load_trace(args.otlp_path, trace_id=args.trace_id, task=args.task)
    except OtelTraceError as e:
        raise SystemExit(f"otel import failed: {e}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(trace_to_dict(trace), f, indent=2, ensure_ascii=False)
    print(f"Imported {len(trace.steps)} steps to {args.output_path}")
    for gap in gaps:
        print(f"  warning: {gap}")


if __name__ == "__main__":
    main()
