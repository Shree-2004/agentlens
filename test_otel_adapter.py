"""Fixture tests for the OpenTelemetry (OTLP JSON) -> AgentLens mapping.

No network, no Collector: spans_to_trace() is a pure function over span
dicts, and these fixtures are shaped like the Collector's file exporter
output (hex ids, typed attribute values, intValue as a string, status code
as an int). The last test runs the real capture in real_traces/.

As with the Langfuse tests, the point is to pin mapping decisions that fail
silently -- ordering, the root span, error status, Opt-In attributes that
weren't recorded -- because a wrong mapping doesn't crash. It hands the
judge a trajectory the agent never ran.

Run:  python3 test_otel_adapter.py
"""

from __future__ import annotations

import json
import os
import tempfile

from otel_adapter import (
    ARGS_NOT_RECORDED,
    RESULT_NOT_RECORDED,
    OtelTraceError,
    _any_value,
    load_otlp_file,
    load_trace,
    recording_gaps,
    select_trace,
    spans_to_trace,
)
from rules import run_rules

_failures: list[str] = []


def check(label: str, got, want) -> None:
    if got == want:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")
        _failures.append(label)


def _attr(key, value):
    if isinstance(value, bool):
        typed = {"boolValue": value}
    elif isinstance(value, int):
        typed = {"intValue": str(value)}  # OTLP JSON encodes int64 as a string
    else:
        typed = {"stringValue": value}
    return {"key": key, "value": typed}


def span(sid, name, start, parent=None, trace="t1", error=None, **attrs):
    """One span in the shape the Collector's file exporter writes it."""
    s = {
        "traceId": trace,
        "spanId": sid,
        "parentSpanId": parent or "",
        "name": name,
        "startTimeUnixNano": str(1_700_000_000_000_000_000 + start),
        "attributes": [_attr(k.replace("__", "."), v) for k, v in attrs.items()],
        "status": {},
    }
    if error:
        s["status"] = {"message": error, "code": 2}
    return s


def msgs(role, text):
    return json.dumps([{"role": role, "parts": [{"type": "text", "content": text}]}])


# A plan-tier lookup like sample_traces/stale_plan_tier.json, arriving as
# GenAI-convention spans: a root agent, one sub-agent, two model calls and a
# tool call between them. Deliberately out of chronological order.
AGENT_RUN = [
    span("s4", "chat gemini", 4, parent="s1",
         gen_ai__operation__name="chat",
         gen_ai__output__messages=msgs("assistant", "U-4471 is on the Free tier, so bulk export isn't available.")),
    span("s3", "execute_tool check_user_plan", 3, parent="s1",
         gen_ai__operation__name="execute_tool", gen_ai__tool__name="check_user_plan",
         gen_ai__tool__call__arguments='{"user_id": "U-4471"}',
         gen_ai__tool__call__result='{"tier": "free", "source": "plan_cache", "cached_at": "2026-01-14"}'),
    span("s5", "GET /internal/health", 5, parent="s1"),  # non-GenAI span -- skipped
    span("s2", "chat gemini", 2, parent="s1",
         gen_ai__operation__name="chat",
         gen_ai__output__messages=msgs("assistant", "Bulk export is Pro-only. I should check the user's plan first.")),
    span("s1", "invoke_agent Support", 1, parent="s0",
         gen_ai__operation__name="invoke_agent", gen_ai__agent__name="Support"),
    span("s0", "invoke_agent Helpdesk", 0,
         gen_ai__operation__name="invoke_agent", gen_ai__agent__name="Helpdesk",
         gen_ai__input__messages=msgs("user", "Can U-4471 use bulk export?"),
         gen_ai__output__messages=msgs("assistant", "You're on the Free tier, which doesn't include bulk export.")),
]


def test_full_run() -> None:
    print("full agent run")
    trace = spans_to_trace(AGENT_RUN)

    check("task comes from the root span's input messages", trace.task, "Can U-4471 use bulk export?")
    check(
        "chronological order; root, sub-agent and non-GenAI spans aren't steps",
        [s.type for s in trace.steps],
        ["reasoning", "tool_call", "tool_result", "reasoning", "final_answer"],
    )
    check("indices are contiguous", [s.index for s in trace.steps], [0, 1, 2, 3, 4])
    check("model output is labelled with its sub-agent",
          trace.steps[0].content, "[Support] Bulk export is Pro-only. I should check the user's plan first.")
    check("tool name carries over", trace.steps[1].tool_name, "check_user_plan")
    check("arguments are parsed into tool_args", trace.steps[1].tool_args, {"user_id": "U-4471"})
    check("result is parsed into tool_result",
          trace.steps[2].tool_result, {"tier": "free", "source": "plan_cache", "cached_at": "2026-01-14"})
    check("root output becomes the final answer",
          trace.steps[4].content, "You're on the Free tier, which doesn't include bulk export.")


def test_tool_error_status() -> None:
    print("tool error status")
    trace = spans_to_trace([
        span("e1", "execute_tool fetch_invoice", 0, gen_ai__operation__name="execute_tool",
             gen_ai__tool__name="fetch_invoice", gen_ai__tool__call__arguments='{"id": 7}',
             error="NotFound: invoice 7", error__type="NotFound"),
    ])
    check("ERROR status sets is_error", trace.steps[1].is_error, True)
    check("the status message becomes the result", trace.steps[1].tool_result, {"error": "NotFound: invoice 7"})

    ok = spans_to_trace([
        span("e2", "execute_tool fetch_invoice", 0, gen_ai__operation__name="execute_tool",
             gen_ai__tool__name="fetch_invoice", gen_ai__tool__call__arguments='{"id": 7}',
             gen_ai__tool__call__result='{"ok": true}'),
    ])
    check("unset status leaves is_error false", ok.steps[1].is_error, False)


def test_unrecorded_opt_in_attributes() -> None:
    """Arguments and results are Opt-In. Missing ones must be marked, not
    left empty -- rules.py reads empty tool_args as a malformed call."""
    print("opt-in attributes not recorded")
    spans = [span("o1", "execute_tool search", 0, gen_ai__operation__name="execute_tool",
                  gen_ai__tool__name="search")]
    trace = spans_to_trace(spans)
    check("missing arguments get a placeholder", trace.steps[0].tool_args, ARGS_NOT_RECORDED)
    check("missing result gets a placeholder", trace.steps[1].tool_result, RESULT_NOT_RECORDED)
    check("no invented missing_tool_arguments finding",
          "missing_tool_arguments" in [f.category for f in run_rules(trace)], False)
    gaps = recording_gaps(spans)
    check("the missing result is reported", any("no recorded result" in g for g in gaps), True)
    check("the missing arguments are reported", any("no recorded arguments" in g for g in gaps), True)


def test_failed_model_call_ends_run() -> None:
    """A run that crashed has no final answer -- the failed call must not be
    relabelled as one."""
    print("failed model call")
    trace = spans_to_trace([
        span("r0", "invoke_agent Pipeline", 0, error="ResourceExhausted: 429 quota",
             gen_ai__operation__name="invoke_agent", gen_ai__input__messages=msgs("user", "Research X")),
        span("r1", "chat gemini", 1, parent="r0", gen_ai__operation__name="chat",
             gen_ai__output__messages=msgs("assistant", "Plan: search for X.")),
        span("r2", "chat gemini", 2, parent="r0", error="ResourceExhausted: 429 quota",
             gen_ai__operation__name="chat", error__type="ResourceExhausted"),
    ])
    check("no final_answer is invented", [s.type for s in trace.steps], ["reasoning", "reasoning"])
    check("the failed call is marked as an error", trace.steps[1].is_error, True)
    check("error type is not repeated", trace.steps[1].content, "(model call failed: ResourceExhausted: 429 quota)")


def test_trailing_generation_becomes_final_answer() -> None:
    print("no run-level output")
    trace = spans_to_trace([
        span("g1", "chat m", 1, gen_ai__operation__name="chat", gen_ai__output__messages=msgs("assistant", "First.")),
        span("g2", "chat m", 2, gen_ai__operation__name="chat", gen_ai__output__messages=msgs("assistant", "Final.")),
    ])
    check("last generation is relabelled, not duplicated", [s.type for s in trace.steps], ["reasoning", "final_answer"])


def test_truncated_values() -> None:
    print("values cut at the attribute length limit")
    cut = json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": "Long report\nline two"}]}])[:-5]
    spans = [span("t1", "chat m", 1, gen_ai__operation__name="chat", gen_ai__output__messages=cut)]
    trace = spans_to_trace(spans)
    check("readable text is salvaged and marked", trace.steps[0].content, "Long report\nline two …[truncated]")
    check("truncation is reported", any("truncated" in g for g in recording_gaps(spans)), True)


def test_any_value_decoding() -> None:
    print("OTLP AnyValue decoding")
    check("intValue string -> int", _any_value({"intValue": "42"}), 42)
    check("arrayValue -> list", _any_value({"arrayValue": {"values": [{"stringValue": "a"}, {"boolValue": True}]}}),
          ["a", True])
    check("kvlistValue -> dict", _any_value({"kvlistValue": {"values": [{"key": "k", "value": {"doubleValue": 1.5}}]}}),
          {"k": 1.5})


def test_trace_selection() -> None:
    """Two runs in one file must never be merged into one trajectory."""
    print("several traces in one file")
    mixed = [span("a", "chat m", 1, trace="aaa", gen_ai__operation__name="chat"),
             span("b", "chat m", 2, trace="bbb", gen_ai__operation__name="chat")]
    try:
        select_trace(mixed)
    except OtelTraceError:
        check("ambiguous file raises", True, True)
    else:
        check("ambiguous file raises", False, True)
    check("an explicit trace id selects one", [s["spanId"] for s in select_trace(mixed, "bbb")], ["b"])


def test_empty_inputs_raise() -> None:
    """Nothing usable must raise, never return a clean-looking empty trace."""
    print("empty inputs")
    for label, call in [
        ("no GenAI spans raises", lambda: spans_to_trace([span("h", "GET /", 0)])),
        ("no spans raises", lambda: spans_to_trace([])),
    ]:
        try:
            call()
        except OtelTraceError:
            check(label, True, True)
        else:
            check(label, False, True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"resourceSpans": []}, f)
    try:
        load_otlp_file(f.name)
    except OtelTraceError:
        check("file with no spans raises", True, True)
    else:
        check("file with no spans raises", False, True)
    finally:
        os.unlink(f.name)


def test_round_trips_through_the_schema() -> None:
    print("round-trip through Trace.from_dict")
    from langfuse_adapter import trace_to_dict
    from schema import Trace

    original = spans_to_trace(AGENT_RUN)
    reloaded = Trace.from_dict(json.loads(json.dumps(trace_to_dict(original))))
    check("step types survive", [s.type for s in reloaded.steps], [s.type for s in original.steps])
    check("tool args survive", reloaded.steps[1].tool_args, {"user_id": "U-4471"})


def test_real_capture() -> None:
    """The live Research Assistant run from capture_otel_trace.py, as the
    Collector wrote it. It crashed on the Gemini quota during revision."""
    print("real capture (real_traces/multi_agent_research_otel_quota_crash.jsonl)")
    path = os.path.join(os.path.dirname(__file__), "real_traces", "multi_agent_research_otel_quota_crash.jsonl")
    trace, gaps = load_trace(path)
    tool_names = [s.tool_name for s in trace.steps if s.type == "tool_call"]

    check("task is the research topic", trace.task, "Produce a research report on: latest advances in protein folding AI")
    check("every tool call is kept", tool_names, ["search_web"] * 3 + ["fetch_arxiv_papers"])
    check("all five agents appear",
          {s.content.split("]")[0] + "]" for s in trace.steps if s.type == "reasoning"},
          {"[Researcher]", "[Analyst]", "[Writer]", "[Verifier]", "[Critic]"})
    check("the run ends on the failed call", (trace.steps[-1].type, trace.steps[-1].is_error), ("reasoning", True))
    check("no final answer for a crashed run", any(s.type == "final_answer" for s in trace.steps), False)
    check("rules invent no missing-argument finding",
          "missing_tool_arguments" in [f.category for f in run_rules(trace)], False)
    check("truncation at the 8000-char limit is reported", any("truncated" in g for g in gaps), True)


def main() -> None:
    for test in (
        test_full_run,
        test_tool_error_status,
        test_unrecorded_opt_in_attributes,
        test_failed_model_call_ends_run,
        test_trailing_generation_becomes_final_answer,
        test_truncated_values,
        test_any_value_decoding,
        test_trace_selection,
        test_empty_inputs_raise,
        test_round_trips_through_the_schema,
        test_real_capture,
    ):
        test()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
