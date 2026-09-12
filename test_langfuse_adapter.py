"""Fixture tests for the Langfuse -> AgentLens mapping.

No network, no Langfuse account, no API key: observations_to_trace() is a
pure function over observation dicts, and these fixtures are shaped like
what the observations API actually returns (camelCase keys, input/output
as JSON strings, newest-first ordering).

The point is to pin the mapping decisions that are easy to get silently
wrong -- ordering, the root span, error flags, empty tool_args -- because
a wrong mapping doesn't crash. It quietly hands the judge a different
trajectory than the agent ran, and every conclusion after that is about a
trace that never happened.

Run:  python3 test_langfuse_adapter.py
"""

from __future__ import annotations

import json

from langfuse_adapter import _as_args, _maybe_json, observations_to_trace
from rules import run_rules

_failures: list[str] = []


def check(label: str, got, want) -> None:
    if got == want:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")
        _failures.append(label)


def obs(oid, otype, name, start, parent=None, inp=None, out=None, level="DEFAULT", root=False):
    """One observation in the shape the Langfuse API returns it."""
    return {
        "id": oid,
        "type": otype,
        "name": name,
        "startTime": start,
        "parentObservationId": parent,
        "isRootObservation": root,
        "input": inp,
        "output": out,
        "level": level,
    }


# A plan-tier lookup, shaped like sample_traces/stale_plan_tier.json but
# arriving as Langfuse observations: a root span wrapping one generation,
# one tool span, and a second generation.
AGENT_RUN = [
    # Deliberately newest-first and out of order, the way the API returns it.
    obs("o4", "GENERATION", "answer", "2026-09-12T10:00:03Z", parent="o0",
        out="U-4471 is on the Free tier, so bulk export isn't available."),
    obs("o3", "SPAN", "check_user_plan", "2026-09-12T10:00:02Z", parent="o0",
        inp='{"user_id": "U-4471"}',
        out='{"tier": "free", "source": "plan_cache", "cached_at": "2026-01-14"}'),
    obs("o2", "SPAN", "setup", "2026-09-12T10:00:01Z", parent="o0"),  # no io -- organizational
    obs("o1", "GENERATION", "plan", "2026-09-12T10:00:00Z", parent="o0",
        out="Bulk export is Pro-only. I should check the user's plan first."),
    obs("o0", "SPAN", "agent_run", "2026-09-12T09:59:59Z", parent=None, root=True,
        inp="Can U-4471 use bulk export?",
        out="You're on the Free tier, which doesn't include bulk export."),
]


def test_full_run() -> None:
    print("full agent run")
    trace = observations_to_trace(AGENT_RUN)

    check("task comes from the root span's input", trace.task, "Can U-4471 use bulk export?")
    check(
        "chronological order, root excluded, empty span dropped",
        [s.type for s in trace.steps],
        ["reasoning", "tool_call", "tool_result", "reasoning", "final_answer"],
    )
    check("indices are renumbered contiguously", [s.index for s in trace.steps], [0, 1, 2, 3, 4])
    check("generation output becomes reasoning content",
          trace.steps[0].content, "Bulk export is Pro-only. I should check the user's plan first.")
    check("tool span name carries over", trace.steps[1].tool_name, "check_user_plan")
    check("JSON-string input is parsed into tool_args",
          trace.steps[1].tool_args, {"user_id": "U-4471"})
    check("JSON-string output is parsed into tool_result",
          trace.steps[2].tool_result, {"tier": "free", "source": "plan_cache", "cached_at": "2026-01-14"})
    check("root output becomes the final answer",
          trace.steps[4].content, "You're on the Free tier, which doesn't include bulk export.")


def test_error_level() -> None:
    print("error levels")
    trace = observations_to_trace([
        obs("e1", "SPAN", "fetch_invoice", "2026-09-12T10:00:00Z",
            inp='{"id": 7}', out='{"error": "not found"}', level="ERROR"),
    ])
    check("ERROR level sets is_error", trace.steps[1].is_error, True)

    ok = observations_to_trace([
        obs("e2", "SPAN", "fetch_invoice", "2026-09-12T10:00:00Z", inp='{"id": 7}', out='{"ok": true}'),
    ])
    check("DEFAULT level leaves is_error false", ok.steps[1].is_error, False)


def test_no_false_missing_arguments() -> None:
    """A non-dict tool input must not read as 'called with no arguments'.

    rules.py flags empty tool_args as a malformed call, so wrapping rather
    than dropping a plain-string input is what keeps the adapter from
    inventing a rule finding for a call that was perfectly well-formed.
    """
    print("non-dict inputs don't fake a rule finding")
    check("plain string input is wrapped, not dropped",
          _as_args("U-4471"), {"input": "U-4471"})

    trace = observations_to_trace([
        obs("s1", "SPAN", "lookup", "2026-09-12T10:00:00Z", inp="U-4471", out='{"tier": "pro"}'),
    ])
    findings = [f.category for f in run_rules(trace)]
    check("no missing_tool_arguments finding", "missing_tool_arguments" in findings, False)


def test_trailing_generation_becomes_final_answer() -> None:
    print("no trace-level output")
    trace = observations_to_trace([
        obs("g1", "GENERATION", "step", "2026-09-12T10:00:00Z", out="First thought."),
        obs("g2", "GENERATION", "step", "2026-09-12T10:00:01Z", out="Final thought."),
    ])
    check("last generation is relabelled, not duplicated",
          [s.type for s in trace.steps], ["reasoning", "final_answer"])
    check("its content is preserved", trace.steps[1].content, "Final thought.")


def test_single_observation_keeps_its_content() -> None:
    """A lone observation is the whole trace, not a root wrapper to skip."""
    print("single-observation trace")
    trace = observations_to_trace([
        obs("only", "SPAN", "search", "2026-09-12T10:00:00Z", root=True,
            inp='{"q": "protein folding"}', out='{"hits": 3}'),
    ])
    check("it still produces a tool_call/tool_result pair",
          [s.type for s in trace.steps], ["tool_call", "tool_result"])


def test_malformed_json_passes_through() -> None:
    print("malformed payloads")
    check("unparseable JSON stays a string", _maybe_json('{"broken": '), '{"broken": ')
    check("plain prose is left alone", _maybe_json("just text"), "just text")
    check("non-strings are untouched", _maybe_json({"a": 1}), {"a": 1})


def test_round_trips_through_the_schema() -> None:
    """The output has to survive the same JSON -> Trace.from_dict() path
    every other AgentLens entry point uses, or analyze.py can't read it."""
    print("round-trip through Trace.from_dict")
    from langfuse_adapter import trace_to_dict
    from schema import Trace

    original = observations_to_trace(AGENT_RUN)
    reloaded = Trace.from_dict(json.loads(json.dumps(trace_to_dict(original))))
    check("step count survives", len(reloaded.steps), len(original.steps))
    check("step types survive", [s.type for s in reloaded.steps], [s.type for s in original.steps])
    check("tool args survive", reloaded.steps[1].tool_args, {"user_id": "U-4471"})


class _StubPage:
    def __init__(self, data, total_pages):
        self.data = data
        self.meta = type("Meta", (), {"total_pages": total_pages})()


class _StubClient:
    """Stands in for the Langfuse SDK client so fetch_trace's paging and
    fallback logic can be exercised without an account or the package."""

    def __init__(self, pages, trace_detail=None, trace_raises=False):
        self._pages = pages
        self._trace_detail = trace_detail
        self._trace_raises = trace_raises
        self.requested_pages: list[int] = []
        outer = self

        class _Observations:
            def get_many(self, trace_id, fields, page, limit):
                outer.requested_pages.append(page)
                return _StubPage(outer._pages[page - 1], len(outer._pages))

        class _TraceApi:
            def get(self, trace_id):
                if outer._trace_raises:
                    raise RuntimeError("endpoint unavailable")
                return outer._trace_detail

        self.api = type("Api", (), {"observations": _Observations(), "trace": _TraceApi()})()


def test_fetch_pages_and_uses_trace_detail() -> None:
    print("fetch_trace paging + trace-level detail")
    from langfuse_adapter import fetch_trace

    client = _StubClient(
        pages=[AGENT_RUN[:3], AGENT_RUN[3:]],
        trace_detail={"name": "plan check", "input": "Can U-4471 use bulk export?",
                      "output": "You're on the Free tier."},
    )
    trace = fetch_trace("t-1", client=client)
    check("both pages were requested", client.requested_pages, [1, 2])
    check("observations from every page are kept", len(trace.steps), 5)
    check("trace name becomes the task", trace.task, "plan check")
    check("trace output becomes the final answer", trace.steps[-1].content, "You're on the Free tier.")


def test_fetch_falls_back_when_trace_endpoint_fails() -> None:
    """The trace-level endpoint is the older one. If it's gone, the import
    must still succeed off the root observation instead of dying."""
    print("fetch_trace fallback")
    from langfuse_adapter import fetch_trace

    trace = fetch_trace("t-2", client=_StubClient(pages=[AGENT_RUN], trace_raises=True))
    check("still imports every step", len(trace.steps), 5)
    check("task falls back to the root span's input", trace.task, "Can U-4471 use bulk export?")


def test_empty_trace_raises() -> None:
    """An empty fetch must raise, never return a clean-looking empty trace."""
    print("empty trace")
    from langfuse_adapter import LangfuseUnavailableError, fetch_trace

    try:
        fetch_trace("t-3", client=_StubClient(pages=[[]]))
    except LangfuseUnavailableError:
        check("raises LangfuseUnavailableError", True, True)
    else:
        check("raises LangfuseUnavailableError", False, True)


def main() -> None:
    for test in (
        test_full_run,
        test_error_level,
        test_no_false_missing_arguments,
        test_trailing_generation_becomes_final_answer,
        test_single_observation_keeps_its_content,
        test_malformed_json_passes_through,
        test_round_trips_through_the_schema,
        test_fetch_pages_and_uses_trace_detail,
        test_fetch_falls_back_when_trace_endpoint_fails,
        test_empty_trace_raises,
    ):
        test()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
