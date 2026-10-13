"""Tests for find_aborted_run, the rule for a run that crashed before answering.

Found by running the rules over a real OpenTelemetry capture
(real_traces/multi_agent_research_otel_quota_crash.jsonl): the run died on a
Gemini quota error mid-revision, and rules-only analysis reported "No failure
found." The other rules only look inside runs that finished.

The rule is deliberately narrow, so most of these checks are about when it
must stay quiet.

Run:  python3 test_rules.py
"""

from __future__ import annotations

import glob
import os

from otel_adapter import load_trace
from rules import find_aborted_run, run_rules
from schema import Step, Trace

_failures: list[str] = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label: str, got, want) -> None:
    if got == want:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")
        _failures.append(label)


def trace_of(*steps: Step) -> Trace:
    return Trace(task="t", steps=[Step(index=i, type=s.type, content=s.content, tool_name=s.tool_name,
                                       tool_args=s.tool_args, tool_result=s.tool_result, is_error=s.is_error)
                                  for i, s in enumerate(steps)])


def test_crash_on_model_call() -> None:
    print("run ends on a failed model call")
    trace = trace_of(
        Step(0, "reasoning", content="Plan: search."),
        Step(0, "tool_call", tool_name="search", tool_args={"q": "x"}),
        Step(0, "tool_result", tool_name="search", tool_result=[{"title": "x"}]),
        Step(0, "reasoning", content="(model call failed: ResourceExhausted: 429)", is_error=True),
    )
    findings = find_aborted_run(trace)
    check("flagged once", len(findings), 1)
    check("at the failed step", findings[0].step_index, 3)
    check("as run_aborted, high confidence", (findings[0].category, findings[0].confidence), ("run_aborted", "high"))
    check("the error text is the detail", findings[0].detail, "(model call failed: ResourceExhausted: 429)")


def test_crash_on_tool() -> None:
    print("run ends on a failed tool call")
    trace = trace_of(
        Step(0, "tool_call", tool_name="fetch", tool_args={"id": 7}),
        Step(0, "tool_result", tool_name="fetch", tool_result={"error": "timeout"}, is_error=True),
    )
    check("flagged at the tool result", [f.step_index for f in find_aborted_run(trace)], [1])


def test_quiet_cases() -> None:
    print("cases the rule must leave alone")
    answered = trace_of(
        Step(0, "reasoning", content="(model call failed: 503)", is_error=True),
        Step(0, "reasoning", content="Retrying."),
        Step(0, "final_answer", content="Done."),
    )
    check("a run that recovered and answered", find_aborted_run(answered), [])

    partial = trace_of(Step(0, "reasoning", content="Still working..."))
    check("a trace that just stops, with no error (maybe a partial capture)", find_aborted_run(partial), [])

    check("an empty trace", find_aborted_run(Trace(task="t", steps=[])), [])


def test_existing_traces_unchanged() -> None:
    """Every trace that finished must get exactly the findings it got before."""
    print("no new findings on traces that finished")
    paths = [p for p in sorted(glob.glob(os.path.join(HERE, "sample_traces", "*.json")))
             if not p.endswith("ground_truth.json")]
    paths.append(os.path.join(HERE, "real_traces", "multi_agent_research_live.json"))
    flagged = [os.path.basename(p) for p in paths if find_aborted_run(Trace.load(p))]
    check(f"none of {len(paths)} sample/real traces is flagged", flagged, [])


def test_real_otel_crash() -> None:
    print("real capture (multi_agent_research_otel_quota_crash.jsonl)")
    trace, _ = load_trace(os.path.join(HERE, "real_traces", "multi_agent_research_otel_quota_crash.jsonl"))
    findings = run_rules(trace)
    check("rules now find the crash", [(f.step_index, f.category) for f in findings],
          [(len(trace.steps) - 1, "run_aborted")])
    check("and name the cause", "ResourceExhausted" in findings[0].detail, True)


def main() -> None:
    for test in (
        test_crash_on_model_call,
        test_crash_on_tool,
        test_quiet_cases,
        test_existing_traces_unchanged,
        test_real_otel_crash,
    ):
        test()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
