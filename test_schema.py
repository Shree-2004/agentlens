"""Tests for trace validation and input limits.

Traces are untrusted input. Before Trace.validate, malformed ones failed in
two bad ways: loudly but uselessly (KeyError: 'task', a RecursionError from
the JSON parser) or -- worse -- silently. A step whose type the judge's
formatter has no branch for was simply left out of the prompt, so the
diagnosis covered a trajectory with a hole in it; duplicate indices made
"critical step 3" point at two steps.

Run:  python3 test_schema.py
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile

import schema
from analyze import diagnose
from schema import MAX_DEPTH, MAX_STEPS, Step, Trace, TraceFormatError

_failures: list[str] = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label: str, got, want) -> None:
    if got == want:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")
        _failures.append(label)


def outcome(fn) -> str:
    """'ok', 'TraceFormatError', or the unexpected exception that escaped."""
    try:
        fn()
    except TraceFormatError:
        return "TraceFormatError"
    except Exception as e:  # noqa: BLE001 -- an escaping exception is the failure being checked
        return f"{type(e).__name__}: {e}"
    return "ok"


def step(index, type_="reasoning", **fields) -> dict:
    return {"index": index, "type": type_, **fields}


def nested(depth: int):
    """`depth` objects nested inside each other, around a plain value."""
    value: object = "leaf"
    for _ in range(depth):
        value = {"k": value}
    return value


def test_malformed_traces_are_rejected() -> None:
    print("malformed traces")
    bad = {
        "top level is a list": [1, 2],
        "missing task": {"steps": []},
        "task is not text": {"task": 42, "steps": []},
        "steps is not a list": {"task": "t", "steps": "oops"},
        "a step is not an object": {"task": "t", "steps": ["reasoning"]},
        "a step has no index": {"task": "t", "steps": [{"type": "reasoning"}]},
        "index is a string": {"task": "t", "steps": [step("0")]},
        "index is a bool": {"task": "t", "steps": [step(True)]},
        "duplicate indices": {"task": "t", "steps": [step(0), step(0, "final_answer")]},
        "indices go backwards": {"task": "t", "steps": [step(1), step(0)]},
        "unknown step type (used to vanish from the prompt)": {"task": "t", "steps": [step(0, "thinking")]},
        "content is not text": {"task": "t", "steps": [step(0, content=["a"])]},
        "tool_args is a list": {"task": "t", "steps": [step(0, "tool_call", tool_name="x", tool_args=[1])]},
        "tool_name is not text": {"task": "t", "steps": [step(0, "tool_call", tool_name=7, tool_args={"a": 1})]},
        "is_error is a string": {"task": "t", "steps": [step(0, "tool_result", is_error="false")]},
    }
    for label, data in bad.items():
        check(label, outcome(lambda d=data: Trace.from_dict(d)), "TraceFormatError")


def test_valid_variations_still_load() -> None:
    print("valid variations")
    check("null content means no text",
          Trace.from_dict({"task": "t", "steps": [step(0, content=None)]}).steps[0].content, "")
    check("gaps in indices are fine", outcome(lambda: Trace.from_dict({"task": "t", "steps": [step(0), step(5)]})), "ok")
    check("no steps at all is fine", outcome(lambda: Trace.from_dict({"task": "t"})), "ok")
    check("nesting at the limit loads",
          outcome(lambda: Trace.from_dict({"task": "t", "steps": [step(0, "tool_result", tool_result=nested(MAX_DEPTH))]})),
          "ok")
    check("one level past it doesn't",
          outcome(lambda: Trace.from_dict({"task": "t", "steps": [step(0, "tool_result", tool_result=nested(MAX_DEPTH + 1))]})),
          "TraceFormatError")
    check("step count over the limit is rejected",
          outcome(lambda: Trace.from_dict({"task": "t", "steps": [step(i) for i in range(MAX_STEPS + 1)]})),
          "TraceFormatError")


def test_file_level_limits() -> None:
    print("file loading")
    with tempfile.TemporaryDirectory() as d:
        broken = os.path.join(d, "broken.json")
        with open(broken, "w", encoding="utf-8") as f:
            f.write('{"task": "t", "steps": [')
        check("invalid JSON", outcome(lambda: Trace.load(broken)), "TraceFormatError")

        deep = os.path.join(d, "deep.json")
        with open(deep, "w", encoding="utf-8") as f:
            f.write('{"task":"t","steps":[{"index":0,"type":"tool_result","tool_result":'
                    + "[" * 100_000 + "]" * 100_000 + "}]}")
        check("nesting deep enough to break the JSON parser", outcome(lambda: Trace.load(deep)), "TraceFormatError")

        small = os.path.join(d, "small.json")
        with open(small, "w", encoding="utf-8") as f:
            json.dump({"task": "t", "steps": [step(0)]}, f)
        saved = schema.MAX_TRACE_BYTES
        schema.MAX_TRACE_BYTES = 10
        try:
            check("file over the size limit is rejected before parsing", outcome(lambda: Trace.load(small)),
                  "TraceFormatError")
        finally:
            schema.MAX_TRACE_BYTES = saved


def test_adapter_built_traces_are_checked_too() -> None:
    """Adapters build Trace objects directly, never passing through from_dict."""
    print("traces built in code")
    built = Trace(task="t", steps=[Step(index=0, type="thinking")])
    check("diagnose validates before the rules run",
          outcome(lambda: diagnose(built, use_judge=False)), "TraceFormatError")


def test_cli_reports_instead_of_crashing() -> None:
    print("analyze.py")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"steps": []}, f)
        run = subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"), path, os.path.join(d, "out.json"),
                              "--no-judge"], capture_output=True, text=True, cwd=HERE)
        check("exits non-zero", run.returncode != 0, True)
        check("with a one-line reason, not a traceback",
              ("can't read trace: trace has no 'task' string" in run.stderr, "Traceback" in run.stderr), (True, False))
        check("and writes no diagnosis", os.path.exists(os.path.join(d, "out.json")), False)


def test_every_trace_in_the_repo_loads() -> None:
    """The limits must not reject anything real."""
    print("existing traces")
    files = [f for pattern in ("sample_traces/**/*.json", "real_traces/*.json", "distillation/synthetic_traces/*.json")
             for f in glob.glob(os.path.join(HERE, pattern), recursive=True)
             if not any(skip in os.path.basename(f) for skip in ("ground_truth", "diagnosis", "verdict"))]
    rejected = [os.path.relpath(f, HERE) for f in files if outcome(lambda f=f: Trace.load(f)) != "ok"]
    check(f"all {len(files)} trace files load", rejected, [])


def main() -> None:
    for test in (
        test_malformed_traces_are_rejected,
        test_valid_variations_still_load,
        test_file_level_limits,
        test_adapter_built_traces_are_checked_too,
        test_cli_reports_instead_of_crashing,
        test_every_trace_in_the_repo_loads,
    ):
        test()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
