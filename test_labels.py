"""Consistency checks for every labels file under sample_traces/.

A label that points at a step the trace doesn't have, or whose acceptable
steps leave out the labeled step, doesn't fail loudly -- it just turns into a
wrong benchmark number. These run without an API key, so a bad label is
caught before any judge sees the trace.

Run:  python3 test_labels.py
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile

from schema import Trace

_failures: list[str] = []
HERE = os.path.dirname(os.path.abspath(__file__))

VERDICTS = ("failure", "clean", "insufficient_evidence")
# Closed category list for schema v2 labels (EVAL_PROTOCOL.md). Older labels
# predate it and use a few names of their own.
CATEGORIES = ("stale_fact", "contradicted_assumption", "misread_tool_result", "unsupported_commitment",
              "dropped_caveat", "computational_error", "wrong_tool_selection", "silent_tool_error",
              "tool_call_loop", "run_aborted", "other")


def check(label: str, got, want) -> None:
    if got == want:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")
        _failures.append(label)


def label_files() -> list[str]:
    return sorted(glob.glob(os.path.join(HERE, "sample_traces", "**", "ground_truth.json"), recursive=True))


def problems_in(path: str) -> list[str]:
    folder = os.path.dirname(path)
    labels = json.load(open(path, encoding="utf-8"))["labels"]
    found = []
    for name, label in labels.items():
        where = f"{os.path.relpath(path, HERE)}:{name}"
        trace_path = os.path.join(folder, name)
        if not os.path.exists(trace_path):
            found.append(f"{where}: trace file missing")
            continue
        steps = {s.index for s in Trace.load(trace_path).steps}
        verdict = label.get("verdict") or ("failure" if label["has_failure"] else "clean")
        if verdict not in VERDICTS:
            found.append(f"{where}: verdict {verdict!r}")
            continue
        if label.get("has_failure") is not None and label["has_failure"] != (verdict == "failure"):
            found.append(f"{where}: has_failure contradicts verdict")
        step = label.get("critical_step")
        if verdict == "failure":
            if step not in steps:
                found.append(f"{where}: critical_step {step} not in trace")
            acceptable = label.get("acceptable_steps")
            if acceptable is not None and (step not in acceptable or not set(acceptable) <= steps):
                found.append(f"{where}: acceptable_steps {acceptable} must include {step} and exist")
            symptom = label.get("first_symptom_step")
            if symptom is not None and (symptom not in steps or symptom < step):
                found.append(f"{where}: first_symptom_step {symptom} must exist and not precede the cause")
        elif step is not None:
            found.append(f"{where}: {verdict} label has critical_step {step}")
        if not set(label.get("evidence_steps") or []) <= steps:
            found.append(f"{where}: evidence_steps outside the trace")
        if "acceptable_steps" in label and label.get("category") not in (None, *CATEGORIES):
            found.append(f"{where}: category {label['category']!r} not in the closed list")
        if not (label.get("notes") or "").strip():
            found.append(f"{where}: no rationale")
    return found


def test_every_labels_file_is_consistent() -> None:
    print("labels files")
    files = label_files()
    check("found the dev, adversarial, abstention and validation sets", len(files) >= 4, True)
    for path in files:
        check(os.path.relpath(path, HERE), problems_in(path), [])


def test_no_task_is_reused_across_sets() -> None:
    """A validation trace that's a copy of a dev trace measures nothing new."""
    print("leakage")
    seen: dict[str, str] = {}
    duplicates = []
    for path in label_files():
        folder = os.path.dirname(path)
        for name in json.load(open(path, encoding="utf-8"))["labels"]:
            task = Trace.load(os.path.join(folder, name)).task.strip().lower()
            where = os.path.relpath(os.path.join(folder, name), HERE)
            # Controls are copies by design: the same trace with only the
            # injected text removed. ctl_x pairs with inj_x and nothing else.
            stem = lambda n: os.path.basename(n).replace("inj_", "").replace("ctl_", "")  # noqa: E731
            is_control_pair = (task in seen and stem(seen[task]) == stem(name)
                               and {os.path.basename(seen[task])[:4], name[:4]} == {"inj_", "ctl_"})
            if task in seen and not is_control_pair:
                duplicates.append((seen[task], where))
            seen[task] = where
    check("no task text appears in two traces", duplicates, [])


def test_unreviewed_labels_cannot_be_scored() -> None:
    print("review gate")
    with tempfile.TemporaryDirectory() as d:
        labels = os.path.join(d, "ground_truth.json")
        with open(os.path.join(d, "t.json"), "w", encoding="utf-8") as f:
            json.dump({"task": "t", "steps": [{"index": 0, "type": "final_answer", "content": "done"}]}, f)
        with open(labels, "w", encoding="utf-8") as f:
            json.dump({"review": {"status": "pending"},
                       "labels": {"t.json": {"verdict": "clean", "has_failure": False, "critical_step": None,
                                             "notes": "x"}}}, f)
        run = subprocess.run([sys.executable, os.path.join(HERE, "run_benchmark.py"), "1", "--labels", labels,
                              "--output", os.path.join(d, "r.json")], capture_output=True, text=True, cwd=HERE,
                             env={**os.environ, "GROQ_API_KEY": "", "ANTHROPIC_API_KEY": "", "GOOGLE_API_KEY": ""})
        check("pending labels: refuses, before any judge call", (run.returncode != 0, "not approved" in run.stderr),
              (True, True))
        check("and writes no report", os.path.exists(os.path.join(d, "r.json")), False)


def main() -> None:
    for test in (test_every_labels_file_is_consistent, test_no_task_is_reused_across_sets,
                 test_unreviewed_labels_cannot_be_scored):
        test()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
