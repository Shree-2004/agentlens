"""Tests for evidence-linked verdicts and abstention.

A judge's summary can sound right without being right -- the objection this
project started from. So a failure verdict now has to quote the trace, and
code checks every quote against the step it cites. The judge can also say
the trace doesn't settle the question: before, a tool result that simply
wasn't recorded got the agent accused of an "unsupported commitment" 3/3
times (output/benchmark_report_v7_abstention_baseline.json).

No API calls: verdicts are stubbed.

Run:  python3 test_evidence.py
"""

from __future__ import annotations

import judge as judge_mod
from analyze import diagnose
from judge import MalformedVerdictError, _quote_in_step, _verdict_to_findings, check_evidence
from run_benchmark import score_run
from schema import Trace

_failures: list[str] = []


def check(label: str, got, want) -> None:
    if got == want:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")
        _failures.append(label)


def raises(fn, exc=MalformedVerdictError) -> str:
    try:
        fn()
    except exc:
        return exc.__name__
    except Exception as e:  # noqa: BLE001 -- the wrong exception is the failure being checked
        return f"{type(e).__name__}: {e}"
    return "no error"


TRACE = Trace.from_dict({
    "task": "Text passenger P-5 the gate for UA-88.",
    "steps": [
        {"index": 0, "type": "tool_call", "tool_name": "get_gate", "tool_args": {"flight": "UA-88"}},
        {"index": 1, "type": "tool_result", "tool_name": "get_gate",
         "tool_result": {"flight": "UA-88", "gate": "C3", "note": "Gate changed – see board"}},
        {"index": 2, "type": "reasoning", "content": "UA-88 leaves from gate B3 in terminal 2. I'll text the passenger."},
        {"index": 3, "type": "final_answer", "content": "Texted P-5: UA-88 departs from gate B3."},
    ],
})
STEP = {s.index: s for s in TRACE.steps}

FAILURE = {
    "verdict": "failure", "critical_step": 2, "category": "misread_tool_result", "confidence": "high",
    "summary": "The agent read gate C3 as B3.", "contradictions": ["C3 vs B3"],
    "evidence": [{"step": 1, "quote": '"gate": "C3"'}, {"step": 2, "quote": "leaves from gate B3"}],
}


def test_quote_matching() -> None:
    print("quote matching")
    check("exact text", _quote_in_step("leaves from gate B3", STEP[2]), True)
    check("case and whitespace don't matter", _quote_in_step("LEAVES  from\ngate b3", STEP[2]), True)
    check("a JSON fragment from a tool result", _quote_in_step('"gate": "C3"', STEP[1]), True)
    check("typographic quotes for straight ones", _quote_in_step("“gate”: “C3”", STEP[1]), True)
    check("a plain hyphen for the en dash in the data", _quote_in_step("Gate changed - see board", STEP[1]), True)
    check("the en dash itself (the judge saw it escaped)", _quote_in_step("Gate changed – see board", STEP[1]), True)
    check("ellipsis for elided text, pieces in order", _quote_in_step("leaves from ... terminal 2", STEP[2]), True)
    check("pieces out of order fail", _quote_in_step("terminal 2 ... leaves from", STEP[2]), False)
    check("a paraphrase fails", _quote_in_step("departs via gate B3", STEP[2]), False)
    check("a value that isn't there fails", _quote_in_step('"gate": "B3"', STEP[1]), False)
    check("an empty quote fails", _quote_in_step(" ... ", STEP[2]), False)
    # From the first real run: verbatim text the judge wrapped in quotation marks.
    check("quotation marks added around verbatim text", _quote_in_step('"UA-88 leaves from gate B3 in terminal 2."', STEP[2]), True)
    check("typographic ones too", _quote_in_step("“leaves from gate B3”", STEP[2]), True)
    check("but a wrapped paraphrase still fails", _quote_in_step('"UA-88 departs gate B3."', STEP[2]), False)
    check("and a JSON object with a key dropped from the middle fails",
          _quote_in_step('{"flight": "UA-88", "note": "Gate changed – see board"}', STEP[1]), False)


def test_check_evidence() -> None:
    print("check_evidence")
    got = check_evidence([{"step": 1, "quote": '"gate": "C3"'},
                          {"step": 2, "quote": '"gate": "C3"'},
                          {"step": 9, "quote": "anything"}], TRACE)
    check("right step verified", got[0]["verified"], True)
    check("real text cited at the wrong step is not", got[1]["verified"], False)
    check("a step that doesn't exist is not", got[2]["verified"], False)


def test_findings_carry_checked_evidence() -> None:
    print("findings")
    f = _verdict_to_findings(dict(FAILURE), TRACE)[0]
    check("evidence travels on the finding", [e["verified"] for e in f.evidence], [True, True])
    check("confidence kept when evidence checks out", f.confidence, "high")

    fabricated = {**FAILURE, "evidence": [{"step": 2, "quote": "the passenger asked for B3"}]}
    check("nothing verifiable: confidence drops to low", _verdict_to_findings(fabricated, TRACE)[0].confidence, "low")
    check("no evidence at all: low as well",
          _verdict_to_findings({**FAILURE, "evidence": []}, TRACE)[0].confidence, "low")
    check("the step is kept either way", _verdict_to_findings(fabricated, TRACE)[0].step_index, 2)
    unchecked = _verdict_to_findings(dict(FAILURE))[0]
    check("without the trace, evidence is unchecked and confidence untouched",
          (unchecked.evidence[0]["verified"], unchecked.confidence), (None, "high"))


def test_three_way_verdict() -> None:
    print("three-way verdict")
    abstain = {"verdict": "insufficient_evidence", "critical_step": None, "summary": "warranty terms not recorded",
               "missing_evidence": "the get_warranty result"}
    check("insufficient_evidence produces no finding", _verdict_to_findings(dict(abstain), TRACE), [])
    v = dict(abstain)
    _verdict_to_findings(v, TRACE)
    check("and normalizes has_failure to false", v["has_failure"], False)

    legacy = {"has_failure": True, "critical_step": 2, "confidence": "high", "summary": "s",
              "evidence": [{"step": 2, "quote": "gate B3"}]}
    _verdict_to_findings(legacy, TRACE)
    check("a has_failure-only verdict reads as failure", legacy["verdict"], "failure")
    check("clean is clean", _verdict_to_findings({"verdict": "clean"}, TRACE), [])

    bad = {
        "unknown verdict": {"verdict": "unsure"},
        "verdict contradicts has_failure": {"verdict": "clean", "has_failure": True},
        "evidence not a list": {**FAILURE, "evidence": "gate B3"},
        "evidence item without a step": {**FAILURE, "evidence": [{"quote": "gate B3"}]},
        "evidence step is a string": {**FAILURE, "evidence": [{"step": "2", "quote": "gate B3"}]},
        "missing_evidence not text": {"verdict": "insufficient_evidence", "missing_evidence": ["x"]},
        "failure with no step": {**FAILURE, "critical_step": None},
    }
    for label, verdict in bad.items():
        check(label, raises(lambda v=verdict: _verdict_to_findings(dict(v), TRACE)), "MalformedVerdictError")


def _stubbed(verdict, fn):
    original = judge_mod.run_judge_raw
    judge_mod.run_judge_raw = lambda trace, provider=None, model=None: dict(verdict)
    try:
        return fn()
    finally:
        judge_mod.run_judge_raw = original


def test_diagnosis_records_the_verdict() -> None:
    print("diagnosis")
    abstain = {"verdict": "insufficient_evidence", "summary": "s", "missing_evidence": "the get_warranty result"}
    diag = _stubbed(abstain, lambda: diagnose(TRACE, use_judge=True))
    check("abstention is recorded, not read as clean",
          (diag.judge["verdict"], diag.judge["missing_evidence"]), ("insufficient_evidence", "the get_warranty result"))
    check("and invents no critical step", diag.critical_step, None)
    payload = _stubbed(FAILURE, lambda: diagnose(TRACE, use_judge=True)).to_dict()
    check("failure verdict recorded", payload["judge"]["verdict"], "failure")
    check("checked evidence reaches the JSON the viewer reads",
          payload["critical_finding"]["evidence"][1], {"step": 2, "quote": "leaves from gate B3", "verified": True})


def test_benchmark_scoring() -> None:
    print("benchmark scoring")
    f = _verdict_to_findings(dict(FAILURE), TRACE)
    failure_label = {"has_failure": True, "critical_step": 2}
    abstain_label = {"verdict": "insufficient_evidence", "has_failure": None, "critical_step": None}
    check("right step", score_run("failure", f, failure_label), True)
    check("wrong step", score_run("failure", f, {**failure_label, "critical_step": 1}), False)
    check("abstaining on a decidable failure is a miss", score_run("insufficient_evidence", [], failure_label), False)
    check("abstaining where the label says abstain is correct", score_run("insufficient_evidence", [], abstain_label), True)
    check("accusing where the label says abstain is a miss", score_run("failure", f, abstain_label), False)
    check("clean on an old has_failure=false label", score_run("clean", [], {"has_failure": False}), True)


def main() -> None:
    for test in (
        test_quote_matching,
        test_check_evidence,
        test_findings_carry_checked_evidence,
        test_three_way_verdict,
        test_diagnosis_records_the_verdict,
        test_benchmark_scoring,
    ):
        test()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
