"""Checks that the judge's extracted claims survive the trip to the viewer.

The claims array is the evidence the extraction pass actually ran -- the
structural fix that turned the real-trace miss into a catch. It used to be
discarded the moment a verdict was flattened into a Finding, which meant
the one artifact proving the judge did the work was invisible everywhere
except debug_judge.py.

These checks pin the plumbing end to end, with a stubbed verdict instead
of an API call, so a refactor can't quietly drop claims again.

Run:  python3 test_claims_plumbing.py
"""

from __future__ import annotations

import json

import judge as judge_mod
from analyze import diagnose
from judge import _verdict_to_findings, run_judge_with_claims
from schema import Trace

_failures: list[str] = []


def check(label: str, got, want) -> None:
    if got == want:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")
        _failures.append(label)


TRACE = Trace.from_dict({
    "task": "Check whether U-4471 can use bulk export.",
    "steps": [
        {"index": 0, "type": "reasoning", "content": "Bulk export is Pro-only; check the plan."},
        {"index": 1, "type": "tool_call", "tool_name": "check_user_plan", "tool_args": {"user_id": "U-4471"}},
        {"index": 2, "type": "tool_result", "tool_name": "check_user_plan",
         "tool_result": {"tier": "free", "source": "plan_cache"}},
        {"index": 3, "type": "final_answer", "content": "You're on the Free tier."},
    ],
})

FAILURE_VERDICT = {
    "claims": [
        {"step": 2, "claim": "U-4471's cached tier is 'free'."},
        {"step": 3, "claim": "U-4471 cannot use bulk export."},
    ],
    "has_failure": True,
    "critical_step": 2,
    "category": "stale_fact",
    "confidence": "high",
    "summary": "Answered from a stale plan cache.",
    "contradictions": ["step 2 says free; billing says pro"],
}

CLEAN_VERDICT = {
    "claims": [{"step": 2, "claim": "U-4471's tier is 'free'."}],
    "has_failure": False,
    "critical_step": None,
    "category": None,
    "confidence": "high",
    "summary": "No contradiction found.",
    "contradictions": [],
}


def _with_stub_verdict(verdict, fn):
    """Swap run_judge_raw for a stub so no API call happens."""
    original = judge_mod.run_judge_raw
    judge_mod.run_judge_raw = lambda trace, provider=None, model=None: verdict
    try:
        return fn()
    finally:
        judge_mod.run_judge_raw = original


def test_claims_returned_alongside_findings() -> None:
    print("run_judge_with_claims")
    findings, claims = _with_stub_verdict(FAILURE_VERDICT, lambda: run_judge_with_claims(TRACE))
    check("finding still produced", [f.step_index for f in findings], [2])
    check("claims come back too", len(claims), 2)


def test_claims_survive_a_clean_verdict() -> None:
    """The false-positive case is the one that matters most here: with no
    finding to attach to, claims are the only evidence the judge looked at
    the trace at all rather than returning nothing."""
    print("clean verdict")
    findings, claims = _with_stub_verdict(CLEAN_VERDICT, lambda: run_judge_with_claims(TRACE))
    check("no findings on a clean trace", findings, [])
    check("claims are still returned", len(claims), 1)


def test_run_judge_signature_unchanged() -> None:
    """check_consistency.py and run_benchmark.py call run_judge directly."""
    print("run_judge back-compat")
    findings = _with_stub_verdict(FAILURE_VERDICT, lambda: judge_mod.run_judge(TRACE))
    check("still returns a plain list of findings", [f.step_index for f in findings], [2])
    check("finding detail still joins contradictions",
          findings[0].detail, "step 2 says free; billing says pro")


def test_diagnosis_carries_claims_to_json() -> None:
    print("diagnose -> JSON")
    diag = _with_stub_verdict(FAILURE_VERDICT, lambda: diagnose(TRACE, use_judge=True))
    check("diagnosis holds the claims", len(diag.claims), 2)

    payload = json.loads(json.dumps(diag.to_dict()))
    check("claims are serialized for the viewer", payload["claims"], FAILURE_VERDICT["claims"])
    check("every claim is tagged with a step",
          all("step" in c and "claim" in c for c in payload["claims"]), True)
    check("claim steps exist in the rendered trace",
          {c["step"] for c in payload["claims"]} <= {s["index"] for s in payload["steps"]}, True)


def test_no_judge_yields_empty_claims() -> None:
    """Rules-only must not imply the judge ran and found nothing to extract."""
    print("rules-only")
    diag = diagnose(TRACE, use_judge=False)
    check("claims list is empty", diag.claims, [])
    check("key is still present for the viewer", "claims" in diag.to_dict(), True)


def test_verdict_without_claims_field() -> None:
    """Older recorded verdicts predate the claims field; don't crash on them."""
    print("legacy verdict")
    legacy = {k: v for k, v in FAILURE_VERDICT.items() if k != "claims"}
    findings, claims = _with_stub_verdict(legacy, lambda: run_judge_with_claims(TRACE))
    check("findings unaffected", [f.step_index for f in findings], [2])
    check("claims default to empty", claims, [])
    check("null claims are handled too",
          _with_stub_verdict({**FAILURE_VERDICT, "claims": None},
                             lambda: run_judge_with_claims(TRACE))[1], [])


def test_verdict_to_findings_is_pure() -> None:
    print("_verdict_to_findings")
    check("clean verdict yields nothing", _verdict_to_findings(CLEAN_VERDICT), [])
    check("missing category falls back",
          _verdict_to_findings({**FAILURE_VERDICT, "category": None})[0].category, "state_corruption")


def main() -> None:
    for test in (
        test_claims_returned_alongside_findings,
        test_claims_survive_a_clean_verdict,
        test_run_judge_signature_unchanged,
        test_diagnosis_carries_claims_to_json,
        test_no_judge_yields_empty_claims,
        test_verdict_without_claims_field,
        test_verdict_to_findings_is_pure,
    ):
        test()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
