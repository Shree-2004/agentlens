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


def test_provider_registry() -> None:
    """Groq sits ahead of Gemini in auto-detect order: Gemini's 20/day free
    cap is what forced this project's benchmark into fragments."""
    print("provider registry")
    order = list(judge_mod._PROVIDERS)
    check("all three providers registered", order, ["anthropic", "groq", "gemini"])
    check("groq is preferred over gemini", order.index("groq") < order.index("gemini"), True)

    env_var, default_model, call_fn = judge_mod._PROVIDERS["groq"]
    check("groq reads GROQ_API_KEY", env_var, "GROQ_API_KEY")
    check("groq has a default model", bool(default_model), True)
    check("groq call is wired", callable(call_fn), True)


def test_auto_detect_skips_providers_without_keys(monkeypatch_env=None) -> None:
    """Auto-detect must only consider providers whose key is actually set."""
    print("auto-detect")
    import os

    saved = {v: os.environ.pop(v, None) for v, _, _ in judge_mod._PROVIDERS.values()}
    try:
        os.environ["GROQ_API_KEY"] = "test-key"
        seen = {}

        def fake_retry(call_fn, prompt, api_key, model, provider, max_attempts=4):
            seen["provider"] = provider
            seen["model"] = model
            return json.dumps(CLEAN_VERDICT)

        original = judge_mod._call_with_retry
        judge_mod._call_with_retry = fake_retry
        try:
            verdict = judge_mod.run_judge_raw(TRACE)
        finally:
            judge_mod._call_with_retry = original

        check("picks the only provider with a key", seen["provider"], "groq")
        check("uses its default model", seen["model"], judge_mod._PROVIDERS["groq"][1])
        check("stamps the provider on the verdict", verdict["_provider"], "groq")
        check("stamps the model too", verdict["_model"], seen["model"])
    finally:
        os.environ.pop("GROQ_API_KEY", None)
        for var, val in saved.items():
            if val is not None:
                os.environ[var] = val


def test_groq_call_shape() -> None:
    """Exercise _call_groq against a stubbed SDK.

    No Groq key is available here, so this cannot prove the call succeeds
    against the live API -- only that the request is assembled the way the
    SDK expects and the response is read from the right place. Those are
    the parts that fail silently or with a confusing AttributeError rather
    than a clear API error.
    """
    print("groq call shape")
    import sys
    import types as pytypes

    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = pytypes.SimpleNamespace(content='{"has_failure": false}')
            return pytypes.SimpleNamespace(choices=[pytypes.SimpleNamespace(message=msg)])

    class _Groq:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.chat = pytypes.SimpleNamespace(completions=_Completions())

    stub = pytypes.ModuleType("groq")
    stub.Groq = _Groq
    saved = sys.modules.get("groq")
    sys.modules["groq"] = stub
    try:
        out = judge_mod._call_groq("TRACE TEXT", "gsk_test", "llama-3.3-70b-versatile")
    finally:
        if saved is None:
            sys.modules.pop("groq", None)
        else:
            sys.modules["groq"] = saved

    check("returns the message content", out, '{"has_failure": false}')
    check("api key reaches the client", captured["api_key"], "gsk_test")
    check("model is passed through", captured["model"], "llama-3.3-70b-versatile")
    check("asks for a JSON object", captured["response_format"], {"type": "json_object"})
    check("system prompt goes in the system role", captured["messages"][0]["role"], "system")
    check("the judge prompt is the system message",
          captured["messages"][0]["content"], judge_mod.JUDGE_SYSTEM_PROMPT)
    check("trace goes in the user role", captured["messages"][1],
          {"role": "user", "content": "TRACE TEXT"})
    check("max_tokens leaves room for the claims array", captured["max_tokens"] >= 1024, True)


def test_no_keys_means_disabled_not_broken() -> None:
    """No key at all is a deliberate 'judge off' state, not an error."""
    print("no keys set")
    import os

    saved = {v: os.environ.pop(v, None) for v, _, _ in judge_mod._PROVIDERS.values()}
    try:
        check("returns None rather than raising", judge_mod.run_judge_raw(TRACE), None)
        check("run_judge returns no findings", judge_mod.run_judge(TRACE), [])
    finally:
        for var, val in saved.items():
            if val is not None:
                os.environ[var] = val


def main() -> None:
    for test in (
        test_claims_returned_alongside_findings,
        test_claims_survive_a_clean_verdict,
        test_run_judge_signature_unchanged,
        test_diagnosis_carries_claims_to_json,
        test_no_judge_yields_empty_claims,
        test_verdict_without_claims_field,
        test_verdict_to_findings_is_pure,
        test_provider_registry,
        test_auto_detect_skips_providers_without_keys,
        test_groq_call_shape,
        test_no_keys_means_disabled_not_broken,
    ):
        test()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
