"""Tests for the judge's retry policy.

Found in real runs: a 17-minute daily-quota wait was retried 4 times, a
dropped connection (no status code) ended a run on its first occurrence, a
Gemini call hung with no timeout, and once Groq's daily cap was reached
every remaining benchmark run was recorded as an error. The error messages
below are the ones the providers actually sent.

No API calls and no real sleeping: the providers are fakes and time.sleep
is recorded instead.

Run:  python3 test_retry.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import types

import judge as judge_mod
import run_benchmark
from judge import (JudgeQuotaExhaustedError, JudgeUnavailableError, _call_with_retry, _classify,
                   _duration_seconds, _wait_hint_seconds)

_failures: list[str] = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label: str, got, want) -> None:
    if got == want:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")
        _failures.append(label)


class StatusError(Exception):
    def __init__(self, status_code, message, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = types.SimpleNamespace(headers=headers or {})


class APIConnectionError(Exception):
    """Named like the SDKs' class: no status code at all."""


class APITimeoutError(Exception):
    pass


GROQ_DAILY = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in "
              "organization `org_x` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 198747, "
              "Requested 3638. Please try again in 17m10.319999999s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}")
GROQ_PER_MINUTE = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` on "
                   "tokens per minute (TPM): Limit 8000, Used 4999, Requested 3090. Please try again in 667.5ms.'}}")
GEMINI_DAILY = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota. "
                "Please retry in 26.58s.', 'details': [{'violations': [{'quotaId': "
                "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, {'retryDelay': '26s'}]}}")
GEMINI_BUSY = "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'The model is overloaded. Please try again later.'}}"


class Script:
    """A fake provider that raises each queued error, then returns."""

    def __init__(self, *errors):
        self.errors, self.calls = list(errors), 0

    def __call__(self, prompt, api_key, model):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return '{"verdict": "clean"}'


def with_recorded_sleep(fn):
    waits = []
    original = time.sleep
    time.sleep = waits.append
    try:
        return fn(), waits
    finally:
        time.sleep = original


def outcome(fn):
    try:
        return fn()
    except JudgeQuotaExhaustedError as e:
        return ("JudgeQuotaExhaustedError", str(e))
    except JudgeUnavailableError as e:
        return ("JudgeUnavailableError", str(e))


def test_classification() -> None:
    print("classification")
    check("Groq tokens-per-day is daily", _classify(StatusError(429, GROQ_DAILY)), "daily")
    check("Gemini free-tier PerDay quota is daily", _classify(StatusError(429, GEMINI_DAILY)), "daily")
    check("Groq tokens-per-minute is transient", _classify(StatusError(429, GROQ_PER_MINUTE)), "transient")
    check("503 overload is transient", _classify(StatusError(503, GEMINI_BUSY)), "transient")
    check("a dropped connection is transient", _classify(APIConnectionError("Connection error.")), "transient")
    check("a timeout is transient", _classify(APITimeoutError("Request timed out.")), "transient")
    check("400 bad request is fatal", _classify(StatusError(400, "model_not_found")), "fatal")
    check("401 bad key is fatal", _classify(StatusError(401, "Invalid API Key")), "fatal")
    gemini_style = Exception(GEMINI_DAILY)
    gemini_style.code = 429  # google-genai puts the status on .code
    check("status read from .code as well", _classify(gemini_style), "daily")


def test_wait_hints() -> None:
    print("wait hints")
    check("17m10.32s", round(_duration_seconds("17m10.32s"), 2), 1030.32)
    check("667.5ms", _duration_seconds("667.5ms"), 0.6675)
    check("1h2m3s", _duration_seconds("1h2m3s"), 3723.0)
    check("from Groq's message", round(_wait_hint_seconds(StatusError(429, GROQ_DAILY))), 1030)
    check("from Gemini's message", _wait_hint_seconds(StatusError(429, GEMINI_DAILY)), 26.58)
    check("Retry-After header wins", _wait_hint_seconds(StatusError(429, GROQ_PER_MINUTE, {"retry-after": "7"})), 7.0)
    check("no hint at all", _wait_hint_seconds(StatusError(503, GEMINI_BUSY)), None)


def test_retry_policy() -> None:
    print("retry policy")
    script = Script(StatusError(429, GROQ_DAILY))
    (result, waits) = with_recorded_sleep(lambda: outcome(lambda: _call_with_retry(script, "p", "k", "m", "groq")))
    check("daily quota: stops after one call", (result[0], script.calls, waits), ("JudgeQuotaExhaustedError", 1, []))
    check("and says how long the provider asked for", "retry in 17 min" in result[1], True)

    script = Script(StatusError(429, GROQ_PER_MINUTE), APIConnectionError("Connection error."))
    (result, waits) = with_recorded_sleep(lambda: _call_with_retry(script, "p", "k", "m", "groq"))
    check("per-minute limit then a dropped connection: retried through to a result",
          (result, script.calls), ('{"verdict": "clean"}', 3))
    check("backoff, not the 0.7s hint, when backoff is longer", waits, [2.0, 4.0])

    script = Script(StatusError(503, "busy", {"retry-after": "30"}))
    (_, waits) = with_recorded_sleep(lambda: _call_with_retry(script, "p", "k", "m", "gemini"))
    check("a longer Retry-After is honoured", waits, [30.0])

    script = Script(StatusError(429, "Rate limit reached. Please try again in 5m0s."))
    (result, waits) = with_recorded_sleep(lambda: outcome(lambda: _call_with_retry(script, "p", "k", "m", "groq")))
    check("a wait over the cap fails instead of sleeping", (result[0], waits), ("JudgeUnavailableError", []))

    script = Script(StatusError(400, "model_not_found: openai/gpt-oss-999b"))
    (result, waits) = with_recorded_sleep(lambda: outcome(lambda: _call_with_retry(script, "p", "k", "m", "groq")))
    check("fatal: one call, no sleep", (result[0], script.calls, waits), ("JudgeUnavailableError", 1, []))
    check("and the reason reaches the error (it used to be 'failed after retries')",
          "model_not_found" in result[1], True)

    script = Script(*[StatusError(503, GEMINI_BUSY)] * 5)
    (result, waits) = with_recorded_sleep(lambda: outcome(lambda: _call_with_retry(script, "p", "k", "m", "gemini")))
    check("transient every time: gives up after max attempts", (result[0], script.calls, waits),
          ("JudgeUnavailableError", 4, [2.0, 4.0, 8.0]))


def test_benchmark_stops_on_daily_quota() -> None:
    """Before: the cap hit mid-run and every remaining run became an error."""
    print("benchmark")
    calls = {"n": 0}

    def fake_judge(trace, provider=None, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"verdict": "clean", "_provider": "groq", "_model": "m"}
        raise JudgeQuotaExhaustedError("groq daily quota exhausted (provider says retry in 17 min)")

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "report.json")
        saved = (run_benchmark.run_judge_raw, sys.argv, sys.stdout)
        run_benchmark.run_judge_raw = fake_judge
        sys.argv = ["run_benchmark.py", "3", "--output", out,
                    "--labels", os.path.join(HERE, "sample_traces", "ground_truth.json")]
        sys.stdout = open(os.devnull, "w")
        try:
            run_benchmark.main()
        finally:
            sys.stdout.close()
            run_benchmark.run_judge_raw, sys.argv, sys.stdout = saved
        report = json.load(open(out, encoding="utf-8"))
    check("stopped at the second call", calls["n"], 2)
    check("the report says why", "daily quota" in (report["stopped_early"] or ""), True)
    check("one completed, one errored", (report["total_completed_runs"], report["total_errored_runs"]), (1, 1))
    check("the other 28 are not attempted, not errors", report["runs_not_attempted"], 28)


def test_configurable_timeout() -> None:
    """The real ~15k-token trace outgrew the 180 s default on Gemini."""
    print("timeout")
    saved = os.environ.get("AGENTLENS_JUDGE_TIMEOUT_S")

    def from_env(value):
        if value is None:
            os.environ.pop("AGENTLENS_JUDGE_TIMEOUT_S", None)
        else:
            os.environ["AGENTLENS_JUDGE_TIMEOUT_S"] = value
        try:
            return judge_mod._timeout_from_env()
        except ValueError:
            return "ValueError"

    try:
        check("default when unset", from_env(None), 180.0)
        check("set in seconds", from_env("420"), 420.0)
        check("blank means default", from_env("  "), 180.0)
        check("not a number is an error, not a silent default", from_env("7min"), "ValueError")
        check("zero is an error", from_env("0"), "ValueError")
    finally:
        if saved is None:
            os.environ.pop("AGENTLENS_JUDGE_TIMEOUT_S", None)
        else:
            os.environ["AGENTLENS_JUDGE_TIMEOUT_S"] = saved

    captured = {}

    class _Groq:
        def __init__(self, api_key=None, **options):
            captured.update(options)
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(
                create=lambda **kw: types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="{}"))])))

    stub = types.ModuleType("groq")
    stub.Groq = _Groq
    saved_module, saved_timeout = sys.modules.get("groq"), judge_mod.JUDGE_TIMEOUT_S
    sys.modules["groq"] = stub
    judge_mod.JUDGE_TIMEOUT_S = 420.0  # what analyze.py --timeout 420 does
    try:
        judge_mod._call_groq("p", "k", "m")
    finally:
        judge_mod.JUDGE_TIMEOUT_S = saved_timeout
        if saved_module is None:
            sys.modules.pop("groq", None)
        else:
            sys.modules["groq"] = saved_module
    check("a changed timeout reaches the client at call time", captured.get("timeout"), 420.0)


def main() -> None:
    for test in (test_classification, test_wait_hints, test_retry_policy, test_benchmark_stops_on_daily_quota,
                 test_configurable_timeout):
        test()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
