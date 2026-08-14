"""LLM judge for the failure mode rules can't see: state corruption.

A fact gets established early (a cache lookup, a tool result) and is stale
or wrong. Every step after it looks locally coherent in isolation -- the
agent is reasoning validly, just from a wrong premise -- so a step-by-step
check would never notice. The judge instead reads the entire trajectory in
one pass, specifically prompted to compare early facts against later ones.
"""

from __future__ import annotations

import json
import os

from schema import Finding, Trace


class JudgeUnavailableError(Exception):
    """Raised when the judge API call itself failed or returned something
    unparseable -- distinct from the judge running successfully and finding
    no failure. Callers that score judge output (benchmarks, consistency
    checks) need to tell these apart: silently treating "the call never
    happened" the same as "the judge looked and found nothing" corrupts the
    exact kind of signal this whole tool exists to protect."""

JUDGE_SYSTEM_PROMPT = """You are diagnosing why an AI agent's run went wrong.

You will be given a task and the full sequence of steps the agent took
(reasoning, tool calls, tool results, and its final answer). Read the whole
trajectory before judging anything.

Your specific job is to catch state corruption: a fact established early
(a cache lookup, a tool result, an assumption) that was stale or wrong, and
then treated as true for the rest of the trajectory. Every step after it
may look locally coherent -- the agent is reasoning validly, just from a
wrong premise. Compare early facts against later facts and against the
final answer, and flag any contradiction.

Respond with a single JSON object and nothing else:
{
  "has_failure": bool,
  "critical_step": int | null,   // where the wrong premise was INTRODUCED, not where it became visible
  "category": string | null,     // e.g. "stale_fact", "contradicted_assumption", "misread_tool_result", "other"
  "confidence": "high" | "medium" | "low",
  "summary": string,             // one sentence: what went wrong and why this step is the root cause
  "contradictions": [string]     // the early fact vs. the later fact that conflicts with it
}

If you find no state corruption, set has_failure to false and leave the
other fields null / empty."""


def _format_trace(trace: Trace) -> str:
    lines = [f"TASK: {trace.task}", ""]
    for step in trace.steps:
        if step.type == "tool_call":
            lines.append(f"[{step.index}] TOOL_CALL {step.tool_name}({json.dumps(step.tool_args)})")
        elif step.type == "tool_result":
            marker = "ERROR" if step.is_error else "RESULT"
            lines.append(f"[{step.index}] TOOL_{marker}: {json.dumps(step.tool_result)}")
        elif step.type == "reasoning":
            lines.append(f"[{step.index}] REASONING: {step.content}")
        elif step.type == "final_answer":
            lines.append(f"[{step.index}] FINAL_ANSWER: {step.content}")
    return "\n".join(lines)


def _parse_verdict(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            text = rest if first_line.strip().lower() in ("json", "") else text
    return json.loads(text)


def _call_anthropic(user_prompt: str, api_key: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def _call_gemini(user_prompt: str, api_key: str, model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=JUDGE_SYSTEM_PROMPT),
    )
    return response.text


# provider -> (env var to look for, model default, call function)
_PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "claude-sonnet-5", _call_anthropic),
    "gemini": ("GOOGLE_API_KEY", "gemini-flash-latest", _call_gemini),
}

_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 529)


def _call_with_retry(call_fn, user_prompt: str, api_key: str, model: str, provider: str, max_attempts: int = 4) -> str | None:
    """Retries transient server-side errors (rate limits, overload) with
    backoff. Returns None (not raises) if every attempt fails, so a flaky
    API call doesn't take down a whole benchmark run over one bad request."""
    import time

    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return call_fn(user_prompt, api_key, model)
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            is_retryable = status in _RETRYABLE_STATUS_CODES
            if not is_retryable or attempt == max_attempts:
                print(f"judge ({provider}): giving up after {attempt} attempt(s) ({e})")
                return None
            print(f"judge ({provider}): attempt {attempt} failed ({e}), retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    return None


def run_judge(trace: Trace, provider: str | None = None, model: str | None = None) -> list[Finding]:
    """Calls an LLM once with the full trajectory to check for state corruption.

    `provider` picks which API to use ("anthropic" or "gemini"); if omitted,
    picks whichever of ANTHROPIC_API_KEY / GOOGLE_API_KEY is set (Anthropic
    first). Returns [] if no key is set at all -- that's a deliberate
    "judge disabled" state, not a failure. Raises JudgeUnavailableError if a
    key IS set but the call failed or the response couldn't be parsed, so
    callers don't mistake "the API broke" for "the judge found nothing."
    """
    if provider:
        providers_to_try = [provider]
    else:
        providers_to_try = [p for p, (env_var, _, _) in _PROVIDERS.items() if os.environ.get(env_var)]

    if not providers_to_try:
        return []

    chosen = providers_to_try[0]
    env_var, default_model, call_fn = _PROVIDERS[chosen]
    api_key = os.environ.get(env_var)
    if not api_key:
        return []

    response_text = _call_with_retry(call_fn, _format_trace(trace), api_key, model or default_model, chosen)
    if response_text is None:
        raise JudgeUnavailableError(f"{chosen} judge call failed after retries")

    try:
        verdict = _parse_verdict(response_text)
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        raise JudgeUnavailableError(f"{chosen} judge returned an unparseable response: {e}") from e

    if not verdict.get("has_failure"):
        return []

    return [
        Finding(
            source="judge",
            step_index=verdict["critical_step"],
            category=verdict.get("category") or "state_corruption",
            confidence=verdict.get("confidence", "medium"),
            summary=verdict.get("summary", ""),
            detail="; ".join(verdict.get("contradictions", [])),
        )
    ]
