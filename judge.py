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
(a cache lookup, a tool result, an assumption, or a claim made in prose)
that was stale or wrong, and then treated as true for the rest of the
trajectory. Every step after it may look locally coherent -- the agent is
reasoning validly, just from a wrong premise.

Contradictions are not limited to structured data. A contradiction is just
as real between two claims written in free-form prose -- one step's
paragraph asserting a fact, a later step's paragraph (a draft, a review, a
report) asserting the opposite -- as it is between two tool_result fields.
A single contradicting sentence can be buried inside thousands of words of
otherwise-correct reasoning or report text; don't let the surrounding
length hide it.

You MUST do this in three passes, and the first pass is a required part of
your output, not just internal reasoning you can skip:

PASS 1 -- EXTRACT. Go through every step in order. For each one, pull out
every concrete, checkable claim it makes -- a date, a name, a number, a
named event, a verdict -- whether that claim sits in a tool_result field
or is written into a sentence of prose. List each one in the "claims"
array below, tagged with the step it came from. Do this even for steps
that seem purely narrative (a draft report, a review) -- those are exactly
where a prose-buried contradiction hides. An empty or near-empty claims
list on a long trace is a sign you skipped this pass; do not skip it.

PASS 2 -- COMPARE. Only after PASS 1 is complete, compare every pair of
claims that refer to the same real-world thing (the same event, the same
entity, the same fact) and check whether they agree.

PASS 3 -- CHECK SUPPORT. A trajectory can be wrong without ever
contradicting itself. Go back over the claims a second time and ask two
further questions:

  (a) UNSUPPORTED COMMITMENT. Does any claim settle on one option where an
      earlier claim established that several were possible, with nothing
      in between justifying that particular choice?

      Before flagging this, apply one test: ARE THE OPTIONS
      INTERCHANGEABLE FOR THIS TASK? Picking an arbitrary one of several
      equally valid ways to satisfy the request is what a competent agent
      does; it is not a failure and must not be flagged.

        - A warehouse reports four identical units of a part in stock and
          the task is to ship one: interchangeable. Any unit completes the
          task correctly. NOT a failure.
        - A search returns two unrelated invoices that happen to carry the
          same reference number, and the task is to void that invoice: NOT
          interchangeable. They belong to different accounts, only one is
          the right one, and picking either without checking risks voiding
          the wrong one.

      The question is not "did the agent explain its choice" -- agents
      rarely do. It is "could this choice be wrong in a way that matters?"
      If every option would satisfy the task equally, there is no failure.
      Flag it only when the options are materially different and the
      agent had no basis for choosing between them.

  (b) DROPPED CAVEAT. Does any claim restate a value while leaving behind
      a qualifier attached to it at its source? A tool result that carries
      a note, warning, default-applied flag, or partial-result marker is
      making two statements, not one. If a later step repeats the headline
      number as if the qualifier were not there, the qualifier was lost.
      The conflict here sits inside a single step's payload, so PASS 2
      will not see it.

Be strict about what counts. An agent choosing between genuinely
equivalent options, or taking a normal next step that nothing casts doubt
on, is not a failure -- it is an agent doing its job. Only flag (a) when
the ambiguity was explicitly surfaced and left unresolved, and (b) when
the source actually carried a qualifier that the later claim dropped. If
you are reaching for a reason, there is no failure: say so.

has_failure is true if PASS 2 or PASS 3 finds something.

Respond with a single JSON object and nothing else:
{
  "claims": [{"step": int, "claim": string}],  // PASS 1 output -- required, fill this in first
  "has_failure": bool,
  "critical_step": int | null,   // where the wrong premise was INTRODUCED, not where it became visible
  "category": string | null,     // e.g. "stale_fact", "contradicted_assumption", "misread_tool_result",
                                 // "unsupported_commitment", "dropped_caveat", "other"
  "confidence": "high" | "medium" | "low",
  "summary": string,             // one sentence: what went wrong and why this step is the root cause
  "contradictions": [string]     // the conflicting facts, or for PASS 3 the commitment and what was
                                 // left unresolved / the value and the qualifier dropped from it
}

If PASS 2 and PASS 3 both find nothing, set has_failure to false and leave
critical_step/category/contradictions null / empty -- but "claims" is
still required either way."""


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


def _call_anthropic(
    user_prompt: str, api_key: str, model: str, system_prompt: str = JUDGE_SYSTEM_PROMPT, max_tokens: int = 1024
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def _call_gemini(user_prompt: str, api_key: str, model: str, system_prompt: str = JUDGE_SYSTEM_PROMPT) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    return response.text


def _call_groq(
    user_prompt: str, api_key: str, model: str, system_prompt: str = JUDGE_SYSTEM_PROMPT, max_tokens: int = 2048
) -> str:
    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        # The judge's contract is a single JSON object. Groq can enforce that
        # server-side, which removes the fenced-markdown case _parse_verdict
        # otherwise has to strip off.
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


# provider -> (env var to look for, model default, call function)
#
# Order matters: this is also the auto-detect precedence when no --provider
# is given. Gemini's free tier allows 20 requests/day, which is the reason
# this project's benchmark had to be run in fragments across many sessions;
# Groq's free tier is orders of magnitude larger, so it goes ahead of Gemini
# as the default free option.
_PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "claude-sonnet-5", _call_anthropic),
    "groq": ("GROQ_API_KEY", "openai/gpt-oss-120b", _call_groq),
    "gemini": ("GOOGLE_API_KEY", "gemini-flash-latest", _call_gemini),
}

# Public: every CLI derives its --provider choices from this, so adding a
# provider to _PROVIDERS is all it takes. The previous hardcoded lists
# silently went stale the moment groq was registered.
PROVIDER_NAMES = tuple(_PROVIDERS)


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


def run_judge_raw(trace: Trace, provider: str | None = None, model: str | None = None) -> dict | None:
    """Calls an LLM once with the full trajectory and returns the raw parsed
    verdict dict (including the "claims" extraction-pass field), or None if
    no API key is configured at all.

    Exists mainly for debugging the judge itself -- e.g. checking whether it
    actually performed the claim-extraction pass, not just whether its final
    has_failure verdict was right. run_judge() below is what normal callers
    should use; it wraps this and returns Findings.

    Raises JudgeUnavailableError if a key IS set but the call failed or the
    response couldn't be parsed.
    """
    if provider:
        providers_to_try = [provider]
    else:
        providers_to_try = [p for p, (env_var, _, _) in _PROVIDERS.items() if os.environ.get(env_var)]

    if not providers_to_try:
        return None

    chosen = providers_to_try[0]
    env_var, default_model, call_fn = _PROVIDERS[chosen]
    api_key = os.environ.get(env_var)
    if not api_key:
        return None

    chosen_model = model or default_model
    response_text = _call_with_retry(call_fn, _format_trace(trace), api_key, chosen_model, chosen)
    if response_text is None:
        raise JudgeUnavailableError(f"{chosen} judge call failed after retries")

    try:
        verdict = _parse_verdict(response_text)
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        raise JudgeUnavailableError(f"{chosen} judge returned an unparseable response: {e}") from e

    # Stamp who answered. With more than one provider available, "89% correct"
    # is meaningless unless you know which judge produced it -- and since
    # auto-detect picks whatever key happens to be set, the provider can change
    # between sessions without anyone choosing to change it. Recorded on the
    # verdict so every downstream consumer gets it for free.
    verdict["_provider"] = chosen
    verdict["_model"] = chosen_model
    return verdict


def _verdict_to_findings(verdict: dict) -> list[Finding]:
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


def run_judge_with_claims(
    trace: Trace, provider: str | None = None, model: str | None = None
) -> tuple[list[Finding], list[dict]]:
    """Same call as run_judge, but also returns the PASS 1 claims.

    The claims array is the evidence the extraction pass actually ran --
    it's what turned the real-trace miss into a catch, and a near-empty
    list on a long trace is the tell that the judge skipped extraction and
    answered from a single read. Flattening a verdict into one Finding
    throws that away, so callers that want to show their work use this.

    Claims come back even when has_failure is false: on a clean trace they
    are the difference between "the judge checked and found nothing" and
    "the judge returned nothing," which is the whole point of the
    false-positive tests.
    """
    verdict = run_judge_raw(trace, provider=provider, model=model)
    if verdict is None:
        return [], []
    return _verdict_to_findings(verdict), list(verdict.get("claims") or [])


def run_judge(trace: Trace, provider: str | None = None, model: str | None = None) -> list[Finding]:
    """Calls an LLM once with the full trajectory to check for state corruption.

    `provider` picks which API to use ("anthropic" or "gemini"); if omitted,
    picks whichever of ANTHROPIC_API_KEY / GOOGLE_API_KEY is set (Anthropic
    first). Returns [] if no key is set at all -- that's a deliberate
    "judge disabled" state, not a failure. Raises JudgeUnavailableError if a
    key IS set but the call failed or the response couldn't be parsed, so
    callers don't mistake "the API broke" for "the judge found nothing."
    """
    return run_judge_with_claims(trace, provider=provider, model=model)[0]
