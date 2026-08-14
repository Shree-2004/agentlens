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
    "gemini": ("GOOGLE_API_KEY", "gemini-2.5-flash", _call_gemini),
}


def run_judge(trace: Trace, provider: str | None = None, model: str | None = None) -> list[Finding]:
    """Calls an LLM once with the full trajectory to check for state corruption.

    `provider` picks which API to use ("anthropic" or "gemini"); if omitted,
    picks whichever of ANTHROPIC_API_KEY / GOOGLE_API_KEY is set (Anthropic
    first). Returns [] if no key is set, the judge finds nothing, or its
    response can't be parsed -- this layer is additive, never fatal.
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

    response_text = call_fn(_format_trace(trace), api_key, model or default_model)

    try:
        verdict = _parse_verdict(response_text)
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"judge ({chosen}): could not parse verdict ({e}), skipping judge findings")
        return []

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
