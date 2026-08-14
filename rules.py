"""Deterministic, rule-based checks for mechanical agent failures.

These run first: no LLM call, no cost, high precision. They catch the
failures that don't require understanding meaning — a tool hammered with
the same arguments, an error the agent barrelled past, a call missing the
arguments it needed. Anything semantic (a stale fact quietly corrupting
later reasoning) is out of scope here; that's judge.py's job.
"""

from __future__ import annotations

from collections import defaultdict

from schema import Finding, Trace

LOOP_THRESHOLD = 3
_ERROR_ACK_WORDS = ("error", "fail", "unable", "retry", "couldn't", "could not", "denied")


def _args_key(tool_name: str, tool_args: dict | None) -> tuple:
    args = tool_args or {}
    return (tool_name, tuple(sorted(args.items())))


def find_tool_loops(trace: Trace) -> list[Finding]:
    """Flag a tool called with identical arguments LOOP_THRESHOLD+ times."""
    seen: dict[tuple, list[int]] = defaultdict(list)
    for step in trace.steps:
        if step.type != "tool_call" or not step.tool_name:
            continue
        seen[_args_key(step.tool_name, step.tool_args)].append(step.index)

    findings = []
    for (tool_name, _args), indices in seen.items():
        if len(indices) >= LOOP_THRESHOLD:
            findings.append(
                Finding(
                    source="rule",
                    step_index=indices[LOOP_THRESHOLD - 1],
                    category="tool_call_loop",
                    confidence="high",
                    summary=f"'{tool_name}' called with identical arguments {len(indices)} times",
                    detail=f"Repeated at steps {indices}",
                )
            )
    return findings


def find_silent_errors(trace: Trace) -> list[Finding]:
    """Flag a tool error the agent never acknowledges in the steps right after it."""
    findings = []
    for i, step in enumerate(trace.steps):
        if step.type != "tool_result" or not step.is_error:
            continue
        acknowledged = False
        for later in trace.steps[i + 1 : i + 3]:
            if later.type in ("reasoning", "final_answer"):
                text = (later.content or "").lower()
                acknowledged = any(w in text for w in _ERROR_ACK_WORDS)
                break
        if not acknowledged:
            findings.append(
                Finding(
                    source="rule",
                    step_index=step.index,
                    category="silent_tool_error",
                    confidence="high",
                    summary=f"Tool error at step {step.index} was not acknowledged afterward",
                    detail=str(step.tool_result),
                )
            )
    return findings


def find_missing_arguments(trace: Trace) -> list[Finding]:
    """Flag tool calls made with no arguments at all — usually a malformed call."""
    findings = []
    for step in trace.steps:
        if step.type == "tool_call" and step.tool_name and not step.tool_args:
            findings.append(
                Finding(
                    source="rule",
                    step_index=step.index,
                    category="missing_tool_arguments",
                    confidence="medium",
                    summary=f"'{step.tool_name}' called with no arguments",
                    detail="tool_args was empty or missing",
                )
            )
    return findings


def run_rules(trace: Trace) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(find_tool_loops(trace))
    findings.extend(find_silent_errors(trace))
    findings.extend(find_missing_arguments(trace))
    return sorted(findings, key=lambda f: f.step_index)
