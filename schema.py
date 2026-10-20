"""Trace schema for AgentLens.

A trace is a task description plus an ordered list of steps. Each step is
one of: reasoning, tool_call, tool_result, final_answer. This shape stays
close to OpenTelemetry's GenAI semantic conventions on purpose, so logs
from LangChain, a custom agent loop, etc. can be adapted into it later
without a rewrite.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

StepType = Literal["reasoning", "tool_call", "tool_result", "final_answer"]
Confidence = Literal["high", "medium", "low"]
STEP_TYPES = ("reasoning", "tool_call", "tool_result", "final_answer")

# Bounds on what a trace may be. Traces are untrusted input, and each of these
# failed badly before: a step type the judge has no branch for was silently
# left out of the prompt, duplicate indices made "critical step 3" ambiguous,
# and deep nesting surfaced as a bare RecursionError from the JSON parser.
# Generous for real runs: across every trace in this repo, the largest is 17
# steps and ~64 KB, with tool args/results nested at most 3 deep.
MAX_TRACE_BYTES = 20 * 1024 * 1024
MAX_STEPS = 2000
MAX_DEPTH = 64


class TraceFormatError(ValueError):
    """The input isn't a trace AgentLens can diagnose faithfully. Raised
    instead of a KeyError deep in the pipeline -- or, worse, a quietly
    partial diagnosis of a trajectory with steps missing from it."""


def _check_depth(value: Any, where: str) -> None:
    """Depth counts nested objects/arrays ({"a": 1} is 1 deep; a bare value 0).
    Iterative, so a hostile document can't exhaust the stack here either."""
    stack = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if not isinstance(node, (dict, list)):
            continue
        if depth > MAX_DEPTH:
            raise TraceFormatError(f"{where} is nested more than {MAX_DEPTH} levels deep")
        children = node.values() if isinstance(node, dict) else node
        stack.extend((v, depth + 1) for v in children)


@dataclass
class Step:
    index: int
    type: StepType
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Optional[Any] = None
    is_error: bool = False

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Step":
        if not isinstance(d, dict):
            raise TraceFormatError(f"a step is {type(d).__name__}, not an object")
        if "index" not in d or "type" not in d:
            raise TraceFormatError(f"a step is missing 'index' or 'type' (keys: {', '.join(d) or 'none'})")
        return Step(
            index=d["index"],
            type=d["type"],
            # null content is common in exported logs; it means "no text", not an error
            content=d.get("content") or "",
            tool_name=d.get("tool_name"),
            tool_args=d.get("tool_args"),
            tool_result=d.get("tool_result"),
            is_error=d.get("is_error", False),
        )


@dataclass
class Trace:
    task: str
    steps: list[Step] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Trace":
        if not isinstance(d, dict):
            raise TraceFormatError(f"a trace is a JSON object, got {type(d).__name__}")
        if not isinstance(d.get("task"), str):
            raise TraceFormatError("trace has no 'task' string")
        steps = d.get("steps", [])
        if not isinstance(steps, list):
            raise TraceFormatError(f"'steps' is {type(steps).__name__}, not a list")
        if len(steps) > MAX_STEPS:
            raise TraceFormatError(f"trace has {len(steps)} steps; the limit is {MAX_STEPS}")
        trace = Trace(task=d["task"], steps=[Step.from_dict(s) for s in steps])
        trace.validate()
        return trace

    @staticmethod
    def load(path: str) -> "Trace":
        size = os.path.getsize(path)
        if size > MAX_TRACE_BYTES:
            raise TraceFormatError(f"{path} is {size:,} bytes; the limit is {MAX_TRACE_BYTES:,}")
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise TraceFormatError(f"{path} is not valid JSON: {e}") from e
            except RecursionError as e:
                raise TraceFormatError(f"{path} is nested too deeply to parse") from e
        return Trace.from_dict(data)

    def validate(self) -> None:
        """Check what the rules, the judge and the viewer all rely on.

        Called by from_dict, and by analyze.diagnose for traces the adapters
        build directly, so every route into a diagnosis is checked once.
        """
        if len(self.steps) > MAX_STEPS:
            raise TraceFormatError(f"trace has {len(self.steps)} steps; the limit is {MAX_STEPS}")
        previous = None
        for position, step in enumerate(self.steps):
            where = f"step at position {position}"
            if not isinstance(step.index, int) or isinstance(step.index, bool):
                raise TraceFormatError(f"{where} has index {step.index!r}, not an integer")
            where = f"step {step.index}"
            if previous is not None and step.index <= previous:
                # Gaps are fine; going backwards or repeating makes "critical
                # step N" ambiguous and the timeline order a lie.
                raise TraceFormatError(f"{where} follows step {previous}: indices must strictly increase")
            previous = step.index
            if step.type not in STEP_TYPES:
                raise TraceFormatError(f"{where} has type {step.type!r}; expected one of {', '.join(STEP_TYPES)}")
            if not isinstance(step.content, str):
                raise TraceFormatError(f"{where} content is {type(step.content).__name__}, not text")
            if step.tool_name is not None and not isinstance(step.tool_name, str):
                raise TraceFormatError(f"{where} tool_name is {type(step.tool_name).__name__}, not text")
            if step.tool_args is not None and not isinstance(step.tool_args, dict):
                raise TraceFormatError(f"{where} tool_args is {type(step.tool_args).__name__}, not an object")
            if not isinstance(step.is_error, bool):
                raise TraceFormatError(f"{where} is_error is {step.is_error!r}, not true/false")
            _check_depth(step.tool_args, f"{where} tool_args")
            _check_depth(step.tool_result, f"{where} tool_result")


@dataclass
class Finding:
    """A single diagnostic finding pointing at a step in the trace."""

    source: Literal["rule", "judge"]
    step_index: int
    category: str
    confidence: Confidence
    summary: str
    detail: str = ""


# Root cause matters more than the downstream symptom, so findings are
# ranked by confidence first and earliest step second.
_CONFIDENCE_RANK: dict[str, int] = {"high": 0, "medium": 1, "low": 2}


def rank_key(f: Finding) -> tuple[int, int]:
    return (_CONFIDENCE_RANK.get(f.confidence, 3), f.step_index)


@dataclass
class Diagnosis:
    task: str
    trace_length: int
    steps: list[dict[str, Any]]
    findings: list[Finding]
    critical_step: Optional[int]
    critical_finding: Optional[Finding]
    # The judge's PASS 1 output: every concrete claim it pulled out of the
    # trace, tagged by step ({"step": int, "claim": str}). Carried through to
    # the diagnosis so the verdict can be audited against the evidence it was
    # drawn from, rather than taken on the strength of one summary sentence.
    # Empty when the judge was skipped or no key was set.
    claims: list[dict[str, Any]] = field(default_factory=list)
    # Whether the judge actually read this trace: {"status": "ran" | "off" |
    # "not_configured" | "failed", "provider", "model", "error"}. Without it,
    # a rules-only diagnosis with no findings is indistinguishable from a
    # judge that checked the trace and found it clean.
    judge: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "trace_length": self.trace_length,
            "steps": self.steps,
            "findings": [vars(f) for f in self.findings],
            "critical_step": self.critical_step,
            "critical_finding": vars(self.critical_finding) if self.critical_finding else None,
            "claims": self.claims,
            "judge": self.judge,
        }
