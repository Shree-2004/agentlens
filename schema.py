"""Trace schema for AgentLens.

A trace is a task description plus an ordered list of steps. Each step is
one of: reasoning, tool_call, tool_result, final_answer. This shape stays
close to OpenTelemetry's GenAI semantic conventions on purpose, so logs
from LangChain, a custom agent loop, etc. can be adapted into it later
without a rewrite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

StepType = Literal["reasoning", "tool_call", "tool_result", "final_answer"]
Confidence = Literal["high", "medium", "low"]


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
        return Step(
            index=d["index"],
            type=d["type"],
            content=d.get("content", ""),
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
        return Trace(task=d["task"], steps=[Step.from_dict(s) for s in d.get("steps", [])])

    @staticmethod
    def load(path: str) -> "Trace":
        with open(path, "r", encoding="utf-8") as f:
            return Trace.from_dict(json.load(f))


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "trace_length": self.trace_length,
            "steps": self.steps,
            "findings": [vars(f) for f in self.findings],
            "critical_step": self.critical_step,
            "critical_finding": vars(self.critical_finding) if self.critical_finding else None,
        }
