"""CLI entry point: run rules + judge over a trace, merge the findings,
and write a diagnosis JSON the viewer can render directly.

Usage:
    python3 analyze.py sample_traces/stale_plan_tier.json output/diagnosis.json
    python3 analyze.py sample_traces/tool_loop.json output/diagnosis.json --no-judge
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from judge import JudgeUnavailableError, run_judge
from rules import run_rules
from schema import Diagnosis, Finding, Trace, rank_key


def _step_summary(step) -> dict:
    if step.type == "tool_call":
        text = f"{step.tool_name}({json.dumps(step.tool_args)})"
    elif step.type == "tool_result":
        prefix = "ERROR: " if step.is_error else ""
        text = prefix + json.dumps(step.tool_result)
    else:
        text = step.content
    return {"index": step.index, "type": step.type, "text": text}


def merge_findings(findings: list[Finding]) -> Finding | None:
    """Prefer the earliest high-confidence finding: the upstream root cause
    matters more than a downstream symptom of the same failure."""
    if not findings:
        return None
    return sorted(findings, key=rank_key)[0]


def diagnose(trace: Trace, use_judge: bool = True, provider: str | None = None) -> Diagnosis:
    findings = run_rules(trace)
    if use_judge:
        try:
            findings.extend(run_judge(trace, provider=provider))
        except JudgeUnavailableError as e:
            print(f"judge unavailable, continuing with rules-only findings: {e}")
    findings.sort(key=lambda f: f.step_index)

    critical = merge_findings(findings)
    return Diagnosis(
        task=trace.task,
        trace_length=len(trace.steps),
        steps=[_step_summary(s) for s in trace.steps],
        findings=findings,
        critical_step=critical.step_index if critical else None,
        critical_finding=critical,
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Diagnose an agent trace")
    parser.add_argument("trace_path")
    parser.add_argument("output_path")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge, rules only")
    parser.add_argument("--provider", choices=["anthropic", "gemini"], help="which judge LLM to use (default: auto-detect from env keys)")
    args = parser.parse_args()

    trace = Trace.load(args.trace_path)
    diagnosis = diagnose(trace, use_judge=not args.no_judge, provider=args.provider)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(diagnosis.to_dict(), f, indent=2)

    if diagnosis.critical_finding:
        cf = diagnosis.critical_finding
        print(f"Critical step: {cf.step_index} ({cf.category}, {cf.confidence} confidence, via {cf.source})")
        print(cf.summary)
    else:
        print("No failure found.")
    print(f"Wrote diagnosis to {args.output_path}")


if __name__ == "__main__":
    main()
