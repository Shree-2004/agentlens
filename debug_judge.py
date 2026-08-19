"""Inspect the judge's full raw verdict on a trace, including the
"claims" extraction-pass field -- not just the final Finding.

Costs exactly one API call. Meant for diagnosing *why* the judge got
something wrong, not for scoring -- use run_benchmark.py/check_consistency.py
for that.

Usage:
    python3 debug_judge.py real_traces/multi_agent_research_live.json
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from judge import JudgeUnavailableError, run_judge_raw
from schema import Trace


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Inspect the judge's raw verdict, including extracted claims")
    parser.add_argument("trace_path")
    parser.add_argument("--provider", choices=["anthropic", "gemini"])
    args = parser.parse_args()

    trace = Trace.load(args.trace_path)

    try:
        verdict = run_judge_raw(trace, provider=args.provider)
    except JudgeUnavailableError as e:
        print(f"Judge call failed: {e}")
        return

    if verdict is None:
        print("No API key configured -- nothing to inspect.")
        return

    claims = verdict.get("claims", [])
    print(f"has_failure: {verdict.get('has_failure')}")
    print(f"critical_step: {verdict.get('critical_step')}")
    print(f"category: {verdict.get('category')}")
    print(f"summary: {verdict.get('summary')}")
    print()
    print(f"Extracted {len(claims)} claims:")
    for c in claims:
        print(f"  [step {c.get('step')}] {c.get('claim')}")

    print()
    print("Full raw verdict:")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
