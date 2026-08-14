"""Run the LLM judge N times on the same trace and check whether it lands
on the same critical step every time.

This is the check Ruchit D. suggested in a LinkedIn comment on this
project: a single convincing-sounding diagnosis proves nothing on its own
-- if the judge can't reproduce its own answer across repeated runs on an
unchanged trace, the diagnosis isn't trustworthy no matter how well-argued
it reads.

Usage:
    python3 check_consistency.py sample_traces/stale_plan_tier.json 5
    python3 check_consistency.py sample_traces/stale_plan_tier.json 5 --provider gemini
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from dotenv import load_dotenv

from judge import JudgeUnavailableError, run_judge
from schema import Trace


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Check judge consistency across repeated runs")
    parser.add_argument("trace_path")
    parser.add_argument("n_runs", type=int)
    parser.add_argument("--provider", choices=["anthropic", "gemini"], help="which judge LLM to use")
    parser.add_argument("--output", default=None, help="where to write the results JSON")
    args = parser.parse_args()

    trace = Trace.load(args.trace_path)
    runs = []

    for i in range(1, args.n_runs + 1):
        try:
            findings = run_judge(trace, provider=args.provider)
        except JudgeUnavailableError as e:
            result = {"run": i, "critical_step": None, "category": None, "confidence": None, "summary": None, "error": str(e)}
            runs.append(result)
            print(f"run {i}: ERROR ({e})")
            continue

        if findings:
            f = findings[0]
            result = {
                "run": i,
                "critical_step": f.step_index,
                "category": f.category,
                "confidence": f.confidence,
                "summary": f.summary,
                "error": None,
            }
        else:
            result = {"run": i, "critical_step": None, "category": None, "confidence": None, "summary": None, "error": None}
        runs.append(result)
        print(f"run {i}: step={result['critical_step']} category={result['category']} confidence={result['confidence']}")

    # errored runs never reached a real verdict -- excluded from agreement math,
    # not counted as "no failure found"
    completed_runs = [r for r in runs if r["error"] is None]
    error_count = len(runs) - len(completed_runs)

    steps = [r["critical_step"] for r in completed_runs]
    step_counts = Counter(steps)

    report = {
        "trace_path": args.trace_path,
        "n_runs": args.n_runs,
        "runs": runs,
        "error_count": error_count,
        "step_distribution": dict(step_counts),
    }

    if step_counts:
        majority_step, majority_count = step_counts.most_common(1)[0]
        report["majority_step"] = majority_step
        report["agreement_rate"] = majority_count / len(completed_runs)
        report["all_agree"] = len(step_counts) == 1
    else:
        report["majority_step"] = None
        report["agreement_rate"] = None
        report["all_agree"] = None

    output_path = args.output or "output/consistency_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    if error_count:
        print(f"{error_count}/{args.n_runs} runs errored (API failures, not judge verdicts) -- excluded from agreement stats.")
    if not completed_runs:
        print("No completed runs -- nothing to report.")
    elif report["all_agree"]:
        print(f"Consistent: all {len(completed_runs)} completed runs picked step {report['majority_step']}.")
    else:
        print(f"Inconsistent: step distribution across completed runs = {dict(step_counts)} "
              f"(agreement rate {report['agreement_rate']:.0%})")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
