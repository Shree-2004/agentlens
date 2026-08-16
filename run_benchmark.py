"""Score the LLM judge against the hand-labeled traces in
sample_traces/ground_truth.json.

For each labeled trace, runs the judge N times and checks, per run:
  - did it correctly detect failure vs. no-failure (false positive / false
    negative tracking on the clean traces)
  - when a failure was expected, did it land on the exact labeled step

Reports per-trace and aggregate accuracy, plus whether repeated runs on
the same trace agree with each other (consistency is a separate question
from correctness -- a judge can be consistently wrong).

Usage:
    python3 run_benchmark.py 3
    python3 run_benchmark.py 3 --provider gemini
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from judge import JudgeUnavailableError, run_judge
from schema import Trace

TRACES_DIR = Path("sample_traces")
GROUND_TRUTH_PATH = TRACES_DIR / "ground_truth.json"


def score_run(findings, label: dict) -> bool:
    predicted_has_failure = bool(findings)
    if predicted_has_failure != label["has_failure"]:
        return False
    if not label["has_failure"]:
        return True
    return findings[0].step_index == label["critical_step"]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Benchmark the judge against ground_truth.json")
    parser.add_argument("n_runs", type=int, nargs="?", default=3)
    parser.add_argument("--provider", choices=["anthropic", "gemini"])
    parser.add_argument("--output", default="output/benchmark_report.json")
    parser.add_argument("--only", help="comma-separated trace filenames to run, instead of the full set")
    args = parser.parse_args()

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))["labels"]
    if args.only:
        wanted = set(args.only.split(","))
        ground_truth = {k: v for k, v in ground_truth.items() if k in wanted}

    per_trace = []
    total_correct = 0
    total_completed = 0
    total_errored = 0
    false_positives = 0
    false_negatives = 0

    for filename, label in ground_truth.items():
        trace = Trace.load(str(TRACES_DIR / filename))
        run_results = []

        for i in range(args.n_runs):
            try:
                findings = run_judge(trace, provider=args.provider)
            except JudgeUnavailableError as e:
                run_results.append({
                    "run": i + 1,
                    "predicted_step": None,
                    "predicted_category": None,
                    "predicted_summary": None,
                    "correct": None,
                    "error": str(e),
                })
                total_errored += 1
                print(f"{filename} run {i + 1}: ERROR ({e})")
                continue

            correct = score_run(findings, label)
            predicted_step = findings[0].step_index if findings else None
            predicted_category = findings[0].category if findings else None
            predicted_summary = findings[0].summary if findings else None

            if not findings and label["has_failure"]:
                false_negatives += 1
            if findings and not label["has_failure"]:
                false_positives += 1

            run_results.append({
                "run": i + 1,
                "predicted_step": predicted_step,
                "predicted_category": predicted_category,
                "predicted_summary": predicted_summary,
                "correct": correct,
                "error": None,
            })
            total_completed += 1
            total_correct += int(correct)
            print(f"{filename} run {i + 1}: predicted={predicted_step} expected={label['critical_step']} "
                  f"{'OK' if correct else 'MISS'}")

        completed = [r for r in run_results if r["error"] is None]
        errored = len(run_results) - len(completed)
        step_votes = Counter(r["predicted_step"] for r in completed)
        trace_entry = {
            "trace": filename,
            "expected_step": label["critical_step"],
            "expected_has_failure": label["has_failure"],
            "notes": label["notes"],
            "runs": run_results,
            "errored_runs": errored,
            "accuracy": (sum(r["correct"] for r in completed) / len(completed)) if completed else None,
        }
        if step_votes:
            _, majority_count = step_votes.most_common(1)[0]
            trace_entry["self_consistency"] = majority_count / len(completed)
        else:
            trace_entry["self_consistency"] = None
        per_trace.append(trace_entry)

    report = {
        "n_runs_per_trace": args.n_runs,
        "n_traces": len(ground_truth),
        "total_completed_runs": total_completed,
        "total_errored_runs": total_errored,
        "overall_accuracy": (total_correct / total_completed) if total_completed else None,
        "false_positive_count": false_positives,
        "false_negative_count": false_negatives,
        "per_trace": per_trace,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    if total_errored:
        print(f"{total_errored} run(s) errored (API failures, not judge verdicts) -- excluded from accuracy.")
    if total_completed:
        print(f"Overall accuracy: {report['overall_accuracy']:.0%} ({total_correct}/{total_completed} completed runs)")
    else:
        print("No completed runs -- nothing to score.")
    print(f"False positives (clean trace flagged as failed): {false_positives}")
    print(f"False negatives (real failure missed entirely, judge actually ran): {false_negatives}")
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
