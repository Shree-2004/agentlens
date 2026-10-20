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
import hashlib
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from console import enable_utf8_output
from judge import JUDGE_SYSTEM_PROMPT, PROVIDER_NAMES, JudgeUnavailableError, _verdict_to_findings, run_judge_raw
from schema import Trace

GROUND_TRUTH_PATH = Path("sample_traces") / "ground_truth.json"


def expected_verdict(label: dict) -> str:
    """Labels written before the three-way verdict only carry has_failure."""
    return label.get("verdict") or ("failure" if label["has_failure"] else "clean")


def score_run(verdict: str, findings, label: dict) -> bool:
    """Right verdict, and for a failure, the right step. Abstaining is only
    correct where the label says the trace can't be decided -- elsewhere it
    is a miss, but counted separately from a false positive or negative."""
    if verdict != expected_verdict(label):
        return False
    if verdict != "failure":
        return True
    return findings[0].step_index == label["critical_step"]


def main() -> None:
    enable_utf8_output()
    load_dotenv()
    parser = argparse.ArgumentParser(description="Benchmark the judge against ground_truth.json")
    parser.add_argument("n_runs", type=int, nargs="?", default=3)
    parser.add_argument("--provider", choices=PROVIDER_NAMES)
    parser.add_argument("--output", default="output/benchmark_report.json")
    parser.add_argument("--only", help="comma-separated trace filenames to run, instead of the full set")
    parser.add_argument("--labels", default=str(GROUND_TRUTH_PATH),
                        help="labels file; traces are read from its folder (e.g. sample_traces/adversarial/ground_truth.json)")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    traces_dir = labels_path.parent
    ground_truth = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
    if args.only:
        wanted = set(args.only.split(","))
        ground_truth = {k: v for k, v in ground_truth.items() if k in wanted}

    per_trace = []
    total_correct = 0
    total_completed = 0
    total_errored = 0
    false_positives = 0
    false_negatives = 0
    abstentions = 0
    wrong_abstentions = 0
    quotes_total = 0
    quotes_verified = 0
    failure_calls = 0
    failure_calls_quoting_critical = 0

    for filename, label in ground_truth.items():
        trace = Trace.load(str(traces_dir / filename))
        run_results = []

        for i in range(args.n_runs):
            try:
                verdict = run_judge_raw(trace, provider=args.provider)
                if verdict is None:
                    # `or {}` used to go here, and {} reads as "no failure":
                    # with no key set, every clean trace scored as correct.
                    raise SystemExit("No judge API key is set -- nothing to benchmark.")
                findings = _verdict_to_findings(verdict, trace)
            except JudgeUnavailableError as e:
                run_results.append({
                    "run": i + 1,
                    "predicted_verdict": None,
                    "predicted_step": None,
                    "predicted_category": None,
                    "predicted_summary": None,
                    "correct": None,
                    "error": str(e),
                    "provider": args.provider,
                    "model": None,
                })
                total_errored += 1
                print(f"{filename} run {i + 1}: ERROR ({e})")
                continue

            predicted_verdict = verdict["verdict"]  # filled in by _verdict_to_findings
            expected = expected_verdict(label)
            correct = score_run(predicted_verdict, findings, label)
            predicted_step = findings[0].step_index if findings else None
            predicted_category = findings[0].category if findings else None
            predicted_summary = findings[0].summary if findings else verdict.get("summary")

            if expected == "failure" and predicted_verdict == "clean":
                false_negatives += 1
            if expected != "failure" and predicted_verdict == "failure":
                false_positives += 1
            if predicted_verdict == "insufficient_evidence":
                abstentions += 1
                wrong_abstentions += expected != "insufficient_evidence"

            # Does the evidence the judge quoted actually exist where it says?
            evidence = findings[0].evidence if findings else []
            verified = sum(1 for e in evidence if e["verified"])
            quotes_critical = any(e["verified"] and e["step"] == predicted_step for e in evidence)
            quotes_total += len(evidence)
            quotes_verified += verified
            if findings:
                failure_calls += 1
                failure_calls_quoting_critical += quotes_critical

            run_results.append({
                "run": i + 1,
                "predicted_verdict": predicted_verdict,
                "predicted_step": predicted_step,
                "predicted_category": predicted_category,
                "predicted_summary": predicted_summary,
                "missing_evidence": verdict.get("missing_evidence"),
                "evidence": evidence,
                "evidence_verified": verified,
                "correct": correct,
                "error": None,
                # Which judge actually answered. Auto-detect picks whichever
                # key is set, so this can change between sessions without
                # anyone deciding to change it -- and a score is meaningless
                # without knowing which model produced it.
                "provider": verdict.get("_provider"),
                "model": verdict.get("_model"),
            })
            total_completed += 1
            total_correct += int(correct)
            shown = predicted_step if predicted_verdict == "failure" else predicted_verdict
            wanted = label["critical_step"] if expected == "failure" else expected
            quotes = f" quotes {verified}/{len(evidence)}" if evidence else ""
            print(f"{filename} run {i + 1}: predicted={shown} expected={wanted} "
                  f"{'OK' if correct else 'MISS'}{quotes}")

        completed = [r for r in run_results if r["error"] is None]
        errored = len(run_results) - len(completed)
        step_votes = Counter(r["predicted_step"] for r in completed)
        trace_entry = {
            "trace": filename,
            "expected_step": label["critical_step"],
            "expected_has_failure": label["has_failure"],
            "expected_verdict": expected_verdict(label),
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

    # Roll the per-run stamps up to the report. A report whose runs came from
    # more than one judge is not one measurement -- averaging it would blend
    # two models into a single number that describes neither, which is the
    # same class of mistake as scoring a failed call as "found nothing."
    judges = sorted({
        f"{r['provider']}/{r['model']}"
        for t in per_trace for r in t["runs"]
        if r.get("provider") and r.get("model")
    })

    report = {
        "n_runs_per_trace": args.n_runs,
        "n_traces": len(ground_truth),
        "labels": labels_path.as_posix(),
        # Which prompt produced these numbers. Pre- and post-change reports
        # are different measurements; the hash keeps them from being pooled.
        "prompt_sha256": hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
        "total_completed_runs": total_completed,
        "total_errored_runs": total_errored,
        "overall_accuracy": (total_correct / total_completed) if total_completed else None,
        "false_positive_count": false_positives,
        "false_negative_count": false_negatives,
        "abstention_count": abstentions,
        "wrong_abstention_count": wrong_abstentions,
        # Evidence support: share of quoted passages that code found in the
        # step they cite, and share of failure verdicts that quote their own
        # critical step verifiably.
        "evidence_quotes": quotes_total,
        "evidence_quotes_verified": quotes_verified,
        "evidence_support_rate": (quotes_verified / quotes_total) if quotes_total else None,
        "failure_verdicts": failure_calls,
        "failure_verdicts_quoting_critical_step": failure_calls_quoting_critical,
        "judges": judges,
        "mixed_judges": len(judges) > 1,
        "per_trace": per_trace,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    if len(judges) > 1:
        print(f"WARNING: runs came from more than one judge ({', '.join(judges)}).")
        print("The overall accuracy below averages across them and describes neither.")
    elif judges:
        print(f"Judge: {judges[0]}")
    if total_errored:
        print(f"{total_errored} run(s) errored (API failures, not judge verdicts) -- excluded from accuracy.")
    if total_completed:
        print(f"Overall accuracy: {report['overall_accuracy']:.0%} ({total_correct}/{total_completed} completed runs)")
    else:
        print("No completed runs -- nothing to score.")
    print(f"False positives (clean trace flagged as failed): {false_positives}")
    print(f"False negatives (real failure missed entirely, judge actually ran): {false_negatives}")
    print(f"Abstained (insufficient evidence): {abstentions}, of which on decidable traces: {wrong_abstentions}")
    if quotes_total:
        print(f"Evidence: {quotes_verified}/{quotes_total} quotes found in the step they cite "
              f"({quotes_verified / quotes_total:.0%}); {failure_calls_quoting_critical}/{failure_calls} "
              f"failure verdicts quote their critical step")
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
