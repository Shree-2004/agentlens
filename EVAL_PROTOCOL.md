# Evaluation protocol

How AgentLens's numbers are produced, and the rules that keep them from being fitted to the test. Written 2026-10-21, after the third prompt pass was found to be partly in-sample: a benchmark used to build a fix can't also certify it.

## The sets

| Set | Folder | What it's for | How it may be used |
|---|---|---|---|
| **dev** | `sample_traces/` (10), `adversarial/`, `abstention/` | Building and debugging the judge | Freely. Every number from it is in-sample. |
| **validation** | `sample_traces/validation/` (15) | Choosing between prompt or engine variants | Run a few times per change. A change that helps dev but not validation doesn't ship. |
| **holdout** | not in this repo until locked | The number that gets quoted | Once per release, reported whatever it shows. A holdout trace that informs a change moves to validation. |
| **external** | Who&When ([ag2ai/Agents_Failure_Attribution](https://github.com/ag2ai/Agents_Failure_Attribution)) | Comparison with published methods under the same task definition | Reported separately: public data may be in a model's training set. |

## Rules

1. **Labels before results.** A set's labels are written and reviewed before the judge ever runs on it. `run_benchmark.py` refuses to score a labels file whose `review.status` isn't `approved`.
2. **Ambiguity is decided up front.** When more than one step is a defensible root cause, the label says so in `acceptable_steps` when it's written. It is never widened after seeing an answer. Exact and lenient scores are both reported; the exact one is the headline.
3. **No relabeling after a run.** A label that turns out to be wrong is corrected in a separate, explained commit, and the run it affected is re-scored and reported both ways. (Example: `inj_verdict_override` disagrees with the judge on every run, with or without injection. It stays as labeled.)
4. **No test-set examples in the prompt.** Every example in `JUDGE_SYSTEM_PROMPT` is unrelated to every trace in every set.
5. **Every report names its judge and its prompt.** Provider, model and a prompt hash are recorded per run; reports from different prompts are never pooled (`output/benchmark_manifest.json`).
6. **Errors are not answers.** A failed or malformed call is excluded and counted, never scored. A run cut short by a daily quota is reported as `stopped_early` with `runs_not_attempted`.

## Annotation schema (v2)

```json
{
  "verdict": "failure | clean | insufficient_evidence",
  "has_failure": "true | false | null (for insufficient_evidence)",
  "critical_step": "int | null -- where the wrong premise was introduced",
  "acceptable_steps": "[int] -- other defensible root-cause steps, decided when labeling; includes critical_step",
  "first_symptom_step": "int | null -- first step where the error is visible downstream",
  "category": "closed list below",
  "responsible_agent": "string | null -- for multi-agent traces",
  "evidence_steps": "[int] -- steps that hold the evidence for the label",
  "difficulty": "easy | medium | hard",
  "notes": "the reasoning behind the label"
}
```

Each labels file also carries `review: {status, reviewer, date, changes}`.

Categories: `stale_fact`, `contradicted_assumption`, `misread_tool_result`, `unsupported_commitment`, `dropped_caveat`, `computational_error`, `wrong_tool_selection`, `silent_tool_error`, `tool_call_loop`, `run_aborted`, `other`.

## Metrics

| Metric | Definition |
|---|---|
| Exact accuracy | right verdict and, for a failure, `critical_step` |
| Lenient accuracy | right verdict and, for a failure, any of `acceptable_steps` |
| False-positive rate | clean or undecidable traces flagged as failures |
| False negatives | failures called clean |
| Abstention | *insufficient evidence* on undecidable traces (should), and on decidable ones (shouldn't) |
| Evidence support | quoted passages found by code in the step they cite; failure verdicts quoting their own critical step |
| Error rates | API failures and malformed verdicts, counted separately |

With sets this small, a single percentage says little. Results are reported as counts (28/30, not 93%). Comparisons between prompt versions use a paired test on the same traces. Confidence intervals are bootstrapped over traces, not runs, since runs of one trace aren't independent.

## Building the holdout

The validation set was written by the same process that has been tuning the prompt, so it's fit for choosing between variants but not for certifying them. The holdout is built so that the prompt's author never sees it:

1. **Traces are generated, not hand-written.** `distillation/generate_synthetic_traces.py` writes them with a model that is not used as a judge here, across the categories above plus clean and undecidable runs. The generator's intended category is bookkeeping, never a label.
2. **A person labels them blind.** They label using this schema, without seeing any judge output. Traces that turn out to be ambiguous get `acceptable_steps`, or are dropped and the drop recorded.
3. **The labels are committed before any run** — the commit is the pre-registration — with the traces outside the repo or in a private one.
4. **It's run once per release** and reported whatever it shows.

## Status

| Set | Traces | Labels | Run? |
|---|---|---|---|
| dev (main) | 10 | hand-written, pre-run | yes — 28/30 on the current prompt |
| adversarial | 4 (+2 controls) | pre-run | yes — 9/12 |
| abstention | 3 | pre-run | yes — 7/9 |
| validation | 15 | proposed 2026-10-21, **review pending** | no |
| holdout | — | — | not built |
| Who&When | 184 logs | expert-labeled (external) | not run |
