# AgentLens

A trace debugger for AI agents. Point it at a failed run, get back the exact step where things went wrong — and why — instead of reading raw JSON logs.

## The problem

Agent debugging doesn't look like normal software debugging. When a deterministic service breaks, you get a stack trace. When an agent breaks, you get a confident, well-formatted, silently wrong answer — and the actual mistake is often buried many steps before the bad output appears. A wrong final answer at step 15 is frequently caused by a bad assumption locked in at step 3.

The hardest version of this is state corruption: a fact gets established early (a cache lookup, a tool result), it's stale or wrong, and every step after it looks locally correct in isolation — the agent is reasoning coherently, just from the wrong premise. Standard evals miss this because they score the final output, not whether the trajectory stayed honest to its own evidence along the way.

AgentLens is a small, opinionated tool for finding that step.

## How it works

Two layers, run in sequence:

- **Rule-based checks** ([rules.py](rules.py)) — fast, deterministic, no LLM call. Catches: tool-call loops (same tool + args called 3+ times), tool errors that get silently treated as success, missing required tool arguments. High precision, but blind to anything semantic.
- **LLM judge** ([judge.py](judge.py)) — reads the entire trajectory in one pass (not step-by-step) and is specifically prompted to compare early facts against later ones, since that comparison is what catches state corruption. Returns a structured verdict: the critical failure step, a failure category, and any contradictions it found.

[analyze.py](analyze.py) merges both into one diagnosis, preferring the earliest high-confidence finding — because the upstream root cause matters more than the downstream symptom.

[output/viewer.html](output/viewer.html) renders the diagnosis as a timeline that visually "breaks" at the critical failure step, so you can see at a glance where a 15-step trajectory actually went wrong.

[check_consistency.py](check_consistency.py) runs the judge N times on the same trace and reports whether it lands on the same critical step every time — a single convincing-sounding diagnosis proves nothing on its own, since a fluent explanation and a correct one look identical unless you check reproducibility.

[run_benchmark.py](run_benchmark.py) scores the judge against [sample_traces/ground_truth.json](sample_traces/ground_truth.json) — a hand-labeled set of traces with the correct critical step recorded *before* the judge is run. It runs each trace N times, checks correctness against the label, and tracks false positives/negatives separately from API errors (a call that never completed is not the same as the judge examining a trace and finding nothing — conflating those two would corrupt the exact kind of signal this tool exists to protect, so `judge.py` raises `JudgeUnavailableError` for the former rather than silently returning no findings).

## Running it

```bash
pip install -r requirements.txt

# either works -- the judge auto-detects whichever key is set
# (Anthropic is checked first if both are present)
export ANTHROPIC_API_KEY=your_key_here
export GOOGLE_API_KEY=your_gemini_key_here

python3 analyze.py sample_traces/stale_plan_tier.json output/diagnosis.json
python3 analyze.py sample_traces/stale_plan_tier.json output/diagnosis.json --provider gemini
python3 run_benchmark.py 3   # score the judge against the labeled ground-truth set
```

A `.env` file in the project root (gitignored) also works, via `python-dotenv`.

**Gemini free tier note:** `generativelanguage.googleapis.com` caps free-tier usage at **20 requests/day per Google Cloud project per model**. The full benchmark (10 traces × 3 runs = 30 calls) doesn't fit in one day on a fresh free-tier project — plan accordingly, or use `--provider anthropic` (paid, no free daily cap) if you need it to finish in one sitting.

Then open `output/viewer.html` in a browser (edit the `DATA_URL` constant at the top of the `<script>` block to point at whatever diagnosis file you want to look at). Browsers block `fetch()` against files opened directly from disk, so serve the folder instead:

```bash
python3 -m http.server
# then open http://localhost:8000/output/viewer.html
```

`sample_traces/tool_loop.json` is a mechanical failure the rule layer catches on its own (no API key needed): `python3 analyze.py sample_traces/tool_loop.json output/diagnosis.json --no-judge`. `stale_plan_tier_with_judge_example.json` is a worked example of judge output, for demoing the UI without an API key.

The other 10 traces are the hand-labeled ground-truth set (`sample_traces/ground_truth.json`), covering 8 distinct failure shapes plus 2 clean/no-failure traces (to test false positives) — stale cached facts, a contradicted assumption, misread tool output (unit confusion), ambiguous entity resolution, a computational error with no explicit contradiction to spot, a misattributed source, a stale permission, and a silently-applied wrong default. See `ground_truth.json` for the full list and the reasoning behind each one.

## Trace format

Traces are plain JSON: a task description plus an ordered list of steps (`reasoning` / `tool_call` / `tool_result` / `final_answer`). See [schema.py](schema.py). This is intentionally close to OpenTelemetry's GenAI semantic conventions, so real logs from LangChain, a custom agent loop, etc. can be adapted into it without much friction — that adapter is the natural next thing to build.

## Status

Still not a validated tool, but there's now a real (if incomplete) benchmark in progress against the 10-trace hand-labeled ground-truth set, using Gemini (`gemini-flash-latest`).

**What's actually been measured**, counting only calls that genuinely completed (excludes API errors — see the false-negative caveat below):

| Trace | Real calls | Correct |
|---|---|---|
| `stale_plan_tier` (stale cached fact) | 11 | 10 — one run picked step 3, the reasoning step where the agent *voiced* the wrong conclusion, instead of step 2 where the bad premise was introduced |
| `timezone_assumption` (contradicted assumption) | 11 | 11 — though the *category label* wobbled once even when the step was right |
| `misread_units` (misread tool output) | 6 | 6 |
| `duplicate_entity_confusion` (ambiguous entity resolution) | 3 | 3 |
| `compounding_rounding_error` (computational error, no explicit contradiction — the deliberately hard case) | 2 | 2 |
| `misattributed_source`, `outdated_permission_role`, `silent_wrong_default`, both `clean_success_*` (false-positive tests) | 0 | untested |

32/33 on the traces that got to run is a genuinely good early signal, including on the case designed to be hardest for a contradiction-spotting judge. But **half the ground-truth set — including both false-positive tests — has zero real data**, so this is not yet a benchmark result, just a promising partial one.

**Why it's incomplete, and why that's a useful finding on its own:** Gemini's free tier caps usage at 20 requests/day per Google Cloud project per model. Running the full 30-call benchmark hit that wall twice in one session (including on a freshly created project), and every call after the wall failed with a 429. The first version of `run_benchmark.py` didn't distinguish "the API call never completed" from "the judge ran and found nothing" — so those failures were silently scored as false negatives (and, on the clean traces, as coincidentally-correct true negatives). That's now fixed: `judge.py` raises `JudgeUnavailableError` for a failed call instead of returning no findings, and the benchmark/consistency scripts exclude errored runs from accuracy math rather than miscounting them. Worth naming directly: this is the exact silent-failure pattern AgentLens's own rule layer checks for in *other* agents' traces (`find_silent_errors` in `rules.py`) — it's a reasonable case study in why that check exists, not just an inconvenience in getting a number.

The remaining 5 traces need either a quota reset or a different provider to finish — see "Running it" above.

## Where this goes next

- **Finish the benchmark.** Get real (non-errored) runs on the remaining 5 traces, including both false-positive tests, which currently have zero data. This is the highest-priority item — everything else assumes the judge is accurate, and the false-positive rate in particular is completely unverified right now.
- **Adapters** for real trace formats (OpenTelemetry GenAI spans, LangSmith exports, raw OpenAI/Anthropic tool-use logs) instead of the hand-rolled JSON schema.
- **A pytest-style regression harness**: turn each diagnosed failure into a fixture that re-runs automatically, so a fixed bug that regresses gets caught in CI instead of production.
- **Batch mode**: run the pipeline over a folder of traces and surface the most common failure category across a whole eval run, not just one trace at a time.
