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

## Running it

```bash
pip install -r requirements.txt

# either works -- the judge auto-detects whichever key is set
# (Anthropic is checked first if both are present)
export ANTHROPIC_API_KEY=your_key_here
export GOOGLE_API_KEY=your_gemini_key_here

python3 analyze.py sample_traces/stale_plan_tier.json output/diagnosis.json
python3 analyze.py sample_traces/stale_plan_tier.json output/diagnosis.json --provider gemini
```

A `.env` file in the project root (gitignored) also works, via `python-dotenv`.

Then open `output/viewer.html` in a browser (edit the `DATA_URL` constant at the top of the `<script>` block to point at whatever diagnosis file you want to look at). Browsers block `fetch()` against files opened directly from disk, so serve the folder instead:

```bash
python3 -m http.server
# then open http://localhost:8000/output/viewer.html
```

Three sample traces are included:

- `sample_traces/tool_loop.json` — a mechanical failure the rule layer catches on its own (no API key needed): `python3 analyze.py sample_traces/tool_loop.json output/diagnosis.json --no-judge`
- `sample_traces/stale_plan_tier.json` — a state-corruption failure that needs the LLM judge to catch; the rule layer alone finds nothing on it. `stale_plan_tier_with_judge_example.json` is a worked example showing what the judge is designed to return, for demoing the UI without an API key.
- `sample_traces/timezone_assumption.json` — a different flavor of state corruption: an unchecked early assumption, later explicitly contradicted by a tool result, that never gets reconciled.

## Trace format

Traces are plain JSON: a task description plus an ordered list of steps (`reasoning` / `tool_call` / `tool_result` / `final_answer`). See [schema.py](schema.py). This is intentionally close to OpenTelemetry's GenAI semantic conventions, so real logs from LangChain, a custom agent loop, etc. can be adapted into it without much friction — that adapter is the natural next thing to build.

## Status

This is still a demo of the architecture, not a validated tool — but it now has two real consistency tests, run 5 times each with the Gemini judge (`gemini-2.5-flash`):

- `stale_plan_tier.json`: correct step (2) in 4/5 runs; the 5th picked step 3 instead — the reasoning step where the agent *voiced* the wrong conclusion, not the step where the bad premise was actually introduced. ([full results](output/consistency_stale_plan_tier.json))
- `timezone_assumption.json`: correct step (0) in 5/5 runs — but the *category label* wasn't fully stable even though the step was: one run called it `stale_fact` instead of `contradicted_assumption`. So step-localization can be more reliable than the judge's own explanation of why. ([full results](output/consistency_timezone_assumption.json))

Combined: 9/10 runs across both traces landed on the correct step. All runs, right or wrong, read equally confident — which is the exact risk with an LLM judge: a fluent explanation and a correct one are indistinguishable unless you check reproducibility.

9/10 across two hand-built traces is not a real accuracy number — it's an early signal that the judge is close but not fully reliable, and that off-by-one-step errors (picking the moment a wrong belief was *stated* over the moment it was *created*) are a real, recurring failure mode. The next step is a proper hand-labeled set (15-20 real or realistic failed trajectories, labeled with the true critical step before running the tool) run through this same consistency check, to get an actual precision number — see below.

## Where this goes next

- **Benchmark it.** Hand-label a small set of failed trajectories (or use public failed-trajectory datasets like τ-bench / GAIA) and score critical-failure-step localization *and* run-to-run consistency against ground truth. This is the highest-priority item — everything else assumes the judge is accurate, and right now that's only checked on two hand-built traces.
- **Adapters** for real trace formats (OpenTelemetry GenAI spans, LangSmith exports, raw OpenAI/Anthropic tool-use logs) instead of the hand-rolled JSON schema.
- **A pytest-style regression harness**: turn each diagnosed failure into a fixture that re-runs automatically, so a fixed bug that regresses gets caught in CI instead of production.
- **Batch mode**: run the pipeline over a folder of traces and surface the most common failure category across a whole eval run, not just one trace at a time.
