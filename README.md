# AgentLens

A trace debugger for AI agents. Point it at a failed run, get back the exact step where things went wrong — and why — instead of reading raw JSON logs.

Want the 30-second version before reading further? Open [demo.html](demo.html) in a browser — a self-contained, self-playing walkthrough of AgentLens catching the real contradiction described below, no server or install required.

## The problem

Agent debugging doesn't look like normal software debugging. When a deterministic service breaks, you get a stack trace. When an agent breaks, you get a confident, well-formatted, silently wrong answer — and the actual mistake is often buried many steps before the bad output appears. A wrong final answer at step 15 is frequently caused by a bad assumption locked in at step 3.

The hardest version of this is state corruption: a fact gets established early (a cache lookup, a tool result), it's stale or wrong, and every step after it looks locally correct in isolation — the agent is reasoning coherently, just from the wrong premise. Standard evals miss this because they score the final output, not whether the trajectory stayed honest to its own evidence along the way.

AgentLens is a small, opinionated tool for finding that step.

## How it works

Two layers, run in sequence:

- **Rule-based checks** ([rules.py](rules.py)) — fast, deterministic, no LLM call. Catches: tool-call loops (same tool + args called 3+ times), tool errors that get silently treated as success, missing required tool arguments. High precision, but blind to anything semantic.
- **LLM judge** ([judge.py](judge.py)) — reads the entire trajectory in one pass (not step-by-step) and is specifically prompted to compare early facts against later ones, since that comparison is what catches state corruption. Returns a structured verdict: the critical failure step, a failure category, and any contradictions it found.

[analyze.py](analyze.py) merges both into one diagnosis, preferring the earliest high-confidence finding — because the upstream root cause matters more than the downstream symptom.

[output/viewer.html](output/viewer.html) renders the diagnosis as a timeline that visually "breaks" at the critical failure step, so you can see at a glance where a 15-step trajectory actually went wrong. It also renders the judge's extracted claims underneath the step each one came from — the verdict and the evidence it was drawn from, in one view. See "Showing the judge's work" below.

[check_consistency.py](check_consistency.py) runs the judge N times on the same trace and reports whether it lands on the same critical step every time — a single convincing-sounding diagnosis proves nothing on its own, since a fluent explanation and a correct one look identical unless you check reproducibility.

[run_benchmark.py](run_benchmark.py) scores the judge against [sample_traces/ground_truth.json](sample_traces/ground_truth.json) — a hand-labeled set of traces with the correct critical step recorded *before* the judge is run. It runs each trace N times, checks correctness against the label, and tracks false positives/negatives separately from API errors (a call that never completed is not the same as the judge examining a trace and finding nothing — conflating those two would corrupt the exact kind of signal this tool exists to protect, so `judge.py` raises `JudgeUnavailableError` for the former rather than silently returning no findings).

[capture_real_trace.py](capture_real_trace.py) takes a different approach entirely: instead of a hand-authored trace with a known answer, it monkeypatches a real, separate agent project's actual tool calls and LLM invocations to record a genuine, unmodified live run — see "The real-trace test" below.

## Running it

```bash
git clone https://github.com/Shree-2004/agentlens.git
cd agentlens
pip install -r requirements.txt

# any one works -- the judge auto-detects whichever key is set, in this order
export GROQ_API_KEY=your_groq_key_here          # best free tier, start here
export ANTHROPIC_API_KEY=your_key_here          # paid, no daily cap
export GOOGLE_API_KEY=your_gemini_key_here      # free but 20 requests/day

python3 analyze.py sample_traces/stale_plan_tier.json output/diagnosis.json
python3 analyze.py sample_traces/stale_plan_tier.json output/diagnosis.json --provider groq
python3 run_benchmark.py 3 --provider groq   # score the judge against the labeled set
```

A `.env` file in the project root (gitignored) also works, via `python-dotenv`.

### Providers

Three are supported, and the judge picks whichever key is set (in the order above) unless `--provider` says otherwise.

| Provider | Default model | Free tier |
|---|---|---|
| `groq` | `openai/gpt-oss-120b` | ~1,000 requests/day; the daily **token** budget binds first on long traces |
| `anthropic` | `claude-sonnet-5` | paid, no daily cap |
| `gemini` | `gemini-flash-latest` | **20 requests/day** per project per model |

**Gemini's 20/day cap is the single fact that shaped all the recorded data in this repo.** The full benchmark is 10 traces × 3 runs = 30 calls, which doesn't fit in one day — which is why the results below were gathered in fragments across many sessions, why so many runs show as errored, and why `output/benchmark.html` had to be built to merge the pieces. Groq was added to lift that constraint; on it or Anthropic the benchmark finishes in one sitting.

**Which judge answered is recorded.** Accuracy from two different models is not one measurement, so `judge.py` stamps `_provider` and `_model` on every verdict, `run_benchmark.py` rolls those into a `judges` array plus a `mixed_judges` flag, and it prints a warning rather than presenting a blended score as a single number. Pass `--provider` explicitly for benchmark runs — auto-detect picks whatever key happens to be set, so the judge can otherwise change between sessions without anyone deciding to change it.

Groq's catalogue varies by account — `llama-3.3-70b-versatile` is listed in the docs but was not available on a fresh free account, so the default is `openai/gpt-oss-120b`. If you get `model_not_found`, list what your key can actually reach:

```python
from groq import Groq; print(sorted(m.id for m in Groq().models.list().data))
```

then pass `--model <id>`.

The eight recorded reports predate that stamping. Their judge is documented in [output/benchmark_manifest.json](output/benchmark_manifest.json) from the project's own records rather than from the data, and the UI labels it as such.

Then open **[index.html](index.html)** — one page with the trace viewer, the benchmark, and the demo behind tabs, plus pickers for which diagnosis and which prompt version to show. It's the front door; `python3 -m http.server` from the project root puts it at `http://localhost:8000/`.

The individual pages still stand alone. `output/viewer.html` defaults to [real_traces/multi_agent_research_live_diagnosis.json](real_traces/multi_agent_research_live_diagnosis.json) — the real Nobel Prize catch, described below — so it shows something real with no API key; append `?file=<path>` to point it at any other diagnosis. Browsers block `fetch()` against files opened directly from disk, so serve the folder rather than opening the file:

```bash
python3 -m http.server
# http://localhost:8000/                      -- everything, behind tabs
# http://localhost:8000/output/viewer.html    -- just the trace viewer
# http://localhost:8000/output/benchmark.html -- just the benchmark
```

`sample_traces/tool_loop.json` is a mechanical failure the rule layer catches on its own (no API key needed): `python3 analyze.py sample_traces/tool_loop.json output/diagnosis.json --no-judge`. `stale_plan_tier_with_judge_example.json` is a worked example of judge output, for demoing the UI without an API key.

The other 10 traces are the hand-labeled ground-truth set (`sample_traces/ground_truth.json`), covering 8 distinct failure shapes plus 2 clean/no-failure traces (to test false positives) — stale cached facts, a contradicted assumption, misread tool output (unit confusion), ambiguous entity resolution, a computational error with no explicit contradiction to spot, a misattributed source, a stale permission, and a silently-applied wrong default. See `ground_truth.json` for the full list and the reasoning behind each one.

## Trace format

Traces are plain JSON: a task description plus an ordered list of steps (`reasoning` / `tool_call` / `tool_result` / `final_answer`). See [schema.py](schema.py). This is intentionally close to OpenTelemetry's GenAI semantic conventions, so real logs from LangChain, a custom agent loop, etc. can be adapted into it without much friction.

## Reading the benchmark

[output/benchmark.html](output/benchmark.html) renders the benchmark reports: headline accuracy, per-trace rows with expected vs. majority-predicted step, self-consistency, and every individual run behind a click.

The reason this needed a UI at all is that the results were never in one file. Gemini's free-tier daily cap forced the set to be run in fragments across many sessions, so the numbers in this README were assembled by hand from eight separate report files. The page merges them — but only within a **prompt version**, never across, because pooling pre-fix and post-fix runs would average away the exact difference the prompt change was made to measure. That grouping lives in [output/benchmark_manifest.json](output/benchmark_manifest.json) as reviewable data rather than a filename-prefix guess buried in JavaScript.

Three things the page is deliberately careful about, all of them the same lesson this project keeps relearning — *a number without its caveat becomes a claim*:

- **An errored run is not a wrong answer.** API failures get their own tile, their own styling, and are excluded from the accuracy denominator rather than scored as "the judge found nothing." That conflation was a real bug here once; the UI is built so it can't be re-introduced visually.
- **A partial set says so.** The pre-fix reports cover only 5 of the 10 traces, weighted toward the hard ones, so merging them gives a *lower* number than the README's 89% — that figure also counts runs recorded in `consistency_*.json` and in sessions that produced no report at all. The era is tagged `partial set` and its caveat sits next to its own numbers.
- **A flagged set gets no green.** On a partial or untrustworthy era, "0 false positives" drops to neutral instead of rendering as a pass. Incomplete data can prove a problem exists; it can't prove one's absence.

The first benchmark run — scored before `judge.py` distinguished a failed call from a real "no failure" verdict — is kept as [benchmark_report_v0_buggy_scorer.json](output/benchmark_report_v0_buggy_scorer.json) and tagged `do not trust`. It's the recorded evidence of the silent-failure bug, not a result.

```bash
python3 -m http.server
# then http://localhost:8000/output/benchmark.html
```

Both pages are linkable. The benchmark takes `?era=v1` / `?report=<file>`, so pointing someone at "the partial pre-fix set" doesn't mean telling them which dropdowns to change. The trace viewer takes `?file=`, so you can point it anywhere without editing the source: `output/viewer.html?file=diagnosis.json`. Absolute and protocol-relative URLs are rejected — a crafted link shouldn't be able to render someone else's JSON inside the page.

Long steps in the viewer are clamped with a `show all (N characters)` toggle, because real traces carry ~10,000-character prose blocks that otherwise bury the timeline — and seeing where a long trajectory broke at a glance is the entire point of the page. The critical step is never clamped; it's the one you came to read.

## Showing the judge's work

The structural fix that made the judge catch the real contradiction was requiring a `claims` array in its output — every concrete, checkable claim pulled from each step, tagged by step, emitted *before* the verdict. That array is the evidence extraction actually ran: a near-empty claims list on a long trace is the tell that the judge skipped the pass and answered from a single read.

It used to be thrown away. `run_judge` flattened the verdict into one `Finding` and dropped everything else, so the only way to see claims was [debug_judge.py](debug_judge.py) making a second, separate API call. The demo animated claims the real viewer couldn't display.

Claims now travel all the way through: `run_judge_with_claims` returns them alongside the findings, `Diagnosis.claims` carries them, and the viewer renders each one under the step it came from, with a count of how many steps the judge actually extracted from. The verdict and its evidence sit in the same view.

Two details that matter:

- **Claims come back even when `has_failure` is false.** On the clean traces, they're the difference between *"the judge checked and found nothing"* and *"the judge returned nothing"* — which is the entire point of the false-positive tests.
- **`run_judge`'s signature is unchanged**, so `check_consistency.py` and `run_benchmark.py` are untouched.

What this looks like on the real trace: step 15 renders red as the critical step, carrying the Critic's claim *"The 2024 Nobel Prizes have not yet been announced"* — directly under the green steps above it that carry the four correct statements that the prize **was** awarded. The contradiction is visible as two claims on the page rather than a sentence asserting one exists.

```bash
python3 test_claims_plumbing.py   # 18 checks, stubbed verdicts, no API call
```

## Importing traces from Langfuse

[langfuse_adapter.py](langfuse_adapter.py) is that adapter. [Langfuse](https://langfuse.com/) is an OpenTelemetry-backed store for exactly these logs, so any agent already instrumented with it — LangChain, LlamaIndex, a hand-rolled loop — becomes an AgentLens input without touching the agent's code.

This matters more than it sounds. [capture_real_trace.py](capture_real_trace.py) monkeypatches one specific project's internals; it works once, for that repo, with its venv active. That's why there is exactly **one** real trace in `real_traces/`, and why "does the judge work on traces nobody designed for it?" is still answered with n=1. The adapter turns capturing real traces into configuration rather than a bespoke script per project.

```bash
pip install langfuse    # optional -- only needed for this path
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL

# diagnose a Langfuse trace directly
python3 analyze.py --langfuse-trace-id <trace-id> output/diagnosis.json

# or export it to the trace format first, to keep or hand-label
python3 langfuse_adapter.py <trace-id> real_traces/imported.json
```

Everything above this section keeps working with no Langfuse install — the import is lazy and the file path is untouched.

**The mapping**, since this is where fidelity gets lost quietly: Langfuse stores a trace as a tree; AgentLens wants the flat sequence the agent actually walked, so observations are flattened by `startTime` (the API returns them newest-first) rather than walked depth-first. A `GENERATION` becomes one `reasoning` step, a tool `SPAN`/`EVENT` becomes a `tool_call` + `tool_result` pair, `level: ERROR` sets `is_error`, and the root span is deliberately *not* treated as a tool — it supplies the task and final answer, since mapping it as one would fabricate a tool call the agent never made and hand the judge the whole trajectory back as a single "result."

A wrong mapping doesn't crash — it silently gives the judge a trajectory the agent never ran, and every conclusion after that is about a trace that didn't happen. So [test_langfuse_adapter.py](test_langfuse_adapter.py) pins the decisions that fail quietly (ordering, root handling, error flags, JSON-string parsing, paging, the empty-fetch case) against fixtures shaped like real API responses:

```bash
python3 test_langfuse_adapter.py   # no network, no account, no API key
```

One mapping choice is worth naming: a non-dict tool input gets wrapped (`"U-4471"` → `{"input": "U-4471"}`) rather than dropped, because `rules.py` flags empty `tool_args` as a malformed call — dropping it would invent a rule finding for a call that was perfectly well-formed. A failed fetch raises `LangfuseUnavailableError` rather than returning an empty trace, for the same reason `judge.py` raises `JudgeUnavailableError`: "the fetch broke" must never score as "clean run, nothing found."

**Status: the mapping is verified against fixtures, not yet against a live Langfuse project.** The next step is instrumenting [Multi-Agent Research Assistant](https://github.com/Shree-2004/Multi-Agent-Research-Assistant) with Langfuse, re-capturing the Nobel Prize run, and diffing the adapter's output against the hand-captured [multi_agent_research_live.json](real_traces/multi_agent_research_live.json) — a real trace with a known answer is the only way to prove the mapping is faithful before trusting it on new ones.

## Status

The benchmark is now complete in the sense that matters most: **every trace in the ground-truth set has real judge data**, including both false-positive tests that took three separate sessions to actually get a call through. Sample sizes are uneven (1 to 11 real calls per trace, driven entirely by when Gemini's free tier let calls through), so treat this as a solid early read, not a final number with tight error bars.

| Trace | Real calls | Correct |
|---|---|---|
| `stale_plan_tier` (stale cached fact) | 11 | 10 — one run picked step 3, the reasoning step where the agent *voiced* the wrong conclusion, instead of step 2 where the bad premise was introduced |
| `timezone_assumption` (contradicted assumption) | 11 | 11 — though the *category label* wobbled once even when the step was right |
| `misread_units` (misread tool output) | 6 | 6 |
| `duplicate_entity_confusion` (ambiguous entity resolution) | 3 | 3 |
| `compounding_rounding_error` (computational error, no explicit contradiction — the deliberately hard case) | 2 | 2 |
| `misattributed_source` (relied on a superseded doc over a current one) | 2 | 2 |
| `outdated_permission_role` (stale cached role) | 2 | 2 |
| `silent_wrong_default` (tool transparently reports a default, agent doesn't register it) | 4 | 0 — **not a random miss.** All 4 runs consistently picked step 1 (the tool call that omitted `to_currency`) over the labeled step 3 (where the agent ignored the tool's own explanatory note). See below. |
| `clean_success_refund` (false-positive test) | 1 | 1 |
| `clean_success_flight` (false-positive test) | 3 | 3 |

**Overall: 40/45 real calls matched the label (89%), zero false positives across 4 real runs on the two clean traces.** The false-positive result is the more important number of the two — a tool that hallucinates problems in clean traces is worse than one that misses real ones, and this is the first real evidence either way.

**The `silent_wrong_default` disagreement turned out to be reproducible, not noise** — 4/4 runs gave the identical answer (step 1), with reasoning like *"the agent failed to pass the target currency to the conversion tool... causing the tool to return an unconverted amount that the agent incorrectly treated as the USD total."* That's internally consistent and not unreasonable: the judge is locating the root cause at the incomplete tool call itself, where the ground-truth label instead points at the agent's failure to read the tool's own warning about it three steps later. Both are defensible readings of "where the wrong premise was introduced" for a trace that — unlike the others — has no second, contradicting tool result to anchor the answer to. That's arguably a real edge in the judge's prompt (traces without an explicit contradiction leave more room for a plausible-but-different root cause) rather than a straightforward wrong answer, and worth flagging as a known limitation rather than quietly excluding it from the count.

Also worth naming: getting here required fixing a real bug. The first benchmark run silently scored "the API call never completed" the same as "the judge found nothing" — inflating false negatives and, on clean traces, faking correct answers by coincidence. `judge.py` now raises `JudgeUnavailableError` for a failed call instead of returning no findings, and every script that scores judge output excludes errored runs rather than miscounting them. Worth naming directly: this is the exact silent-failure pattern AgentLens's own rule layer checks for in *other* agents' traces (`find_silent_errors` in `rules.py`) — a fitting case study in why that check exists.

Full run-by-run data: [output/benchmark_report_remaining.json](output/benchmark_report_remaining.json) and [output/benchmark_report_final.json](output/benchmark_report_final.json).

### The real-trace test

Every trace above was hand-authored — built specifically to contain (or not contain) a failure. That's a structural limitation of any benchmark built this way: a tool that finds problems in traces designed for it to find doesn't prove much about traces nobody designed for it. So [capture_real_trace.py](capture_real_trace.py) does something different: it monkeypatches the actual tool calls and LLM invocations of a separate, real project — [Multi-Agent Research Assistant](https://github.com/Shree-2004/Multi-Agent-Research-Assistant)'s researcher → analyst → writer → critic pipeline — and records what genuinely happens on a live, unmodified run, with no pre-known answer. Output: [real_traces/multi_agent_research_live.json](real_traces/multi_agent_research_live.json).

That live run surfaced a real, unplanned contradiction: the Analyst (step 13) states the 2024 Nobel Prize in Chemistry was awarded for AlphaFold/protein-design work; the Critic (step 15), reviewing the very same report two steps later, states *"the 2024 Nobel Prizes have not yet been announced."* Two directly conflicting claims about the same real-world fact, in the same trace — exactly the shape of thing AgentLens is designed to catch.

**The judge missed it, 3 out of 3 runs.** Both the rule layer (no tool loops or flagged errors here — correctly, since none exist) and the judge (`analyze.py`, then `check_consistency.py` for 2 more independent runs) said no failure found, consistently. This is a genuine, reproducible miss on real data, not a quota artifact.

The likely reason: every synthetic trace in the ground-truth set put its "fact" in a clean, structured `tool_result` (e.g. `{"tier": "free"}`) — easy to diff against a later structured fact. This contradiction instead sits buried in two ~10,000-character prose blocks of AI-generated report text, where the conflicting claim has to be extracted from natural language rather than read off a field. That's a plausible, real gap between "spots structured fact contradictions" and "spots contradictions buried in long free-form reasoning" — and the 8 synthetic failure traces, however varied, never actually tested the latter.

**First prompt-fix attempt: looks like it didn't work.** `judge.py`'s prompt was revised to explicitly say contradictions can be prose-buried, not just structured, and to extract concrete claims from every step before comparing. Re-tested against the same real trace across two sessions (limited by quota to 1 completed call each time) — both post-fix runs still missed the contradiction. That's 5/5 misses total across both prompt versions (3 before the fix, 2 after). Still a small sample, but consistent enough to draw a tentative conclusion: telling the judge to extract claims first, as an instruction, doesn't reliably make it actually do that as a distinct step. It's a one-pass "read everything, then answer" call — a wordier instruction may not change *how* it processes the input, just what it's told to prioritize.

**Second attempt: a structural fix — this one worked.** The judge's output schema now requires a `"claims"` array — every concrete, checkable claim extracted from each step, tagged by step index — that has to come *before* `has_failure` in the response, forcing extraction to be a visible, required part of the output rather than an internal instruction the model could silently skip. First real test, after two prior sessions of transient overload errors: **it caught the contradiction.** `has_failure: true`, `critical_step: 15` (correctly identifying the Critic's step as where the false belief was actually introduced — steps 10, 11, 13, and 14 all correctly stated the Nobel Prize *was* awarded, so step 15 genuinely is where the wrong claim entered the trajectory, not just where it became visible), `category: contradicted_assumption`, and 12 extracted claims that include all four correct Nobel Prize mentions *and* the Critic's contradicting one — clear evidence the extraction pass genuinely ran, not a lucky final answer. Full raw verdict: [real_traces/multi_agent_research_live_judge_verdict_v3.json](real_traces/multi_agent_research_live_judge_verdict_v3.json).

This is one success on one real trace, not proof the fix generalizes — the obvious next step is checking it didn't regress the synthetic benchmark's 89%.

**Regression check: complete. All 10 traces re-verified against the structural-fix prompt, zero unexplained regressions.**

- `stale_plan_tier`: 3/3 correct
- `outdated_permission_role`: 1/3 correct (2 errored)
- `clean_success_refund`: 2/2 correct — **zero false positives**
- `clean_success_flight`: 2/2 correct — **zero false positives**
- `silent_wrong_default`: 0/2 — still picks step 1 over the labeled step 3, identical to its pre-fix behavior. Not a regression from the fix; a pre-existing, well-understood, defensible disagreement (see above) that the fix neither caused nor happened to resolve.
- `timezone_assumption`: 2/2 correct
- `misread_units`: 3/3 correct
- `duplicate_entity_confusion`: 2/2 correct
- `compounding_rounding_error`: 2/2 correct — including the deliberately hard case (no explicit contradiction to anchor to)
- `misattributed_source`: 2/2 correct

**21 completed calls across all 10 traces: 19 correct (90%), with the 2 "misses" both being the same single known edge case, not scattered noise.** Both traces that mattered most for regression risk — the two clean/false-positive tests — came back with zero false positives. Full run-by-run data across every session: [output/benchmark_report_v2_structural_fix.json](output/benchmark_report_v2_structural_fix.json), [output/benchmark_report_v2_priority.json](output/benchmark_report_v2_priority.json), [output/benchmark_report_v2_remaining5.json](output/benchmark_report_v2_remaining5.json), [output/benchmark_report_v2_final2.json](output/benchmark_report_v2_final2.json), and [output/benchmark_report_v2_lasttrace.json](output/benchmark_report_v2_lasttrace.json).

**Where this leaves the project:** the structural fix for prose-buried contradictions is real, tested against the trace that originally exposed it, and confirmed not to have broken anything else across the full ground-truth set. `silent_wrong_default` remains the one open, named limitation — not hidden, not averaged away.

## Where this goes next

- **Capture more real traces**, ideally ones that fail more obviously than the protein-folding one (a genuine crash, not just a subtle prose contradiction), to see how far the fix generalizes beyond "long + prose-buried."
- **Decide on `silent_wrong_default`**: accept it as a legitimate alternate reading (traces without an explicit later contradiction have a wider band of defensible answers), or extend the judge prompt to also weigh "did the agent ignore an explicit warning" as its own signal.
- **Even out the sample sizes** on the synthetic benchmark. 1-3 real calls on some traces vs. 11 on others is a real limitation of "run until the free tier stops you" — more balanced runs (or a paid tier) would tighten this into a number worth quoting more precisely.
- **Even out the sample sizes** on the synthetic benchmark. 1-3 real calls on some traces vs. 11 on others is a real limitation of "run until the free tier stops you" — more balanced runs (or a paid tier) would tighten this into an actual number worth quoting.
- **Adapters** for real trace formats (OpenTelemetry GenAI spans, LangSmith exports, raw OpenAI/Anthropic tool-use logs) instead of the hand-rolled JSON schema.
- **A pytest-style regression harness**: turn each diagnosed failure into a fixture that re-runs automatically, so a fixed bug that regresses gets caught in CI instead of production.
- **Batch mode**: run the pipeline over a folder of traces and surface the most common failure category across a whole eval run, not just one trace at a time.
