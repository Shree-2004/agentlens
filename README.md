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

[output/viewer.html](output/viewer.html) renders the diagnosis as a timeline that visually "breaks" at the critical failure step, so you can see at a glance where a 15-step trajectory actually went wrong.

[check_consistency.py](check_consistency.py) runs the judge N times on the same trace and reports whether it lands on the same critical step every time — a single convincing-sounding diagnosis proves nothing on its own, since a fluent explanation and a correct one look identical unless you check reproducibility.

[run_benchmark.py](run_benchmark.py) scores the judge against [sample_traces/ground_truth.json](sample_traces/ground_truth.json) — a hand-labeled set of traces with the correct critical step recorded *before* the judge is run. It runs each trace N times, checks correctness against the label, and tracks false positives/negatives separately from API errors (a call that never completed is not the same as the judge examining a trace and finding nothing — conflating those two would corrupt the exact kind of signal this tool exists to protect, so `judge.py` raises `JudgeUnavailableError` for the former rather than silently returning no findings).

[capture_real_trace.py](capture_real_trace.py) takes a different approach entirely: instead of a hand-authored trace with a known answer, it monkeypatches a real, separate agent project's actual tool calls and LLM invocations to record a genuine, unmodified live run — see "The real-trace test" below.

## Running it

```bash
git clone https://github.com/Shree-2004/agentlens.git
cd agentlens
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
