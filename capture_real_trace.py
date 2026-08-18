"""One-off script: capture a REAL execution trace from the Multi-Agent
Research Assistant project (a separate repo) and convert it into
AgentLens's trace schema.

Every trace AgentLens has been tested on so far was hand-authored --
built specifically to contain (or not contain) a failure. This script
instead monkeypatches that project's actual tool calls and LLM
invocations to record what really happens on a live run, unmodified,
so the resulting trace reflects genuine agent behavior rather than a
constructed test case.

Not part of the ground-truth benchmark -- there's no pre-known "correct"
answer here, that's the point. Output goes to real_traces/, separate
from sample_traces/'s hand-labeled set.

Usage (run from anywhere, with the target project's venv active):
    python3 capture_real_trace.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TARGET_PROJECT = r"c:\Users\Shree Londhe\Desktop\GITHUB proj\Multi-Agent Research Assistant"
TOPIC = "latest advances in protein folding AI"  # same topic as the project's own known-good benchmark run
OUTPUT_PATH = Path(__file__).parent / "real_traces" / "multi_agent_research_live.json"

sys.path.insert(0, TARGET_PROJECT)
os.chdir(TARGET_PROJECT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import agents.researcher as researcher_mod  # noqa: E402
import agents.analyst as analyst_mod  # noqa: E402
import agents.writer as writer_mod  # noqa: E402
import agents.critic as critic_mod  # noqa: E402
from graph.pipeline import build_pipeline  # noqa: E402

trace_steps: list[dict] = []
_next_index = [0]


def record(step_type: str, **fields) -> None:
    trace_steps.append({"index": _next_index[0], "type": step_type, **fields})
    _next_index[0] += 1


_llm_labels: dict[int, str] = {}
_llm_class_patched = False


def wrap_llm_invoke(module, label: str):
    """Records the LLM's raw response content as a reasoning step, then
    returns the real response unchanged -- doesn't alter agent behavior,
    only observes it.

    ChatGoogleGenerativeAI is a pydantic model, which blocks setting
    arbitrary instance attributes (only declared fields), so per-instance
    monkeypatching isn't possible -- patch the class's `invoke` once
    instead, and use an id(instance) -> label registry to tell the four
    modules' LLM instances apart in the recorded trace.
    """
    global _llm_class_patched
    _llm_labels[id(module.llm)] = label

    if _llm_class_patched:
        return

    llm_class = type(module.llm)
    original_invoke = llm_class.invoke

    def wrapped(self, messages, *args, **kwargs):
        response = original_invoke(self, messages, *args, **kwargs)
        content = getattr(response, "content", str(response))
        who = _llm_labels.get(id(self), "LLM")
        record("reasoning", content=f"[{who}] {content.strip()}")
        return response

    llm_class.invoke = wrapped
    _llm_class_patched = True


def wrap_tool(module, attr_name: str, tool_name: str, arg_names: list[str]):
    original_fn = getattr(module, attr_name)

    def wrapped(*args, **kwargs):
        bound_args = dict(zip(arg_names, args))
        bound_args.update(kwargs)
        record("tool_call", tool_name=tool_name, tool_args=bound_args)
        try:
            result = original_fn(*args, **kwargs)
        except Exception as e:
            record("tool_result", tool_name=tool_name, tool_result={"error": str(e)}, is_error=True)
            raise
        summary = {"count": len(result), "titles": [r.get("title") for r in result[:5]]} if isinstance(result, list) else result
        record("tool_result", tool_name=tool_name, tool_result=summary, is_error=False)
        return result

    setattr(module, attr_name, wrapped)


wrap_llm_invoke(researcher_mod, "Researcher")
wrap_llm_invoke(analyst_mod, "Analyst")
wrap_llm_invoke(writer_mod, "Writer")
wrap_llm_invoke(critic_mod, "Critic")

wrap_tool(researcher_mod, "search_web", "search_web", ["query", "max_results"])
wrap_tool(researcher_mod, "fetch_arxiv_papers", "fetch_arxiv_papers", ["topic", "max_results"])

record("reasoning", content=f"Starting real multi-agent research pipeline run on topic: {TOPIC}")

app = build_pipeline()
initial_state = {
    "topic": TOPIC,
    "raw_sources": [],
    "analysis_notes": "",
    "draft_report": "",
    "critic_feedback": None,
    "final_report": None,
    "iteration_count": 0,
}

final_state = app.invoke(initial_state)

final_report = final_state.get("final_report") or final_state.get("draft_report") or ""
record("final_answer", content=final_report)

trace = {
    "task": f"Produce a research report on: {TOPIC}",
    "steps": trace_steps,
    "_meta": {
        "source": "live capture from Multi-Agent Research Assistant, not hand-authored",
        "iteration_count": final_state.get("iteration_count", 0),
        "final_report_length": len(final_report),
    },
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(trace, indent=2), encoding="utf-8")
print(f"\nCaptured {len(trace_steps)} steps, iteration_count={final_state.get('iteration_count', 0)}")
print(f"Wrote real trace to {OUTPUT_PATH}")
