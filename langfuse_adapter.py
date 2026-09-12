"""Convert Langfuse traces into AgentLens's trace schema.

Every real trace AgentLens has been tested on so far came from
capture_real_trace.py, which monkeypatches one specific project's tool
calls. That works exactly once, for that repo, with its venv active --
which is why there is only one real trace in real_traces/.

Langfuse is an OpenTelemetry-backed store of the same thing: ordered
agent steps with tool inputs/outputs already structured. Anything
instrumented with it (LangChain, LlamaIndex, a hand-rolled loop) becomes
an AgentLens input through this adapter with no patching at all.

The mapping lives in observations_to_trace(), deliberately separated from
the network call so it can be tested against fixtures without a Langfuse
account -- see test_langfuse_adapter.py.

Usage:
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL

    python3 langfuse_adapter.py <trace-id> real_traces/imported.json
    python3 analyze.py --langfuse-trace-id <trace-id> output/diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

from schema import Step, Trace


class LangfuseUnavailableError(RuntimeError):
    """Raised when Langfuse is configured but the trace couldn't be fetched.

    Deliberately distinct from "returned an empty trace", for the same
    reason judge.py separates JudgeUnavailableError from "found nothing":
    a failed fetch scored as an empty trace would look like a clean run.
    """


# Langfuse returns camelCase over the wire and snake_case through the Python
# SDK's model objects. Accept both, plus plain dicts, so fixtures and live
# responses go through the identical mapping code.
def _field(obs: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obs, dict):
            if name in obs:
                return obs[name]
        elif hasattr(obs, name):
            return getattr(obs, name)
    return None


def _maybe_json(value: Any) -> Any:
    """Langfuse returns input/output as strings; parse when they're JSON."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _as_args(value: Any) -> dict[str, Any]:
    """Coerce an observation's input into the dict tool_args expects.

    Non-dict inputs get wrapped rather than dropped: rules.py flags a
    tool_call with empty tool_args as a malformed call, so a lossy mapping
    here would manufacture rule findings for failures that never happened.
    """
    if isinstance(value, dict):
        return value
    if value is None or value == "":
        return {}
    return {"input": value}


def _sort_key(indexed: tuple[int, Any]) -> tuple[int, str, int]:
    """Chronological order, falling back to the response's own order.

    The observations API sorts by startTime descending, and sibling spans
    can share a timestamp, so the original position breaks ties instead of
    letting equal timestamps reshuffle the trajectory.
    """
    position, obs = indexed
    start = _field(obs, "start_time", "startTime")
    return (0 if start is not None else 1, str(start or ""), position)


def _is_root(obs: Any) -> bool:
    explicit = _field(obs, "is_root_observation", "isRootObservation")
    if explicit is not None:
        return bool(explicit)
    return _field(obs, "parent_observation_id", "parentObservationId") is None


def _is_error(obs: Any) -> bool:
    level = _field(obs, "level")
    if level and str(level).upper() in ("ERROR", "CRITICAL"):
        return True
    return bool(_field(obs, "status_message", "statusMessage"))


def observations_to_trace(
    observations: list[Any],
    task: Optional[str] = None,
    trace_input: Any = None,
    trace_output: Any = None,
) -> Trace:
    """Flatten a Langfuse observation tree into AgentLens's ordered step list.

    Langfuse models a trace as a tree (parentObservationId); AgentLens wants
    the flat chronological sequence the agent actually went through, so the
    tree is flattened by start time rather than walked depth-first.

    Per observation type:
      GENERATION    -> one `reasoning` step (the model's own output)
      SPAN / EVENT  -> a `tool_call` + `tool_result` pair
      root span     -> not a tool; supplies `task` and the `final_answer`

    Skipping the root matters: a root span usually wraps the whole run, so
    mapping it as a tool would fabricate a tool call the agent never made
    and hand the judge a duplicate of the entire trajectory as one "result."
    """
    ordered = [obs for _, obs in sorted(enumerate(observations), key=_sort_key)]

    root = next((o for o in ordered if _is_root(o)), None) if len(ordered) > 1 else None
    if root is not None:
        if trace_input is None:
            trace_input = _maybe_json(_field(root, "input"))
        if trace_output is None:
            trace_output = _maybe_json(_field(root, "output"))

    steps: list[Step] = []

    def add(step_type: str, **fields: Any) -> None:
        steps.append(Step(index=len(steps), type=step_type, **fields))

    for obs in ordered:
        if obs is root:
            continue

        obs_type = str(_field(obs, "type") or "SPAN").upper()
        name = _field(obs, "name") or obs_type.lower()
        obs_input = _maybe_json(_field(obs, "input"))
        obs_output = _maybe_json(_field(obs, "output"))

        if obs_type == "GENERATION":
            add("reasoning", content=_as_text(obs_output))
            continue

        # An organizational span with nothing in or out carries no evidence
        # for the judge to reason over -- keep it out of the trajectory.
        if obs_input in (None, "") and obs_output in (None, ""):
            continue

        add("tool_call", tool_name=name, tool_args=_as_args(obs_input))
        add("tool_result", tool_name=name, tool_result=obs_output, is_error=_is_error(obs))

    final_text = _as_text(trace_output)
    if final_text:
        add("final_answer", content=final_text)
    elif steps and steps[-1].type == "reasoning":
        # No trace-level output recorded: the agent's last generation is the
        # closest thing to a final answer, so relabel rather than invent one.
        steps[-1].type = "final_answer"

    resolved_task = task or _as_text(trace_input) or "(no task recorded in Langfuse trace)"
    return Trace(task=resolved_task, steps=steps)


def _fetch_observations(client: Any, trace_id: str) -> list[Any]:
    """Page through every observation belonging to one trace."""
    collected: list[Any] = []
    page = 1
    while True:
        response = client.api.observations.get_many(
            trace_id=trace_id, fields="core,basic,io", page=page, limit=100
        )
        batch = list(getattr(response, "data", None) or [])
        collected.extend(batch)

        meta = getattr(response, "meta", None)
        total_pages = getattr(meta, "total_pages", None) if meta else None
        if not batch or total_pages is None or page >= total_pages:
            break
        page += 1
    return collected


def fetch_trace(trace_id: str, task: Optional[str] = None, client: Any = None) -> Trace:
    """Pull one Langfuse trace and convert it to an AgentLens Trace.

    Credentials come from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
    LANGFUSE_HOST, which the Langfuse SDK reads itself.
    """
    if client is None:
        try:
            from langfuse import Langfuse
        except ImportError as e:
            raise LangfuseUnavailableError(
                "langfuse is not installed -- `pip install langfuse` to import traces"
            ) from e
        if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
            raise LangfuseUnavailableError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set to import traces"
            )
        client = Langfuse()

    try:
        observations = _fetch_observations(client, trace_id)
    except Exception as e:
        raise LangfuseUnavailableError(f"could not fetch observations for trace {trace_id}: {e}") from e

    if not observations:
        raise LangfuseUnavailableError(f"trace {trace_id} has no observations (wrong id, or wrong project?)")

    # Trace-level input/output give the cleanest task and final answer, but
    # this endpoint is the older one -- treat it as a best-effort extra and
    # fall back to the root observation rather than failing the whole import.
    trace_input = trace_output = None
    try:
        detail = client.api.trace.get(trace_id)
        trace_input = _maybe_json(_field(detail, "input"))
        trace_output = _maybe_json(_field(detail, "output"))
        task = task or _field(detail, "name")
    except Exception as e:
        print(f"langfuse: trace-level fetch unavailable ({e}); using the root observation instead")

    return observations_to_trace(observations, task=task, trace_input=trace_input, trace_output=trace_output)


def trace_to_dict(trace: Trace) -> dict[str, Any]:
    return {"task": trace.task, "steps": [vars(s) for s in trace.steps]}


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Export a Langfuse trace into AgentLens's trace format")
    parser.add_argument("trace_id")
    parser.add_argument("output_path")
    parser.add_argument("--task", help="override the task description (default: the Langfuse trace name/input)")
    args = parser.parse_args()

    trace = fetch_trace(args.trace_id, task=args.task)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(trace_to_dict(trace), f, indent=2, ensure_ascii=False)
    print(f"Imported {len(trace.steps)} steps to {args.output_path}")


if __name__ == "__main__":
    main()
