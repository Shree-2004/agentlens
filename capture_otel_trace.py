"""One-off script: capture a REAL run of the Multi-Agent Research Assistant
as standard OpenTelemetry spans, exported over OTLP to a real Collector.

capture_real_trace.py records the same run straight into AgentLens's own
schema. This one deliberately does not: it emits spans following the
OpenTelemetry GenAI semantic conventions (invoke_agent / chat /
execute_tool), ships them to an OpenTelemetry Collector, and keeps the
Collector's file output as-is. That file is the fixture otel_adapter.py
is built against -- the same format any Collector-based stack writes.

Instrumentation is manual (monkeypatching, as in capture_real_trace.py)
because the target pins langchain 0.2 / google-generativeai 0.7, which
predate the GenAI auto-instrumentation libraries. Tool arguments/results
and message contents are Opt-In in the spec; they are switched on here,
and attribute values are capped by OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT the
way a real deployment would cap them.

Usage (from agentlens/, Docker running):
    docker run -d --rm --name agentlens-otel-collector -p 4318:4318 \
        -v <abs>/otel/collector-config.yaml:/etc/otelcol-contrib/config.yaml:ro \
        -v <abs>/otel/out:/out otel/opentelemetry-collector-contrib:latest
    .venv-otel/Scripts/python.exe capture_otel_trace.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Must be set before the TracerProvider is created -- SpanLimits reads it once.
os.environ.setdefault("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "8000")

from opentelemetry import context as otel_context, trace  # noqa: E402
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: E402
from opentelemetry.trace import Status, StatusCode  # noqa: E402

TARGET_PROJECT = r"c:\Users\Shree Londhe\Desktop\GITHUB proj\Multi-Agent Research Assistant"
TOPIC = "latest advances in protein folding AI"  # same topic as capture_real_trace.py, for comparison
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
PROVIDER = "gcp.gen_ai"  # well-known gen_ai.provider.name for the Gemini API

sys.path.insert(0, TARGET_PROJECT)
os.chdir(TARGET_PROJECT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(TARGET_PROJECT, ".env"))

provider = TracerProvider(resource=Resource.create({"service.name": "multi-agent-research-assistant"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT)))
tracer = provider.get_tracer("agentlens.capture_otel_trace")

import google.generativeai as genai  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402

import agents.researcher as researcher_mod  # noqa: E402
import graph.pipeline as pipeline_mod  # noqa: E402


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _fail(span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_attribute("error.type", type(exc).__name__)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


# ── invoke_agent: one span per LangGraph node ──────────────────
# LangGraph may run a node on an executor thread; if the root span's
# context didn't follow it there, parent explicitly so the run stays
# one trace instead of fragmenting into one trace per node.
_root_ctx: list = [None]


def wrap_node(attr_name: str, agent_name: str) -> None:
    original = getattr(pipeline_mod, attr_name)

    def wrapped(state):
        parent = None if trace.get_current_span().get_span_context().is_valid else _root_ctx[0]
        with tracer.start_as_current_span(
            f"invoke_agent {agent_name}",
            context=parent,
            attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": agent_name},
            record_exception=False,
        ) as span:
            try:
                return original(state)
            except Exception as e:
                _fail(span, e)
                raise

    setattr(pipeline_mod, attr_name, wrapped)


# ── chat: LangChain calls (Researcher, Analyst, Writer, Critic) ─
# Patched at _generate so the Critic's with_structured_output() path,
# which never goes through .invoke on the chat model, is captured too.
_ROLE = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}


def _lc_messages(messages) -> tuple[list[dict], list[dict]]:
    system, chat = [], []
    for m in messages:
        role = _ROLE.get(getattr(m, "type", ""), getattr(m, "type", "user"))
        part = {"type": "text", "content": m.content if isinstance(m.content, str) else _json(m.content)}
        (system if role == "system" else chat).append(part if role == "system" else {"role": role, "parts": [part]})
    return system, chat


def _lc_output(result) -> list[dict]:
    out = []
    for gen in result.generations:
        msg = gen.message
        parts = []
        if msg.content:
            parts.append({"type": "text", "content": msg.content if isinstance(msg.content, str) else _json(msg.content)})
        for tc in getattr(msg, "tool_calls", None) or []:
            parts.append({"type": "tool_call", "id": tc.get("id"), "name": tc.get("name"), "arguments": tc.get("args")})
        out.append({"role": "assistant", "parts": parts})
    return out


_original_generate = ChatGoogleGenerativeAI._generate


def _generate(self, messages, *args, **kwargs):
    model = str(self.model).removeprefix("models/")
    with tracer.start_as_current_span(f"chat {model}", record_exception=False) as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", PROVIDER)
        span.set_attribute("gen_ai.request.model", model)
        system, chat = _lc_messages(messages)
        if system:
            span.set_attribute("gen_ai.system_instructions", _json(system))
        span.set_attribute("gen_ai.input.messages", _json(chat))
        try:
            result = _original_generate(self, messages, *args, **kwargs)
        except Exception as e:
            _fail(span, e)
            raise
        span.set_attribute("gen_ai.output.messages", _json(_lc_output(result)))
        usage = getattr(result.generations[0].message, "usage_metadata", None) if result.generations else None
        if usage:
            span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
            span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
        return result


ChatGoogleGenerativeAI._generate = _generate


# ── chat: direct google-generativeai calls (Verifier) ──────────
_original_generate_content = genai.GenerativeModel.generate_content


def _generate_content(self, contents, *args, **kwargs):
    model = str(self.model_name).removeprefix("models/")
    with tracer.start_as_current_span(f"chat {model}", record_exception=False) as span:
        span.set_attribute("gen_ai.operation.name", "generate_content")
        span.set_attribute("gen_ai.provider.name", PROVIDER)
        span.set_attribute("gen_ai.request.model", model)
        instruction = getattr(self, "_system_instruction", None)
        if instruction is not None:
            text = "".join(getattr(p, "text", "") for p in getattr(instruction, "parts", [])) or str(instruction)
            span.set_attribute("gen_ai.system_instructions", _json([{"type": "text", "content": text}]))
        prompt = contents if isinstance(contents, str) else _json(contents)
        span.set_attribute("gen_ai.input.messages", _json([{"role": "user", "parts": [{"type": "text", "content": prompt}]}]))
        try:
            response = _original_generate_content(self, contents, *args, **kwargs)
        except Exception as e:
            _fail(span, e)
            raise
        try:
            text = response.text
        except Exception:  # blocked / empty candidates
            text = ""
        span.set_attribute("gen_ai.output.messages", _json([{"role": "assistant", "parts": [{"type": "text", "content": text}]}]))
        return response


genai.GenerativeModel.generate_content = _generate_content


# ── execute_tool: Tavily + ArXiv ───────────────────────────────
# Both tools catch their own exceptions and return [] -- so a failed
# search reaches the trace as an OK span with an empty result, exactly
# as it would under any real instrumentation. Recorded as-is.
def wrap_tool(attr_name: str, arg_names: list[str]) -> None:
    original = getattr(researcher_mod, attr_name)

    def wrapped(*args, **kwargs):
        bound = dict(zip(arg_names, args))
        bound.update(kwargs)
        with tracer.start_as_current_span(
            f"execute_tool {attr_name}",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": attr_name,
                "gen_ai.tool.type": "function",
                "gen_ai.tool.call.arguments": _json(bound),
            },
            record_exception=False,
        ) as span:
            try:
                result = original(*args, **kwargs)
            except Exception as e:
                _fail(span, e)
                raise
            span.set_attribute("gen_ai.tool.call.result", _json(result))
            return result

    setattr(researcher_mod, attr_name, wrapped)


for attr, name in [
    ("researcher_node", "Researcher"),
    ("analyst_node", "Analyst"),
    ("writer_node", "Writer"),
    ("verifier_node", "Verifier"),
    ("critic_node", "Critic"),
]:
    wrap_node(attr, name)

wrap_tool("search_web", ["query", "max_results"])
wrap_tool("fetch_arxiv_papers", ["query", "max_results"])

app = pipeline_mod.build_pipeline()
initial_state = {
    "topic": TOPIC,
    "raw_sources": [],
    "analysis_notes": "",
    "draft_report": "",
    "critic_feedback": None,
    "final_report": None,
    "iteration_count": 0,
}

with tracer.start_as_current_span(
    "invoke_agent Multi-Agent Research Assistant",
    attributes={
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "Multi-Agent Research Assistant",
        "gen_ai.input.messages": _json([{"role": "user", "parts": [{"type": "text", "content": f"Produce a research report on: {TOPIC}"}]}]),
    },
    record_exception=False,
) as root:
    _root_ctx[0] = otel_context.get_current()
    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        _fail(root, e)
        raise
    final_report = final_state.get("final_report") or final_state.get("draft_report") or ""
    root.set_attribute("gen_ai.output.messages", _json([{"role": "assistant", "parts": [{"type": "text", "content": final_report}]}]))
    trace_id = format(root.get_span_context().trace_id, "032x")

provider.shutdown()  # flushes the batch processor
print(f"\niteration_count={final_state.get('iteration_count', 0)}, report length={len(final_report)}")
print(f"trace_id={trace_id} exported to {OTLP_ENDPOINT}")
print(f"Collector output: {Path(__file__).parent / 'otel' / 'out' / 'traces.jsonl'}")
