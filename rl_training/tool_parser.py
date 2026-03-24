"""
Tool call parsing and argument normalisation for Qwen2.5-7B-Instruct.

Handles two common output formats:
  1. OpenAI Chat Completions tool_calls (from vLLM / API serving)
  2. Raw text containing <tool_call> blocks (from direct HuggingFace generation)

Also provides argument normalisation (ported from baseline_eval.py) to fix
frequent Qwen quirks:
  - Wrapping flat-field tools under a nested 'search_params' key
  - Omitting Optional[str] fields that Pydantic expects
  - Adding extra keys to schemas that declare extra="forbid"
"""

from __future__ import annotations

import json
import re
import typing
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedToolCall:
    """A single parsed tool invocation."""
    name: str
    arguments: dict
    call_id: str = ""


# ─── Parse from Chat Completions response ───────────────────────────────────


def parse_chat_completions_tool_calls(message) -> list[ParsedToolCall]:
    """
    Parse tool calls from an OpenAI Chat Completions response message object.
    Works with both the openai SDK objects and raw dicts.
    """
    calls = []

    # Handle SDK message object
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            calls.append(ParsedToolCall(
                name=tc.function.name,
                arguments=args,
                call_id=getattr(tc, "id", ""),
            ))
        return calls

    # Handle raw dict (e.g. from vLLM JSON output)
    if isinstance(message, dict):
        for tc in message.get("tool_calls", []):
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            calls.append(ParsedToolCall(
                name=fn.get("name", ""),
                arguments=args,
                call_id=tc.get("id", ""),
            ))

    return calls


# ─── Parse from raw text (Qwen tool call format) ───────────────────────────

# Qwen2.5-Instruct uses a specific format for tool calls in raw generation:
#   <tool_call>
#   {"name": "tool_name", "arguments": {"key": "value"}}
#   </tool_call>
# It may also generate Hermes-style:
#   <|tool_call|>
#   {"name": "...", "arguments": {...}}
#   <|/tool_call|>

_TOOL_CALL_PATTERN = re.compile(
    r"<\|?tool_call\|?>\s*"
    r"(\{.*?\})"
    r"\s*<\|?/tool_call\|?>",
    re.DOTALL,
)

# Alternative: Qwen sometimes uses ```json blocks with function_call
_JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(\{[^`]*\})\s*```",
    re.DOTALL,
)


def parse_raw_text_tool_calls(text: str) -> list[ParsedToolCall]:
    """
    Parse tool calls from raw LLM generation text.
    Supports <tool_call> tags and JSON code blocks.
    """
    calls = []

    # Try <tool_call> tags first
    for match in _TOOL_CALL_PATTERN.finditer(text):
        try:
            data = json.loads(match.group(1))
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(ParsedToolCall(name=name, arguments=arguments))
        except (json.JSONDecodeError, TypeError):
            continue

    if calls:
        return calls

    # Fallback: try JSON blocks with function_call structure
    for match in _JSON_BLOCK_PATTERN.finditer(text):
        try:
            data = json.loads(match.group(1))
            if "name" in data:
                arguments = data.get("arguments", data.get("parameters", {}))
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append(ParsedToolCall(name=data["name"], arguments=arguments))
        except (json.JSONDecodeError, TypeError):
            continue

    return calls


# ─── Argument normalisation ────────────────────────────────────────────────


def normalize_tool_args(tool_obj, raw_args: dict) -> dict:
    """
    Normalize model-generated args to match the tool's Pydantic schema.

    Fixes three common Qwen mismatches:

    1. Wrapping flat-field tools under a nested 'search_params' key because
       it has seen fhir_observation_search (which does use search_params).
       Fix: if the schema has no 'search_params' field, flatten the nested
       dict into the top level.

    2. Optional[str] fields declared with no default are treated as required
       by Pydantic v2, but the model omits them.
       Fix: inject None for any Optional field that is absent.

    3. Extra keys for schemas that declare extra="forbid".
       Fix: strip keys not in the schema.
    """
    schema_fields = tool_obj.input_schema.model_fields
    args = dict(raw_args)

    # 1. Flatten nested search_params when the schema doesn't have that key
    if "search_params" not in schema_fields and "search_params" in args:
        nested = args.pop("search_params")
        if isinstance(nested, dict):
            args.update(nested)

    # 2. Supply None for Optional (Union[X, None]) fields that are missing
    for field_name, field_info in schema_fields.items():
        if field_name in args:
            continue
        annotation = field_info.annotation
        origin = getattr(annotation, "__origin__", None)
        if origin is typing.Union:
            type_args = getattr(annotation, "__args__", ())
            if type(None) in type_args:
                args[field_name] = None

    # 3. Strip extra keys for schemas that declare extra="forbid"
    config = getattr(tool_obj.input_schema, "model_config", {})
    if config.get("extra") == "forbid":
        args = {k: v for k, v in args.items() if k in schema_fields}

    return args


# ─── Convenience: detect finish tool call ──────────────────────────────────


def is_finish_call(call: ParsedToolCall) -> bool:
    """Check if a parsed tool call is the 'finish' tool."""
    return call.name == "finish"


def extract_finish_value(call: ParsedToolCall) -> list:
    """Extract the value from a finish tool call."""
    return call.arguments.get("value", [])
