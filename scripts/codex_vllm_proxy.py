from __future__ import annotations

import asyncio
import argparse
from collections import OrderedDict
import copy
from dataclasses import dataclass, field
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response as FastAPIResponse, StreamingResponse
from openai.types.responses import (
    Response as ResponsesResponse,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseInProgressEvent,
    ResponseIncompleteEvent,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_error import ResponseError
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.response_usage import ResponseUsage
from pydantic import TypeAdapter, ValidationError

LOG = logging.getLogger("codex_vllm_proxy")

RESPONSE_INPUT_OUTPUT_ITEM_ADAPTER = TypeAdapter(ResponseInputItemParam | ResponseOutputItem)
RECENT_WARNING_KEYS: OrderedDict[str, None] = OrderedDict()
AUTO_CONTINUE_PROMPT = (
    "Continue from the previous assistant step. "
    "If you intended to call a tool, emit the tool call now. "
    "Otherwise provide the complete final answer instead of a transition sentence."
)
MAX_PREMATURE_FINAL_RETRIES = 1
PREMATURE_FINAL_PREFIX_RE = re.compile(
    r"^(?:"
    r"let me\b|"
    r"now let me\b|"
    r"i(?:'| wi)ll\b|"
    r"next[, ]+i(?:'| wi)ll\b|"
    r"checking\b|"
    r"trying\b|"
    r"让我|"
    r"现在让我|"
    r"让我再|"
    r"让我继续|"
    r"让我检查|"
    r"让我尝试|"
    r"让我对比|"
    r"让我查看|"
    r"让我总结|"
    r"我来|"
    r"接下来让我"
    r")",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ProxySettings:
    upstream_base_url: str
    request_timeout: float
    max_conversation_history: int


@dataclass(slots=True)
class StreamToolState:
    id: str
    name: str = ""
    arguments: str = ""
    output_index: int = -1


@dataclass(slots=True)
class StreamResponseState:
    response: ResponsesResponse
    message_id: str
    sequence_number: int = 0
    assistant_output_index: int | None = None
    assistant_text: str = ""
    reasoning_output_index: int | None = None
    reasoning_id: str | None = None
    reasoning_text: str = ""
    tool_calls: dict[int, StreamToolState] = field(default_factory=dict)
    usage: Any = None


conversation_history: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()


def load_settings(
    *,
    upstream_base_url: str | None = None,
    request_timeout: float | None = None,
    max_conversation_history: int | None = None,
) -> ProxySettings:
    return ProxySettings(
        upstream_base_url=(upstream_base_url or os.getenv("CODEX_VLLM_PROXY_UPSTREAM", "http://127.0.0.1:8000")).rstrip("/"),
        request_timeout=request_timeout or float(os.getenv("CODEX_VLLM_PROXY_TIMEOUT", "600")),
        max_conversation_history=max_conversation_history or int(os.getenv("CODEX_VLLM_PROXY_MAX_HISTORY", "200")),
    )


def current_timestamp() -> float:
    return time.time()


def _response_json(value: Any, *, exclude_none: bool = True) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=exclude_none)
    return value


def _sse(event: ResponseStreamEvent) -> str:
    return f"data: {event.model_dump_json(exclude_none=True)}\n\n"


def _next_sequence(state: StreamResponseState) -> int:
    sequence_number = state.sequence_number
    state.sequence_number += 1
    return sequence_number


def _serialize_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def _truncate_for_log(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key in RECENT_WARNING_KEYS:
        return
    LOG.warning(message, *args)
    RECENT_WARNING_KEYS[key] = None
    RECENT_WARNING_KEYS.move_to_end(key)
    while len(RECENT_WARNING_KEYS) > 256:
        RECENT_WARNING_KEYS.popitem(last=False)


def _format_validation_errors(exc: ValidationError, limit: int = 400) -> str:
    return _truncate_for_log(_json_dumps_compact(exc.errors(include_url=False)), limit=limit)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") in {
            "input_text",
            "output_text",
            "text",
            "reasoning_text",
        }:
            parts.append(item.get("text", ""))
    return "".join(parts)


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _merge_response_usage(*usages: ResponseUsage | None) -> ResponseUsage | None:
    values = [usage for usage in usages if usage is not None]
    if not values:
        return None

    merged = {
        "input_tokens": sum(int(getattr(usage, "input_tokens", 0) or 0) for usage in values),
        "input_tokens_details": {
            "cached_tokens": sum(
                int(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0)
                for usage in values
            )
        },
        "output_tokens": sum(int(getattr(usage, "output_tokens", 0) or 0) for usage in values),
        "output_tokens_details": {
            "reasoning_tokens": sum(
                int(getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0)
                for usage in values
            )
        },
        "total_tokens": sum(int(getattr(usage, "total_tokens", 0) or 0) for usage in values),
    }
    return ResponseUsage.model_validate(merged)


def _normalize_function_arguments(
    arguments: Any,
    *,
    call_id: str | None = None,
    tool_name: str | None = None,
) -> str:
    if arguments is None or arguments == "":
        return "{}"

    if isinstance(arguments, (dict, list, int, float, bool)):
        return _json_dumps_compact(arguments)

    if not isinstance(arguments, str):
        raise HTTPException(
            status_code=400,
            detail=f"invalid function_call.arguments for call_id={call_id or 'unknown'}: expected JSON string/object",
        )

    content = arguments.strip()
    if not content:
        return "{}"

    try:
        parsed = json.loads(content)
        return _json_dumps_compact(parsed)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        values: list[Any] = []
        index = 0

        while index < len(content):
            while index < len(content) and content[index].isspace():
                index += 1
            if index >= len(content):
                break
            try:
                value, next_index = decoder.raw_decode(content, index)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"invalid function_call.arguments for call_id={call_id or 'unknown'}"
                        f" name={tool_name or 'unknown'}: {exc.msg}"
                    ),
                ) from exc
            values.append(value)
            index = next_index

        if not values:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"invalid function_call.arguments for call_id={call_id or 'unknown'}"
                    f" name={tool_name or 'unknown'}: empty JSON payload"
                ),
            )

        _warn_once(
            f"concat-args:{tool_name or 'unknown'}:{hash(content)}",
            "normalized concatenated function_call arguments for call_id=%s name=%s values=%s payload=%s",
            call_id or "unknown",
            tool_name or "unknown",
            len(values),
            _truncate_for_log(content),
        )
        return _json_dumps_compact(values[-1])


def _normalize_message_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    normalized_parts: list[Any] = []
    for part in content:
        if isinstance(part, str):
            normalized_parts.append({"type": "input_text", "text": part})
            continue

        if not isinstance(part, dict):
            normalized_parts.append(part)
            continue

        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text", "reasoning_text"} or "text" in part:
            normalized_parts.append({"type": "input_text", "text": str(part.get("text", ""))})
            continue

        normalized_parts.append(copy.deepcopy(part))

    return normalized_parts


def _normalize_summary_content(summary: Any) -> list[dict[str, str]]:
    if not isinstance(summary, list):
        return []

    normalized_summary: list[dict[str, str]] = []
    for part in summary:
        if isinstance(part, str):
            normalized_summary.append({"type": "summary_text", "text": part})
            continue
        if isinstance(part, dict) and ("text" in part or part.get("type") == "summary_text"):
            normalized_summary.append({"type": "summary_text", "text": str(part.get("text", ""))})
    return normalized_summary


def _normalize_reasoning_content(content: Any) -> list[dict[str, str]] | None:
    if content is None:
        return None
    if not isinstance(content, list):
        return []

    normalized_parts: list[dict[str, str]] = []
    for part in content:
        if isinstance(part, str):
            normalized_parts.append({"type": "reasoning_text", "text": part})
            continue
        if isinstance(part, dict) and ("text" in part or part.get("type") == "reasoning_text"):
            normalized_parts.append({"type": "reasoning_text", "text": str(part.get("text", ""))})

    return normalized_parts


def _canonicalize_input_item(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get("type")
    if item_type == "message":
        return {
            "type": "message",
            "role": item.get("role"),
            "content": _normalize_message_content(item.get("content")),
        }

    if item_type == "reasoning":
        normalized_item: dict[str, Any] = {
            "type": "reasoning",
            "id": str(item.get("id") or f"rs_{uuid.uuid4().hex}"),
            "summary": _normalize_summary_content(item.get("summary")),
        }
        normalized_content = _normalize_reasoning_content(item.get("content"))
        if normalized_content is not None:
            normalized_item["content"] = normalized_content
        if item.get("encrypted_content") is not None:
            normalized_item["encrypted_content"] = item["encrypted_content"]
        if item.get("status") is not None:
            normalized_item["status"] = item["status"]
        return normalized_item

    return copy.deepcopy(item)


def _normalize_input_items(raw_input: Any) -> list[Any]:
    if isinstance(raw_input, str):
        input_items: list[Any] = [raw_input]
    elif isinstance(raw_input, list):
        input_items = list(raw_input)
    else:
        input_items = []

    normalized: list[Any] = []
    for item in input_items:
        if isinstance(item, str):
            normalized.append(item)
            continue

        if not isinstance(item, dict):
            LOG.warning("ignoring non-dict responses input item: %r", item)
            continue

        candidate = _canonicalize_input_item(item)
        item_type = candidate.get("type") or type(candidate).__name__
        try:
            parsed = RESPONSE_INPUT_OUTPUT_ITEM_ADAPTER.validate_python(candidate)
        except ValidationError as exc:
            if item_type == "reasoning":
                continue
            LOG.warning(
                "passing through invalid responses input item type=%s errors=%s",
                item_type,
                _format_validation_errors(exc),
            )
            normalized.append(candidate)
            continue

        normalized.append(_response_json(parsed))

    return normalized


def _normalize_metadata(metadata: Any) -> dict[str, str] | None:
    if not isinstance(metadata, dict):
        return None

    normalized: dict[str, str] = {}
    for key, value in metadata.items():
        if key is None or value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized or None


def _normalize_response_usage(usage: Any) -> ResponseUsage | None:
    if isinstance(usage, ResponseUsage):
        return usage
    if not isinstance(usage, dict):
        return None

    if "input_tokens" in usage and "output_tokens" in usage:
        try:
            return ResponseUsage.model_validate(usage)
        except ValidationError:
            return None

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    normalized_usage = {
        "input_tokens": int(input_tokens or 0),
        "input_tokens_details": {
            "cached_tokens": int(prompt_details.get("cached_tokens", 0)) if isinstance(prompt_details, dict) else 0,
        },
        "output_tokens": int(output_tokens or 0),
        "output_tokens_details": {
            "reasoning_tokens": int(completion_details.get("reasoning_tokens", 0)) if isinstance(completion_details, dict) else 0,
        },
        "total_tokens": int(total_tokens if total_tokens is not None else int(input_tokens or 0) + int(output_tokens or 0)),
    }
    try:
        return ResponseUsage.model_validate(normalized_usage)
    except ValidationError:
        return None


def _extract_reasoning_text(reasoning: Any) -> str:
    if isinstance(reasoning, str):
        return reasoning

    if isinstance(reasoning, dict):
        for key in ("delta", "text", "content", "reasoning"):
            value = reasoning.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return _content_to_text(value)
            if isinstance(value, dict):
                nested = _extract_reasoning_text(value)
                if nested:
                    return nested

    if isinstance(reasoning, list):
        return _content_to_text(reasoning)

    return ""


def _extract_reasoning_delta(delta: dict[str, Any]) -> str:
    return _extract_reasoning_text(delta.get("reasoning"))


def _looks_like_premature_final_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) > 280:
        return False

    collapsed = " ".join(stripped.lower().split())
    if PREMATURE_FINAL_PREFIX_RE.match(stripped) and (stripped.endswith((":", "：")) or len(stripped) <= 96):
        return True
    if stripped.endswith((":", "：")) and any(marker in collapsed for marker in ("let me", "i'll", "i will", "trying", "checking")):
        return True
    if stripped.endswith((":", "：")) and any(marker in stripped for marker in ("让我", "我来", "接下来")):
        return True
    return False


def _should_auto_continue(
    *,
    finish_reason: Any,
    assistant_text: str,
    tool_calls_payload: list[Any],
    retry_count: int,
) -> bool:
    return (
        finish_reason == "stop"
        and not tool_calls_payload
        and retry_count < MAX_PREMATURE_FINAL_RETRIES
        and _looks_like_premature_final_answer(assistant_text)
    )


def _extract_leading_instructions(input_items: list[Any]) -> tuple[list[str], list[Any]]:
    instructions: list[str] = []
    remaining: list[Any] = []
    consumed_prefix = True

    for item in input_items:
        is_instruction = (
            consumed_prefix
            and isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") in {"system", "developer"}
        )
        if is_instruction:
            text = _content_to_text(item.get("content"))
            if text:
                instructions.append(text)
            continue

        consumed_prefix = False
        remaining.append(item)

    skipped = [
        item for item in remaining
        if isinstance(item, dict)
        and item.get("type") == "message"
        and item.get("role") in {"system", "developer"}
    ]
    if skipped:
        LOG.warning("skipping %s non-leading system/developer messages", len(skipped))

    return instructions, remaining


def _ensure_system_message(messages: list[dict[str, Any]], instructions: str) -> None:
    for message in messages:
        if message.get("role") == "system":
            message["content"] = instructions
            return
    messages.insert(0, {"role": "system", "content": instructions})


def _history_has_tool_call(messages: list[dict[str, Any]], call_id: str | None) -> bool:
    if not call_id:
        return False
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []):
            if tool_call.get("id") == call_id:
                return True
    return False


def _flush_pending_tool_calls(
    messages: list[dict[str, Any]],
    pending_tool_calls: list[dict[str, Any]],
    *,
    assistant_text: str | None = None,
) -> None:
    if not pending_tool_calls:
        return
    messages.append(
        {
            "role": "assistant",
            "content": assistant_text if assistant_text else None,
            "tool_calls": copy.deepcopy(pending_tool_calls),
        }
    )
    pending_tool_calls.clear()


def _sanitize_chat_messages(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue

            function["arguments"] = _normalize_function_arguments(
                function.get("arguments"),
                call_id=tool_call.get("id"),
                tool_name=function.get("name"),
            )


def validate_message_sequence(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen_tool_call_ids: set[str] = set()

    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            validated.append(message)
            continue

        tool_call_id = message.get("tool_call_id")
        if not tool_call_id:
            LOG.warning("dropping tool message %s without tool_call_id", index)
            continue

        if tool_call_id in seen_tool_call_ids:
            LOG.warning("dropping duplicate tool message for call_id=%s", tool_call_id)
            continue

        has_preceding_tool_call = False
        for previous in reversed(validated):
            if previous.get("role") != "assistant":
                continue
            for tool_call in previous.get("tool_calls", []):
                if tool_call.get("id") == tool_call_id:
                    has_preceding_tool_call = True
                    break
            break

        if not has_preceding_tool_call:
            LOG.warning("dropping orphaned tool message for call_id=%s", tool_call_id)
            continue

        validated.append(message)
        seen_tool_call_ids.add(tool_call_id)

    return validated


def _normalize_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []

    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue

        function_obj = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function_obj.get("name")
        if not name:
            continue

        function_data = {"name": name}
        if function_obj.get("description"):
            function_data["description"] = function_obj["description"]
        if function_obj.get("parameters") is not None:
            function_data["parameters"] = function_obj["parameters"]

        normalized.append({"type": "function", "function": function_data})

    return normalized


def convert_responses_request(request_data: dict[str, Any]) -> dict[str, Any]:
    chat_request: dict[str, Any] = {
        "model": request_data.get("model"),
        "stream": bool(request_data.get("stream", False)),
        "temperature": request_data.get("temperature", 1.0),
        "top_p": request_data.get("top_p", 1.0),
    }

    if "max_output_tokens" in request_data:
        chat_request["max_tokens"] = request_data["max_output_tokens"]

    messages = copy.deepcopy(conversation_history.get(request_data.get("previous_response_id"), []))
    input_items = _normalize_input_items(request_data.get("input"))

    extracted_instructions, input_items = _extract_leading_instructions(input_items)
    instruction_blocks: list[str] = []
    if request_data.get("instructions"):
        instruction_blocks.append(str(request_data["instructions"]))
    instruction_blocks.extend(extracted_instructions)
    if instruction_blocks:
        _ensure_system_message(messages, "\n\n".join(instruction_blocks))

    pending_tool_calls: list[dict[str, Any]] = []
    for item in input_items:
        if isinstance(item, str):
            _flush_pending_tool_calls(messages, pending_tool_calls)
            messages.append({"role": "user", "content": item})
            continue

        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        role = item.get("role")

        if item_type == "message" and role == "user":
            _flush_pending_tool_calls(messages, pending_tool_calls)
            messages.append({"role": "user", "content": _content_to_text(item.get("content"))})
            continue

        if item_type == "message" and role == "assistant":
            assistant_text = _content_to_text(item.get("content"))
            if assistant_text:
                _flush_pending_tool_calls(messages, pending_tool_calls)
                messages.append({"role": "assistant", "content": assistant_text})
            continue

        if item_type == "message" and role in {"system", "developer"}:
            LOG.warning("ignoring non-leading %s message", role)
            continue

        if item_type == "reasoning":
            continue

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
            name = item.get("name") or item.get("function", {}).get("name")
            if not name:
                LOG.warning("ignoring function_call without name: %s", item)
                continue
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": item.get("arguments")
                        or item.get("function", {}).get("arguments")
                        or "{}",
                    },
                }
            )
            continue

        if item_type == "function_call_output":
            _flush_pending_tool_calls(messages, pending_tool_calls)
            call_id = item.get("call_id") or item.get("id")
            if not call_id:
                LOG.warning("ignoring function_call_output without call_id")
                continue

            if not _history_has_tool_call(messages, call_id):
                tool_name = item.get("name")
                if not tool_name:
                    LOG.warning("ignoring function_call_output without matching call and no name")
                    continue
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": item.get("arguments", "{}"),
                                },
                            }
                        ],
                    }
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _serialize_output(item.get("output", "")),
                }
            )
            continue

        LOG.warning("unsupported responses input item type=%s", item_type)

    _flush_pending_tool_calls(messages, pending_tool_calls)

    if not messages or (len(messages) == 1 and messages[0].get("role") == "system"):
        messages.append({"role": "user", "content": ""})

    _sanitize_chat_messages(messages)
    chat_request["messages"] = validate_message_sequence(messages)

    normalized_tools = _normalize_tools(request_data.get("tools"))
    if normalized_tools:
        chat_request["tools"] = normalized_tools

    if request_data.get("tool_choice") is not None:
        chat_request["tool_choice"] = request_data["tool_choice"]

    reasoning = request_data.get("reasoning")
    if isinstance(reasoning, dict) and any(value is not None for value in reasoning.values()):
        chat_request["reasoning"] = reasoning

    for key in ("user", "metadata"):
        if request_data.get(key) is not None:
            chat_request[key] = request_data[key]

    return chat_request


def build_response_shell(request_data: dict[str, Any], response_id: str) -> ResponsesResponse:
    response_data: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": current_timestamp(),
        "status": "in_progress",
        "model": str(request_data.get("model") or ""),
        "output": [],
        "parallel_tool_calls": bool(request_data.get("parallel_tool_calls", True)),
        "tool_choice": request_data.get("tool_choice") if request_data.get("tool_choice") is not None else "auto",
        "tools": request_data.get("tools") if isinstance(request_data.get("tools"), list) else [],
        "temperature": request_data.get("temperature", 1.0),
        "top_p": request_data.get("top_p", 1.0),
        "truncation": request_data.get("truncation", "disabled"),
        "text": request_data.get("text") if isinstance(request_data.get("text"), dict) else {"format": {"type": "text"}},
    }

    if request_data.get("instructions") is not None:
        response_data["instructions"] = request_data["instructions"]
    if isinstance(request_data.get("max_output_tokens"), int):
        response_data["max_output_tokens"] = request_data["max_output_tokens"]
    if request_data.get("previous_response_id"):
        response_data["previous_response_id"] = str(request_data["previous_response_id"])
    if isinstance(request_data.get("reasoning"), dict) and request_data["reasoning"]:
        response_data["reasoning"] = request_data["reasoning"]
    if isinstance(request_data.get("user"), str) and request_data["user"]:
        response_data["user"] = request_data["user"]

    metadata = _normalize_metadata(request_data.get("metadata"))
    if metadata:
        response_data["metadata"] = metadata

    try:
        return ResponsesResponse.model_validate(response_data)
    except ValidationError as exc:
        LOG.warning(
            "response shell validation failed; falling back to minimal shell errors=%s",
            _format_validation_errors(exc),
        )
        return ResponsesResponse(
            id=response_id,
            created_at=current_timestamp(),
            model=str(request_data.get("model") or ""),
            object="response",
            output=[],
            parallel_tool_calls=bool(request_data.get("parallel_tool_calls", True)),
            tool_choice="auto",
            tools=[],
            status="in_progress",
        )


def _store_history(response_id: str, messages: list[dict[str, Any]], max_history: int) -> None:
    conversation_history[response_id] = copy.deepcopy(messages)
    conversation_history.move_to_end(response_id)
    while len(conversation_history) > max_history:
        conversation_history.popitem(last=False)


def _build_assistant_tool_message(
    tool_calls: dict[int, StreamToolState] | dict[int, dict[str, Any]],
    *,
    assistant_text: str | None = None,
) -> dict[str, Any]:
    ordered_calls = [tool_calls[index] for index in sorted(tool_calls)]
    return {
        "role": "assistant",
        "content": assistant_text if assistant_text else None,
        "tool_calls": [
            {
                "id": tool_call.id if isinstance(tool_call, StreamToolState) else tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call.name if isinstance(tool_call, StreamToolState) else tool_call["name"],
                    "arguments": tool_call.arguments if isinstance(tool_call, StreamToolState) else tool_call["arguments"],
                },
            }
            for tool_call in ordered_calls
        ],
    }


def _build_commentary_output_message(text: str, *, status: str = "completed") -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=f"msg_{uuid.uuid4().hex}",
        type="message",
        role="assistant",
        status=status,
        phase="commentary",
        content=[_new_output_text(text)],
    )


def _new_output_text(text: str = "") -> ResponseOutputText:
    return ResponseOutputText(type="output_text", text=text, annotations=[])


def _append_output_item(response_obj: ResponsesResponse, item: ResponseOutputItem) -> int:
    response_obj.output.append(item)
    return len(response_obj.output) - 1


def _ensure_output_message(
    state: StreamResponseState,
) -> tuple[ResponseOutputMessage, int, bool]:
    if state.assistant_output_index is not None:
        output_item = state.response.output[state.assistant_output_index]
        if isinstance(output_item, ResponseOutputMessage):
            return output_item, state.assistant_output_index, False

    output_item = ResponseOutputMessage(
        id=state.message_id,
        type="message",
        role="assistant",
        status="in_progress",
        phase="commentary",
        content=[_new_output_text(state.assistant_text)],
    )
    output_index = _append_output_item(state.response, output_item)
    state.assistant_output_index = output_index
    return output_item, output_index, True


def _ensure_reasoning_item(
    state: StreamResponseState,
) -> tuple[ResponseReasoningItem, int, bool]:
    if state.reasoning_output_index is not None:
        output_item = state.response.output[state.reasoning_output_index]
        if isinstance(output_item, ResponseReasoningItem):
            return output_item, state.reasoning_output_index, False

    reasoning_item = ResponseReasoningItem(
        id=state.reasoning_id or f"rs_{uuid.uuid4().hex}",
        summary=[],
        type="reasoning",
        status="in_progress",
        content=[ResponseReasoningTextContent(type="reasoning_text", text=state.reasoning_text)],
    )
    output_index = _append_output_item(state.response, reasoning_item)
    state.reasoning_output_index = output_index
    state.reasoning_id = reasoning_item.id
    return reasoning_item, output_index, True


def _ensure_tool_output_item(
    state: StreamResponseState,
    index: int,
) -> tuple[StreamToolState, ResponseFunctionToolCall, bool]:
    tool_state = state.tool_calls.get(index)
    if tool_state is None:
        tool_state = StreamToolState(id=f"call_{uuid.uuid4().hex}")
        state.tool_calls[index] = tool_state

    if tool_state.output_index >= 0:
        output_item = state.response.output[tool_state.output_index]
        if isinstance(output_item, ResponseFunctionToolCall):
            return tool_state, output_item, False

    output_item = ResponseFunctionToolCall(
        id=tool_state.id,
        call_id=tool_state.id,
        name=tool_state.name,
        arguments=tool_state.arguments,
        type="function_call",
        status="in_progress",
    )
    tool_state.output_index = _append_output_item(state.response, output_item)
    return tool_state, output_item, True


def _emit_assistant_message_open_events(state: StreamResponseState) -> list[str]:
    output_item, output_index, created = _ensure_output_message(state)
    if not created:
        return []

    return [
        _sse(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=output_item,
                output_index=output_index,
                sequence_number=_next_sequence(state),
            )
        ),
        _sse(
            ResponseContentPartAddedEvent(
                type="response.content_part.added",
                item_id=output_item.id,
                output_index=output_index,
                content_index=0,
                part=output_item.content[0],
                sequence_number=_next_sequence(state),
            )
        ),
    ]


def _emit_reasoning_open_events(state: StreamResponseState) -> list[str]:
    output_item, output_index, created = _ensure_reasoning_item(state)
    if not created:
        return []

    return [
        _sse(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=output_item,
                output_index=output_index,
                sequence_number=_next_sequence(state),
            )
        ),
        _sse(
            ResponseContentPartAddedEvent(
                type="response.content_part.added",
                item_id=output_item.id,
                output_index=output_index,
                content_index=0,
                part={"type": "reasoning_text", "text": state.reasoning_text},
                sequence_number=_next_sequence(state),
            )
        ),
    ]


def _emit_final_assistant_events(
    state: StreamResponseState,
    *,
    require_item: bool,
    phase: str,
    status: str = "completed",
) -> list[str]:
    if state.assistant_output_index is None and not require_item:
        return []

    events = _emit_assistant_message_open_events(state)
    output_item, output_index, _ = _ensure_output_message(state)
    output_item.status = status
    output_item.phase = phase
    output_item.content[0].text = state.assistant_text

    events.extend(
        [
            _sse(
                ResponseTextDoneEvent(
                    type="response.output_text.done",
                    item_id=output_item.id,
                    output_index=output_index,
                    content_index=0,
                    text=state.assistant_text,
                    logprobs=[],
                    sequence_number=_next_sequence(state),
                )
            ),
            _sse(
                ResponseContentPartDoneEvent(
                    type="response.content_part.done",
                    item_id=output_item.id,
                    output_index=output_index,
                    content_index=0,
                    part=output_item.content[0],
                    sequence_number=_next_sequence(state),
                )
            ),
            _sse(
                ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    item=output_item,
                    output_index=output_index,
                    sequence_number=_next_sequence(state),
                )
            ),
        ]
    )
    return events


def _emit_final_reasoning_events(state: StreamResponseState, *, status: str = "completed") -> list[str]:
    if state.reasoning_output_index is None:
        return []

    output_item = state.response.output[state.reasoning_output_index]
    if not isinstance(output_item, ResponseReasoningItem):
        return []

    output_item.status = status
    output_item.content = [ResponseReasoningTextContent(type="reasoning_text", text=state.reasoning_text)]

    return [
        _sse(
            ResponseReasoningTextDoneEvent(
                type="response.reasoning_text.done",
                item_id=output_item.id,
                output_index=state.reasoning_output_index,
                content_index=0,
                text=state.reasoning_text,
                sequence_number=_next_sequence(state),
            )
        ),
        _sse(
            ResponseContentPartDoneEvent(
                type="response.content_part.done",
                item_id=output_item.id,
                output_index=state.reasoning_output_index,
                content_index=0,
                part={"type": "reasoning_text", "text": state.reasoning_text},
                sequence_number=_next_sequence(state),
            )
        ),
        _sse(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=output_item,
                output_index=state.reasoning_output_index,
                sequence_number=_next_sequence(state),
            )
        ),
    ]


def _emit_final_tool_events(state: StreamResponseState) -> list[str]:
    events: list[str] = []
    for index in sorted(state.tool_calls):
        tool_state = state.tool_calls[index]
        if tool_state.output_index < 0:
            continue

        output_item = state.response.output[tool_state.output_index]
        if not isinstance(output_item, ResponseFunctionToolCall):
            continue

        tool_state.arguments = _normalize_function_arguments(
            tool_state.arguments,
            call_id=tool_state.id,
            tool_name=tool_state.name,
        )
        output_item.id = tool_state.id
        output_item.call_id = tool_state.id
        output_item.name = tool_state.name
        output_item.arguments = tool_state.arguments
        output_item.status = "completed"

        events.extend(
            [
                _sse(
                    ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        item_id=output_item.id or tool_state.id,
                        output_index=tool_state.output_index,
                        arguments=output_item.arguments,
                        name=output_item.name,
                        sequence_number=_next_sequence(state),
                    )
                ),
                _sse(
                    ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        item=output_item,
                        output_index=tool_state.output_index,
                        sequence_number=_next_sequence(state),
                    )
                ),
            ]
        )

    return events


def _mark_response_completed(response_obj: ResponsesResponse, usage: Any = None) -> None:
    response_obj.status = "completed"
    response_obj.completed_at = current_timestamp()
    response_obj.incomplete_details = None
    normalized_usage = _normalize_response_usage(usage)
    if normalized_usage is not None:
        response_obj.usage = normalized_usage


def _mark_response_incomplete(
    response_obj: ResponsesResponse,
    *,
    reason: str = "max_output_tokens",
    usage: Any = None,
) -> None:
    response_obj.status = "incomplete"
    response_obj.completed_at = None
    response_obj.incomplete_details = IncompleteDetails(reason=reason)
    normalized_usage = _normalize_response_usage(usage)
    if normalized_usage is not None:
        response_obj.usage = normalized_usage

    for output_item in response_obj.output:
        if isinstance(output_item, (ResponseOutputMessage, ResponseReasoningItem)):
            output_item.status = "incomplete"
            continue
        if isinstance(output_item, ResponseFunctionToolCall) and output_item.status in {None, "in_progress"}:
            output_item.status = "incomplete"


def _mark_response_failed(response_obj: ResponsesResponse, message: str) -> None:
    response_obj.status = "failed"
    response_obj.error = ResponseError(code="server_error", message=message)


def _build_auto_continue_chat_request(
    chat_request: dict[str, Any],
    history_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    followup_request = copy.deepcopy(chat_request)
    followup_messages = copy.deepcopy(history_messages)
    followup_messages.append({"role": "user", "content": AUTO_CONTINUE_PROMPT})
    _sanitize_chat_messages(followup_messages)
    followup_request["messages"] = validate_message_sequence(followup_messages)
    return followup_request


def _persist_text_history(
    response_id: str,
    chat_messages: list[dict[str, Any]],
    assistant_text: str,
    settings: ProxySettings,
) -> None:
    persisted_messages = copy.deepcopy(chat_messages)
    if assistant_text:
        persisted_messages.append({"role": "assistant", "content": assistant_text})
    _store_history(response_id, persisted_messages, settings.max_conversation_history)


def _finalize_text_response(
    response_obj: ResponsesResponse,
    chat_messages: list[dict[str, Any]],
    assistant_text: str,
    settings: ProxySettings,
) -> None:
    _mark_response_completed(response_obj, response_obj.usage)
    _persist_text_history(response_obj.id, chat_messages, assistant_text, settings)


def _finalize_incomplete_text_response(
    response_obj: ResponsesResponse,
    chat_messages: list[dict[str, Any]],
    assistant_text: str,
    settings: ProxySettings,
    *,
    reason: str = "max_output_tokens",
) -> None:
    _mark_response_incomplete(response_obj, reason=reason, usage=response_obj.usage)
    _persist_text_history(response_obj.id, chat_messages, assistant_text, settings)


def _finalize_tool_response(
    response_obj: ResponsesResponse,
    chat_messages: list[dict[str, Any]],
    tool_calls: dict[int, StreamToolState] | dict[int, dict[str, Any]],
    settings: ProxySettings,
    *,
    assistant_text: str,
) -> None:
    _mark_response_completed(response_obj, response_obj.usage)
    persisted_messages = copy.deepcopy(chat_messages)
    persisted_messages.append(_build_assistant_tool_message(tool_calls, assistant_text=assistant_text))
    _store_history(response_obj.id, persisted_messages, settings.max_conversation_history)


def convert_chat_completion_response(
    request_data: dict[str, Any],
    chat_request: dict[str, Any],
    upstream_data: dict[str, Any],
    settings: ProxySettings,
    *,
    history_messages: list[dict[str, Any]] | None = None,
) -> ResponsesResponse:
    response_id = f"resp_{uuid.uuid4().hex}"
    response_obj = build_response_shell(request_data, response_id)
    if upstream_data.get("model"):
        response_obj.model = upstream_data["model"]

    choice = (upstream_data.get("choices") or [{}])[0]
    finish_reason = choice.get("finish_reason")
    message = choice.get("message") or {}
    assistant_text = message.get("content") if isinstance(message.get("content"), str) else _content_to_text(message.get("content"))
    reasoning_text = _extract_reasoning_text(message.get("reasoning"))
    terminal_item_status = "incomplete" if finish_reason == "length" else "completed"

    tool_calls_payload = message.get("tool_calls") or []
    tool_calls: dict[int, StreamToolState] = {}

    if reasoning_text:
        response_obj.output.append(
            ResponseReasoningItem(
                id=f"rs_{uuid.uuid4().hex}",
                summary=[],
                type="reasoning",
                status=terminal_item_status,
                content=[ResponseReasoningTextContent(type="reasoning_text", text=reasoning_text)],
            )
        )

    if assistant_text:
        response_obj.output.append(
            ResponseOutputMessage(
                id=f"msg_{uuid.uuid4().hex}",
                type="message",
                role="assistant",
                status=terminal_item_status,
                phase="commentary" if tool_calls_payload else "final_answer",
                content=[_new_output_text(assistant_text)],
            )
        )

    for index, tool_call in enumerate(tool_calls_payload):
        function = tool_call.get("function") or {}
        call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex}"
        arguments = _normalize_function_arguments(
            function.get("arguments", ""),
            call_id=call_id,
            tool_name=function.get("name"),
        )
        tool_calls[index] = StreamToolState(
            id=call_id,
            name=function.get("name", ""),
            arguments=arguments,
        )
        response_obj.output.append(
            ResponseFunctionToolCall(
                id=call_id,
                call_id=call_id,
                name=function.get("name", ""),
                arguments=arguments,
                type="function_call",
                status="completed",
            )
        )

    normalized_usage = _normalize_response_usage(upstream_data.get("usage"))
    if normalized_usage is not None:
        response_obj.usage = normalized_usage
    persisted_history = history_messages if history_messages is not None else chat_request["messages"]
    if finish_reason == "length" and not tool_calls:
        _finalize_incomplete_text_response(
            response_obj,
            persisted_history,
            assistant_text,
            settings,
            reason="max_output_tokens",
        )
    elif tool_calls:
        _finalize_tool_response(response_obj, persisted_history, tool_calls, settings, assistant_text=assistant_text)
    else:
        _finalize_text_response(response_obj, persisted_history, assistant_text, settings)

    return response_obj


def _reset_stream_turn_state(state: StreamResponseState) -> None:
    state.message_id = f"msg_{uuid.uuid4().hex}"
    state.assistant_output_index = None
    state.assistant_text = ""
    state.reasoning_output_index = None
    state.reasoning_id = None
    state.reasoning_text = ""
    state.tool_calls = {}
    state.usage = None


async def stream_chat_completion_response(
    client: httpx.AsyncClient,
    upstream_response: httpx.Response,
    request_data: dict[str, Any],
    chat_request: dict[str, Any],
    settings: ProxySettings,
    headers: dict[str, str],
) -> AsyncIterator[str]:
    state = StreamResponseState(
        response=build_response_shell(request_data, f"resp_{uuid.uuid4().hex}"),
        message_id=f"msg_{uuid.uuid4().hex}",
    )
    completed = False
    retry_count = 0
    current_stream = upstream_response
    current_chat_request = copy.deepcopy(chat_request)
    history_messages = copy.deepcopy(chat_request["messages"])
    cumulative_usage: ResponseUsage | None = None

    yield _sse(
        ResponseCreatedEvent(
            type="response.created",
            response=state.response,
            sequence_number=_next_sequence(state),
        )
    )
    yield _sse(
        ResponseInProgressEvent(
            type="response.in_progress",
            response=state.response,
            sequence_number=_next_sequence(state),
        )
    )

    try:
        while True:
            restart_turn = False
            async for raw_line in current_stream.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue

                if line in {"[DONE]", "data: [DONE]"}:
                    if not completed:
                        for event in _emit_final_reasoning_events(state):
                            yield event
                        if state.tool_calls:
                            for event in _emit_final_assistant_events(
                                state,
                                require_item=bool(state.assistant_text),
                                phase="commentary",
                            ):
                                yield event
                            state.response.usage = _merge_response_usage(cumulative_usage, state.usage)
                            _finalize_tool_response(
                                state.response,
                                history_messages,
                                state.tool_calls,
                                settings,
                                assistant_text=state.assistant_text,
                            )
                            for event in _emit_final_tool_events(state):
                                yield event
                        else:
                            for event in _emit_final_assistant_events(
                                state,
                                require_item=bool(state.assistant_text),
                                phase="final_answer",
                            ):
                                yield event
                            state.response.usage = _merge_response_usage(cumulative_usage, state.usage)
                            _finalize_text_response(state.response, history_messages, state.assistant_text, settings)
                        yield _sse(
                            ResponseCompletedEvent(
                                type="response.completed",
                                response=state.response,
                                sequence_number=_next_sequence(state),
                            )
                        )
                    completed = True
                    break

                if line.startswith("data: "):
                    line = line[6:]

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    LOG.warning("skipping non-json upstream chunk: %s", line[:200])
                    continue

                if data.get("model"):
                    state.response.model = data["model"]
                normalized_usage = _normalize_response_usage(data.get("usage"))
                if normalized_usage is not None:
                    state.usage = normalized_usage

                choices = data.get("choices") or []
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta") or {}

                reasoning_delta = _extract_reasoning_delta(delta)
                if reasoning_delta:
                    for event in _emit_reasoning_open_events(state):
                        yield event
                    state.reasoning_text += reasoning_delta
                    output_item = state.response.output[state.reasoning_output_index]
                    if isinstance(output_item, ResponseReasoningItem) and output_item.content:
                        output_item.content[0].text = state.reasoning_text
                    yield _sse(
                        ResponseReasoningTextDeltaEvent(
                            type="response.reasoning_text.delta",
                            item_id=state.reasoning_id or "",
                            output_index=state.reasoning_output_index or 0,
                            content_index=0,
                            delta=reasoning_delta,
                            sequence_number=_next_sequence(state),
                        )
                    )

                content_delta = delta.get("content")
                if isinstance(content_delta, str) and content_delta:
                    for event in _emit_assistant_message_open_events(state):
                        yield event
                    state.assistant_text += content_delta
                    output_item = state.response.output[state.assistant_output_index]
                    if isinstance(output_item, ResponseOutputMessage):
                        output_item.content[0].text = state.assistant_text
                    yield _sse(
                        ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            item_id=state.message_id,
                            output_index=state.assistant_output_index or 0,
                            content_index=0,
                            delta=content_delta,
                            logprobs=[],
                            sequence_number=_next_sequence(state),
                        )
                    )

                function_call_delta = delta.get("function_call")
                if isinstance(function_call_delta, dict):
                    delta.setdefault(
                        "tool_calls",
                        [{"index": 0, "id": None, "function": function_call_delta}],
                    )

                for tool_delta in delta.get("tool_calls") or []:
                    index = tool_delta.get("index", 0)
                    if not isinstance(index, int):
                        index = 0

                    tool_state, output_item, created = _ensure_tool_output_item(state, index)

                    if tool_delta.get("id"):
                        tool_state.id = str(tool_delta["id"])
                        output_item.id = tool_state.id
                        output_item.call_id = tool_state.id

                    function = tool_delta.get("function") or {}
                    if function.get("name"):
                        tool_state.name = str(function["name"])
                        output_item.name = tool_state.name

                    if created:
                        yield _sse(
                            ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                item=output_item,
                                output_index=tool_state.output_index,
                                sequence_number=_next_sequence(state),
                            )
                        )

                    arg_fragment = function.get("arguments")
                    if isinstance(arg_fragment, str) and arg_fragment:
                        tool_state.arguments += arg_fragment
                        output_item.arguments = tool_state.arguments
                        yield _sse(
                            ResponseFunctionCallArgumentsDeltaEvent(
                                type="response.function_call_arguments.delta",
                                item_id=output_item.id or tool_state.id,
                                output_index=tool_state.output_index,
                                delta=arg_fragment,
                                sequence_number=_next_sequence(state),
                            )
                        )

                finish_reason = choice.get("finish_reason")
                if finish_reason == "tool_calls":
                    for event in _emit_final_reasoning_events(state):
                        yield event
                    for event in _emit_final_assistant_events(
                        state,
                        require_item=bool(state.assistant_text),
                        phase="commentary",
                    ):
                        yield event
                    for event in _emit_final_tool_events(state):
                        yield event
                    state.response.usage = _merge_response_usage(cumulative_usage, state.usage)
                    _finalize_tool_response(
                        state.response,
                        history_messages,
                        state.tool_calls,
                        settings,
                        assistant_text=state.assistant_text,
                    )
                    yield _sse(
                        ResponseCompletedEvent(
                            type="response.completed",
                            response=state.response,
                            sequence_number=_next_sequence(state),
                        )
                    )
                    completed = True
                    break

                if finish_reason == "stop":
                    if _should_auto_continue(
                        finish_reason=finish_reason,
                        assistant_text=state.assistant_text,
                        tool_calls_payload=[],
                        retry_count=retry_count,
                    ):
                        LOG.warning("detected probable premature final answer during stream; retrying upstream turn")
                        cumulative_usage = _merge_response_usage(cumulative_usage, state.usage)
                        for event in _emit_final_reasoning_events(state):
                            yield event
                        for event in _emit_final_assistant_events(
                            state,
                            require_item=bool(state.assistant_text),
                            phase="commentary",
                        ):
                            yield event
                        if state.assistant_text:
                            history_messages.append({"role": "assistant", "content": state.assistant_text})
                        await current_stream.aclose()
                        retry_count += 1
                        current_chat_request = _build_auto_continue_chat_request(chat_request, history_messages)
                        followup_request = client.build_request(
                            "POST",
                            "/v1/chat/completions",
                            json=current_chat_request,
                            headers=headers,
                        )
                        current_stream = await client.send(followup_request, stream=True)
                        if current_stream.status_code != 200:
                            payload = await current_stream.aread()
                            _mark_response_failed(
                                state.response,
                                f"auto-continue upstream error: {payload.decode(errors='replace')[:400]}",
                            )
                            yield _sse(
                                ResponseFailedEvent(
                                    type="response.failed",
                                    response=state.response,
                                    sequence_number=_next_sequence(state),
                                )
                            )
                            completed = True
                            break
                        _reset_stream_turn_state(state)
                        restart_turn = True
                        break

                    for event in _emit_final_reasoning_events(state):
                        yield event
                    for event in _emit_final_assistant_events(
                        state,
                        require_item=bool(state.assistant_text),
                        phase="final_answer",
                    ):
                        yield event
                    state.response.usage = _merge_response_usage(cumulative_usage, state.usage)
                    _finalize_text_response(state.response, history_messages, state.assistant_text, settings)
                    yield _sse(
                        ResponseCompletedEvent(
                            type="response.completed",
                            response=state.response,
                            sequence_number=_next_sequence(state),
                        )
                    )
                    completed = True
                    break

                if finish_reason == "length":
                    for event in _emit_final_reasoning_events(state, status="incomplete"):
                        yield event
                    for event in _emit_final_assistant_events(
                        state,
                        require_item=bool(state.assistant_text),
                        phase="final_answer",
                        status="incomplete",
                    ):
                        yield event
                    state.response.usage = _merge_response_usage(cumulative_usage, state.usage)
                    _finalize_incomplete_text_response(
                        state.response,
                        history_messages,
                        state.assistant_text,
                        settings,
                        reason="max_output_tokens",
                    )
                    yield _sse(
                        ResponseIncompleteEvent(
                            type="response.incomplete",
                            response=state.response,
                            sequence_number=_next_sequence(state),
                        )
                    )
                    completed = True
                    break
            if completed:
                break
            if restart_turn:
                continue
            break
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        LOG.exception("stream translation failed")
        _mark_response_failed(state.response, str(exc))
        yield _sse(
            ResponseFailedEvent(
                type="response.failed",
                response=state.response,
                sequence_number=_next_sequence(state),
            )
        )
    finally:
        await current_stream.aclose()


def _forward_headers(request: Request) -> dict[str, str]:
    excluded = {"host", "content-length", "connection", "accept-encoding"}
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in excluded
    }


def _upstream_error_response(exc: httpx.HTTPError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"error": {"message": f"upstream request failed: {exc.__class__.__name__}: {exc}"}},
    )


async def translate_nonstream_chat_completion_response(
    client: httpx.AsyncClient,
    request_data: dict[str, Any],
    chat_request: dict[str, Any],
    headers: dict[str, str],
    settings: ProxySettings,
) -> ResponsesResponse:
    history_messages = copy.deepcopy(chat_request["messages"])
    prefix_output_items: list[ResponseOutputItem] = []
    cumulative_usage: ResponseUsage | None = None
    current_chat_request = copy.deepcopy(chat_request)
    retry_count = 0

    while True:
        upstream = await client.post(
            "/v1/chat/completions",
            json=current_chat_request,
            headers=headers,
        )
        if upstream.status_code != 200:
            raise HTTPException(
                status_code=upstream.status_code,
                detail=upstream.text if hasattr(upstream, "text") else upstream.content.decode(errors="replace"),
            )

        upstream_data = upstream.json()
        choice = (upstream_data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        assistant_text = message.get("content") if isinstance(message.get("content"), str) else _content_to_text(message.get("content"))
        tool_calls_payload = message.get("tool_calls") or []
        if _should_auto_continue(
            finish_reason=choice.get("finish_reason"),
            assistant_text=assistant_text,
            tool_calls_payload=tool_calls_payload,
            retry_count=retry_count,
        ):
            LOG.warning("detected probable premature final answer in non-stream response; retrying upstream turn")
            cumulative_usage = _merge_response_usage(
                cumulative_usage,
                _normalize_response_usage(upstream_data.get("usage")),
            )
            if assistant_text:
                prefix_output_items.append(_build_commentary_output_message(assistant_text))
                history_messages.append({"role": "assistant", "content": assistant_text})
            current_chat_request = _build_auto_continue_chat_request(chat_request, history_messages)
            retry_count += 1
            continue
        break

    response_obj = convert_chat_completion_response(
        request_data,
        current_chat_request,
        upstream_data,
        settings,
        history_messages=history_messages,
    )
    if prefix_output_items:
        response_obj.output = prefix_output_items + response_obj.output
    if cumulative_usage is not None:
        response_obj.usage = _merge_response_usage(cumulative_usage, response_obj.usage)
    return response_obj


def create_app(settings: ProxySettings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.client = httpx.AsyncClient(
            base_url=settings.upstream_base_url,
            timeout=settings.request_timeout,
        )
        LOG.info("proxy started with upstream=%s", settings.upstream_base_url)
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="Codex vLLM Proxy", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request) -> FastAPIResponse:
        try:
            upstream = await app.state.client.get("/v1/models", headers=_forward_headers(request))
        except httpx.HTTPError as exc:
            return _upstream_error_response(exc)
        return FastAPIResponse(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    @app.post("/v1/chat/completions")
    async def passthrough_chat_completions(request: Request) -> FastAPIResponse:
        body = await request.body()
        headers = _forward_headers(request)
        try:
            upstream = await app.state.client.post("/v1/chat/completions", content=body, headers=headers)
        except httpx.HTTPError as exc:
            return _upstream_error_response(exc)
        return FastAPIResponse(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    @app.post("/v1/responses")
    @app.post("/responses")
    async def create_response(request: Request) -> FastAPIResponse:
        try:
            request_data = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc

        chat_request = convert_responses_request(request_data)
        headers = _forward_headers(request)

        if request_data.get("stream"):
            upstream_request = app.state.client.build_request(
                "POST",
                "/v1/chat/completions",
                json=chat_request,
                headers=headers,
            )
            try:
                upstream_stream = await app.state.client.send(upstream_request, stream=True)
            except httpx.HTTPError as exc:
                return _upstream_error_response(exc)

            if upstream_stream.status_code != 200:
                payload = await upstream_stream.aread()
                await upstream_stream.aclose()
                return FastAPIResponse(
                    content=payload,
                    status_code=upstream_stream.status_code,
                    media_type=upstream_stream.headers.get("content-type", "application/json"),
                )

            async def stream() -> AsyncIterator[str]:
                try:
                    async for event in stream_chat_completion_response(
                        app.state.client,
                        upstream_stream,
                        request_data,
                        chat_request,
                        settings,
                        headers,
                    ):
                        yield event
                except asyncio.CancelledError:
                    LOG.info("stream cancelled by downstream client")
                except httpx.HTTPError as exc:
                    LOG.info("stream ended early: %s: %s", exc.__class__.__name__, exc)

            return StreamingResponse(stream(), media_type="text/event-stream")

        try:
            response_obj = await translate_nonstream_chat_completion_response(
                app.state.client,
                request_data,
                chat_request,
                headers,
                settings,
            )
        except httpx.HTTPError as exc:
            return _upstream_error_response(exc)
        except HTTPException as exc:
            return FastAPIResponse(
                content=_json_dumps_compact({"error": {"message": str(exc.detail)}}),
                status_code=exc.status_code,
                media_type="application/json",
            )
        return JSONResponse(_response_json(response_obj))

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight Codex-to-vLLM Responses proxy")
    parser.add_argument("--host", default=os.getenv("CODEX_VLLM_PROXY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CODEX_VLLM_PROXY_PORT", "7740")))
    parser.add_argument("--upstream", default=os.getenv("CODEX_VLLM_PROXY_UPSTREAM", "http://127.0.0.1:8000"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("CODEX_VLLM_PROXY_TIMEOUT", "600")))
    parser.add_argument("--max-history", type=int, default=int(os.getenv("CODEX_VLLM_PROXY_MAX_HISTORY", "200")))
    parser.add_argument("--log-level", default=os.getenv("CODEX_VLLM_PROXY_LOG_LEVEL", "info"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = load_settings(
        upstream_base_url=args.upstream,
        request_timeout=args.timeout,
        max_conversation_history=args.max_history,
    )
    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
