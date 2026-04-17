from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "codex_vllm_proxy.py"
SPEC = importlib.util.spec_from_file_location("codex_vllm_proxy", MODULE_PATH)
proxy = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = proxy
SPEC.loader.exec_module(proxy)


class FakeUpstreamResponse:
    def __init__(
        self,
        lines: list[str],
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> None:
        self._lines = lines
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/event-stream"}
        self._body = body or b""
        self.closed = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self) -> None:
        self.closed = True

    async def aread(self) -> bytes:
        return self._body


class FakeJSONResponse:
    def __init__(self, payload: dict, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)
        self.content = self.text.encode("utf-8")

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    def __init__(
        self,
        *,
        post_responses: list[FakeJSONResponse] | None = None,
        send_responses: list[FakeUpstreamResponse] | None = None,
    ) -> None:
        self.post_responses = list(post_responses or [])
        self.send_responses = list(send_responses or [])
        self.post_calls: list[dict] = []
        self.send_calls: list[dict] = []

    async def post(self, path: str, json: dict | None = None, headers: dict | None = None):
        self.post_calls.append({"path": path, "json": json, "headers": headers})
        return self.post_responses.pop(0)

    def build_request(self, method: str, path: str, json: dict | None = None, headers: dict | None = None):
        return {"method": method, "path": path, "json": json, "headers": headers}

    async def send(self, request: dict, stream: bool = False):
        self.send_calls.append({"request": request, "stream": stream})
        return self.send_responses.pop(0)


class CodexVllmProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        proxy.conversation_history.clear()
        proxy.RECENT_WARNING_KEYS.clear()
        self.settings = proxy.load_settings(
            upstream_base_url="http://127.0.0.1:7730",
            request_timeout=60,
            max_conversation_history=8,
        )

    def test_convert_request_promotes_developer_and_skips_reasoning(self) -> None:
        request_data = {
            "model": "Qwen/test",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Follow repo policy."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "read queue-item.json"}],
                },
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Need to inspect the queue item."}],
                },
                {"type": "message", "role": "assistant", "content": []},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                },
            ],
        }

        chat_request = proxy.convert_responses_request(request_data)
        self.assertEqual(chat_request["messages"][0], {"role": "system", "content": "Follow repo policy."})
        self.assertEqual(chat_request["messages"][1], {"role": "user", "content": "read queue-item.json"})
        self.assertEqual(chat_request["messages"][2]["role"], "assistant")
        self.assertEqual(chat_request["messages"][2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(chat_request["messages"][3], {"role": "tool", "tool_call_id": "call_1", "content": "ok"})

    def test_previous_response_id_reuses_existing_tool_call(self) -> None:
        proxy.conversation_history["resp_prev"] = [
            {"role": "system", "content": "Follow repo policy."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_existing",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": '{"cmd":"pwd"}'},
                    }
                ],
            },
        ]

        request_data = {
            "model": "Qwen/test",
            "previous_response_id": "resp_prev",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_existing",
                    "output": "done",
                }
            ],
        }

        chat_request = proxy.convert_responses_request(request_data)
        self.assertEqual(len(chat_request["messages"]), 3)
        self.assertEqual(chat_request["messages"][2], {"role": "tool", "tool_call_id": "call_existing", "content": "done"})

    def test_convert_request_normalizes_concatenated_function_arguments(self) -> None:
        request_data = {
            "model": "Qwen/test",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run pwd"}]},
                {
                    "type": "function_call",
                    "call_id": "call_dup",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}{"cmd":"pwd"}',
                },
            ],
        }

        chat_request = proxy.convert_responses_request(request_data)
        self.assertEqual(
            chat_request["messages"][1]["tool_calls"][0]["function"]["arguments"],
            '{"cmd":"pwd"}',
        )

    def test_convert_request_normalizes_history_tool_arguments(self) -> None:
        proxy.conversation_history["resp_prev"] = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_existing",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"pwd"}{"cmd":"pwd"}',
                        },
                    }
                ],
            },
        ]

        chat_request = proxy.convert_responses_request(
            {
                "model": "Qwen/test",
                "previous_response_id": "resp_prev",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "next"}]}],
            }
        )

        self.assertEqual(
            chat_request["messages"][0]["tool_calls"][0]["function"]["arguments"],
            '{"cmd":"pwd"}',
        )

    def test_normalize_input_items_coerces_output_text_messages_and_reasoning(self) -> None:
        raw_input = [
            {
                "type": "message",
                "role": "assistant",
                "id": "msg_existing",
                "status": "completed",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "Prior answer."}],
            },
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "Need a plan."}],
            },
        ]

        with self.assertNoLogs(proxy.LOG, level="WARNING"):
            normalized = proxy._normalize_input_items(raw_input)

        self.assertEqual(
            normalized[0],
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "input_text", "text": "Prior answer."}],
            },
        )
        self.assertEqual(normalized[1]["type"], "reasoning")
        self.assertTrue(normalized[1]["id"].startswith("rs_"))
        self.assertEqual(
            normalized[1]["content"],
            [{"type": "reasoning_text", "text": "Need a plan."}],
        )

    def test_convert_request_accepts_output_text_assistant_messages_without_warning(self) -> None:
        request_data = {
            "model": "Qwen/test",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Already checked the queue."}],
                },
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Need to continue."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue"}],
                },
            ],
        }

        with self.assertNoLogs(proxy.LOG, level="WARNING"):
            chat_request = proxy.convert_responses_request(request_data)

        self.assertEqual(
            chat_request["messages"],
            [
                {"role": "assistant", "content": "Already checked the queue."},
                {"role": "user", "content": "continue"},
            ],
        )

    def test_premature_final_detection_matches_transition_sentences(self) -> None:
        self.assertTrue(proxy._looks_like_premature_final_answer(""))
        self.assertTrue(proxy._looks_like_premature_final_answer("现在让我对比一下本地文件系统和数据库中的迁移，看看是否有未应用的："))
        self.assertTrue(proxy._looks_like_premature_final_answer("Let me check that."))
        self.assertFalse(proxy._looks_like_premature_final_answer("已经确认没有未应用的迁移。"))

    def test_convert_chat_completion_response_stores_history(self) -> None:
        request_data = {"model": "Qwen/test", "stream": False}
        chat_request = {"model": "Qwen/test", "messages": [{"role": "user", "content": "hello"}]}
        upstream_data = {
            "model": "Qwen/test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'll call a tool.",
                        "tool_calls": [
                            {
                                "id": "call_tool",
                                "type": "function",
                                "function": {"name": "exec_command", "arguments": '{"cmd":"pwd"}'},
                            }
                        ],
                    }
                }
            ],
        }

        response_obj = proxy.convert_chat_completion_response(
            request_data,
            chat_request,
            upstream_data,
            self.settings,
        )

        self.assertEqual(response_obj.status, "completed")
        self.assertEqual(response_obj.output[0].type, "message")
        self.assertEqual(response_obj.output[0].phase, "commentary")
        self.assertEqual(response_obj.output[1].type, "function_call")
        self.assertEqual(response_obj.output[1].status, "completed")
        stored_messages = proxy.conversation_history[response_obj.id]
        self.assertEqual(stored_messages[-1]["role"], "assistant")
        self.assertEqual(stored_messages[-1]["tool_calls"][0]["id"], "call_tool")

    def test_convert_chat_completion_response_keeps_single_text_message(self) -> None:
        request_data = {"model": "Qwen/test", "stream": False}
        chat_request = {"model": "Qwen/test", "messages": [{"role": "user", "content": "hello"}]}
        upstream_data = {
            "model": "Qwen/test",
            "choices": [{"message": {"role": "assistant", "content": "是"}}],
        }

        response_obj = proxy.convert_chat_completion_response(
            request_data,
            chat_request,
            upstream_data,
            self.settings,
        )

        self.assertEqual(len(response_obj.output), 1)
        self.assertEqual(response_obj.output[0].type, "message")
        self.assertEqual(response_obj.output[0].status, "completed")
        self.assertEqual(response_obj.output[0].phase, "final_answer")
        self.assertEqual(response_obj.output[0].content[0].text, "是")

    def test_convert_chat_completion_response_marks_length_as_incomplete(self) -> None:
        request_data = {"model": "Qwen/test", "stream": False}
        chat_request = {"model": "Qwen/test", "messages": [{"role": "user", "content": "return json"}]}
        upstream_data = {
            "model": "Qwen/test",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "Need more output tokens.",
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        }

        response_obj = proxy.convert_chat_completion_response(
            request_data,
            chat_request,
            upstream_data,
            self.settings,
        )

        self.assertEqual(response_obj.status, "incomplete")
        self.assertEqual(response_obj.incomplete_details.reason, "max_output_tokens")
        self.assertEqual(len(response_obj.output), 1)
        self.assertEqual(response_obj.output[0].type, "reasoning")
        self.assertEqual(response_obj.output[0].status, "incomplete")
        self.assertEqual(proxy.conversation_history[response_obj.id], chat_request["messages"])

    def test_translate_nonstream_auto_continues_probable_premature_final(self) -> None:
        request_data = {"model": "Qwen/test", "stream": False}
        chat_request = {"model": "Qwen/test", "messages": [{"role": "user", "content": "check migrations"}]}
        client = FakeAsyncClient(
            post_responses=[
                FakeJSONResponse(
                    {
                        "model": "Qwen/test",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": "现在让我对比一下本地文件系统和数据库中的迁移，看看是否有未应用的：",
                                },
                            }
                        ],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                    }
                ),
                FakeJSONResponse(
                    {
                        "model": "Qwen/test",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": "已经确认没有未应用的迁移。",
                                },
                            }
                        ],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
                    }
                ),
            ]
        )

        response_obj = asyncio.run(
            proxy.translate_nonstream_chat_completion_response(
                client,
                request_data,
                chat_request,
                headers={"authorization": "Bearer local-demo"},
                settings=self.settings,
            )
        )

        self.assertEqual(len(client.post_calls), 2)
        self.assertEqual(client.post_calls[1]["json"]["messages"][-1]["role"], "user")
        self.assertEqual(client.post_calls[1]["json"]["messages"][-1]["content"], proxy.AUTO_CONTINUE_PROMPT)
        self.assertEqual(response_obj.status, "completed")
        self.assertEqual(len(response_obj.output), 2)
        self.assertEqual(response_obj.output[0].phase, "commentary")
        self.assertEqual(
            response_obj.output[0].content[0].text,
            "现在让我对比一下本地文件系统和数据库中的迁移，看看是否有未应用的：",
        )
        self.assertEqual(response_obj.output[1].phase, "final_answer")
        self.assertEqual(response_obj.output[1].content[0].text, "已经确认没有未应用的迁移。")
        self.assertEqual(response_obj.usage.total_tokens, 14)
        stored_messages = proxy.conversation_history[response_obj.id]
        self.assertEqual(stored_messages[-2]["content"], "现在让我对比一下本地文件系统和数据库中的迁移，看看是否有未应用的：")
        self.assertEqual(stored_messages[-1]["content"], "已经确认没有未应用的迁移。")

    def test_stream_text_response_emits_official_message_events(self) -> None:
        request_data = {"model": "Qwen/test", "stream": True}
        chat_request = {"model": "Qwen/test", "messages": [{"role": "user", "content": "hello"}]}
        upstream_response = FakeUpstreamResponse(
            [
                'data: {"model":"Qwen/test","choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}',
                'data: {"model":"Qwen/test","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}',
                "data: [DONE]",
            ]
        )

        events = asyncio.run(
            self._collect_stream_events(
                upstream_response,
                request_data,
                chat_request,
            )
        )

        self.assertEqual(
            [event["type"] for event in events],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual([event["sequence_number"] for event in events], list(range(len(events))))
        self.assertEqual(events[2]["item"]["type"], "message")
        self.assertEqual(events[6]["text"], "Hello")
        self.assertEqual(events[-1]["response"]["output"][0]["content"][0]["text"], "Hello")
        self.assertEqual(events[-1]["response"]["output"][0]["phase"], "final_answer")
        self.assertEqual(events[-1]["response"]["usage"]["total_tokens"], 3)

    def test_stream_tool_call_response_emits_typed_tool_events(self) -> None:
        request_data = {"model": "Qwen/test", "stream": True}
        chat_request = {"model": "Qwen/test", "messages": [{"role": "user", "content": "pwd"}]}
        upstream_response = FakeUpstreamResponse(
            [
                'data: {"model":"Qwen/test","choices":[{"delta":{"content":"Checking..."},"finish_reason":null}]}',
                'data: {"model":"Qwen/test","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"exec_command","arguments":"{\\"cmd\\":\\"pw"}}]},"finish_reason":null}]}',
                'data: {"model":"Qwen/test","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"d\\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":4,"total_tokens":5}}',
            ]
        )

        events = asyncio.run(
            self._collect_stream_events(
                upstream_response,
                request_data,
                chat_request,
            )
        )

        self.assertIn("response.function_call_arguments.done", [event["type"] for event in events])
        tool_added_event = next(event for event in events if event["type"] == "response.output_item.added" and event["item"]["type"] == "function_call")
        self.assertEqual(tool_added_event["output_index"], 1)
        tool_done_event = next(event for event in events if event["type"] == "response.function_call_arguments.done")
        self.assertEqual(tool_done_event["item_id"], "call_1")
        self.assertEqual(tool_done_event["name"], "exec_command")
        self.assertEqual(tool_done_event["arguments"], '{"cmd":"pwd"}')
        completed = events[-1]["response"]
        self.assertEqual(completed["output"][0]["phase"], "commentary")
        self.assertEqual(completed["output"][1]["type"], "function_call")
        self.assertEqual(completed["output"][1]["status"], "completed")

    def test_stream_reasoning_response_emits_reasoning_events(self) -> None:
        request_data = {"model": "Qwen/test", "stream": True}
        chat_request = {"model": "Qwen/test", "messages": [{"role": "user", "content": "think"}]}
        upstream_response = FakeUpstreamResponse(
            [
                'data: {"model":"Qwen/test","choices":[{"delta":{"reasoning":"Need a quick plan. ","content":"Done."},"finish_reason":"stop"}]}',
            ]
        )

        events = asyncio.run(
            self._collect_stream_events(
                upstream_response,
                request_data,
                chat_request,
            )
        )

        self.assertIn("response.reasoning_text.delta", [event["type"] for event in events])
        self.assertIn("response.reasoning_text.done", [event["type"] for event in events])
        reasoning_added = next(event for event in events if event["type"] == "response.output_item.added" and event["item"]["type"] == "reasoning")
        self.assertEqual(reasoning_added["output_index"], 0)
        completed = events[-1]["response"]
        self.assertEqual(completed["output"][0]["type"], "reasoning")
        self.assertEqual(completed["output"][1]["type"], "message")

    def test_stream_length_response_emits_incomplete_event(self) -> None:
        request_data = {"model": "Qwen/test", "stream": True}
        chat_request = {"model": "Qwen/test", "messages": [{"role": "user", "content": "return json"}]}
        upstream_response = FakeUpstreamResponse(
            [
                'data: {"model":"Qwen/test","choices":[{"delta":{"reasoning":"Need more output."},"finish_reason":"length"}],"usage":{"prompt_tokens":2,"completion_tokens":4,"total_tokens":6}}',
            ]
        )

        events = asyncio.run(
            self._collect_stream_events(
                upstream_response,
                request_data,
                chat_request,
            )
        )

        self.assertEqual(events[-1]["type"], "response.incomplete")
        self.assertEqual(events[-1]["response"]["status"], "incomplete")
        self.assertEqual(events[-1]["response"]["incomplete_details"]["reason"], "max_output_tokens")
        self.assertEqual(events[-1]["response"]["output"][0]["type"], "reasoning")
        self.assertNotIn("response.completed", [event["type"] for event in events])
        self.assertEqual(proxy.conversation_history[events[-1]["response"]["id"]], chat_request["messages"])

    def test_stream_auto_continues_probable_premature_final(self) -> None:
        request_data = {"model": "Qwen/test", "stream": True}
        chat_request = {"model": "Qwen/test", "messages": [{"role": "user", "content": "check migrations"}]}
        upstream_response = FakeUpstreamResponse(
            [
                'data: {"model":"Qwen/test","choices":[{"delta":{"content":"现在让我对比一下本地文件系统和数据库中的迁移，看看是否有未应用的："},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}',
            ]
        )
        client = FakeAsyncClient(
            send_responses=[
                FakeUpstreamResponse(
                    [
                        'data: {"model":"Qwen/test","choices":[{"delta":{"content":"已经确认没有未应用的迁移。"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":5,"total_tokens":9}}',
                    ]
                )
            ]
        )

        events = asyncio.run(
            self._collect_stream_events(
                upstream_response,
                request_data,
                chat_request,
                client=client,
            )
        )

        self.assertEqual(len(client.send_calls), 1)
        self.assertTrue(client.send_calls[0]["stream"])
        self.assertEqual(client.send_calls[0]["request"]["json"]["messages"][-1]["content"], proxy.AUTO_CONTINUE_PROMPT)
        completed = events[-1]["response"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(len(completed["output"]), 2)
        self.assertEqual(completed["output"][0]["phase"], "commentary")
        self.assertEqual(completed["output"][0]["content"][0]["text"], "现在让我对比一下本地文件系统和数据库中的迁移，看看是否有未应用的：")
        self.assertEqual(completed["output"][1]["phase"], "final_answer")
        self.assertEqual(completed["output"][1]["content"][0]["text"], "已经确认没有未应用的迁移。")
        self.assertNotEqual(completed["output"][0]["id"], completed["output"][1]["id"])
        self.assertEqual(completed["usage"]["total_tokens"], 14)
        stored_messages = proxy.conversation_history[completed["id"]]
        self.assertEqual(stored_messages[-2]["content"], "现在让我对比一下本地文件系统和数据库中的迁移，看看是否有未应用的：")
        self.assertEqual(stored_messages[-1]["content"], "已经确认没有未应用的迁移。")

    async def _collect_stream_events(
        self,
        upstream_response: FakeUpstreamResponse,
        request_data: dict,
        chat_request: dict,
        *,
        client: FakeAsyncClient | None = None,
    ) -> list[dict]:
        events: list[dict] = []
        async for chunk in proxy.stream_chat_completion_response(
            client or FakeAsyncClient(),
            upstream_response,
            request_data,
            chat_request,
            self.settings,
            {},
        ):
            self.assertTrue(chunk.startswith("data: "))
            events.append(json.loads(chunk[6:].strip()))
        return events


if __name__ == "__main__":
    unittest.main()
