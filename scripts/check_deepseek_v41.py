"""Opt-in real DeepSeek API checks and measured synthetic serving workloads."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import struct
import time
import urllib.request
import zlib


def png_data_url() -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    size = 560
    raw = (b"\0" + bytes([255, 0, 0]) * size) * size
    image = b"\x89PNG\r\n\x1a\n"
    image += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    image += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(image).decode()


class Client:
    def __init__(self, args: argparse.Namespace):
        self.base = args.base_url.rstrip("/")
        self.model = args.model
        self.timeout = args.timeout
        self.headers = {"Content-Type": "application/json"}
        if args.key_file:
            self.headers["Authorization"] = "Bearer " + args.key_file.read_text().strip()

    def post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers=self.headers,
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def chat(self, messages: list, **kwargs) -> tuple[dict, float]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"thinking": False},
            **kwargs,
        }
        start = time.monotonic()
        result = self.post("/v1/chat/completions", payload)
        return result, time.monotonic() - start

    def stream(self, messages: list, max_tokens: int) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "chat_template_kwargs": {"thinking": False},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers=self.headers,
        )
        start = time.monotonic()
        first = None
        usage = None
        done = False
        text = []
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=self.timeout) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                part = json.loads(data)
                if part.get("usage"):
                    usage = part["usage"]
                for choice in part.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or delta.get("reasoning")
                    if content:
                        first = first or time.monotonic()
                        text.append(content)
        elapsed = time.monotonic() - start
        assert done and usage and first and text, "Incomplete SSE response"
        tokens = usage["completion_tokens"]
        ttft = first - start
        return {
            "usage": usage,
            "elapsed_s": elapsed,
            "ttft_s": ttft,
            "decode_tokens_per_s": (tokens - 1) / max(elapsed - ttft, 0.001),
            "sse_done": done,
        }


def message(prompt: str) -> list:
    return [{"role": "user", "content": prompt}]


def smoke(client: Client) -> dict:
    results = {}
    for label, prompt, expected in [
        ("math_en", "What is 17 * 19? Return only the integer.", "323"),
        ("math_zh", "请计算 123 加 456，只输出整数结果。", "579"),
    ]:
        response, elapsed = client.chat(message(prompt))
        content = response["choices"][0]["message"].get("content") or ""
        assert expected in content, (label, content)
        results[label] = {"passed": True, "elapsed_s": elapsed, "usage": response["usage"]}
    response, elapsed = client.chat(
        [
            {"role": "system", "content": "Always reply with exactly the word ORCHID."},
            {"role": "user", "content": "Hello"},
        ]
    )
    assert "ORCHID" in (response["choices"][0]["message"].get("content") or "")
    results["system_role"] = {"passed": True, "elapsed_s": elapsed}
    response, elapsed = client.chat(
        message("Call get_weather for Beijing."),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="auto",
    )
    calls = response["choices"][0]["message"].get("tool_calls") or []
    assert calls and calls[0]["function"]["name"] == "get_weather", response
    assert "Beijing" in json.loads(calls[0]["function"]["arguments"])["city"]
    results["tool"] = {"passed": True, "elapsed_s": elapsed}
    response, elapsed = client.chat(
        message("Compute 27 * 38. Explain briefly and give the result."),
        max_tokens=2048,
        chat_template_kwargs={"thinking": True, "reasoning_effort": "low"},
    )
    answer = response["choices"][0]["message"]
    assert "1026" in (answer.get("content") or ""), response
    assert answer.get("reasoning") or answer.get("reasoning_content"), response
    results["reasoning"] = {"passed": True, "elapsed_s": elapsed, "usage": response["usage"]}
    response, elapsed = client.chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the solid color? Answer one English word."},
                    {"type": "image_url", "image_url": {"url": png_data_url()}},
                ],
            }
        ]
    )
    assert "red" in (response["choices"][0]["message"].get("content") or "").lower()
    results["vision"] = {"passed": True, "elapsed_s": elapsed, "usage": response["usage"]}
    results["stream"] = client.stream(message("Count upward from 1, separated by spaces."), 128)
    return results


def long_prompt(client: Client, target: int, benchmark: bool = False) -> tuple[list, int]:
    records = max(target // 20, 1)
    markers = ["ORCHID-739125", "COBALT-284673", "MAPLE-961842"]
    for _ in range(6):
        lines = [
            f"Record {i:07d}: district {(i * 71) % 997}, sensor {(i * 23) % 991}, measured {(i * 37) % 983} units."
            for i in range(records)
        ]
        for fraction, value in zip([0.1, 0.5, 0.9], markers):
            lines[int(records * fraction)] = (
                f"SPECIAL AUDIT CODE: {value}. Preserve this exact code."
            )
        prompt = (
            "Read the following synthetic log. Three SPECIAL AUDIT CODE records are important.\n"
            + "\n".join(lines)
            + "\nReturn all three SPECIAL AUDIT CODE values exactly, in document order."
        )
        if benchmark:
            prompt += (
                " Then write a detailed analysis of at least 1,000 words about how to validate "
                "these synthetic sensor records, detect missing observations, compare districts, "
                "and report measurement uncertainty. Explain concrete procedures in full prose."
            )
        messages = message(prompt)
        tokens = client.post(
            "/tokenize",
            {
                "model": client.model,
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"thinking": False},
            },
        )["count"]
        if target - max(128, int(target * 0.005)) <= tokens <= target:
            return messages, tokens
        records = max(1, int(records * (target - 64) / tokens))
    raise RuntimeError(f"Could not size prompt to {target}; last count={tokens}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Server root, without /v1")
    parser.add_argument("--model", default="deepseek-v41")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--suite", choices=["smoke", "benchmark", "context"], default="smoke")
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = Client(args)
    started = time.time()
    result = {"suite": args.suite, "model": args.model, "started_unix": started}
    if args.suite == "smoke":
        result["checks"] = smoke(client)
    else:
        messages, count = long_prompt(
            client, args.input_tokens, benchmark=args.suite == "benchmark"
        )
        result["tokenized_prompt_tokens"] = count
        if args.suite == "context":
            response, elapsed = client.chat(messages, max_tokens=args.output_tokens)
            answer = response["choices"][0]["message"].get("content") or ""
            expected = ["ORCHID-739125", "COBALT-284673", "MAPLE-961842"]
            positions = [answer.find(value) for value in expected]
            result.update(
                {
                    "elapsed_s": elapsed,
                    "usage": response["usage"],
                    "answer": answer,
                    "recall_passed": all(position >= 0 for position in positions)
                    and positions == sorted(positions),
                    "prompt_count_matched": response["usage"]["prompt_tokens"] == count,
                }
            )
        else:
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                runs = list(
                    pool.map(
                        lambda _: client.stream(messages, args.output_tokens), range(args.requests)
                    )
                )
            elapsed = time.monotonic() - start
            result.update(
                {
                    "concurrency": args.concurrency,
                    "requests": runs,
                    "elapsed_s": elapsed,
                    "aggregate_output_tokens_per_s": sum(
                        r["usage"]["completion_tokens"] for r in runs
                    )
                    / elapsed,
                    "median_ttft_s": statistics.median(r["ttft_s"] for r in runs),
                    "shared_prefix": True,
                }
            )
    result["finished_unix"] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    if result.get("recall_passed") is False:
        raise SystemExit("Context request completed but recall failed")
    if result.get("prompt_count_matched") is False:
        raise SystemExit("Served prompt token count differs from tokenizer count")


if __name__ == "__main__":
    main()
