"""Exercise the real model API; write inspectable responses under output/validation."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import struct
import time
import urllib.error
import urllib.request
import zlib

from download_model import ROOT, read_env


def red_png() -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">2I5B", 128, 128, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 128) * 128))
    data += chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()


def main() -> None:
    env = read_env()
    base = f"http://127.0.0.1:{env.get('VLLM_PORT', '7730')}"
    model = env.get("SERVED_MODEL_NAME", "nv-community/Qwen3.8-Flash-Next-NVFP4")
    api_key = env.get("VLLM_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    evidence: dict = {"model": model, "checks": {}}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "output/validation" / (stamp + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)

    def request(name: str, path: str, payload: dict | None = None, url: str = base):
        start = time.monotonic()
        req = urllib.request.Request(
            url + path,
            headers=headers,
            data=None if payload is None else json.dumps(payload).encode(),
        )
        with urllib.request.urlopen(req, timeout=600) as response:
            raw = response.read().decode()
        value = json.loads(raw) if raw else None
        evidence["checks"][name] = {
            "seconds": round(time.monotonic() - start, 3),
            "response": value,
        }
        return value

    def chat(name: str, messages: list, **extra):
        return request(
            name,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                **extra,
            },
        )

    try:
        request("health", "/health")
        models = request("models", "/v1/models")
        assert model in [m["id"] for m in models["data"]], "Model identity mismatch"
        if api_key:
            try:
                with urllib.request.urlopen(base + "/v1/models", timeout=10):
                    raise AssertionError("Unauthenticated model request was accepted")
            except urllib.error.HTTPError as exc:
                assert exc.code == 401, f"Expected 401, got {exc.code}"
                evidence["checks"]["authentication"] = {"enabled": True, "status": 401}
        else:
            # The successful models request above carries no Authorization header.
            evidence["checks"]["authentication"] = {"enabled": False, "status": 200}
        text = chat("text", [{"role": "user", "content": "计算 17 加 25，只输出数字。"}])
        choice = text["choices"][0]
        assert choice["finish_reason"] == "stop" and "42" in choice["message"]["content"]
        system = chat(
            "system",
            [
                {"role": "system", "content": "Always reply with the single word ORCHID."},
                {"role": "user", "content": "Hello"},
            ],
        )
        assert "ORCHID" in system["choices"][0]["message"]["content"]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply exactly: STREAM_OK"}],
            "max_tokens": 64,
            "temperature": 0,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            base + "/v1/chat/completions", headers=headers, data=json.dumps(payload).encode()
        )
        chunks, content, done, finish = [], "", False, None
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    break
                event = json.loads(line[6:])
                chunks.append(event)
                for item in event.get("choices", []):
                    content += item["delta"].get("content") or ""
                    finish = item.get("finish_reason") or finish
        evidence["checks"]["stream"] = {"content": content, "done": done, "events": chunks}
        assert done and finish == "stop" and "STREAM_OK" in content
        result = chat(
            "tools",
            [
                {
                    "role": "user",
                    "content": "Call get_weather for Beijing. Do not answer without calling the tool.",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
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
        choice = result["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        function = choice["message"]["tool_calls"][0]["function"]
        assert function["name"] == "get_weather"
        assert json.loads(function["arguments"])["city"].lower() in {"beijing", "北京"}
        vision = chat(
            "vision",
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the color of this image? Reply with one English color word.",
                        },
                        {"type": "image_url", "image_url": {"url": red_png()}},
                    ],
                }
            ],
        )
        assert vision["choices"][0]["finish_reason"] == "stop"
        assert "red" in vision["choices"][0]["message"]["content"].lower()
        thinking = chat(
            "reasoning",
            [{"role": "user", "content": "What is 12 times 13?"}],
            max_tokens=1024,
            chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "low"},
        )
        message = thinking["choices"][0]["message"]
        assert thinking["choices"][0]["finish_reason"] == "stop"
        assert "156" in message["content"] and "<think>" not in message["content"]
        assert message.get("reasoning") or message.get("reasoning_content")
        records = "\n".join(
            f"Record {i:04d}: region=R{i % 11}; output={1000 + i * 17}; energy={200 + i % 97}; "
            f"water={400 + i % 83}; status={'review' if i % 7 == 0 else 'accepted'}."
            for i in range(384)
        )
        retrieval = chat(
            "long_retrieval",
            [
                {
                    "role": "user",
                    "content": records
                    + "\n只输出 Record 0017 与 Record 0371 的 output 字段值，按此顺序用英文逗号分隔，不要解释。",
                }
            ],
            max_tokens=64,
        )
        answer = retrieval["choices"][0]
        assert answer["finish_reason"] == "stop"
        assert "".join(answer["message"]["content"].split()) == "1289,7307"
        evidence["status"] = "passed"
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["error"] = str(exc)
        raise
    finally:
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        print(f"Evidence: {output.relative_to(ROOT)}")
    print("Passed: " + ", ".join(evidence["checks"]))


if __name__ == "__main__":
    main()
