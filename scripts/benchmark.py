"""Measure real SSE latency/throughput with fixed output length and cold request prefixes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import statistics
import subprocess
import time
import urllib.request

from download_model import ROOT, read_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()
    env = read_env()
    base = "http://127.0.0.1:" + env.get("VLLM_PORT", "7730")
    headers = {"Content-Type": "application/json"}
    if env.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + env["VLLM_API_KEY"]
    model = env.get("SERVED_MODEL_NAME", "nv-community/Qwen3.8-Flash-Next-NVFP4")
    short = (
        "Write a detailed Python implementation of a bounded asynchronous job queue with "
        "retry, cancellation, per-job timeout and graceful shutdown. Explain the invariants "
        "and demonstrate usage with examples."
    )
    records = "\n".join(
        f"Record {i:04d}: region=R{i % 11}; output={1000 + i * 17}; energy={200 + i % 97}; "
        f"water={400 + i % 83}; status={'review' if i % 7 == 0 else 'accepted'}."
        for i in range(384)
    )
    long_prompt = (
        records + "\n用中文详细说明这些记录应如何验证数据质量，给出具体核对规则与实施步骤。"
    )
    cases = [
        ("short_single", short, 1),
        ("long_single", long_prompt, 1),
        ("short_concurrent4", short, 4),
    ]
    evidence = {
        "label": args.label,
        "model": model,
        "output_tokens": args.tokens,
        "repeats": args.repeats,
        "method": "SSE; temperature=0; thinking off; ignore_eos; distinct leading request IDs; per-case warmup excluded",
        "cases": {},
    }
    container = json.loads(subprocess.check_output(["docker", "inspect", "tiangong-vllm-model-1"]))[
        0
    ]
    evidence["runtime"] = {
        "container_id": container["Id"],
        "image_id": container["Image"],
        "image_reference": container["Config"]["Image"],
        "command": container["Config"]["Cmd"],
        "environment": {
            key: value
            for item in container["Config"]["Env"]
            for key, value in [item.split("=", 1)]
            if key
            in {"SPECULATIVE_CONFIG", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "OMP_NUM_THREADS"}
        },
    }

    def metrics():
        with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
            return response.read().decode()

    def generate(prompt: str, request_id: int):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": f"Request {request_id:08d}. Be precise and thorough.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": args.tokens,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            base + "/v1/chat/completions", headers=headers, data=json.dumps(payload).encode()
        )
        start = time.monotonic()
        first = None
        content = ""
        usage = None
        done = False
        finish = None
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    break
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    text = choice.get("delta", {}).get("content") or ""
                    if text and first is None:
                        first = time.monotonic()
                    content += text
                    finish = choice.get("finish_reason") or finish
        elapsed = time.monotonic() - start
        assert done and first is not None and usage is not None, "Incomplete SSE response"
        assert usage["completion_tokens"] == args.tokens and finish == "length"
        return {
            "request_id": request_id,
            "ttft_seconds": first - start,
            "elapsed_seconds": elapsed,
            "output_tokens_per_second": args.tokens / elapsed,
            "usage": usage,
            "finish_reason": finish,
            "content": content,
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "output/benchmarks" / (stamp + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence["metrics_before"] = metrics()
    try:
        for case_index, (name, prompt, concurrency) in enumerate(cases):
            # Warm all lanes and shape-specific kernels before timing.
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(
                    pool.map(lambda i: generate(prompt, case_index * 1000 + i), range(concurrency))
                )
                groups = []
                for repeat in range(args.repeats):
                    start = time.monotonic()
                    responses = list(
                        pool.map(
                            lambda i: generate(prompt, case_index * 1000 + (repeat + 1) * 10 + i),
                            range(concurrency),
                        )
                    )
                    elapsed = time.monotonic() - start
                    groups.append(
                        {
                            "wall_seconds": elapsed,
                            "responses": responses,
                            "aggregate_tokens_per_second": concurrency * args.tokens / elapsed,
                        }
                    )
            rows = [row for group in groups for row in group["responses"]]
            summary = {
                "median_ttft_seconds": statistics.median(row["ttft_seconds"] for row in rows),
                "median_latency_seconds": statistics.median(row["elapsed_seconds"] for row in rows),
                "median_aggregate_tokens_per_second": statistics.median(
                    group["aggregate_tokens_per_second"] for group in groups
                ),
                "prompt_tokens": [row["usage"]["prompt_tokens"] for row in rows],
            }
            evidence["cases"][name] = {
                "concurrency": concurrency,
                "prompt": prompt,
                "summary": summary,
                "groups": groups,
            }
            print(name, json.dumps(summary), flush=True)
        current = json.loads(
            subprocess.check_output(["docker", "inspect", "tiangong-vllm-model-1"])
        )[0]
        assert current["Id"] == container["Id"], "Container changed during measurement"
        evidence["status"] = "passed"
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["error"] = str(exc)
        raise
    finally:
        try:
            evidence["metrics_after"] = metrics()
        except Exception as exc:
            evidence["metrics_error"] = str(exc)
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        print("Evidence:", output.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
