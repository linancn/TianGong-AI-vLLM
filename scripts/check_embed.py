"""Verify the live embedding API with a query/document retrieval example."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import time
import urllib.error
import urllib.request

from download_model import ROOT, read_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("embed", "embed3"), default="embed")
    args = parser.parse_args()
    prefix = "EMBED3" if args.profile == "embed3" else "EMBED"
    default_port = "7731"
    env = read_env()
    max_model_len = int(env.get(prefix + "_MAX_MODEL_LEN", "32768"))
    assert 4 < max_model_len <= 32768, "Unsupported Embed max_model_len"
    host = env.get(prefix + "_HOST", "127.0.0.1")
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host:
        host = f"[{host}]"
    base = f"http://{host}:{env.get(prefix + '_PORT', default_port)}"
    model = env.get(prefix + "_SERVED_MODEL_NAME", "nv-community/Nemotron-3-Embed-8B-BF16")
    api_key = env.get(prefix + "_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "output/validation" / (stamp + f"-{args.profile}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence: dict = {"model": model, "checks": {}}

    def request(path: str, payload: dict | None = None, check_name: str | None = None) -> dict:
        start = time.monotonic()
        req = urllib.request.Request(
            base + path,
            headers=headers,
            data=None if payload is None else json.dumps(payload).encode(),
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = response.read()
        value = json.loads(raw) if raw else {}
        evidence["checks"][check_name or path] = {"seconds": round(time.monotonic() - start, 3)}
        return value

    def embed(
        input_type: str,
        texts: list[str],
        *,
        truncate: str = "END",
        expected_tokens: int | None = None,
    ) -> list[list[float]]:
        check_name = "long_context" if expected_tokens is not None else f"embed_{input_type}"
        response = request(
            "/v2/embed",
            {
                "model": model,
                "input_type": input_type,
                "texts": texts,
                "embedding_types": ["float"],
                "truncate": truncate,
            },
            check_name,
        )
        if expected_tokens is not None:
            billed = response["meta"]["billed_units"]["input_tokens"]
            assert (
                billed == expected_tokens
            ), f"Expected {expected_tokens} input tokens without truncation, got {billed}"
            evidence["checks"][check_name]["billed_input_tokens"] = billed
        vectors = response["embeddings"]["float"]
        assert len(vectors) == len(texts)
        for vector in vectors:
            assert len(vector) == 4096, "Unexpected embedding dimension"
            assert all(math.isfinite(value) for value in vector), "Non-finite embedding value"
            norm = math.sqrt(sum(value * value for value in vector))
            assert abs(norm - 1) < 0.01, f"Unexpected embedding norm: {norm}"
        return vectors

    try:
        request("/health")
        models = request("/v1/models")
        matching = [item for item in models["data"] if item["id"] == model]
        assert matching, "Model identity mismatch"
        actual_max = matching[0]["max_model_len"]
        assert (
            actual_max == max_model_len
        ), f"Expected max_model_len={max_model_len}, got {actual_max}"
        evidence["checks"]["model_config"] = {"max_model_len": actual_max}
        if api_key:
            try:
                with urllib.request.urlopen(base + "/v1/models", timeout=10):
                    raise AssertionError("Unauthenticated model request was accepted")
            except urllib.error.HTTPError as exc:
                assert exc.code == 401, f"Expected 401, got {exc.code}"
        queries = embed("query", ["What is the capital city of France?"])
        documents = embed(
            "document",
            ["Paris is the capital city of France.", "Bananas are yellow fruits."],
        )
        scores = [sum(a * b for a, b in zip(queries[0], item)) for item in documents]
        assert scores[0] > scores[1], "Relevant passage did not rank first"
        evidence["checks"]["retrieval"] = {
            "dimension": 4096,
            "relevant_score": scores[0],
            "irrelevant_score": scores[1],
            "correct_order": True,
        }
        # The pinned Nemotron document prompt accounts for four tokens.
        # A full-length request verifies the configured boundary without END truncation.
        embed(
            "document",
            ["hello " * (max_model_len - 4)],
            truncate="NONE",
            expected_tokens=max_model_len,
        )
        evidence["status"] = "passed"
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["error"] = str(exc)
        raise
    finally:
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        print(f"Evidence: {output.relative_to(ROOT)}")
    print(
        "Passed: Embed model identity, 4096 dimensions, L2 norm, "
        f"retrieval order, and {max_model_len}-token input without truncation"
    )


if __name__ == "__main__":
    main()
