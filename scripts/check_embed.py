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
    default_port = "7732" if args.profile == "embed3" else "7731"
    env = read_env()
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

    def request(path: str, payload: dict | None = None) -> dict:
        start = time.monotonic()
        req = urllib.request.Request(
            base + path,
            headers=headers,
            data=None if payload is None else json.dumps(payload).encode(),
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = response.read()
        value = json.loads(raw) if raw else {}
        evidence["checks"][path] = {"seconds": round(time.monotonic() - start, 3)}
        return value

    def embed(input_type: str, texts: list[str]) -> list[list[float]]:
        response = request(
            "/v2/embed",
            {
                "model": model,
                "input_type": input_type,
                "texts": texts,
                "embedding_types": ["float"],
                "truncate": "END",
            },
        )
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
        assert model in [item["id"] for item in models["data"]], "Model identity mismatch"
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
        evidence["status"] = "passed"
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["error"] = str(exc)
        raise
    finally:
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        print(f"Evidence: {output.relative_to(ROOT)}")
    print("Passed: Embed model identity, 4096 dimensions, L2 norm, and retrieval order")


if __name__ == "__main__":
    main()
