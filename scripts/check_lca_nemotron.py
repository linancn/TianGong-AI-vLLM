"""Real HTTP/reference acceptance inside the pinned LCA inference image.

Read a dedicated gateway key from stdin with --key-stdin. Never log the key,
request text, error response bodies, or complete response vectors. Evidence is
written to a new private output directory. --benchmark regenerates ALL vectors.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

MODEL = "Tiangong-LCA/LCA-Nemotron-Embed-1B"
REVISION = "6d77edaa3b92dd1b7e3dd8a152ea7170a8b581c5"
TOOLS_SHA = "2d0c7d36a00b14dc1cc19d7ac6f94887f9d45d35"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--tools-dir", type=Path, default=Path("/opt/lca/tools"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-stdin", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--gateway-rewrites-model",
        action="store_true",
        help="Verify that the authorized gateway route replaces the submitted model name",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.benchmark and not args.data_dir:
        parser.error("--benchmark requires --data-dir")
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    key = sys.stdin.readline().strip() if args.key_stdin else ""
    if args.key_stdin and not key:
        parser.error("Empty key")
    sys.path.insert(0, str(args.tools_dir))
    from src.lca_benchmark.metrics import evaluate, rank_scores
    from src.lca_benchmark.releases import read_manifest, rows, validate_dataset, verify_files

    model_manifest = read_manifest(args.tools_dir / "config/releases/lca-nemotron-embed-1b.json")
    assert model_manifest["revision"] == REVISION
    verify_files(model_manifest, args.model_dir)
    save(args.output_dir / "model-manifest.json", model_manifest)
    import numpy as np
    import torch

    torch.cuda.set_per_process_memory_fraction(0.15, args.device)
    spec = importlib.util.spec_from_file_location("published_encode", args.model_dir / "encode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.LCAEmbedder(str(args.model_dir), device=args.device, local_files_only=True)
    assert reference.model[1].include_prompt
    timings = []
    proofs = []

    def call(payload, authenticated=True):
        headers = {"Content-Type": "application/json"}
        if key and authenticated:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/embeddings",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=160) as response:
                value = json.load(response)
                return response.status, value, time.monotonic() - started
        except urllib.error.HTTPError as error:
            return error.code, {}, time.monotonic() - started

    def encode_http(texts, mode, timed=True):
        status, value, elapsed = call({"model": MODEL, "input_type": mode, "input": texts})
        assert status == 200, f"HTTP status {status}"
        assert value["model"] == MODEL and value["input_type"] == mode
        assert [v["index"] for v in value["data"]] == list(range(len(texts)))
        result = np.array([v["embedding"] for v in value["data"]], dtype=np.float32)
        assert result.shape == (len(texts), 2048) and np.isfinite(result).all()
        assert np.max(np.abs(np.linalg.norm(result, axis=1) - 1)) < 1e-5
        if timed:
            timings.append(
                {
                    "seconds": elapsed,
                    "mode": mode,
                    "batch": len(texts),
                    "characters": [len(t) for t in texts],
                }
            )
        return result

    def compare(name, texts, mode):
        actual = encode_http(texts, mode)
        expected = reference.encode(texts, mode, batch_size=8)
        cosines = np.sum(actual * expected, axis=1)
        max_error = float(np.max(np.abs(actual - expected)))
        proof = {
            "name": name,
            "mode": mode,
            "count": len(texts),
            "input_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in texts],
            "min_cosine": float(cosines.min()),
            "max_abs": max_error,
        }
        np.save(args.output_dir / (name + "-http.npy"), actual)
        np.save(args.output_dir / (name + "-reference.npy"), expected)
        proofs.append(proof)
        assert proof["min_cosine"] >= 0.999 and max_error <= 0.02, proof
        return actual, expected

    samples = [
        "生命周期评价与碳排放",
        "Life cycle assessment of recycled aluminium.",
        "electricity production, photovoltaic",
        "水泥生产过程的温室气体排放",
    ]
    encode_http(samples[:1], "query", timed=False)
    compare("query", samples, "query")
    compare("document", samples, "document")
    compare("query-truncation", ["生命周期 emissions " * 800, "short"], "query")
    compare("document-truncation", ["生命周期 emissions " * 1600, "short"], "document")
    compare("batch32", (samples * 8), "document")
    valid = {"model": MODEL, "input_type": "query", "input": ["validation"]}
    invalid = [
        None,
        {"model": MODEL, "input": ["missing mode"]},
        {**valid, "input_type": "other"},
        {**valid, "input": []},
        {**valid, "input": [""]},
        {**valid, "input": [1]},
        {**valid, "input": ["x"] * 33},
        {**valid, "input": "x" * 100001},
        {**valid, "dimensions": 1024},
        {**valid, "encoding_format": "base64"},
    ]
    if args.gateway_rewrites_model:
        status, value, _ = call({**valid, "model": "wrong"})
        assert status == 200 and value["model"] == MODEL
    else:
        invalid.append({**valid, "model": "wrong"})
    invalid_statuses = [call(payload)[0] for payload in invalid]
    assert all(status == 400 for status in invalid_statuses), invalid_statuses
    oversize_status = call({**valid, "input": ["x" * 99999] * 32})[0]
    assert oversize_status == 413, oversize_status
    unauth = call(valid, authenticated=False)[0] if key else None
    if key:
        assert unauth in (401, 403), unauth
    # Warm latency at concurrency 1 and 3, alternating real query/document calls.
    performance = []
    for concurrency in (1, 3):
        start = len(timings)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(
                pool.map(
                    lambda i: encode_http(samples, "query" if i % 2 else "document"), range(30)
                )
            )
        values = [r["seconds"] for r in timings[start:]]
        performance.append(
            {
                "concurrency": concurrency,
                "requests": len(values),
                "batch": 4,
                "p50_seconds": float(np.percentile(values, 50)),
                "p95_seconds": float(np.percentile(values, 95)),
                "errors": 0,
            }
        )
    result = {
        "model_revision": REVISION,
        "tools_sha": TOOLS_SHA,
        "device": args.device,
        "cuda_dtype": "bfloat16",
        "dimension": 2048,
        "runtime": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "transformers",
                "sentence-transformers",
                "huggingface-hub",
                "safetensors",
            )
        },
        "numerical": proofs,
        "invalid_statuses": invalid_statuses,
        "oversize_status": oversize_status,
        "unauthenticated_status": unauth,
        "performance": performance,
    }
    save(args.output_dir / "sample-results.json", result)
    print(json.dumps({"phase": "samples", "passed": True, "performance": performance}), flush=True)
    if args.benchmark:
        manifest = read_manifest(args.tools_dir / "config/datasets/lca-retrieval-benchmark-v1.json")
        verify_files(manifest, args.data_dir)
        validate_dataset(manifest, args.data_dir)
        save(args.output_dir / "benchmark-manifest.json", manifest)
        docs = sorted(rows(args.data_dir / "corpus.jsonl"), key=lambda r: r["document_id"])
        queries = sorted(rows(args.data_dir / "queries.jsonl"), key=lambda r: r["query_id"])
        vectors = {}
        with (args.output_dir / "input-hashes.jsonl").open("w") as ledger:
            for mode, records in (("document", docs), ("query", queries)):
                api_parts, ref_parts = [], []
                for begin in range(0, len(records), 32):
                    batch = records[begin : begin + 32]
                    texts = [row["text"] for row in batch]
                    actual = encode_http(texts, mode)
                    expected = reference.encode(texts, mode, batch_size=8)
                    cosine = float(np.sum(actual * expected, axis=1).min())
                    error = float(np.max(np.abs(actual - expected)))
                    assert cosine >= 0.999 and error <= 0.02, (mode, begin, cosine, error)
                    for row in batch:
                        ledger.write(
                            json.dumps(
                                {
                                    "id": row[mode + "_id"],
                                    "mode": mode,
                                    "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                                }
                            )
                            + "\n"
                        )
                    api_parts.append(actual)
                    ref_parts.append(expected)
                    if begin % 1024 == 0:
                        print(
                            json.dumps(
                                {
                                    "phase": "benchmark",
                                    "mode": mode,
                                    "completed": begin + len(batch),
                                }
                            ),
                            flush=True,
                        )
                vectors[mode] = (np.concatenate(api_parts), np.concatenate(ref_parts))
                for index, label in enumerate(("http", "reference")):
                    np.save(args.output_dir / (mode + "-" + label + ".npy"), vectors[mode][index])
        results = {}
        rankings = {}
        for index, label in enumerate(("http", "reference")):
            dvec = torch.from_numpy(vectors["document"][index])
            qvec = torch.from_numpy(vectors["query"][index])
            rankings[label] = [
                {
                    "query_id": q["query_id"],
                    "document_ids": rank_scores(
                        [d["document_id"] for d in docs], (qvec[i] @ dvec.T).tolist()
                    ),
                }
                for i, q in enumerate(queries)
            ]
            (args.output_dir / (label + "-rankings.jsonl")).write_text(
                "".join(json.dumps(r) + "\n" for r in rankings[label])
            )
            results[label] = evaluate(args.data_dir, rankings[label])
            save(args.output_dir / (label + "-metrics.json"), results[label])
        delta = {
            metric: abs(value - results["reference"]["metrics"][metric])
            for metric, value in results["http"]["metrics"].items()
        }
        differing = [
            {"query_id": a["query_id"], "http": a["document_ids"], "reference": b["document_ids"]}
            for a, b in zip(rankings["http"], rankings["reference"])
            if a != b
        ]
        save(args.output_dir / "ranking-differences.json", differing)
        result["benchmark"] = {
            "revision": manifest["revision"],
            "documents": len(docs),
            "queries": len(queries),
            "metrics": results["http"]["metrics"],
            "absolute_reference_delta": delta,
            "different_rankings": len(differing),
            "batch_size": 8,
            "request_batch": 32,
            "similarity": "CPU float32 inner product, document_id tie-break",
        }
        assert all(value <= 1e-4 for value in delta.values()), delta
    save(args.output_dir / "timings.json", timings)
    result["files"] = {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}
    save(args.output_dir / "acceptance.json", result)
    print(json.dumps({"passed": True, "benchmark": result.get("benchmark")}), flush=True)


if __name__ == "__main__":
    main()
