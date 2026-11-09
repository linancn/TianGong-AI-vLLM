"""Bounded HTTP adapter around the immutable published LCAEmbedder.

One supervised child owns one GPU model. HTTP threads never touch the model.
An inference deadline kills only this child and exits for GPUStack recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import ipaddress
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "Tiangong-LCA/LCA-Nemotron-Embed-1B"
REVISION = "6d77edaa3b92dd1b7e3dd8a152ea7170a8b581c5"
TOOLS_SHA = "2d0c7d36a00b14dc1cc19d7ac6f94887f9d45d35"
MAX_BODY = 1_000_000
MAX_TEXT = 100_000
MAX_ITEMS = 32


def validate_request(value, served_model_name=None):
    if not isinstance(value, dict) or set(value) != {"model", "input_type", "input"}:
        raise ValueError("Exactly model, input_type and input are required")
    if value["model"] != MODEL_ID and (
        not served_model_name or value["model"] != served_model_name
    ):
        raise ValueError("Unknown model")
    if value["input_type"] not in ("query", "document"):
        raise ValueError("input_type must be query or document")
    texts = [value["input"]] if isinstance(value["input"], str) else value["input"]
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_ITEMS:
        raise ValueError("input must contain 1 to 32 texts")
    if any(not isinstance(t, str) or not t.strip() or len(t) > MAX_TEXT for t in texts):
        raise ValueError("Texts must be nonblank strings of at most 100000 characters")
    return texts, value["input_type"]


class BodyTooLarge(ValueError):
    pass


def read_body(headers, stream):
    """Accept the gateway's chunked framing without weakening the body limit."""
    lengths = headers.get_all("Content-Length", [])
    encodings = headers.get_all("Transfer-Encoding", [])
    if encodings:
        if lengths or len(encodings) != 1 or encodings[0].lower().strip() != "chunked":
            raise ValueError("Ambiguous or unsupported framing")
        body = bytearray()
        for _ in range(10000):
            line = stream.readline(1025)
            if len(line) > 1024 or not line.endswith(b"\r\n"):
                raise ValueError("Invalid chunk header")
            size_text = line[:-2].split(b";", 1)[0]
            if not size_text or any(c not in b"0123456789abcdefABCDEF" for c in size_text):
                raise ValueError("Invalid chunk size")
            size = int(size_text, 16)
            if len(body) + size > MAX_BODY:
                raise BodyTooLarge()
            if size == 0:
                # This API does not accept trailers or empty JSON bodies.
                if stream.readline(3) != b"\r\n" or not body:
                    raise ValueError("Invalid chunk terminator")
                return bytes(body)
            chunk = stream.read(size)
            if len(chunk) != size or stream.read(2) != b"\r\n":
                raise ValueError("Incomplete chunk")
            body.extend(chunk)
        raise ValueError("Too many chunks")
    if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
        raise ValueError("A single Content-Length is required")
    size = int(lengths[0])
    if not 0 < size <= MAX_BODY:
        raise BodyTooLarge()
    body = stream.read(size)
    if len(body) != size:
        raise ValueError("Incomplete request body")
    return body


def verify_release(model_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["revision"] != REVISION or manifest["repo_id"] != MODEL_ID:
        raise ValueError("Release identity mismatch")
    for name, expected in manifest["files"].items():
        path = Path(model_path) / name
        if path.stat().st_size != expected["bytes"]:
            raise ValueError("Release size mismatch")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected["sha256"]:
            raise ValueError("Release hash mismatch")


def model_worker(connection, config):
    # No input, token, exception message or traceback is written to logs.
    try:
        verify_release(config.model, config.manifest)
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one assigned CUDA device is required")
        torch.cuda.set_per_process_memory_fraction(config.memory_fraction, 0)
        spec = importlib.util.spec_from_file_location(
            "lca_release", Path(config.model) / "encode.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        embedder = module.LCAEmbedder(config.model, device="cuda:0", local_files_only=True)
        if not embedder.model[1].include_prompt:
            raise RuntimeError("Prompt pooling mismatch")
        # Warm both modes before readiness.
        for mode in ("query", "document"):
            embedder.encode(["readiness"], mode, batch_size=config.batch_size)
        connection.send({"ready": True})
        while True:
            texts, mode = connection.recv()
            vectors = embedder.encode(texts, mode, batch_size=config.batch_size).tolist()
            if len(vectors) != len(texts) or any(
                len(v) != 2048 or any(not math.isfinite(x) for x in v) for v in vectors
            ):
                raise RuntimeError("Invalid vectors")
            connection.send({"vectors": vectors, "peak_bytes": torch.cuda.max_memory_allocated()})
    except (Exception, KeyboardInterrupt) as exc:
        try:
            connection.send({"error": type(exc).__name__})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


@dataclass
class Job:
    texts: list
    mode: str
    enqueued: float = field(default_factory=time.monotonic)
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    status: int = 503
    result: dict = field(default_factory=lambda: {"error": "Service unavailable"})


class Engine:
    def __init__(self, config):
        self.config = config
        self.jobs = queue.Queue(config.queue_size)
        self.ready = threading.Event()
        self.failed = threading.Event()
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.stats = {
            "requests": 0,
            "texts": 0,
            "errors": 0,
            "queue_expired": 0,
            "rejected": 0,
            "inference_seconds": 0.0,
            "peak_bytes": 0,
            "active": 0,
        }
        context = mp.get_context("spawn")
        self.parent, child = context.Pipe()
        self.process = context.Process(target=model_worker, args=(child, config), daemon=True)
        self.process.start()
        child.close()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def increment(self, **values):
        with self.lock:
            for key, value in values.items():
                self.stats[key] += value

    def run(self):
        try:
            if not self.parent.poll(self.config.load_timeout):
                raise RuntimeError("Load deadline")
            if self.parent.recv() != {"ready": True}:
                raise RuntimeError("Load failure")
            self.ready.set()
            print(json.dumps({"event": "ready", "model_revision": REVISION}), flush=True)
            while not self.stopped.is_set():
                try:
                    job = self.jobs.get(timeout=0.2)
                except queue.Empty:
                    if not self.process.is_alive():
                        raise RuntimeError("Inference process exited")
                    continue
                try:
                    if (
                        job.cancelled.is_set()
                        or time.monotonic() - job.enqueued > self.config.queue_timeout
                    ):
                        self.increment(queue_expired=1)
                        job.result = {"error": "Queue deadline exceeded"}
                        continue
                    started = time.monotonic()
                    self.increment(active=1)
                    self.parent.send((job.texts, job.mode))
                    if not self.parent.poll(self.config.inference_timeout):
                        job.status = 504
                        job.result = {"error": "Inference deadline exceeded"}
                        raise RuntimeError("Inference deadline")
                    result = self.parent.recv()
                    if "error" in result:
                        raise RuntimeError("Inference process failed")
                    job.result = {
                        "object": "list",
                        "model": MODEL_ID,
                        "input_type": job.mode,
                        "data": [
                            {"object": "embedding", "index": i, "embedding": v}
                            for i, v in enumerate(result["vectors"])
                        ],
                    }
                    job.status = 200
                    self.increment(
                        requests=1,
                        texts=len(job.texts),
                        inference_seconds=time.monotonic() - started,
                    )
                    with self.lock:
                        self.stats["peak_bytes"] = max(
                            self.stats["peak_bytes"], result["peak_bytes"]
                        )
                finally:
                    with self.lock:
                        self.stats["active"] = 0
                    job.done.set()
                    self.jobs.task_done()
        except (Exception, KeyboardInterrupt) as exc:
            self.ready.clear()
            if not self.stopped.is_set():
                self.increment(errors=1)
                self.failed.set()
                print(
                    json.dumps({"event": "engine_failed", "type": type(exc).__name__}), flush=True
                )
        finally:
            self.ready.clear()
            while True:
                try:
                    job = self.jobs.get_nowait()
                    job.done.set()
                    self.jobs.task_done()
                except queue.Empty:
                    break

    def close(self):
        self.ready.clear()
        self.stopped.set()
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)
        self.thread.join(timeout=1)
        self.parent.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, address, engine, allowed_clients):
        self.engine = engine
        self.allowed_clients = {str(ipaddress.ip_address(x)) for x in allowed_clients}
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        request.settimeout(10)
        if not self.slots.acquire(blocking=False):
            self.engine.increment(rejected=1)
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        # Socket and malformed request errors must not emit user payloads.
        pass


class Handler(BaseHTTPRequestHandler):
    server_version = "LCAEmbedding/1"

    def log_message(self, *args):
        pass

    def respond(self, status, value):
        body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("X-LCA-Model-Revision", REVISION)
        if status in (429, 503):
            self.send_header("Retry-After", "1")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def allowed(self):
        if self.client_address[0] in self.server.allowed_clients:
            return True
        self.respond(403, {"error": "Use the authenticated model gateway"})
        return False

    def do_GET(self):
        if not self.allowed():
            return
        engine = self.server.engine
        if self.path in ("/health", "/readyz"):
            self.respond(
                200 if engine.ready.is_set() else 503,
                {"ready": engine.ready.is_set(), "model_revision": REVISION},
            )
        elif self.path == "/metrics":
            with engine.lock:
                stats = dict(engine.stats)
            self.respond(
                200,
                {
                    **stats,
                    "queued": engine.jobs.qsize(),
                    "model_revision": REVISION,
                    "tools_sha": TOOLS_SHA,
                },
            )
        elif self.path == "/v1/models":
            self.respond(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        else:
            self.respond(404, {"error": "Not found"})

    def do_POST(self):
        if not self.allowed():
            return
        if self.path != "/v1/embeddings":
            self.respond(404, {"error": "Not found"})
            return
        engine = self.server.engine
        if not engine.ready.is_set():
            self.respond(503, {"error": "Model not ready"})
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            body = read_body(self.headers, self.rfile)
            texts, mode = validate_request(
                json.loads(body), getattr(engine.config, "served_model_name", None)
            )
        except BodyTooLarge:
            engine.increment(rejected=1)
            self.respond(413, {"error": "Request body limit exceeded"})
            return
        except (ValueError, UnicodeError, OSError):
            engine.increment(rejected=1)
            self.respond(
                400, {"error": "Invalid request; require model, input_type and 1-32 nonblank texts"}
            )
            return
        job = Job(texts, mode)
        try:
            engine.jobs.put_nowait(job)
        except queue.Full:
            engine.increment(rejected=1)
            self.respond(429, {"error": "Inference queue full"})
            return
        if not job.done.wait(engine.config.queue_timeout + engine.config.inference_timeout + 2):
            job.cancelled.set()
            self.respond(504, {"error": "Request deadline exceeded"})
            return
        self.respond(job.status, job.result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", help="Exact internal GPUStack deployment alias")
    parser.add_argument(
        "--manifest", default="/opt/lca/tools/config/releases/lca-nemotron-embed-1b.json"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allowed-clients", nargs="+", default=["127.0.0.1"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--queue-timeout", type=float, default=30)
    parser.add_argument("--inference-timeout", type=float, default=120)
    parser.add_argument("--load-timeout", type=float, default=300)
    parser.add_argument("--memory-fraction", type=float, default=0.15)
    parser.add_argument(
        "--read-only-model-mount",
        action="store_true",
        help="Remount the dedicated GPUStack model bind mount read-only",
    )
    args = parser.parse_args()
    if not (
        1 <= args.batch_size <= 32
        and 1 <= args.queue_size <= 32
        and 0 < args.memory_fraction <= 0.25
        and min(args.queue_timeout, args.inference_timeout, args.load_timeout) > 0
    ):
        parser.error("Invalid resource limits")
    if args.read_only_model_mount:
        mount = Path(args.model).resolve().parent
        if not os.path.ismount(mount):
            parser.error("Expected a dedicated parent model bind mount")
        subprocess.run(
            ["mount", "-o", "remount,bind,ro", "--", str(mount)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    engine = Engine(args)
    httpd = Server((args.host, args.port), engine, args.allowed_clients)
    stopping = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopping.set())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        while not stopping.wait(0.2) and not engine.failed.is_set():
            pass
    finally:
        engine.close()
        httpd.shutdown()
        httpd.server_close()
    return 1 if engine.failed.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
