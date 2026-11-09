"""HTTP/queue contract tests; numerical GPU acceptance is check_lca_nemotron.py."""

import importlib.util
import http.client
import io
from email.message import Message
import json
from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

SPEC = importlib.util.spec_from_file_location(
    "lca_server", Path(__file__).resolve().parents[1] / "deploy/lca-nemotron/server.py"
)
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


def fake_worker(connection, config):
    connection.send({"ready": True})
    while True:
        texts, mode = connection.recv()
        if texts == ["block"]:
            time.sleep(5)
        if texts == ["exit"]:
            return
        connection.send({"vectors": [[1.0] + [0.0] * 2047 for _ in texts], "peak_bytes": 1})


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(server, "model_worker", fake_worker)
    config = SimpleNamespace(queue_size=1, load_timeout=5, queue_timeout=0.2, inference_timeout=0.3)
    value = server.Engine(config)
    assert value.ready.wait(5)
    yield value
    value.close()


@pytest.fixture
def endpoint(engine):
    httpd = server.Server(("127.0.0.1", 0), engine, ["127.0.0.1"])
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:" + str(httpd.server_port), httpd
    httpd.shutdown()
    httpd.server_close()


def request(endpoint, payload=None, path="/v1/embeddings", headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        endpoint + path, data=data, headers=headers or {"Content-Type": "application/json"}
    )
    try:
        response = urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def valid(mode="query", texts=None):
    return {"model": server.MODEL_ID, "input_type": mode, "input": texts or ["test"]}


@pytest.mark.parametrize(
    "patch",
    [
        {"input_type": "wrong"},
        {"model": "wrong"},
        {"input": []},
        {"input": [""]},
        {"input": [" "]},
        {"input": [1]},
        {"input": ["x"] * 33},
        {"input": "x" * 100001},
        {"dimensions": 1024},
        {"encoding_format": "base64"},
    ],
)
def test_reject_invalid_without_queueing(endpoint, engine, patch):
    status, _ = request(endpoint[0], {**valid(), **patch})
    assert status == 400
    assert engine.stats["requests"] == 0


def test_mixed_modes_and_batch_order(endpoint):
    for mode in ("query", "document", "query"):
        status, result = request(endpoint[0], valid(mode, ["one", "two"]))
        assert status == 200 and result["input_type"] == mode
        assert [item["index"] for item in result["data"]] == [0, 1]
        assert all(len(item["embedding"]) == 2048 for item in result["data"])


def test_gateway_chunked_deployment_alias(endpoint, engine):
    engine.config.served_model_name = "lca-deployment-gpu0"
    body = json.dumps({**valid(), "model": engine.config.served_model_name}).encode()
    connection = http.client.HTTPConnection("127.0.0.1", endpoint[1].server_port, timeout=5)
    connection.request(
        "POST",
        "/v1/embeddings",
        [body[:20], body[20:]],
        {"Content-Type": "application/json"},
        encode_chunked=True,
    )
    response = connection.getresponse()
    assert response.status == 200
    assert json.load(response)["model"] == server.MODEL_ID
    connection.close()
    assert request(endpoint[0], valid())[0] == 200
    assert request(endpoint[0], {**valid(), "model": "other-deployment"})[0] == 400


@pytest.mark.parametrize(
    "extra,body,error",
    [
        ({"Content-Length": "1"}, b"1\r\nx\r\n0\r\n\r\n", ValueError),
        ({}, b"-1\r\n", ValueError),
        ({}, b"2\r\nx", ValueError),
        ({}, b"1\r\nx!\n0\r\n\r\n", ValueError),
        ({}, b"1\r\nx\r\n0\r\nTrailer: no\r\n\r\n", ValueError),
        ({}, b"f4241\r\n", server.BodyTooLarge),
        ({"Transfer-Encoding": "gzip, chunked"}, b"", ValueError),
    ],
)
def test_chunked_framing_limits(extra, body, error):
    headers = Message()
    for key, value in {"Transfer-Encoding": "chunked", **extra}.items():
        headers[key] = value
    with pytest.raises(error):
        server.read_body(headers, io.BytesIO(body))


def test_readiness_allowlist_and_content_type(endpoint, engine):
    assert request(endpoint[0], path="/readyz")[0] == 200
    assert request(endpoint[0], valid(), headers={"Content-Type": "text/plain"})[0] == 400
    engine.ready.clear()
    assert request(endpoint[0], valid())[0] == 503
    assert request(endpoint[0], path="/readyz")[0] == 503
    endpoint[1].allowed_clients.clear()
    assert request(endpoint[0], path="/readyz")[0] == 403


def test_queue_bound_and_deadline_release_waiters(engine):
    active = server.Job(["block"], "document")
    engine.jobs.put_nowait(active)
    deadline = time.monotonic() + 2
    while not engine.stats["active"] and time.monotonic() < deadline:
        time.sleep(0.005)
    pending = server.Job(["pending"], "query")
    engine.jobs.put_nowait(pending)
    with pytest.raises(queue.Full):
        engine.jobs.put_nowait(server.Job(["overflow"], "query"))
    assert active.done.wait(2) and active.status == 504
    assert pending.done.wait(2) and pending.status == 503
    assert engine.failed.is_set() and not engine.ready.is_set()
    engine.close()
    assert not engine.process.is_alive()


def test_expired_job_never_runs(engine):
    job = server.Job(["expired"], "query", enqueued=time.monotonic() - 10)
    engine.jobs.put_nowait(job)
    assert job.done.wait(2) and job.status == 503
    assert engine.stats["requests"] == 0
    assert engine.stats["queue_expired"] == 1


def test_idle_child_failure_removes_readiness(engine):
    engine.process.terminate()
    engine.process.join(2)
    assert engine.failed.wait(2)
    assert not engine.ready.is_set()
