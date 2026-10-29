from __future__ import annotations

import http.client
import io
import json
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from localai.context import AppContext
from localai.modules.bridge import BridgeManager, ImageJob


def manager_for_test(tmp_path: Path) -> BridgeManager:
    config = {
        "llamacpp": {"api_key": "test-secret"},
        "gateway": {"host": "127.0.0.1", "port": 0},
        "comfyui": {"root": str(tmp_path), "python_path": str(tmp_path / "python.exe"),
                    "checkpoint": "model.safetensors"},
    }
    return BridgeManager(AppContext(tmp_path, config))


def test_image_job_restores_text_after_success_and_failure(tmp_path: Path) -> None:
    manager = manager_for_test(tmp_path)
    transitions: list[str] = []
    manager.activate = lambda mode: transitions.append(mode) or {"mode": mode}  # type: ignore[method-assign]
    target = tmp_path / "image.png"
    manager.comfy.generate = lambda *_args, **_kwargs: target  # type: ignore[method-assign]
    job = ImageJob("success")
    manager._run_image(job, "cat", "", 512, 512, 10, 1)
    assert job.state == "completed" and job.path == target
    assert transitions == ["image", "text"]

    def fail(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError("generation failed")

    manager.comfy.generate = fail  # type: ignore[method-assign]
    transitions.clear()
    job = ImageJob("failure")
    manager._run_image(job, "cat", "", 512, 512, 10, 1)
    assert job.state == "failed" and job.error == "generation failed"
    assert transitions == ["image", "text"]


def test_image_request_limits(tmp_path: Path) -> None:
    manager = manager_for_test(tmp_path)
    with pytest.raises(ValueError):
        manager.submit_image({"prompt": "cat", "width": 2048})
    with pytest.raises(ValueError):
        manager.submit_image({"prompt": ""})
    with pytest.raises(ValueError):
        manager.submit_image({"prompt": "cat", "n": 2})
    with pytest.raises(ValueError):
        manager.submit_image({"prompt": "cat", "model": "unknown"})


def test_gateway_requires_key(tmp_path: Path) -> None:
    manager = manager_for_test(tmp_path)
    manager.serve()
    try:
        port = manager.server.server_address[1]  # type: ignore[union-attr]
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/admin/status")
        assert conn.getresponse().status == 401
        conn.close()
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/admin/status", headers={"Authorization": "Bearer test-secret"})
        response = conn.getresponse()
        assert response.status == 200
        assert b"stopped" in response.read()
        conn.close()
    finally:
        manager.shutdown()


def test_gateway_proxies_chat_request(tmp_path: Path) -> None:
    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            incoming = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert incoming["messages"][0]["content"] == "你好"
            assert len(incoming["image_payload"]) == 2_000_000
            result = json.dumps({"choices": [{"message": {"content": "可用"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

        def log_message(self, *_args: object) -> None:
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    manager = manager_for_test(tmp_path)
    manager.mode = "text"
    manager.client = SimpleNamespace(config=SimpleNamespace(port=upstream.server_address[1]))  # type: ignore[assignment]
    manager.serve()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", manager.server.server_address[1])  # type: ignore[union-attr]
        conn.request("POST", "/v1/chat/completions", body=json.dumps({"messages": [{"content": "你好"}], "image_payload": "a" * 2_000_000}),
                     headers={"Authorization": "Bearer test-secret", "Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["choices"][0]["message"]["content"] == "可用"
        conn.close()
    finally:
        manager.client = None
        manager.shutdown()
        upstream.shutdown()
        upstream.server_close()


def test_comfy_workflow_saves_generated_png(tmp_path: Path) -> None:
    backend = manager_for_test(tmp_path).comfy
    checkpoint = tmp_path / "models/checkpoints/model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    submitted: list[dict] = []

    def fake_json(path: str, payload: dict | None = None) -> dict:
        if path == "/prompt":
            submitted.append(payload["prompt"])
            return {"prompt_id": "job-1"}
        return {"job-1": {"outputs": {"7": {"images": [
            {"filename": "result.png", "subfolder": "", "type": "output"}
        ]}}}}

    backend._json = fake_json  # type: ignore[method-assign]
    with patch("localai.modules.image_backend.urlopen", return_value=io.BytesIO(b"png-bytes")):
        path = backend.generate("a cat", "blur", 768, 768, 20, 123)
    assert path.read_bytes() == b"png-bytes"
    assert submitted[0]["2"]["inputs"]["text"] == "a cat"
    assert submitted[0]["4"]["inputs"]["batch_size"] == 1
    assert submitted[0]["5"]["inputs"]["seed"] == 123


def test_qwen_workflow_uses_separate_model_encoder_and_vae(tmp_path: Path) -> None:
    backend = manager_for_test(tmp_path).comfy
    for folder, filename in (("diffusion_models", "qwen_image_2512_fp8_e4m3fn.safetensors"),
                             ("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors"),
                             ("vae", "qwen_image_vae.safetensors")):
        path = tmp_path / "models" / folder / filename
        path.parent.mkdir(parents=True)
        path.touch()
    submitted: list[dict] = []

    def fake_json(path: str, payload: dict | None = None) -> dict:
        if path == "/prompt":
            submitted.append(payload["prompt"])
            return {"prompt_id": "job-2"}
        return {"job-2": {"outputs": {"10": {"images": [
            {"filename": "result.png", "subfolder": "", "type": "output"}
        ]}}}}

    backend._json = fake_json  # type: ignore[method-assign]
    with patch("localai.modules.image_backend.urlopen", return_value=io.BytesIO(b"png-bytes")):
        result = backend.generate("中文标题", "模糊", 1024, 1024, 50, 42, model="qwen-image-2512")
    assert result.read_bytes() == b"png-bytes"
    assert submitted[0]["1"]["class_type"] == "UNETLoader"
    assert submitted[0]["2"]["inputs"]["type"] == "qwen_image"
    assert submitted[0]["8"]["inputs"]["steps"] == 50
