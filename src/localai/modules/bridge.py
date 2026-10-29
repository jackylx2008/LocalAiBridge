"""Single public API and serialized GPU mode management."""

from __future__ import annotations

import http.client
import hmac
import json
import logging
import secrets
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from flows.server_flow import start_owned
from localai.context import AppContext
from localai.modules.image_backend import ComfyBackend
from localai.modules.llamacpp_client import LlamaCppClient


@dataclass
class ImageJob:
    id: str
    model: str = "sdxl"
    state: str = "queued"
    error: str | None = None
    path: Path | None = None
    seed: int | None = None


class BridgeManager:
    def __init__(self, context: AppContext) -> None:
        self.context = context
        self.lock = threading.RLock()
        self.mode = "stopped"
        self.client: LlamaCppClient | None = None
        self.comfy = ComfyBackend(context.config["comfyui"], context.project_root / "outputs")
        self.jobs: dict[str, ImageJob] = {}
        self.server: ThreadingHTTPServer | None = None
        self.server_thread: threading.Thread | None = None

    def status(self) -> dict:
        return {"mode": self.mode, "jobs": {key: value.state for key, value in self.jobs.items()}}

    def activate(self, mode: str) -> dict:
        if mode not in {"text", "image"}:
            raise ValueError("mode 只能是 text 或 image")
        with self.lock:
            if mode == "text" and self.mode == mode and self.client and self.client.is_healthy():
                return self.status()
            if mode == "image" and self.mode == mode and self.comfy.process and self.comfy.process.poll() is None:
                return self.status()
            self.mode = "switching"
            try:
                if self.client:
                    self.client.shutdown_server()
                    self.client = None
                self.comfy.stop()
                if mode == "text":
                    self.client, _ = start_owned(self.context)
                else:
                    self.comfy.start()
                self.mode = mode
            except Exception:
                self.mode = "stopped"
                raise
            return self.status()

    def stop(self) -> None:
        with self.lock:
            if self.client:
                self.client.shutdown_server()
                self.client = None
            self.comfy.stop()
            self.mode = "stopped"

    def submit_image(self, payload: dict) -> ImageJob:
        model = str(payload.get("model", "sdxl"))
        if model not in {"sdxl", "qwen-image-2512"}:
            raise ValueError("model 只能是 sdxl 或 qwen-image-2512")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
            raise ValueError("prompt 必须是 1–4000 字符的文本")
        if int(payload.get("n", 1)) != 1:
            raise ValueError("首版每次只能生成 1 张图片")
        width, height = int(payload.get("width", 1024)), int(payload.get("height", 1024))
        steps = int(payload.get("steps", 50 if model == "qwen-image-2512" else 20))
        if width % 64 or height % 64 or not (512 <= width <= 1024 and 512 <= height <= 1024):
            raise ValueError("宽高必须为 512–1024 范围内的 64 倍数")
        max_steps = 50 if model == "qwen-image-2512" else 40
        if width * height > 1024 * 1024 or not 1 <= steps <= max_steps:
            raise ValueError(f"像素数不能超过 1024×1024，steps 必须为 1–{max_steps}")
        negative = payload.get("negative_prompt", "")
        if not isinstance(negative, str) or len(negative) > 4000:
            raise ValueError("negative_prompt 无效")
        seed = int(payload.get("seed", secrets.randbelow(2**32)))
        if not 0 <= seed < 2**64:
            raise ValueError("seed 超出范围")
        job = ImageJob(str(uuid.uuid4()), model=model, seed=seed)
        self.jobs[job.id] = job
        threading.Thread(target=self._run_image, args=(job, prompt, negative, width, height, steps, seed), daemon=True).start()
        return job

    def _run_image(self, job: ImageJob, prompt: str, negative: str, width: int, height: int, steps: int, seed: int) -> None:
        with self.lock:
            try:
                job.state = "running"
                self.activate("image")
                job.path = self.comfy.generate(prompt, negative, width, height, steps, seed, model=job.model)
            except Exception as exc:
                logging.getLogger(__name__).exception("生图失败")
                job.error = str(exc)
            finally:
                try:
                    job.state = "restoring"
                    self.activate("text")
                except Exception as exc:
                    logging.getLogger(__name__).exception("生图后恢复文字模型失败")
                    job.error = (job.error + "; " if job.error else "") + f"恢复文字模式失败：{exc}"
                job.state = "failed" if job.error else "completed"

    def serve(self) -> None:
        manager = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, format: str, *args: object) -> None:
                logging.getLogger(__name__).info("bridge: " + format, *args)

            def reply(self, status: int, data: dict) -> None:
                raw = json.dumps(data, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def authorized(self) -> bool:
                key = str(manager.context.config["llamacpp"].get("api_key", ""))
                return bool(key) and hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key)

            def handle_request(self) -> None:
                if not self.authorized():
                    return self.reply(401, {"error": "Unauthorized"})
                path = urlsplit(self.path).path
                try:
                    if path == "/admin/status" and self.command == "GET":
                        return self.reply(200, manager.status())
                    if path == "/admin/mode" and self.command == "POST":
                        return self.reply(200, manager.activate(self.body()["mode"]))
                    if path == "/v1/images/generations" and self.command == "POST":
                        job = manager.submit_image(self.body())
                        return self.reply(202, {"id": job.id, "status_url": f"/v1/images/jobs/{job.id}"})
                    if path.startswith("/v1/images/jobs/") and self.command == "GET":
                        parts = path.split("/")
                        job = manager.jobs.get(parts[4]) if len(parts) >= 5 else None
                        if not job:
                            return self.reply(404, {"error": "Job not found"})
                        if len(parts) == 6 and parts[5] == "content":
                            if not job.path or job.state != "completed":
                                return self.reply(409, {"error": "Image not ready"})
                            raw = job.path.read_bytes()
                            self.send_response(200)
                            self.send_header("Content-Type", "image/png")
                            self.send_header("Content-Length", str(len(raw)))
                            self.end_headers()
                            self.wfile.write(raw)
                            return
                        return self.reply(200, {"id": job.id, "model": job.model, "state": job.state, "error": job.error,
                                                "seed": job.seed, "image_url": f"/v1/images/jobs/{job.id}/content" if job.path else None})
                    if path.startswith("/v1/") or path == "/health":
                        return self.proxy()
                    return self.reply(404, {"error": "Unknown endpoint"})
                except (ValueError, KeyError, TypeError) as exc:
                    return self.reply(400, {"error": str(exc)})
                except Exception as exc:
                    logging.getLogger(__name__).exception("Bridge request failed")
                    return self.reply(503, {"error": str(exc)})

            def body_bytes(self, max_size: int = 1024 * 1024) -> bytes:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > max_size:
                    raise ValueError("请求体过大")
                return self.rfile.read(length)

            def body(self) -> dict:
                value = json.loads(self.body_bytes())
                if not isinstance(value, dict):
                    raise ValueError("请求体必须为 JSON 对象")
                return value

            def proxy(self) -> None:
                with manager.lock:
                    if manager.mode != "text" or not manager.client:
                        return self.reply(503, {"error": "Text model unavailable"})
                    port = manager.client.config.port
                    upstream = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
                    try:
                        headers = {"Content-Type": self.headers.get("Content-Type", "application/json"),
                                   "Authorization": self.headers.get("Authorization", "")}
                        upstream.request(self.command, self.path, body=self.body_bytes(64 * 1024 * 1024) if self.command == "POST" else None,
                                         headers=headers)
                        response = upstream.getresponse()
                        self.send_response(response.status)
                        for name, value in response.getheaders():
                            if name.lower() not in {"transfer-encoding", "connection", "content-length", "content-encoding"}:
                                self.send_header(name, value)
                        self.send_header("Connection", "close")
                        self.end_headers()
                        while chunk := response.read1(65536):
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    finally:
                        upstream.close()

            do_GET = handle_request
            do_POST = handle_request

        port = int(self.context.config["gateway"]["port"])
        host = str(self.context.config["gateway"]["host"])
        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def shutdown(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        self.stop()
