"""llama.cpp OpenAI 兼容服务的配置、进程和 HTTP 客户端。"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CUDA_RUNTIME_DLLS = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")


def subprocess_creation_flags(persistent: bool = False) -> int:
    """返回不会为控制台程序创建可见窗口的 Windows 进程标志。"""
    if os.name != "nt":
        return 0
    flags = subprocess.CREATE_NO_WINDOW
    if persistent:
        flags |= subprocess.CREATE_NEW_PROCESS_GROUP
    return flags


@dataclass(frozen=True)
class LlamaCppConfig:
    base_url: str
    host: str
    port: int
    model: str
    api_key: str
    autostart: bool
    server_path: Path
    model_path: Path
    mmproj_path: Path | None
    extra_dll_dirs: tuple[Path, ...]
    n_gpu_layers: int
    ctx_size: int
    reasoning: str
    reasoning_budget: int
    startup_timeout: float

    @classmethod
    def from_config(cls, data: dict[str, Any], project_root: Path) -> "LlamaCppConfig":
        def as_path(value: object) -> Path:
            path = Path(str(value)).expanduser()
            return path if path.is_absolute() else (project_root / path).resolve()

        mmproj_value = str(data.get("mmproj_path", "")).strip()
        dll_values = [part.strip() for part in str(data.get("extra_dll_dirs", "")).split(os.pathsep)]
        return cls(
            base_url=str(data["base_url"]).rstrip("/"),
            host=str(data.get("host", "127.0.0.1")),
            port=int(data.get("port", 8080)),
            model=str(data["model"]),
            api_key=str(data.get("api_key", "")),
            autostart=_as_bool(data.get("autostart", True)),
            server_path=as_path(data.get("server_path", "")),
            model_path=as_path(data.get("model_path", "")),
            mmproj_path=as_path(mmproj_value) if mmproj_value else None,
            extra_dll_dirs=tuple(as_path(value) for value in dll_values if value),
            n_gpu_layers=int(data.get("n_gpu_layers", 999)),
            ctx_size=int(data.get("ctx_size", 8192)),
            reasoning=str(data.get("reasoning", "off")),
            reasoning_budget=int(data.get("reasoning_budget", 0)),
            startup_timeout=float(data.get("startup_timeout", 600)),
        )

    def validate(self) -> None:
        missing = [path for path in (self.server_path, self.model_path) if not path.is_file()]
        if self.mmproj_path and not self.mmproj_path.is_file():
            missing.append(self.mmproj_path)
        if missing:
            raise FileNotFoundError("缺少运行文件: " + ", ".join(str(path) for path in missing))
        missing_dll_dirs = [path for path in self.extra_dll_dirs if not path.is_dir()]
        if missing_dll_dirs:
            raise FileNotFoundError("缺少 DLL 目录: " + ", ".join(str(path) for path in missing_dll_dirs))
        if os.name == "nt" and self.extra_dll_dirs:
            missing_dlls = [
                name
                for name in CUDA_RUNTIME_DLLS
                if not any((directory / name).is_file() for directory in self.extra_dll_dirs)
            ]
            if missing_dlls:
                raise FileNotFoundError(
                    "共享 CUDA DLL 目录不完整，缺少: " + ", ".join(missing_dlls)
                )
        if self.host not in {"127.0.0.1", "localhost", "::1"} and not self.api_key:
            raise ValueError("监听局域网地址时必须配置 LLAMACPP_API_KEY")

    def command(self) -> list[str]:
        command = [
            str(self.server_path), "-m", str(self.model_path), "--alias", self.model,
            "-c", str(self.ctx_size), "-ngl", str(self.n_gpu_layers),
            "--reasoning", self.reasoning, "--reasoning-budget", str(self.reasoning_budget),
            "--host", self.host, "--port", str(self.port),
        ]
        if self.mmproj_path:
            command.extend(["--mmproj", str(self.mmproj_path)])
        if self.api_key:
            command.extend(["--api-key", self.api_key])
        return command


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class LlamaCppClient:
    def __init__(self, config: LlamaCppConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.process: subprocess.Popen[bytes] | None = None
        self.pid_file = project_root / "run" / "llama-server.pid"

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _request(self, path: str, payload: dict[str, Any] | None = None, timeout: float = 10) -> Any:
        url = self.config.base_url + path
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
        request = Request(url, data=data, headers=self.headers, method="POST" if data else "GET")
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self, timeout: float = 3) -> dict[str, Any]:
        base = self.config.base_url.removesuffix("/v1")
        request = Request(base + "/health", headers=self.headers)
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def models(self) -> dict[str, Any]:
        return self._request("/models")

    def chat(self, prompt: str, max_tokens: int = 64) -> str:
        result = self._request(
            "/chat/completions",
            {
                "model": self.config.model,
                "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=max(120, self.config.startup_timeout),
        )
        return str(result["choices"][0]["message"].get("content", ""))

    def is_healthy(self) -> bool:
        try:
            return self.health().get("status") == "ok"
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            return False

    def start_server(self, persistent: bool = False) -> subprocess.Popen[bytes]:
        self.config.validate()
        if self.is_healthy():
            raise RuntimeError(f"端口 {self.config.port} 已有可用服务")
        env = os.environ.copy()
        path_parts = [str(self.config.server_path.parent), *(str(path) for path in self.config.extra_dll_dirs)]
        env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_handle = (logs_dir / "llama_server.out.log").open("ab")
        err_handle = (logs_dir / "llama_server.err.log").open("ab")
        creationflags = subprocess_creation_flags(persistent)
        try:
            self.process = subprocess.Popen(
                self.config.command(), cwd=self.project_root, env=env,
                stdout=out_handle, stderr=err_handle, creationflags=creationflags,
            )
        finally:
            out_handle.close()
            err_handle.close()
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(self.process.pid), encoding="ascii")
        return self.process

    def wait_until_ready(
        self,
        wait_callback: Callable[[float], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started_at = time.monotonic()
        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f"llama-server 已异常退出，退出码 {self.process.returncode}")
            try:
                health = self.health()
                models = self.models()
                return health, models
            except (HTTPError, URLError, TimeoutError, OSError, ValueError):
                if wait_callback:
                    wait_callback(time.monotonic() - started_at)
                time.sleep(1)
        raise TimeoutError(f"等待 llama-server 启动超过 {self.config.startup_timeout:.0f} 秒")

    def ensure_server(self) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            return self.health(), self.models()
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            if not self.config.autostart:
                raise RuntimeError("llama-server 不可用，且自动启动已关闭")
            self.start_server()
            return self.wait_until_ready()

    def assert_model_available(self, models: dict[str, Any]) -> None:
        ids = {str(item.get("id")) for item in models.get("data", [])}
        if self.config.model not in ids:
            raise RuntimeError(f"模型 {self.config.model!r} 不在服务模型列表中: {sorted(ids)}")

    def shutdown_server(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.process and self.pid_file.exists():
            self.pid_file.unlink()
