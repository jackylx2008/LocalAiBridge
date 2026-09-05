"""CUDA、服务、模型列表和对话请求的自检编排。"""

from __future__ import annotations

import subprocess
from typing import Any

from localai.context import AppContext
from flows.server_flow import build_client


def run(context: AppContext, prompt: str | None, max_tokens: int) -> dict[str, Any]:
    cuda = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    client = build_client(context)
    try:
        health, models = client.ensure_server()
        client.assert_model_available(models)
        result: dict[str, Any] = {
            "cuda_check": {"command": "nvidia-smi", "ok": cuda.returncode == 0},
            "llamacpp": {
                "base_url": client.config.base_url,
                "model": client.config.model,
                "health": health,
                "available_models": [item.get("id") for item in models.get("data", [])],
            },
        }
        if prompt is not None:
            result["answer"] = client.chat(prompt, max_tokens=max_tokens)
        return result
    finally:
        client.shutdown_server()
