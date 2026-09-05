from __future__ import annotations

from pathlib import Path

from localai.config_loader import load_config
from localai.modules.llamacpp_client import LlamaCppConfig


def test_example_config_loads(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLAMACPP_API_KEY", "test-secret")
    monkeypatch.setenv("LLAMACPP_SERVER_PATH", str(tmp_path / "server.exe"))
    monkeypatch.setenv("LLAMACPP_MODEL_PATH", str(tmp_path / "model.gguf"))
    config = load_config(Path(__file__).parents[1])
    llama = LlamaCppConfig.from_config(config["llamacpp"], Path(__file__).parents[1])
    assert llama.model == "Qwen3.8-27B-Q4_K_M"
    assert "--api-key" in llama.command()
    assert llama.host == "0.0.0.0"
    assert llama.extra_dll_dirs == (Path(__file__).parents[2] / "vendor" / "cuda12",)
