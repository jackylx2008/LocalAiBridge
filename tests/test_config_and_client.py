from __future__ import annotations

from pathlib import Path
import subprocess

from localai.config_loader import load_config
from localai.context import resolve_runtime_paths
from localai.modules.llamacpp_client import LlamaCppConfig, subprocess_creation_flags


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


def test_resolve_runtime_paths_uses_project_root_for_source(monkeypatch, tmp_path: Path) -> None:
    entry_file = tmp_path / "main.py"
    monkeypatch.setattr("localai.context.sys.frozen", False, raising=False)

    project_root, config_path = resolve_runtime_paths(str(entry_file))

    assert project_root == tmp_path
    assert config_path == tmp_path / "config.yaml"


def test_resolve_runtime_paths_finds_config_above_dist(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "dist" / "LocalAiBridge.exe"
    executable.parent.mkdir()
    (tmp_path / "config.yaml").touch()
    monkeypatch.setattr("localai.context.sys.frozen", True, raising=False)
    monkeypatch.setattr("localai.context.sys.executable", str(executable))

    project_root, config_path = resolve_runtime_paths("unused")

    assert project_root == tmp_path
    assert config_path == tmp_path / "config.yaml"


def test_resolve_runtime_paths_uses_bundled_config_as_fallback(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "deploy" / "LocalAiBridge.exe"
    bundle_root = tmp_path / "bundle"
    executable.parent.mkdir()
    bundle_root.mkdir()
    monkeypatch.setattr("localai.context.sys.frozen", True, raising=False)
    monkeypatch.setattr("localai.context.sys.executable", str(executable))
    monkeypatch.setattr("localai.context.sys._MEIPASS", str(bundle_root), raising=False)

    project_root, config_path = resolve_runtime_paths("unused")

    assert project_root == executable.parent
    assert config_path == bundle_root / "config.yaml"


def test_windows_subprocesses_are_started_without_a_console(monkeypatch) -> None:
    monkeypatch.setattr("localai.modules.llamacpp_client.os.name", "nt")

    session_flags = subprocess_creation_flags()
    persistent_flags = subprocess_creation_flags(persistent=True)

    assert session_flags & subprocess.CREATE_NO_WINDOW
    assert persistent_flags & subprocess.CREATE_NO_WINDOW
    assert persistent_flags & subprocess.CREATE_NEW_PROCESS_GROUP
