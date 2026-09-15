"""入口和编排层共享的项目上下文。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localai.config_loader import load_config


@dataclass(frozen=True)
class AppContext:
    project_root: Path
    config: dict[str, Any]


def bootstrap_context(entry_file: str) -> AppContext:
    """定位项目根目录、加入 src 路径并加载配置。"""
    project_root, config_path = resolve_runtime_paths(entry_file)
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return AppContext(
        project_root=project_root,
        config=load_config(project_root, str(config_path)),
    )


def resolve_runtime_paths(entry_file: str) -> tuple[Path, Path]:
    """返回可写运行目录和配置路径，兼容源码及 PyInstaller 单文件程序。"""
    source_root = Path(entry_file).resolve().parent
    if not getattr(sys, "frozen", False):
        return source_root, source_root / "config.yaml"

    executable_root = Path(sys.executable).resolve().parent
    candidates = (executable_root, executable_root.parent)
    project_root = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "common.env").is_file() or (candidate / "config.yaml").is_file()
        ),
        executable_root,
    )
    external_config = project_root / "config.yaml"
    if external_config.is_file():
        return project_root, external_config

    bundle_root = Path(getattr(sys, "_MEIPASS", executable_root))
    return project_root, bundle_root / "config.yaml"
