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
    project_root = Path(entry_file).resolve().parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return AppContext(project_root=project_root, config=load_config(project_root))
