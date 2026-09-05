"""项目配置和本机环境变量加载。"""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")


def load_dotenv(path: Path) -> None:
    """读取简单 dotenv 文件，且不覆盖进程中已有变量。"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _set_cloudstation_root() -> None:
    if os.environ.get("CLOUDSTATION_ROOT"):
        return
    names = {
        "Windows": "CLOUDSTATION_ROOT_WINDOWS",
        "Darwin": "CLOUDSTATION_ROOT_MACOS",
        "Linux": "CLOUDSTATION_ROOT_LINUX",
    }
    value = os.environ.get(names.get(platform.system(), ""), "")
    if value:
        os.environ["CLOUDSTATION_ROOT"] = str(Path(value).expanduser())


def _expand_env(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default or "")

    return _ENV_PATTERN.sub(replace, text)


def load_config(project_root: Path, config_file: str = "config.yaml") -> dict[str, Any]:
    """加载 common.env，展开配置占位符并解析 YAML。"""
    load_dotenv(project_root / "common.env")
    _set_cloudstation_root()
    path = project_root / config_file
    data = yaml.safe_load(_expand_env(path.read_text(encoding="utf-8")))
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是对象: {path}")
    return data
