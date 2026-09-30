"""项目配置和本机环境变量加载。"""

from __future__ import annotations

import os
import platform
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")
_CTX_SIZE_LINE = re.compile(rb"(?m)^([ \t]*LLAMACPP_CTX_SIZE[ \t]*=)[^\r\n]*")


def save_ctx_size(project_root: Path, ctx_size: int) -> None:
    """将 GUI 选择的上下文长度写入本机配置，保留其他设置。"""
    if ctx_size <= 0:
        raise ValueError("上下文 token 数必须是正整数")
    path = project_root / "common.env"
    original = path.read_bytes() if path.exists() else b""
    replacement = rb"\g<1>" + str(ctx_size).encode("ascii")
    if _CTX_SIZE_LINE.search(original):
        updated = _CTX_SIZE_LINE.sub(replacement, original)
    else:
        newline = b"\r\n" if b"\r\n" in original else b"\n"
        separator = b"" if not original or original.endswith((b"\r", b"\n")) else newline
        updated = original + separator + b"LLAMACPP_CTX_SIZE=" + str(ctx_size).encode("ascii") + newline
    if updated == original:
        return
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=project_root, prefix=".common.env.", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(updated)
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


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
