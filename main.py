"""LocalAiBridge 图形界面主入口

用途：
  启动本地 AI 桌面窗口，自动加载 Qwen3.8，并管理健康检查、重新加载、停止和窗口关闭时的模型卸载。

配置文件：
  config.yaml 保存通用配置；common.env 保存本机 llama.cpp、模型路径、监听地址与 API Key。

示例：
  .venv/Scripts/python.exe main.py
  .venv/Scripts/pythonw.exe main.py

输出：
  窗口显示加载进度、运行状态、局域网地址和实时日志；磁盘日志写入 logs/main.log 和
  logs/llama_server.*.log。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from localai.context import bootstrap_context
from localai.gui.app import run_gui
from logging_config import configure_utf8_stdio, setup_logger


def main() -> int:
    configure_utf8_stdio()
    context = bootstrap_context(__file__)
    setup_logger(context.config["app"]["log_level"])
    run_gui(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
