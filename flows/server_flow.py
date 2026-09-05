"""本地 AI 服务的启动、状态和停止编排。"""

from __future__ import annotations

import os
import signal
import socket
import time
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Callable

from localai.context import AppContext
from localai.modules.llamacpp_client import LlamaCppClient, LlamaCppConfig
from logging_config import get_logger


logger = get_logger(__name__)


def build_client(context: AppContext) -> LlamaCppClient:
    config = LlamaCppConfig.from_config(context.config["llamacpp"], context.project_root)
    return LlamaCppClient(config, context.project_root)


def start(context: AppContext) -> dict[str, Any]:
    client = build_client(context)
    logger.info(
        "检查本地 AI 服务：base_url=%s, listen=%s:%s, model=%s",
        client.config.base_url,
        client.config.host,
        client.config.port,
        client.config.model,
    )
    if client.is_healthy():
        models = client.models()
        client.assert_model_available(models)
        inference_answer = _verify_inference(client)
        logger.info("本地 AI 服务已在运行，模型和推理校验通过")
        result = _status_result(client, models, already_running=True)
        result["inference_check"] = {"status": "passed", "answer": inference_answer}
        return result
    logger.info("服务当前不可用，正在启动 llama-server")
    process = client.start_server(persistent=True)
    logger.info("llama-server 进程已创建：pid=%s", process.pid)
    _, models = client.wait_until_ready()
    client.assert_model_available(models)
    inference_answer = _verify_inference(client)
    logger.info("本地 AI 服务启动完成，模型和推理均可用：%s", client.config.model)
    result = _status_result(client, models, already_running=False)
    result["pid"] = process.pid
    result["inference_check"] = {"status": "passed", "answer": inference_answer}
    return result


def restart(context: AppContext) -> dict[str, Any]:
    """停止本项目记录的已有服务，再重新加载模型并完成推理校验。"""
    client = build_client(context)
    if client.is_healthy():
        if not client.pid_file.exists():
            raise RuntimeError("检测到外部 llama-server，但没有本项目 PID；为避免误杀，拒绝自动重启")
        logger.info("正式启动模式：先停止本项目已有服务")
        stop(context)
        deadline = time.monotonic() + 30
        while client.is_healthy() and time.monotonic() < deadline:
            time.sleep(0.2)
        if client.is_healthy():
            raise TimeoutError("旧 llama-server 在 30 秒内未停止")
    else:
        client.pid_file.unlink(missing_ok=True)
        logger.info("正式启动模式：当前没有运行中的服务")
    result = start(context)
    result["startup_mode"] = "fresh"
    return result


def start_owned(
    context: AppContext,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[LlamaCppClient, dict[str, Any]]:
    """启动由当前 Python 会话持有的服务，以便退出时可靠释放模型。"""
    report = progress_callback or (lambda _percent, _message: None)
    report(5, "正在读取并校验本机配置")
    client = build_client(context)
    if client.is_healthy():
        if not client.pid_file.exists():
            raise RuntimeError("检测到外部 llama-server，但没有本项目 PID；为避免误杀，拒绝自动重启")
        report(10, "正在停止已有本地 AI 服务")
        logger.info("会话模式：先停止本项目已有服务")
        stop(context)
    else:
        client.pid_file.unlink(missing_ok=True)

    report(20, "正在创建 llama-server 进程")
    logger.info("会话模式：正在创建由当前脚本持有的 llama-server")
    process = client.start_server(persistent=False)
    report(30, f"服务进程已创建（PID {process.pid}），正在加载模型")
    try:
        _, models = client.wait_until_ready(
            lambda elapsed: report(min(82, 30 + int(elapsed * 4)), f"正在加载 GGUF 模型（{elapsed:.0f} 秒）")
        )
        report(86, "服务健康检查通过，正在校验模型列表")
        client.assert_model_available(models)
        report(94, "模型列表校验通过，正在发送“你好”测试")
        inference_answer = _verify_inference(client)
    except Exception:
        client.shutdown_server()
        raise
    result = _status_result(client, models, already_running=False)
    result.update(
        {
            "pid": process.pid,
            "inference_check": {"status": "passed", "answer": inference_answer},
            "startup_mode": "fresh",
        }
    )
    report(100, "模型加载完成，服务运行正常")
    logger.info("会话模式启动完成：pid=%s, model=%s", process.pid, client.config.model)
    return client, result


def status(context: AppContext) -> dict[str, Any]:
    client = build_client(context)
    if not client.is_healthy():
        logger.warning("本地 AI 服务不可用：%s", client.config.base_url)
        return {"running": False, "base_url": client.config.base_url}
    models = client.models()
    client.assert_model_available(models)
    logger.info("本地 AI 服务状态正常，模型校验通过")
    return _status_result(client, models, already_running=True)


def stop(context: AppContext) -> dict[str, Any]:
    client = build_client(context)
    if not client.pid_file.exists():
        logger.warning("未找到由本项目记录的 llama-server PID")
        return {"stopped": False, "message": "没有由本项目记录的 llama-server PID"}
    pid = int(client.pid_file.read_text(encoding="ascii").strip())
    if os.name == "nt":
        import subprocess

        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, check=False,
        )
        if completed.returncode not in {0, 128}:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    while client.is_healthy() and time.monotonic() < deadline:
        time.sleep(0.2)
    if client.is_healthy():
        raise TimeoutError("llama-server 已终止，但服务在 30 秒后仍可访问")
    client.pid_file.unlink(missing_ok=True)
    logger.info("本地 AI 服务已停止，模型已从内存/显存卸载：pid=%s", pid)
    return {"stopped": True, "pid": pid, "model": client.config.model, "model_unloaded": True}


def _status_result(client: LlamaCppClient, models: dict[str, Any], already_running: bool) -> dict[str, Any]:
    lan_ip = _get_primary_lan_ipv4()
    return {
        "running": True,
        "already_running": already_running,
        "base_url": client.config.base_url,
        "lan_ip": lan_ip,
        "lan_base_url": f"http://{lan_ip}:{client.config.port}/v1" if lan_ip else None,
        "port": client.config.port,
        "listen": f"{client.config.host}:{client.config.port}",
        "model": client.config.model,
        "available_models": [item.get("id") for item in models.get("data", [])],
        "authentication": "bearer" if client.config.api_key else "none",
    }


def _get_primary_lan_ipv4() -> str | None:
    """返回默认路由使用的 IPv4，并在失败时回退到本机私有地址。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            candidate = probe.getsockname()[0]
        if ip_address(candidate).is_private:
            return candidate
    except OSError:
        pass

    try:
        candidates = {
            item[4][0]
            for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_DGRAM)
        }
    except OSError:
        return None
    private = sorted(address for address in candidates if ip_address(address).is_private)
    return private[0] if private else None


def _verify_inference(client: LlamaCppClient) -> str:
    logger.info("发送真实推理请求以验证模型：prompt=你好")
    answer = client.chat("你好", max_tokens=32).strip()
    if not answer:
        raise RuntimeError("推理验证失败：模型返回空内容")
    logger.info("真实推理请求通过：answer=%s", answer)
    return answer
