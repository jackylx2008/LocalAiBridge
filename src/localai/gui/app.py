"""本地 AI 服务桌面窗口实现。"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import timedelta
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

from localai.context import AppContext
from flows.server_flow import build_client, start_owned
from localai.modules.llamacpp_client import LlamaCppClient


class QueueLogHandler(logging.Handler):
    """把后台日志投递到 Tk 主线程处理。"""

    def __init__(self, events: queue.Queue[tuple[str, Any]]) -> None:
        super().__init__()
        self.events = events

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.events.put(("log", (record.levelname, self.format(record))))
        except Exception:
            self.handleError(record)


class LocalAiApp:
    def __init__(self, root: tk.Tk, context: AppContext) -> None:
        self.root = root
        self.context = context
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.client: LlamaCppClient | None = None
        self.worker: threading.Thread | None = None
        self.busy = False
        self.running = False
        self.close_requested = False
        self.started_monotonic: float | None = None

        self.status_var = tk.StringVar(value="准备启动")
        self.detail_var = tk.StringVar(value="窗口初始化完成，正在准备加载模型")
        self.progress_var = tk.DoubleVar(value=0)
        self.model_var = tk.StringVar(value="尚未加载")
        self.ip_var = tk.StringVar(value="检测中")
        self.port_var = tk.StringVar(value=str(context.config["llamacpp"].get("port", 8080)))
        self.url_var = tk.StringVar(value="等待模型启动")
        self.pid_var = tk.StringVar(value="—")
        self.elapsed_var = tk.StringVar(value="00:00:00")
        self.environment_var = tk.StringVar(value=f"Python {sys.version_info.major}.{sys.version_info.minor} / Tk {tk.TkVersion}")

        self._configure_window()
        self._build_ui()
        self._install_log_handler()
        self.root.protocol("WM_DELETE_WINDOW", self._request_close)
        self.root.after(100, self._drain_events)
        self.root.after(1000, self._update_elapsed)
        self.root.after(250, self.start_model)

    def _configure_window(self) -> None:
        self.root.title("LocalAiBridge - Qwen3.8 本地 AI 服务")
        self.root.geometry("920x720")
        self.root.minsize(760, 600)
        style = ttk.Style(self.root)
        style.configure("Hero.Horizontal.TProgressbar", thickness=26)
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 17, "bold"))
        style.configure("Value.TLabel", font=("Microsoft YaHei UI", 10, "bold"))

    def _build_ui(self) -> None:
        self.root.rowconfigure(0, weight=3)
        self.root.rowconfigure(1, weight=2)
        self.root.columnconfigure(0, weight=1)

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))
        service_tab = ttk.Frame(notebook, padding=16)
        config_tab = ttk.Frame(notebook, padding=16)
        notebook.add(service_tab, text="服务控制")
        notebook.add(config_tab, text="配置概览")
        self._build_service_tab(service_tab)
        self._build_config_tab(config_tab)

        log_frame = ttk.LabelFrame(self.root, text="运行日志与实时输出", padding=8)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = ScrolledText(
            log_frame,
            height=12,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
            background="#101418",
            foreground="#d9e2ec",
            insertbackground="white",
        )
        self.log_text.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.log_text.tag_configure("INFO", foreground="#d9e2ec")
        self.log_text.tag_configure("WARNING", foreground="#ffd166")
        self.log_text.tag_configure("ERROR", foreground="#ff6b6b")
        self.log_text.tag_configure("SUCCESS", foreground="#63d471")
        ttk.Button(log_frame, text="清空界面日志", command=self._clear_log).grid(row=1, column=1, sticky="e", pady=(8, 0))

        status_bar = ttk.Frame(self.root, padding=(12, 7))
        status_bar.grid(row=2, column=0, sticky="ew")
        status_bar.columnconfigure(1, weight=1)
        ttk.Label(status_bar, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status_bar, textvariable=self.detail_var).grid(row=0, column=1, sticky="w", padx=16)
        ttk.Label(status_bar, textvariable=self.elapsed_var).grid(row=0, column=2, padx=12)
        ttk.Label(status_bar, textvariable=self.environment_var).grid(row=0, column=3, sticky="e")

    def _build_service_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(parent, textvariable=self.detail_var).grid(row=1, column=0, sticky="w", pady=(4, 10))
        self.progress = ttk.Progressbar(
            parent,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            style="Hero.Horizontal.TProgressbar",
        )
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 16))

        info = ttk.LabelFrame(parent, text="服务运行信息", padding=12)
        info.grid(row=3, column=0, sticky="nsew")
        info.columnconfigure(1, weight=1)
        rows = [
            ("内存加载模型", self.model_var),
            ("本机局域网 IP", self.ip_var),
            ("服务端口", self.port_var),
            ("Mac 访问地址", self.url_var),
            ("服务进程 PID", self.pid_var),
        ]
        for row, (label, variable) in enumerate(rows):
            ttk.Label(info, text=f"{label}：").grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
            ttk.Label(info, textvariable=variable, style="Value.TLabel").grid(row=row, column=1, sticky="w", pady=4)

        buttons = ttk.Frame(parent)
        buttons.grid(row=4, column=0, sticky="ew", pady=(16, 0))
        self.start_button = ttk.Button(buttons, text="重新加载模型", command=self.start_model)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="停止模型", command=self.stop_model, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.health_button = ttk.Button(buttons, text="健康检查", command=self.health_check, state="disabled")
        self.health_button.pack(side="left")

    def _build_config_tab(self, parent: ttk.Frame) -> None:
        config = self.context.config["llamacpp"]
        parent.columnconfigure(1, weight=1)
        items = [
            ("模型别名", config.get("model")),
            ("监听地址", f"{config.get('host')}:{config.get('port')}"),
            ("本机 API", config.get("base_url")),
            ("上下文长度", config.get("ctx_size")),
            ("GPU Offload 层", config.get("n_gpu_layers")),
            ("API 鉴权", "已启用 Bearer Key" if config.get("api_key") else "未启用"),
            ("模型文件", config.get("model_path")),
        ]
        for row, (label, value) in enumerate(items):
            ttk.Label(parent, text=f"{label}：").grid(row=row, column=0, sticky="nw", padx=(0, 12), pady=6)
            ttk.Label(parent, text=str(value), wraplength=650).grid(row=row, column=1, sticky="nw", pady=6)

    def _install_log_handler(self) -> None:
        handler = QueueLogHandler(self.events)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(handler)

    def start_model(self) -> None:
        if self.busy:
            return
        self.busy = True
        self.running = False
        self.started_monotonic = time.monotonic()
        self.progress_var.set(0)
        self.status_var.set("正在启动")
        self.detail_var.set("准备加载 Qwen3.8 模型")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.health_button.configure(state="disabled")
        self._append_log("INFO", "开始自动加载本地模型")
        self.worker = threading.Thread(target=self._start_worker, daemon=True)
        self.worker.start()

    def _start_worker(self) -> None:
        try:
            if self.client is not None:
                self.events.put(("progress", (8, "正在停止当前模型以便重新加载")))
                self.client.shutdown_server()
                self.client = None
            client, result = start_owned(
                self.context,
                lambda percent, message: self.events.put(("progress", (percent, message))),
            )
            self.events.put(("started", (client, result)))
        except Exception as exc:
            logging.getLogger(__name__).exception("GUI 自动加载模型失败")
            self.events.put(("error", str(exc)))

    def stop_model(self) -> None:
        if self.busy or not self.client:
            return
        self.busy = True
        self.status_var.set("正在停止")
        self.detail_var.set("正在释放模型占用的内存和显存")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.health_button.configure(state="disabled")
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self) -> None:
        try:
            if self.client:
                self.client.shutdown_server()
            self.events.put(("stopped", None))
        except Exception as exc:
            logging.getLogger(__name__).exception("停止本地模型失败")
            self.events.put(("error", str(exc)))

    def health_check(self) -> None:
        if self.busy:
            return
        self.health_button.configure(state="disabled")
        self.detail_var.set("正在执行健康检查")
        threading.Thread(target=self._health_worker, daemon=True).start()

    def _health_worker(self) -> None:
        try:
            client = self.client or build_client(self.context)
            health = client.health()
            models = client.models()
            client.assert_model_available(models)
            self.events.put(("health", health))
        except Exception as exc:
            self.events.put(("health_error", str(exc)))

    def _drain_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                self._handle_event(event, payload)
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(100, self._drain_events)

    def _handle_event(self, event: str, payload: Any) -> None:
        if event == "log":
            level, message = payload
            self._append_log(level, message)
        elif event == "progress":
            percent, message = payload
            self.progress_var.set(percent)
            self.detail_var.set(message)
        elif event == "started":
            self.client, result = payload
            if self.close_requested:
                threading.Thread(target=self._close_after_start, daemon=True).start()
                return
            self.busy = False
            self.running = True
            self.status_var.set("运行正常")
            self.detail_var.set("健康检查、模型校验和“你好”推理均已通过")
            self.progress_var.set(100)
            self.model_var.set(str(result["model"]))
            self.ip_var.set(str(result.get("lan_ip") or "未检测到"))
            self.port_var.set(str(result["port"]))
            self.url_var.set(str(result.get("lan_base_url") or "未检测到"))
            self.pid_var.set(str(result.get("pid") or "—"))
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="normal")
            self.health_button.configure(state="normal")
            self._append_log("SUCCESS", f"模型加载完成；测试回答：{result['inference_check']['answer']}")
        elif event == "stopped":
            self.client = None
            self.busy = False
            self.running = False
            self.status_var.set("已停止")
            self.detail_var.set("模型已从内存和显存卸载")
            self.progress_var.set(0)
            self.model_var.set("尚未加载")
            self.pid_var.set("—")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.health_button.configure(state="disabled")
            self._append_log("SUCCESS", "本地 AI 服务已停止，模型资源已释放")
            if self.close_requested:
                self.root.destroy()
        elif event == "health":
            self.health_button.configure(state="normal")
            self.detail_var.set("健康检查通过，配置模型可用")
            self._append_log("SUCCESS", f"健康检查通过：{payload}")
            messagebox.showinfo("健康检查", "服务健康检查通过，模型列表校验正常。")
        elif event == "health_error":
            self.health_button.configure(state="normal" if self.running else "disabled")
            self.detail_var.set("健康检查失败")
            self._append_log("ERROR", f"健康检查失败：{payload}")
            messagebox.showerror("健康检查失败", str(payload))
        elif event == "error":
            self.busy = False
            self.running = False
            self.status_var.set("启动失败")
            self.detail_var.set(str(payload))
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.health_button.configure(state="disabled")
            self._append_log("ERROR", str(payload))
            if self.close_requested:
                self.root.destroy()
            else:
                messagebox.showerror("本地 AI 服务错误", str(payload))

    def _append_log(self, level: str, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n", level if level in {"INFO", "WARNING", "ERROR", "SUCCESS"} else "INFO")
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 1000:
            self.log_text.delete("1.0", f"{lines - 1000}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _update_elapsed(self) -> None:
        if self.started_monotonic is not None and (self.busy or self.running):
            elapsed = int(time.monotonic() - self.started_monotonic)
            self.elapsed_var.set(str(timedelta(seconds=elapsed)))
        if self.root.winfo_exists():
            self.root.after(1000, self._update_elapsed)

    def _request_close(self) -> None:
        if self.close_requested:
            return
        self.close_requested = True
        self.status_var.set("正在关闭")
        self.detail_var.set("正在停止服务并卸载模型")
        if self.busy:
            self._append_log("WARNING", "将在当前加载步骤安全结束后关闭服务")
        elif self.client:
            self.busy = True
            threading.Thread(target=self._stop_worker, daemon=True).start()
        else:
            self.root.destroy()

    def _close_after_start(self) -> None:
        if self.client:
            self.client.shutdown_server()
        self.events.put(("stopped", None))


def run_gui(context: AppContext) -> None:
    root = tk.Tk()
    LocalAiApp(root, context)
    root.mainloop()
