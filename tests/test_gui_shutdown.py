from __future__ import annotations

import logging
import queue
from pathlib import Path
from typing import Any
from unittest.mock import patch

import tkinter as tk

from localai.gui.app import LocalAiApp, QueueLogHandler, configure_process_identity, configure_window_icon, parse_context_tokens


class FakeRoot:
    def __init__(self) -> None:
        self.after_calls: list[tuple[int, Any]] = []

    def after(self, delay: int, callback: Any) -> str:
        self.after_calls.append((delay, callback))
        return "after-id"


def test_event_drain_does_not_reschedule_after_destroy() -> None:
    app = LocalAiApp.__new__(LocalAiApp)
    app.root = FakeRoot()
    app.events = queue.Queue()
    app.events.put(("close", None))
    app.destroyed = False
    app._event_after_id = "current-callback"

    def handle_event(_event: str, _payload: Any) -> None:
        app.destroyed = True

    app._handle_event = handle_event
    app._drain_events()

    assert app.root.after_calls == []
    assert app._event_after_id is None


def test_context_token_input_requires_positive_integer() -> None:
    assert parse_context_tokens(" 16384 ") == 16384
    for value in ("", "0", "-1", "16K", "3.5"):
        try:
            parse_context_tokens(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"无效 token 数未被拒绝: {value!r}")


def test_16k_shortcut_sets_value_and_reloads() -> None:
    app = LocalAiApp.__new__(LocalAiApp)
    app.busy = False
    values: list[str] = []
    app.ctx_size_var = tk.StringVar(master=tk.Tcl(), value="32768")
    app.start_model = lambda: values.append(app.ctx_size_var.get())

    app.set_context_size(16384)

    assert app.ctx_size_var.get() == "16384"
    assert values == ["16384"]


class FakeIconRoot:
    def __init__(self, error: bool = False) -> None:
        self.error = error
        self.icon_paths: list[str] = []

    def iconbitmap(self, default: str) -> None:
        self.icon_paths.append(default)
        if self.error:
            raise tk.TclError("unsupported icon")


def test_configure_window_icon_uses_windows_icon(tmp_path: Path) -> None:
    icon_path = tmp_path / "icons" / "windows" / "LocalAIBridge.ico"
    icon_path.parent.mkdir(parents=True)
    icon_path.touch()
    root = FakeIconRoot()

    with patch("localai.gui.app.sys.platform", "win32"):
        loaded = configure_window_icon(root, tmp_path)

    assert loaded is True
    assert root.icon_paths == [str(icon_path)]


def test_configure_window_icon_is_non_fatal_when_tk_rejects_icon(tmp_path: Path) -> None:
    icon_path = tmp_path / "icons" / "windows" / "LocalAIBridge.ico"
    icon_path.parent.mkdir(parents=True)
    icon_path.touch()

    with patch("localai.gui.app.sys.platform", "win32"):
        loaded = configure_window_icon(FakeIconRoot(error=True), tmp_path)

    assert loaded is False


def test_configure_process_identity_is_skipped_outside_windows() -> None:
    with patch("localai.gui.app.sys.platform", "darwin"):
        configured = configure_process_identity()

    assert configured is False


def test_queue_log_handler_omits_traceback_from_gui() -> None:
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    handler = QueueLogHandler(events)
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    try:
        raise RuntimeError("concise cause")
    except RuntimeError:
        record = logging.getLogger("test").makeRecord(
            "test", logging.ERROR, __file__, 1, "startup failed: %s", ("concise cause",),
            exc_info=True,
        )
    handler.emit(record)

    event, payload = events.get_nowait()
    assert event == "log"
    assert "startup failed: concise cause" in payload[1]
    assert "Traceback" not in payload[1]
