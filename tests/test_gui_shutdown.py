from __future__ import annotations

import queue
from typing import Any

from localai.gui.app import LocalAiApp


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
