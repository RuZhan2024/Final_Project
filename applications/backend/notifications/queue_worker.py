from __future__ import annotations

import logging
import queue
import threading
import time

from dataclasses import dataclass
from typing import Callable


logger = logging.getLogger(__name__)


@dataclass
class DispatchJob:
    fn: Callable[[], None]
    event_id: str


class NotificationQueueWorker:
    """Simple daemon thread worker for non-blocking outbound dispatch."""

    def __init__(self, maxsize: int, poll_interval_s: float):
        self._queue: queue.Queue[DispatchJob] = queue.Queue(maxsize=maxsize)
        self._poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="safe-guard-notify", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def submit(self, job: DispatchJob) -> bool:
        self.start()
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            logger.warning("safe_guard queue full; dropping event_id=%s", job.event_id)
            return False

    def stop(self) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=1.5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=self._poll_interval_s)
            except queue.Empty:
                continue
            try:
                job.fn()
            except Exception as exc:
                logger.exception("safe_guard job failed event_id=%s err=%s", job.event_id, exc)
            finally:
                self._queue.task_done()
