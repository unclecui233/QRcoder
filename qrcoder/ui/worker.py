"""Background analysis so the GUI never blocks.

A ``QThread`` runs :func:`qrcoder.core.analysis.analyze_image`; results travel
back through signals as Python objects.  Only one analysis runs at a time and a
new request supersedes the previous one, which keeps a rapid drag-drop burst from
queueing up work.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from ..core.analysis import analyze_symbols
from ..core.symbols import ImageReport

__all__ = ["AnalysisWorker", "AnalysisController"]


class AnalysisWorker(QThread):
    """Runs one analysis and emits the report."""

    finishedReport = Signal(object)     # ImageReport
    failed = Signal(str)

    def __init__(self, image: np.ndarray, time_budget: float = 12.0,
                 keys=None, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._image = image
        self._time_budget = time_budget
        self._keys = keys

    def run(self) -> None:  # pragma: no cover - thread body
        try:
            report = analyze_symbols(self._image, keys=self._keys,
                                     time_budget=self._time_budget)
        except Exception as exc:  # never let a crash escape the thread
            import traceback

            traceback.print_exc()
            self.failed.emit("%s: %s" % (type(exc).__name__, exc))
            return
        self.finishedReport.emit(report)


class AnalysisController(QObject):
    """Serialises analysis requests."""

    started = Signal()
    finished = Signal(object)           # AnalysisReport
    failed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._worker: Optional[AnalysisWorker] = None
        self._pending: Optional[np.ndarray] = None

    @property
    def busy(self) -> bool:
        return self._worker is not None

    def request(self, image: np.ndarray, time_budget: float = 14.0,
                keys=None) -> None:
        if self._worker is not None:
            # Remember only the newest request; older ones are stale by now.
            self._pending = (image, time_budget, keys)
            return
        self._start(image, time_budget, keys)

    def _start(self, image: np.ndarray, time_budget: float, keys=None) -> None:
        worker = AnalysisWorker(image, time_budget, keys, parent=self)
        worker.finishedReport.connect(self._on_report)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_thread_finished)
        self._worker = worker
        self.started.emit()
        worker.start()

    def _on_report(self, report: ImageReport) -> None:
        self.finished.emit(report)

    def _on_failed(self, message: str) -> None:
        self.failed.emit(message)

    def _on_thread_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        pending = self._pending
        self._pending = None
        if pending is not None:
            image, budget, keys = pending
            self._start(image, budget, keys)

    def wait(self, timeout_ms: int = 20000) -> bool:
        """Block until the current worker finishes (used on shutdown)."""
        worker = self._worker
        if worker is None:
            return True
        return bool(worker.wait(timeout_ms))
