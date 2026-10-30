"""Drag-and-drop routing.

Qt delivers drag events to the *deepest* widget under the cursor.  A
``QGraphicsView`` accepts drags itself (its viewport is a child widget), so a
``MainWindow.dragEnterEvent`` override never sees drops that land on the canvas.
Instead of relying on event bubbling, this module installs a single application
level event filter that normalises every drag over the main window.

That also covers the side panel, the tab bar, the status bar and text editors,
and it can never be defeated by a child widget calling ``acceptDrops(False)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QWidget

__all__ = [
    "IMAGE_SUFFIXES",
    "DropPayload",
    "DropRouter",
    "DropOverlay",
    "qimage_to_numpy",
    "classify_mime",
    "local_image_paths",
]

#: Extensions accepted from the file manager / browser.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
                  ".gif", ".jp2", ".pbm", ".pgm", ".ppm")


@dataclass
class DropPayload:
    """A normalised drag-and-drop payload."""

    kind: str                      # "image" | "files" | "rejected"
    paths: List[str] = field(default_factory=list)
    image: Optional[np.ndarray] = None
    rejected: List[str] = field(default_factory=list)
    ignored: int = 0

    @property
    def usable(self) -> bool:
        return self.kind in ("image", "files") and (bool(self.paths) or self.image is not None)

    def status_text(self) -> str:
        if self.kind == "rejected":
            names = "、".join(self.rejected[:3])
            if len(self.rejected) > 3:
                names += " 等 %d 个文件" % len(self.rejected)
            return "不支持的文件类型：%s（支持 %s）" % (
                names, " ".join(s.lstrip(".") for s in IMAGE_SUFFIXES[:7]))
        if self.kind == "image":
            text = "已从拖拽内容载入"
        elif self.paths:
            text = "已载入 %s" % os.path.basename(self.paths[0])
        else:
            text = ""
        if self.ignored:
            text += "（已忽略其余 %d 个）" % self.ignored
        return text


def qimage_to_numpy(qimage: QImage) -> np.ndarray:
    """Convert a QImage to a BGR uint8 array."""
    if qimage.isNull():
        return np.zeros((1, 1, 3), np.uint8)
    if qimage.format() != QImage.Format.Format_RGB888:
        qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
    width, height = qimage.width(), qimage.height()
    buffer = np.frombuffer(qimage.constBits(), dtype=np.uint8).reshape(
        (height, qimage.bytesPerLine()))
    rgb = buffer[:, : width * 3].reshape((height, width, 3))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _suffix_ok(path: str) -> bool:
    return path.lower().endswith(IMAGE_SUFFIXES)


def local_image_paths(mime) -> List[str]:
    """Return local file paths in *mime* that look like images."""
    if mime is None or not mime.hasUrls():
        return []
    paths = []
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        local = url.toLocalFile()
        if local and _suffix_ok(local) and os.path.isfile(local):
            paths.append(local)
    return paths


def _all_local_names(mime) -> List[str]:
    if mime is None or not mime.hasUrls():
        return []
    return [url.toLocalFile() or url.toString() for url in mime.urls()]


def classify_mime(mime) -> Optional[DropPayload]:
    """Decide what a drag contains.

    Returns ``None`` when the drag is not ours at all (plain text being moved
    inside a text editor, for example) so the widget keeps its own behaviour.
    """
    if mime is None:
        return None
    paths = local_image_paths(mime)
    if paths:
        total = len(_all_local_names(mime))
        return DropPayload(kind="files", paths=paths,
                           ignored=max(0, total - len(paths)))
    if mime.hasImage():
        image = mime.imageData()
        array = None
        if isinstance(image, QImage):
            array = qimage_to_numpy(image)
        elif image is not None and hasattr(image, "toImage"):
            array = qimage_to_numpy(image.toImage())
        if array is not None and array.size > 3:
            return DropPayload(kind="image", image=array)
    if mime.hasUrls():
        names = _all_local_names(mime)
        if names:
            return DropPayload(kind="rejected", rejected=names)
    return None


class DropRouter(QObject):
    """Application-level event filter that normalises drags over one window."""

    dropped = Signal(object)          # DropPayload
    dragActiveChanged = Signal(bool)

    def __init__(self, window: QWidget, parent: Optional[QObject] = None) -> None:
        super().__init__(parent or window)
        self._window = window
        self._active = False

    def attach(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def detach(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)

    # -- helpers ----------------------------------------------------------
    def _belongs_to_window(self, obj) -> bool:
        if not isinstance(obj, QWidget):
            return False
        if obj is self._window:
            return True
        try:
            return self._window.isAncestorOf(obj)
        except RuntimeError:          # widget already destroyed
            return False

    def _set_active(self, active: bool) -> None:
        if active != self._active:
            self._active = active
            self.dragActiveChanged.emit(active)

    # -- event filter -----------------------------------------------------
    def eventFilter(self, obj, event) -> bool:
        kind = event.type()
        if kind not in (QEvent.Type.DragEnter, QEvent.Type.DragMove,
                        QEvent.Type.DragLeave, QEvent.Type.Drop):
            return False
        if not self._belongs_to_window(obj):
            return False

        if kind == QEvent.Type.DragLeave:
            self._set_active(False)
            return False

        payload = classify_mime(event.mimeData())
        if payload is None:
            self._set_active(False)
            return False

        # Accept every drag we understand so that the drop always reaches us,
        # regardless of which child widget happens to be under the cursor.
        event.acceptProposedAction()

        if kind == QEvent.Type.Drop:
            self._set_active(False)
            self.dropped.emit(payload)
            return True

        self._set_active(True)
        return True


class DropOverlay(QWidget):
    """Translucent "drop here" hint painted over the whole window."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setVisible(False)
        self._accepted = True

    def set_accepted(self, accepted: bool) -> None:
        if accepted != self._accepted:
            self._accepted = accepted
            if self.isVisible():
                self.update()

    def show_for_parent(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())
        self.show()
        self.raise_()

    def paintEvent(self, event) -> None:
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QFont, QPainter, QPen

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._accepted:
            fill = QColor(64, 158, 255, 46)
            border = QColor(64, 158, 255, 220)
            text = "松开以加载并分析图片"
        else:
            fill = QColor(220, 80, 80, 46)
            border = QColor(220, 80, 80, 220)
            text = "不支持的文件类型"

        painter.fillRect(self.rect(), fill)
        pen = QPen(border)
        pen.setWidth(3)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        rect = QRectF(self.rect()).adjusted(10, 10, -10, -10)
        painter.drawRoundedRect(rect, 14, 14)

        font = QFont(self.font())
        font.setPointSize(max(12, int(self.height() / 26)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255, 235)))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()
