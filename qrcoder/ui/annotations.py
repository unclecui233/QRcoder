"""Graphics items for the functional-region overlay.

Two item types do all the drawing:

* :class:`RegionItem` - a semi-transparent polygon with a border plus a label
  child that ignores the view transform, so labels stay legible at any zoom.
* :class:`GridItem` - the module grid, drawn as one path for cheap updates.

Status colours follow the spec: green = read/recognised, yellow = geometric
estimate, red = unreadable, blue = candidate.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QPainter,
                           QPainterPath, QPen, QPolygonF)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPolygonItem

__all__ = [
    "STATUS_COLORS",
    "STATUS_NAMES",
    "RegionItem",
    "GridItem",
    "make_polygon",
    "build_grid_path",
]

#: Fill, border and label colour per confidence status.
STATUS_COLORS = {
    "certain": (QColor(46, 204, 113, 46), QColor(39, 174, 96), QColor(16, 92, 50)),
    "estimated": (QColor(241, 196, 15, 56), QColor(214, 158, 0), QColor(120, 86, 0)),
    "unreadable": (QColor(231, 76, 60, 56), QColor(200, 45, 35), QColor(140, 22, 18)),
    "candidate": (QColor(52, 152, 219, 56), QColor(41, 128, 185), QColor(20, 78, 120)),
}

STATUS_NAMES = {
    "certain": "确定",
    "estimated": "几何估计",
    "unreadable": "无法读取",
    "candidate": "候选",
}


class _LabelItem(QGraphicsItem):
    """A text label that keeps a constant on-screen size.

    Drawing the pill background and the text in one item is what makes the
    ``ItemIgnoresTransformations`` flag work: the item's own painter is already
    unscaled, so the rounded rectangle tracks the text exactly.
    """

    def __init__(self, text: str, color: QColor, parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self._text = text
        self._color = QColor(color)
        font = QFont()
        font.setPointSizeF(8.5)
        font.setBold(True)
        self._font = font
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setZValue(20)
        self._metrics = QFontMetricsF(font)

    def set_text(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self.prepareGeometryChange()
            self.update()

    def text(self) -> str:
        return self._text

    def boundingRect(self) -> QRectF:
        width = self._metrics.horizontalAdvance(self._text) + 8.0
        height = self._metrics.height() + 2.0
        return QRectF(0.0, 0.0, width, height)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.boundingRect()
        painter.setBrush(QBrush(QColor(255, 255, 255, 225)))
        painter.setPen(QPen(QColor(self._color.red(), self._color.green(),
                                  self._color.blue(), 200), 1.0))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 3.0, 3.0)
        painter.setPen(QPen(self._color))
        painter.setFont(self._font)
        painter.drawText(rect.adjusted(4.0, 0.0, -4.0, 0.0),
                         int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         self._text)


# One accent colour per detected code so overlapping symbols stay distinguishable.
SYMBOL_COLORS = ("#3aa0ff", "#ff9f43", "#a55eea", "#2bcbba", "#fd79a8",
                 "#fdcb6e", "#45aaf2", "#26de81")


def _tint(accent: str, fill, border) -> tuple:
    """Blend a per-symbol accent into the status colours, keeping the status legible."""
    from PySide6.QtGui import QColor

    base = QColor(accent)
    fill_colour = QColor(base)
    fill_colour.setAlpha(fill.alpha() if hasattr(fill, "alpha") else 46)
    border_colour = base.lighter(115)
    text_colour = QColor(base)
    return fill_colour, border_colour, text_colour


class RegionItem(QGraphicsPolygonItem):
    """One annotated functional region: a polygon plus a floating label."""

    def __init__(self, key: str, label: str, status: str,
                 polygon: QPolygonF, detail: str = "",
                 parent: Optional[QGraphicsItem] = None,
                 accent: Optional[str] = None) -> None:
        super().__init__(polygon, parent)
        self.key = key
        self.status = status
        self.label_text = label
        self.detail = detail
        self.accent = accent

        fill, border, text = STATUS_COLORS.get(status, STATUS_COLORS["unreadable"])
        if accent:
            fill, border, text = _tint(accent, fill, border)
        self._fill = QColor(fill)
        self._border = QColor(border)
        self._text_color = QColor(text)

        self.setBrush(QBrush(self._fill))
        pen = QPen(self._border)
        pen.setWidthF(1.6)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

        tip = label
        if detail:
            tip = "%s\n%s" % (label, detail)
        tip = "%s\n状态：%s" % (tip, STATUS_NAMES.get(status, status))
        self.setToolTip(tip)

        self._label = _LabelItem(label, self._text_color, self)
        rect = self.boundingRect()
        self._label.setPos(rect.left(), rect.top())

    # -- state ------------------------------------------------------------
    def set_label_visible(self, visible: bool) -> None:
        self._label.setVisible(visible)

    def label_item(self) -> _LabelItem:
        return self._label

    def set_highlighted(self, highlighted: bool) -> None:
        pen = QPen(self._border)
        pen.setWidthF(3.0 if highlighted else 1.6)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(12 if highlighted else 10)
        self.update()

    # -- painting ---------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        brush = QColor(self._fill)
        if self.isUnderMouse():
            brush.setAlpha(min(150, brush.alpha() + 55))
        painter.setBrush(QBrush(brush))
        painter.setPen(self.pen())
        painter.drawPolygon(self.polygon())

    def hoverEnterEvent(self, event) -> None:
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.update()
        super().hoverLeaveEvent(event)


class GridItem(QGraphicsItem):
    """The module grid, drawn as a single path."""

    def __init__(self, path: QPainterPath, size: int,
                 parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self._path = path
        self._size = size
        self.setZValue(5)
        pen = QPen(QColor(60, 130, 220, 110))
        pen.setWidthF(0.0)
        pen.setCosmetic(True)
        self._pen = pen
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:
        return self._path.boundingRect().adjusted(-2, -2, 2, 2)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setPen(self._pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self._path)


def make_polygon(points: Sequence[Tuple[float, float]]) -> QPolygonF:
    return QPolygonF([QPointF(float(x), float(y)) for x, y in points])


def build_grid_path(transform, size: int, step: int = 1) -> QPainterPath:
    """Build the module grid path from a module->pixel homography."""
    import numpy as np

    path = QPainterPath()
    if transform is None:
        return path

    def to_pixel(mc: float, mr: float) -> QPointF:
        pt = transform @ np.array([mc, mr, 1.0])
        denom = pt[2] if abs(pt[2]) > 1e-12 else 1e-12
        return QPointF(float(pt[0] / denom), float(pt[1] / denom))

    step = max(1, int(step))
    indices = list(range(0, size + 1, step))
    if indices[-1] != size:
        indices.append(size)

    for index in indices:
        path.moveTo(to_pixel(float(index), 0.0))
        path.lineTo(to_pixel(float(index), float(size)))
    for index in indices:
        path.moveTo(to_pixel(0.0, float(index)))
        path.lineTo(to_pixel(float(size), float(index)))
    return path
