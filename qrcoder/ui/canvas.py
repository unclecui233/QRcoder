"""The central image canvas: zoom, pan, overlay layers, region focus."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView)

from ..core.symbols import ImageReport
from .annotations import (SYMBOL_COLORS, GridItem, RegionItem,
                          build_grid_path, make_polygon)

__all__ = ["ImageCanvas", "numpy_to_qimage"]


def numpy_to_qimage(image: np.ndarray) -> QImage:
    """Convert a BGR/gray uint8 numpy array to a QImage that owns its memory."""
    if image.ndim == 2:
        gray = np.ascontiguousarray(image)
        height, width = gray.shape
        qimage = QImage(gray.data, width, height, width,
                        QImage.Format.Format_Grayscale8)
        return qimage.copy()
    if image.shape[2] == 4:
        bgra = np.ascontiguousarray(image)
        height, width = bgra.shape[:2]
        qimage = QImage(bgra.data, width, height, bgra.strides[0],
                        QImage.Format.Format_ARGB32)
        return qimage.copy()
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    height, width = rgb.shape[:2]
    qimage = QImage(rgb.data, width, height, rgb.strides[0],
                    QImage.Format.Format_RGB888)
    return qimage.copy()


#: Which annotation groups the "only functional patterns" toggle keeps.
FUNCTIONAL_PREFIXES = (
    "finder_", "timing_", "align_", "format_", "version_",
)


class ImageCanvas(QGraphicsView):
    """Shows the source image with the analysis overlay on top."""

    regionClicked = Signal(str)
    zoomChanged = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing
                            | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor(38, 40, 44))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._grid_item: Optional[GridItem] = None
        self._regions: Dict[str, RegionItem] = {}
        self._symbol_items: Dict[int, list] = {}
        self._symbol_quads: Dict[int, list] = {}
        self._hidden_symbols: set = set()
        self._report: Optional[ImageReport] = None
        self._coord_scale = 1.0
        self._image: Optional[np.ndarray] = None

        # Layer switches.
        self._grid_visible = False
        self._labels_visible = True
        self._functional_only = False
        self._hidden_groups: set = set()

    # -- content ----------------------------------------------------------
    def set_image(self, image: np.ndarray, keep_view: bool = False) -> None:
        self._image = image
        qimage = numpy_to_qimage(image)
        pixmap = QPixmap.fromImage(qimage)
        if self._pixmap_item is None:
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._pixmap_item.setZValue(0)
            self._scene.addItem(self._pixmap_item)
        else:
            self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        if not keep_view:
            self.fit_to_window()

    def clear_overlay(self) -> None:
        for item in self._regions.values():
            self._scene.removeItem(item)
        self._regions.clear()
        self._symbol_items = {}
        self._symbol_quads = {}
        self._hidden_symbols = set()
        if self._grid_item is not None:
            self._scene.removeItem(self._grid_item)
            self._grid_item = None
        self._report = None

    def apply_report(self, report: AnalysisReport, image: Optional[np.ndarray] = None,
                     fit: bool = True) -> None:
        """Draw the annotation overlay for *report*."""
        if image is not None:
            self.set_image(image, keep_view=not fit)
        self.clear_overlay()
        self._report = report
        self._coord_scale = report.coord_scale or 1.0

        # Annotations are expressed in the analysed (possibly downscaled) image,
        # so scale them back into original pixel space for display.
        factor = 1.0 / self._coord_scale if self._coord_scale else 1.0
        for annotation in report.annotations:
            points = [(x * factor, y * factor) for x, y in annotation.polygon]
            if len(points) < 3:
                continue
            item = RegionItem(annotation.key, annotation.label, annotation.status,
                              make_polygon(points), annotation.detail)
            item.set_label_visible(self._labels_visible)
            self._scene.addItem(item)
            self._regions[annotation.key] = item

        if report.transform is not None and report.matrix is not None:
            size = int(report.matrix.shape[0])
            scaled = np.array(report.transform, dtype=np.float64)
            # Module->pixel in analysis space; convert to display space.
            scaled = scaled @ np.array([[factor, 0.0, 0.0],
                                        [0.0, factor, 0.0],
                                        [0.0, 0.0, 1.0]])
            step = 1
            if size > 80:
                step = 4
            elif size > 45:
                step = 2
            path = build_grid_path(scaled, size, step)
            self._grid_item = GridItem(path, size)
            self._grid_item.setVisible(self._grid_visible)
            self._scene.addItem(self._grid_item)

        self._apply_group_visibility()
        if fit:
            self.fit_to_window()

    # -- view -------------------------------------------------------------
    def fit_to_window(self) -> None:
        if self._pixmap_item is None:
            return
        rect = self._pixmap_item.boundingRect()
        if rect.isEmpty():
            return
        self.fitInView(rect.adjusted(-8, -8, 8, 8), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(self.current_scale())

    def reset_zoom(self) -> None:
        self.resetTransform()
        self.zoomChanged.emit(1.0)

    def current_scale(self) -> float:
        return float(self.transform().m11())

    def zoom_by(self, factor: float) -> None:
        scale = self.current_scale()
        target = max(0.05, min(40.0, scale * factor))
        factor = target / scale if scale else 1.0
        self.scale(factor, factor)
        self.zoomChanged.emit(self.current_scale())

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            self.zoom_by(1.15 if delta > 0 else 1 / 1.15)
            event.accept()
            return
        # Plain wheel also zooms: this is an image viewer, not a document.
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(1.15 if delta > 0 else 1 / 1.15)
            event.accept()
            return
        super().wheelEvent(event)

    # -- layer switches ---------------------------------------------------
    def apply_image_report(self, report, image=None, fit: bool = True) -> None:
        """Draw annotations for every code found in *report* (an ImageReport)."""
        if image is not None:
            self.set_image(image, keep_view=not fit)
        self.clear_overlay()
        self._report = report
        self._symbol_items = {}
        self._coord_scale = getattr(report, "coord_scale", 1.0) or 1.0

        for symbol in report.symbols:
            accent = SYMBOL_COLORS[symbol.index % len(SYMBOL_COLORS)]
            items = []
            for annotation in symbol.annotations:
                points = list(annotation.polygon)
                if len(points) < 3:
                    continue
                # Multi-code labels carry the symbology name to avoid confusion.
                label = annotation.label
                if len(report.symbols) > 1:
                    label = "%s｜%s" % (symbol.symbology_name, label)
                item = RegionItem(annotation.key, label, annotation.status,
                                  make_polygon(points), annotation.detail,
                                  accent=accent)
                item.set_label_visible(self._labels_visible)
                self._scene.addItem(item)
                self._regions[annotation.key] = item
                items.append(item)
            self._symbol_items[symbol.index] = items
            self._symbol_quads[symbol.index] = list(symbol.quad)

        self._grid_item = None
        primary = report.primary
        if primary is not None and getattr(primary, "transform", None) is not None \
                and getattr(primary, "matrix", None) is not None:
            size = int(primary.matrix.shape[0])
            factor = 1.0 / self._coord_scale if self._coord_scale else 1.0
            scaled = np.array(primary.transform, dtype=np.float64)
            scaled = scaled @ np.array([[factor, 0.0, 0.0],
                                        [0.0, factor, 0.0],
                                        [0.0, 0.0, 1.0]])
            step = 4 if size > 80 else (2 if size > 45 else 1)
            self._grid_item = GridItem(build_grid_path(scaled, size, step), size)
            self._grid_item.setVisible(self._grid_visible)
            self._scene.addItem(self._grid_item)

        self._apply_group_visibility()
        if fit:
            self.fit_to_window()

    def focus_symbol(self, index: int) -> None:
        """Highlight and zoom to the code with the given index."""
        items = self._symbol_items.get(index)
        for other_index, other_items in self._symbol_items.items():
            for item in other_items:
                item.set_highlighted(other_index == index)
        quad = self._symbol_quads.get(index)
        if not quad:
            return
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        if rect.width() <= 0 or rect.height() <= 0:
            return
        margin = max(rect.width(), rect.height()) * 0.6
        self.fitInView(rect.adjusted(-margin, -margin, margin, margin),
                       Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(self.current_scale())

    def set_symbol_visible(self, index: int, visible: bool) -> None:
        if visible:
            self._hidden_symbols.discard(index)
        else:
            self._hidden_symbols.add(index)
        self._apply_group_visibility()

    def symbol_indexes(self) -> list:
        return sorted(self._symbol_items)

    def set_grid_visible(self, visible: bool) -> None:
        self._grid_visible = visible
        if self._grid_item is not None:
            self._grid_item.setVisible(visible)

    def set_labels_visible(self, visible: bool) -> None:
        self._labels_visible = visible
        for item in self._regions.values():
            item.set_label_visible(visible)

    def set_functional_only(self, enabled: bool) -> None:
        self._functional_only = enabled
        self._apply_group_visibility()

    def set_group_visible(self, group: str, visible: bool) -> None:
        if visible:
            self._hidden_groups.discard(group)
        else:
            self._hidden_groups.add(group)
        self._apply_group_visibility()

    def _apply_group_visibility(self) -> None:
        for index, items in self._symbol_items.items():
            if index in self._hidden_symbols:
                for item in items:
                    item.setVisible(False)
                continue
            for item in items:
                visible = True
                if self._functional_only and not item.key.startswith(FUNCTIONAL_PREFIXES):
                    visible = False
                if _group_of(item.key) in self._hidden_groups:
                    visible = False
                item.setVisible(visible)
        # single-symbol overlay (legacy path)
        for key, item in self._regions.items():
            if any(item in items for items in self._symbol_items.values()):
                continue
            visible = True
            if self._functional_only and not key.startswith(FUNCTIONAL_PREFIXES):
                visible = False
            if _group_of(key) in self._hidden_groups:
                visible = False
            item.setVisible(visible)

    # -- focus ------------------------------------------------------------
    def focus_region(self, key: str) -> None:
        item = self._regions.get(key)
        if item is None:
            return
        for other in self._regions.values():
            other.set_highlighted(other is item)
        rect = item.boundingRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        margin = max(rect.width(), rect.height()) * 1.6
        target = rect.adjusted(-margin, -margin, margin, margin)
        self.fitInView(target, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(self.current_scale())

    def clear_highlight(self) -> None:
        for item in self._regions.values():
            item.set_highlighted(False)


def _group_of(key: str) -> str:
    for prefix in ("finder_", "timing_", "align_", "format_", "version_", "data_"):
        if key.startswith(prefix):
            return prefix.rstrip("_")
    return "other"
