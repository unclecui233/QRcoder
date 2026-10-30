"""The main window: menus, toolbars, input handling, tabs, status bar."""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QAction, QActionGroup, QGuiApplication, QIcon,
                           QImage, QKeySequence, QPixmap)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFrame, QGroupBox, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPushButton,
                               QSizePolicy, QSplitter, QStatusBar, QTabWidget,
                               QToolBar, QVBoxLayout, QWidget)

from ..core import __version__ as CORE_VERSION
from ..core import symbologies
from ..core.symbols import ImageReport
from ..core.generator import GenerationResult
from ..core.preprocess import load_image, to_gray
from .canvas import ImageCanvas, numpy_to_qimage
from .dnd import (IMAGE_SUFFIXES, DropOverlay, DropPayload, DropRouter,
                  qimage_to_numpy)
from .generator_panel import GeneratorPanel
from .info_panel import InfoPanel
from .worker import AnalysisController

__all__ = ["MainWindow", "run"]

SUPPORTED_SUFFIXES = IMAGE_SUFFIXES


class LayerControls(QWidget):
    """Layer visibility switches above the canvas."""

    gridToggled = Signal(bool)
    labelsToggled = Signal(bool)
    functionalOnlyToggled = Signal(bool)
    groupToggled = Signal(str, bool)
    fitRequested = Signal()
    zoomInRequested = Signal()
    zoomOutRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        self._grid = QCheckBox("显示网格")
        self._grid.toggled.connect(self.gridToggled)
        self._labels = QCheckBox("显示标签")
        self._labels.setChecked(True)
        self._labels.toggled.connect(self.labelsToggled)
        self._functional = QCheckBox("只显示功能图案")
        self._functional.toggled.connect(self.functionalOnlyToggled)
        for widget in (self._grid, self._labels, self._functional):
            layout.addWidget(widget)

        layout.addWidget(_vline())

        self._groups = {}
        for key, label in (("finder", "定位"), ("timing", "时序"),
                           ("align", "对齐"), ("format", "格式"),
                           ("version", "版本"), ("data", "数据区")):
            box = QCheckBox(label)
            box.setChecked(True)
            box.toggled.connect(lambda checked, k=key: self.groupToggled.emit(k, checked))
            self._groups[key] = box
            layout.addWidget(box)

        layout.addWidget(_vline())

        fit_button = QPushButton("适应窗口")
        fit_button.clicked.connect(self.fitRequested)
        layout.addWidget(fit_button)
        zoom_in = QPushButton("＋")
        zoom_in.setFixedWidth(30)
        zoom_in.clicked.connect(self.zoomInRequested)
        layout.addWidget(zoom_in)
        zoom_out = QPushButton("－")
        zoom_out.setFixedWidth(30)
        zoom_out.clicked.connect(self.zoomOutRequested)
        layout.addWidget(zoom_out)

        layout.addStretch(1)
        self._zoom_label = QLabel("100%")
        self._zoom_label.setMinimumWidth(52)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._zoom_label)

    def set_zoom_text(self, scale: float) -> None:
        self._zoom_label.setText("%d%%" % int(round(scale * 100)))


def _vline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


class MainWindow(QMainWindow):
    """Application shell."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("QR 码分析与生成器")
        self.resize(1360, 900)
        self.setAcceptDrops(True)

        self._image: Optional[np.ndarray] = None
        self._image_path: Optional[str] = None
        self._report: Optional[ImageReport] = None
        self._scope = symbologies.SCOPE_ALL
        self._custom_keys: List[str] = []
        self._controller = AnalysisController(self)
        self._controller.started.connect(self._on_analysis_started)
        self._controller.finished.connect(self._on_analysis_finished)
        self._controller.failed.connect(self._on_analysis_failed)

        self._build_tabs()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_status_bar()
        self._build_dnd()

        clipboard = QApplication.clipboard()
        clipboard.dataChanged.connect(self._on_clipboard_changed)
        self._clipboard_timer = QTimer(self)
        self._clipboard_timer.setInterval(700)
        self._clipboard_timer.timeout.connect(self._poll_clipboard)
        self._clipboard_timer.start()
        self._last_clipboard_image_hash: Optional[int] = None

        self._update_actions()

    # -- construction ------------------------------------------------------
    def _build_tabs(self) -> None:
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # analysis page
        analysis_page = QWidget()
        page_layout = QVBoxLayout(analysis_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        self._layers = LayerControls()
        self._layers.gridToggled.connect(self._on_grid_toggled)
        self._layers.labelsToggled.connect(self._on_labels_toggled)
        self._layers.functionalOnlyToggled.connect(self._on_functional_toggled)
        self._layers.groupToggled.connect(self._on_group_toggled)
        self._layers.fitRequested.connect(self._on_fit)
        self._layers.zoomInRequested.connect(lambda: self._canvas.zoom_by(1.25))
        self._layers.zoomOutRequested.connect(lambda: self._canvas.zoom_by(1 / 1.25))
        page_layout.addWidget(self._layers)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._canvas = ImageCanvas()
        self._canvas.zoomChanged.connect(self._layers.set_zoom_text)
        splitter.addWidget(self._canvas)

        self._info = InfoPanel()
        self._info.regionActivated.connect(self._on_region_activated)
        self._info.symbolActivated.connect(self._on_symbol_activated)
        self._info.setMinimumWidth(330)
        self._info.setMaximumWidth(520)
        splitter.addWidget(self._info)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([960, 400])
        page_layout.addWidget(splitter, 1)
        self._tabs.addTab(analysis_page, "分析")

        # generation page
        self._generator = GeneratorPanel()
        self._generator.generated.connect(self._on_generated)
        self._tabs.addTab(self._generator, "生成")

        self._tabs.currentChanged.connect(self._on_tab_changed)

    def _build_scope_menu(self) -> None:
        menu = self.menuBar().addMenu("码制(&S)")
        group = QActionGroup(self)
        group.setExclusive(True)
        self._scope_actions = {}
        for scope in (symbologies.SCOPE_ALL, symbologies.SCOPE_QR,
                      symbologies.SCOPE_CUSTOM):
            action = QAction(symbologies.SCOPE_LABELS[scope], self)
            action.setCheckable(True)
            action.setChecked(scope == self._scope)
            action.triggered.connect(lambda _=False, s=scope: self.set_scope(s))
            group.addAction(action)
            menu.addAction(action)
            self._scope_actions[scope] = action
        menu.addSeparator()
        pick = QAction("自定义码制…", self)
        pick.triggered.connect(self._pick_custom_symbologies)
        menu.addAction(pick)

    def set_scope(self, scope: str) -> None:
        self._scope = scope
        action = self._scope_actions.get(scope)
        if action is not None:
            action.setChecked(True)
        self._status_label.setText("识别范围：%s" % symbologies.SCOPE_LABELS.get(scope, scope))
        if self._image is not None:
            self._controller.request(self._image,
                                     keys=symbologies.keys_for_scope(scope,
                                                                    self._custom_keys))

    def _pick_custom_symbologies(self) -> None:
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("自定义识别的码制")
        layout = QVBoxLayout(dialog)
        boxes = {}
        for symbology in symbologies.all():
            if not symbology.engine_supported:
                continue
            box = QCheckBox("%s（%s）" % (symbology.name, symbology.capability_label))
            box.setChecked(not self._custom_keys or symbology.key in self._custom_keys)
            layout.addWidget(box)
            boxes[symbology.key] = box
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._custom_keys = [key for key, box in boxes.items() if box.isChecked()]
        self.set_scope(symbologies.SCOPE_CUSTOM)

    def _build_actions(self) -> None:
        self._open_action = QAction("打开图片…", self)
        self._open_action.setShortcut(QKeySequence.StandardKey.Open)
        self._open_action.triggered.connect(self.open_file_dialog)

        self._paste_action = QAction("从剪贴板粘贴", self)
        self._paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        self._paste_action.triggered.connect(self.paste_from_clipboard)

        self._export_json_action = QAction("导出报告 (JSON)…", self)
        self._export_json_action.triggered.connect(lambda: self._export_report("json"))

        self._export_text_action = QAction("导出报告 (文本)…", self)
        self._export_text_action.triggered.connect(lambda: self._export_report("text"))

        self._copy_content_action = QAction("复制解码内容", self)
        self._copy_content_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self._copy_content_action.triggered.connect(self._copy_content)

        self._fit_action = QAction("适应窗口", self)
        self._fit_action.setShortcut(QKeySequence("Ctrl+0"))
        self._fit_action.triggered.connect(self._on_fit)

        self._grid_action = QAction("显示网格", self)
        self._grid_action.setCheckable(True)
        self._grid_action.setShortcut(QKeySequence("Ctrl+G"))
        self._grid_action.toggled.connect(self._layers._grid.setChecked)
        self._layers._grid.toggled.connect(self._grid_action.setChecked)

        self._quit_action = QAction("退出", self)
        self._quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self._quit_action.triggered.connect(self.close)

        self._about_action = QAction("关于", self)
        self._about_action.triggered.connect(self._show_about)

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件(&F)")
        file_menu.addAction(self._open_action)
        file_menu.addAction(self._paste_action)
        file_menu.addSeparator()
        file_menu.addAction(self._export_json_action)
        file_menu.addAction(self._export_text_action)
        file_menu.addSeparator()
        file_menu.addAction(self._quit_action)

        edit_menu = menubar.addMenu("编辑(&E)")
        edit_menu.addAction(self._copy_content_action)

        view_menu = menubar.addMenu("视图(&V)")
        view_menu.addAction(self._fit_action)
        view_menu.addAction(self._grid_action)

        self._build_scope_menu()

        help_menu = menubar.addMenu("帮助(&H)")
        help_menu.addAction(self._about_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(self._open_action)
        toolbar.addAction(self._paste_action)
        toolbar.addSeparator()
        toolbar.addAction(self._export_json_action)
        toolbar.addAction(self._export_text_action)
        toolbar.addSeparator()
        toolbar.addAction(self._fit_action)
        toolbar.addAction(self._grid_action)
        self._toolbar = toolbar

    def _build_dnd(self) -> None:
        # Drag events land on the deepest widget under the cursor, which for the
        # main area is QGraphicsView.viewport().  Rather than relying on bubbling
        # (which QGraphicsView blocks), an application-level filter takes over
        # every drag inside this window: canvas, panels, tabs, status bar.
        self._drop_overlay = DropOverlay(self)
        self._drop_router = DropRouter(self, self)
        self._drop_router.dropped.connect(self._on_dropped)
        self._drop_router.dragActiveChanged.connect(self._on_drag_active)
        self._drop_router.attach()

    def _on_drag_active(self, active: bool) -> None:
        if active:
            self._drop_overlay.show_for_parent()
        else:
            self._drop_overlay.hide()

    def _on_dropped(self, payload: DropPayload) -> None:
        self._drop_overlay.hide()
        if not payload.usable:
            self._status_label.setText(payload.status_text())
            return
        if payload.image is not None:
            self.set_image(payload.image, payload.status_text())
            return
        path = payload.paths[0]
        try:
            image = load_image(path)
        except Exception as exc:
            self._status_label.setText("无法读取 %s：%s"
                                       % (os.path.basename(path), exc))
            return
        self._image_path = path
        self.setWindowTitle("QR 码分析与生成器 — %s" % os.path.basename(path))
        self.set_image(image, payload.status_text())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._drop_overlay.isVisible():
            self._drop_overlay.setGeometry(self.rect())

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self._status_label = QLabel("拖入图片、粘贴图片或使用「文件 → 打开图片」")
        bar.addWidget(self._status_label, 1)
        self._timing_label = QLabel("")
        bar.addPermanentWidget(self._timing_label)
        self._engine_label = QLabel("")
        bar.addPermanentWidget(self._engine_label)

    # -- input -------------------------------------------------------------
    def open_file_dialog(self) -> None:
        filters = "图像文件 (%s);;所有文件 (*)" % " ".join("*" + s for s in SUPPORTED_SUFFIXES)
        path, _ = QFileDialog.getOpenFileName(self, "打开二维码图片", "", filters)
        if path:
            self.load_path(path)

    def load_path(self, path: str) -> None:
        try:
            image = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "无法加载图像", str(exc))
            return
        self._image_path = path
        self.setWindowTitle("QR 码分析与生成器 — %s" % os.path.basename(path))
        self.set_image(image)

    def set_image(self, image: np.ndarray, source: str = "") -> None:
        self._image = image
        self._tabs.setCurrentIndex(0)
        self._canvas.set_image(image)
        self._info.clear()
        self._report = None
        self._status_label.setText(source or "正在分析…")
        self._timing_label.setText("")
        self._engine_label.setText("")
        self._update_actions()
        self._controller.request(
            image, keys=symbologies.keys_for_scope(self._scope, self._custom_keys))

    def paste_from_clipboard(self) -> None:
        image = self._clipboard_image()
        if image is None:
            self._status_label.setText("剪贴板中没有图像")
            return
        self._last_clipboard_image_hash = hash(image.tobytes())
        self.set_image(image, "已从剪贴板载入")

    def _clipboard_image(self) -> Optional[np.ndarray]:
        clipboard = QApplication.clipboard()
        qimage = clipboard.image()
        if not qimage.isNull():
            return self._qimage_to_numpy(qimage)
        # Some applications expose the image only as a URL list.
        mime = clipboard.mimeData()
        if mime is not None and mime.hasUrls():
            for url in mime.urls():
                local = url.toLocalFile()
                if local and local.lower().endswith(SUPPORTED_SUFFIXES):
                    try:
                        return load_image(local)
                    except Exception:
                        continue
        return None

    @staticmethod
    def _qimage_to_numpy(qimage: QImage) -> np.ndarray:
        return qimage_to_numpy(qimage)

    def _on_clipboard_changed(self) -> None:
        # Debounced: applications fire dataChanged several times per copy.
        QTimer.singleShot(120, self._poll_clipboard)

    def _poll_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()
        if mime is None:
            return
        if not (mime.hasImage() or (mime.hasUrls() and any(
                u.toLocalFile().lower().endswith(SUPPORTED_SUFFIXES)
                for u in mime.urls()))):
            return
        qimage = clipboard.image()
        if qimage.isNull():
            return
        digest = hash(qimage.bits().tobytes() if hasattr(qimage.bits(), "tobytes")
                      else bytes(qimage.constBits()))
        if digest == self._last_clipboard_image_hash:
            return
        self._last_clipboard_image_hash = digest
        self._status_label.setText("剪贴板中有图像，按 Ctrl+V 载入分析")

    # -- drag and drop -----------------------------------------------------
    # Handled centrally by DropRouter (see _build_dnd); the window keeps
    # acceptDrops for platforms that probe it before starting a drag.

    # -- analysis callbacks ------------------------------------------------
    def _on_analysis_started(self) -> None:
        self._status_label.setText("正在分析…")

    def _on_analysis_finished(self, report: ImageReport) -> None:
        self._report = report
        self._canvas.apply_image_report(report, fit=True)
        self._info.show_image_report(report)
        self._status_label.setText(report.headline)
        stages = "，".join("%s %.2fs" % (name, seconds)
                          for name, seconds in report.stage_times.items())
        self._timing_label.setText("%.2f 秒%s" % (
            report.elapsed, ("（%s）" % stages) if stages else ""))
        primary = report.primary
        if primary is None:
            self._engine_label.setText("")
        else:
            engines = primary.engine or "—"
            if primary.engines_agree and primary.decoded:
                engines += "（双引擎一致）"
            if len(report.symbols) > 1:
                engines = "%d 个码｜%s" % (len(report.symbols), engines)
            self._engine_label.setText("引擎：%s" % engines)
        self._update_actions()

    def _on_analysis_failed(self, message: str) -> None:
        self._status_label.setText("分析失败：%s" % message)
        QMessageBox.warning(self, "分析失败",
                            "分析过程出现异常，但程序仍可继续使用。\n\n%s" % message)

    # -- view callbacks ----------------------------------------------------
    def _on_fit(self) -> None:
        self._canvas.fit_to_window()

    def _on_grid_toggled(self, visible: bool) -> None:
        self._canvas.set_grid_visible(visible)

    def _on_labels_toggled(self, visible: bool) -> None:
        self._canvas.set_labels_visible(visible)

    def _on_functional_toggled(self, enabled: bool) -> None:
        self._canvas.set_functional_only(enabled)

    def _on_group_toggled(self, group: str, visible: bool) -> None:
        self._canvas.set_group_visible(group, visible)

    def _on_region_activated(self, key: str) -> None:
        if self._report is not None and len(self._report.symbols) > 1:
            for symbol in self._report.symbols:
                if any(a.key == key for a in symbol.annotations):
                    self._info._populate_symbol(symbol)
                    self._canvas.focus_symbol(symbol.index)
                    return
        self._canvas.focus_region(key)

    def _on_symbol_activated(self, index: int) -> None:
        if self._report is None:
            return
        for symbol in self._report.symbols:
            if symbol.index == index:
                self._info._populate_symbol(symbol)
                break
        self._canvas.focus_symbol(index)

    def _on_tab_changed(self, index: int) -> None:
        if index == 1:
            self._status_label.setText("生成模式")

    def _on_generated(self, result: GenerationResult) -> None:
        self._status_label.setText(
            "已生成：版本 %d，%s，掩码 %d，%s"
            % (result.version, result.error, result.mask, result.mode))

    # -- export ------------------------------------------------------------
    def _export_report(self, kind: str) -> None:
        if self._report is None:
            QMessageBox.information(self, "暂无报告", "请先加载并分析一张图片。")
            return
        suffix = "json" if kind == "json" else "txt"
        filt = "JSON 文件 (*.json)" if kind == "json" else "文本文件 (*.txt)"
        default = os.path.join(os.path.expanduser("~"), "qrcode-report.%s" % suffix)
        path, _ = QFileDialog.getSaveFileName(self, "导出报告", default, filt)
        if not path:
            return
        try:
            if kind == "json":
                payload = json.dumps(self._report.to_dict(), ensure_ascii=False, indent=2)
            else:
                payload = self._report.to_text()
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(payload)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", "%s: %s" % (type(exc).__name__, exc))
            return
        self._status_label.setText("报告已导出：%s" % path)

    def _copy_content(self) -> None:
        if self._report is None or not self._report.decoded_count:
            QMessageBox.information(self, "暂无内容", "尚未成功解码。")
            return
        texts = [s.text for s in self._report.symbols if s.decoded]
        QApplication.clipboard().setText("\n".join(texts))
        self._status_label.setText("解码内容已复制到剪贴板")

    def _update_actions(self) -> None:
        has_report = self._report is not None
        self._export_json_action.setEnabled(has_report)
        self._export_text_action.setEnabled(has_report)
        self._copy_content_action.setEnabled(
            bool(self._report and self._report.decoded_count))

    def _show_about(self) -> None:
        from ..core.engines import available_engines

        QMessageBox.about(
            self, "关于",
            "<h3>二维码分析与生成器</h3>"
            "<p>目前可以识别大部分的二维码</p>"
            "<p>解码引擎：%s</p>"
            "<p>版本 %s</p>"
            "<p>应该也不会更新了(</p>" % ("、".join(available_engines()), CORE_VERSION))

    # -- shutdown ----------------------------------------------------------
    def closeEvent(self, event) -> None:
        self._clipboard_timer.stop()
        self._drop_router.detach()
        self._controller.wait(8000)
        super().closeEvent(event)


def run(argv: Optional[List[str]] = None) -> int:
    """Start the GUI (or analyse a path given on the command line)."""
    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("QR 码分析与生成器")
    app.setOrganizationName("QRcoder")

    window = MainWindow()
    window.show()

    for argument in argv[1:]:
        if argument.lower().endswith(SUPPORTED_SUFFIXES) and os.path.isfile(argument):
            window.load_path(argument)
            break

    return app.exec()
