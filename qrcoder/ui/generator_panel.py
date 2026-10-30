"""The generation tab: payload builder, options, preview, save."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,
                               QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QScrollArea,
                               QSpinBox, QStackedWidget, QVBoxLayout,
                               QWidget)

from ..core.generator import (PAYLOAD_TYPES, GenerationError, GenerationRequest,
                              GenerationResult, build_payload, generate,
                              parse_payload, payload_type)

__all__ = ["GeneratorPanel"]


class GeneratorPanel(QWidget):
    """Full generation workflow, self-contained."""

    generated = Signal(object)      # GenerationResult

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result: Optional[GenerationResult] = None
        self._field_edits: Dict[str, QLineEdit] = {}

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        # ---- left: form --------------------------------------------------
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(0, 0, 6, 0)
        form_layout.setSpacing(10)
        form_scroll.setWidget(form_container)

        type_box = QGroupBox("内容类型")
        type_layout = QVBoxLayout(type_box)
        self._type_combo = QComboBox()
        for spec in PAYLOAD_TYPES:
            self._type_combo.addItem(spec.label, spec.key)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_layout.addWidget(self._type_combo)
        self._type_help = QLabel()
        self._type_help.setWordWrap(True)
        self._type_help.setStyleSheet("color:#8a8f98;")
        type_layout.addWidget(self._type_help)
        form_layout.addWidget(type_box)

        payload_box = QGroupBox("内容")
        payload_layout = QVBoxLayout(payload_box)
        self._field_stack = QStackedWidget()
        for spec in PAYLOAD_TYPES:
            page = QWidget()
            page_layout = QFormLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(6)
            for name, label, placeholder in spec.fields:
                if name == "text":
                    edit = QPlainTextEdit()
                    edit.setPlaceholderText(placeholder)
                    edit.setMinimumHeight(120)
                    setattr(self, "_text_edit", edit)
                    page_layout.addRow(label, edit)
                else:
                    edit = QLineEdit()
                    edit.setPlaceholderText(placeholder)
                    self._field_edits[(spec.key, name)] = edit
                    page_layout.addRow(label, edit)
            self._field_stack.addWidget(page)
        payload_layout.addWidget(self._field_stack)
        form_layout.addWidget(payload_box)

        option_box = QGroupBox("生成参数")
        option_layout = QFormLayout(option_box)
        self._error_combo = QComboBox()
        for level in ("L", "M", "Q", "H"):
            self._error_combo.addItem(level, level)
        self._error_combo.setCurrentIndex(1)
        option_layout.addRow("纠错等级", self._error_combo)

        self._auto_check = QCheckBox("自动选择版本（最小可行）")
        self._auto_check.setChecked(True)
        self._auto_check.toggled.connect(self._on_auto_toggled)
        option_layout.addRow(self._auto_check)

        self._version_spin = QSpinBox()
        self._version_spin.setRange(1, 40)
        self._version_spin.setValue(1)
        self._version_spin.setEnabled(False)
        option_layout.addRow("版本", self._version_spin)

        self._mask_combo = QComboBox()
        self._mask_combo.addItem("自动（按罚分规则）", None)
        for index in range(8):
            self._mask_combo.addItem("掩码 %d" % index, index)
        option_layout.addRow("掩码", self._mask_combo)

        self._scale_spin = QSpinBox()
        self._scale_spin.setRange(1, 40)
        self._scale_spin.setValue(8)
        option_layout.addRow("模块像素", self._scale_spin)

        self._border_spin = QSpinBox()
        self._border_spin.setRange(0, 16)
        self._border_spin.setValue(4)
        option_layout.addRow("静区宽度", self._border_spin)

        self._dark_edit = QLineEdit("#000000")
        self._light_edit = QLineEdit("#ffffff")
        option_layout.addRow("前景色", self._dark_edit)
        option_layout.addRow("背景色", self._light_edit)
        form_layout.addWidget(option_box)

        buttons = QHBoxLayout()
        self._generate_button = QPushButton("生成二维码")
        self._generate_button.setDefault(True)
        self._generate_button.clicked.connect(self._on_generate)
        buttons.addWidget(self._generate_button)
        self._load_button = QPushButton("从文本填充")
        self._load_button.setToolTip("把当前文本框内容解析回结构化字段")
        self._load_button.clicked.connect(self._on_load_from_text)
        buttons.addWidget(self._load_button)
        form_layout.addLayout(buttons)
        form_layout.addStretch(1)

        root.addWidget(form_scroll, 3)

        # ---- right: preview ---------------------------------------------
        preview_box = QGroupBox("预览")
        preview_layout = QVBoxLayout(preview_box)
        self._preview = QLabel("尚未生成")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(260, 260)
        self._preview.setStyleSheet(
            "background:#1e2024; border:1px solid #33363d; border-radius:6px;")
        preview_layout.addWidget(self._preview, 1)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        preview_layout.addWidget(self._summary)

        save_row = QHBoxLayout()
        self._save_png = QPushButton("保存 PNG")
        self._save_svg = QPushButton("保存 SVG")
        self._copy_button = QPushButton("复制到剪贴板")
        for button in (self._save_png, self._save_svg, self._copy_button):
            button.setEnabled(False)
            save_row.addWidget(button)
        self._save_png.clicked.connect(lambda: self._save("png"))
        self._save_svg.clicked.connect(lambda: self._save("svg"))
        self._copy_button.clicked.connect(self._copy_to_clipboard)
        preview_layout.addLayout(save_row)
        root.addWidget(preview_box, 2)

        self._on_type_changed(0)

    # -- state ------------------------------------------------------------
    def current_type(self) -> str:
        return str(self._type_combo.currentData() or "text")

    def _on_type_changed(self, index: int) -> None:
        self._field_stack.setCurrentIndex(index)
        spec = PAYLOAD_TYPES[index]
        self._type_help.setText(spec.help_text)

    def _on_auto_toggled(self, checked: bool) -> None:
        self._version_spin.setEnabled(not checked)

    def _collect_values(self) -> Dict[str, str]:
        kind = self.current_type()
        spec = payload_type(kind)
        values: Dict[str, str] = {}
        for name, _label, _placeholder in spec.fields:
            if name == "text":
                values[name] = self._text_edit.toPlainText()
            else:
                edit = self._field_edits.get((kind, name))
                if edit is not None:
                    values[name] = edit.text()
        return values

    def set_text(self, text: str) -> None:
        """Load *text* into the editor, switching type when it is recognised."""
        kind, values = parse_payload(text)
        for index, spec in enumerate(PAYLOAD_TYPES):
            if spec.key == kind:
                self._type_combo.setCurrentIndex(index)
                break
        if kind == "text":
            self._text_edit.setPlainText(values.get("text", text))
            return
        for name, value in values.items():
            if name == "text":
                self._text_edit.setPlainText(value)
                continue
            edit = self._field_edits.get((kind, name))
            if edit is not None:
                edit.setText(value)

    def _on_load_from_text(self) -> None:
        text = self._text_edit.toPlainText()
        if text.strip():
            self.set_text(text)

    # -- generation -------------------------------------------------------
    def _build_request(self) -> GenerationRequest:
        values = self._collect_values()
        content = build_payload(self.current_type(), values)
        mask = self._mask_combo.currentData()
        request = GenerationRequest(
            content=content,
            error=str(self._error_combo.currentData()),
            version=None if self._auto_check.isChecked() else int(self._version_spin.value()),
            mask=int(mask) if mask is not None else None,
            scale=int(self._scale_spin.value()),
            border=int(self._border_spin.value()),
            dark=self._dark_edit.text().strip() or "#000000",
            light=self._light_edit.text().strip() or "#ffffff",
        )
        return request

    def _on_generate(self) -> None:
        try:
            request = self._build_request()
        except GenerationError as exc:
            QMessageBox.warning(self, "无法生成", str(exc))
            return
        try:
            result = generate(request)
        except GenerationError as exc:
            QMessageBox.warning(self, "无法生成", str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.critical(self, "生成失败", "%s: %s" % (type(exc).__name__, exc))
            return

        self._result = result
        self._render_preview(result)
        for button in (self._save_png, self._save_svg, self._copy_button):
            button.setEnabled(True)
        self.generated.emit(result)

    def _render_preview(self, result: GenerationResult) -> None:
        size = result.matrix.shape[0]
        scale = max(1, int(300 / max(1, size)))
        img = np.where(result.matrix > 0, 0, 255).astype(np.uint8)
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
        height, width = img.shape
        qimage = QImage(img.data, width, height, width,
                        QImage.Format.Format_Grayscale8).copy()
        self._preview.setPixmap(QPixmap.fromImage(qimage))

        lines = [
            "版本：%d　尺寸：%d × %d" % (result.version, result.size, result.size),
            "纠错等级：%s　掩码：%d" % (result.error, result.mask),
            "编码模式：%s" % result.mode,
            "字符数：%d　字节数：%d" % (result.char_count, result.byte_count),
        ]
        if result.notes:
            lines.extend(result.notes)
        self._summary.setText("\n".join(lines))

    # -- export -----------------------------------------------------------
    def _save(self, kind: str) -> None:
        if self._result is None:
            return
        suffix = "png" if kind == "png" else "svg"
        filt = "PNG 图像 (*.png)" if kind == "png" else "SVG 矢量图 (*.svg)"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存二维码", os.path.join(os.path.expanduser("~"),
                                             "qrcode.%s" % suffix), filt)
        if not path:
            return
        try:
            data = (self._result.png_bytes() if kind == "png"
                    else self._result.svg_bytes())
            with open(path, "wb") as handle:
                handle.write(data)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", "%s: %s" % (type(exc).__name__, exc))
            return
        QMessageBox.information(self, "已保存", path)

    def _copy_to_clipboard(self) -> None:
        if self._result is None:
            return
        try:
            png = self._result.png_bytes()
        except Exception as exc:
            QMessageBox.critical(self, "复制失败", str(exc))
            return
        image = QImage()
        if not image.loadFromData(png, "PNG"):
            QMessageBox.critical(self, "复制失败", "图像编码失败")
            return
        QApplication.clipboard().setImage(image)
        QMessageBox.information(self, "已复制", "二维码图像已复制到剪贴板")

    # -- external ---------------------------------------------------------
    def result_image(self) -> Optional[np.ndarray]:
        """The generated matrix as a displayable grayscale array."""
        if self._result is None:
            return None
        size = self._result.matrix.shape[0]
        scale = max(1, int(400 / max(1, size)))
        img = np.where(self._result.matrix > 0, 0, 255).astype(np.uint8)
        return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
