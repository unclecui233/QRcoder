"""The right-hand information panel: decode status, QR parameters, regions."""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QPushButton,
                               QScrollArea, QSizePolicy, QTextEdit,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from ..core import symbologies
from ..core.symbols import ImageReport, SymbolReport
from ..core.analysis import (AnalysisReport, CERTAIN, CANDIDATE, ESTIMATED,
                             STATUS_LABELS, UNREADABLE)
from .annotations import STATUS_COLORS

__all__ = ["InfoPanel"]

_STATUS_DOT = {
    CERTAIN: "#27ae60",
    ESTIMATED: "#d69e00",
    UNREADABLE: "#c82333",
    CANDIDATE: "#2980b9",
}


def _dot_pixmap(color: str, size: int = 10) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return pixmap


class _FieldRow(QWidget):
    """A ``label : value`` row with a status dot."""

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)

        self._dot = QLabel()
        self._dot.setFixedWidth(12)
        self._dot.setPixmap(_dot_pixmap("#888888"))
        self._label = QLabel(label)
        self._label.setMinimumWidth(76)
        self._label.setStyleSheet("color:#8a8f98;")
        self._value = QLabel("—")
        self._value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._value.setWordWrap(True)

        layout.addWidget(self._dot)
        layout.addWidget(self._label)
        layout.addWidget(self._value, 1)

    def set(self, value: str, status: str = "", tooltip: str = "") -> None:
        self._value.setText(value)
        color = _STATUS_DOT.get(status, "#888888")
        self._dot.setPixmap(_dot_pixmap(color))
        bold = status in (CERTAIN,)
        font = self._value.font()
        font.setBold(bold)
        self._value.setFont(font)
        self._value.setToolTip(tooltip)
        self._dot.setToolTip(tooltip)


class InfoPanel(QScrollArea):
    """Structured report view."""

    regionActivated = Signal(str)
    symbolActivated = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(10)
        self.setWidget(container)

        self._headline = QLabel("尚未加载图像")
        head_font = QFont()
        head_font.setPointSizeF(12.0)
        head_font.setBold(True)
        self._headline.setFont(head_font)
        self._headline.setWordWrap(True)
        self._headline.setStyleSheet("padding:6px; border-radius:6px; background:#2c2f36;")
        self._layout.addWidget(self._headline)

        # --- decode status --------------------------------------------
        self._decode_box = QGroupBox("解码状态")
        decode_layout = QVBoxLayout(self._decode_box)
        decode_layout.setSpacing(2)
        self._row_symbology = _FieldRow("码制类型")
        self._row_capability = _FieldRow("识别能力")
        self._row_decoded = _FieldRow("完整解码")
        self._row_engine = _FieldRow("采纳引擎")
        self._row_agree = _FieldRow("引擎一致性")
        self._row_variant = _FieldRow("预处理")
        self._row_geometry = _FieldRow("网格定位")
        self._row_stage = _FieldRow("失败阶段")
        self._row_reason = _FieldRow("原因")
        for row in (self._row_symbology, self._row_capability,
                    self._row_decoded, self._row_engine, self._row_agree,
                    self._row_variant, self._row_geometry, self._row_stage,
                    self._row_reason):
            decode_layout.addWidget(row)
        self._layout.addWidget(self._decode_box)

        # --- detected codes (hidden unless more than one) --------------
        self._symbol_box = QGroupBox("检测到的码（点击定位）")
        symbol_layout = QVBoxLayout(self._symbol_box)
        self._symbol_list = QListWidget()
        self._symbol_list.setAlternatingRowColors(True)
        self._symbol_list.setMaximumHeight(132)
        self._symbol_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._symbol_list.itemClicked.connect(self._on_symbol_clicked)
        symbol_layout.addWidget(self._symbol_list)
        self._symbol_box.setVisible(False)
        self._layout.addWidget(self._symbol_box)

        # --- parameters (rows rebuilt per symbology) -------------------
        self._param_box = QGroupBox("码参数")
        self._param_layout = QVBoxLayout(self._param_box)
        self._param_layout.setSpacing(2)
        self._param_rows: Dict[str, _FieldRow] = {}
        self._layout.addWidget(self._param_box)

        # --- content ---------------------------------------------------
        self._content_box = QGroupBox("内容")
        content_layout = QVBoxLayout(self._content_box)
        self._row_ctype = _FieldRow("内容类型")
        content_layout.addWidget(self._row_ctype)
        self._content = QTextEdit()
        self._content.setReadOnly(True)
        self._content.setMinimumHeight(84)
        self._content.setPlaceholderText("解码成功后在此显示内容")
        content_layout.addWidget(self._content)
        buttons = QHBoxLayout()
        self._copy_button = QPushButton("复制内容")
        self._copy_button.setEnabled(False)
        buttons.addWidget(self._copy_button)
        buttons.addStretch(1)
        content_layout.addLayout(buttons)
        self._layout.addWidget(self._content_box)

        # --- region summary -------------------------------------------
        self._region_box = QGroupBox("区域识别汇总")
        region_layout = QVBoxLayout(self._region_box)
        region_layout.setSpacing(2)
        self._row_finders = _FieldRow("定位图案")
        self._row_align = _FieldRow("对齐图案")
        self._row_format = _FieldRow("格式信息")
        self._row_verinfo = _FieldRow("版本信息")
        self._row_data = _FieldRow("数据区")
        self._row_corrected = _FieldRow("纠错字数")
        for row in (self._row_finders, self._row_align, self._row_format,
                    self._row_verinfo, self._row_data, self._row_corrected):
            region_layout.addWidget(row)
        self._layout.addWidget(self._region_box)

        # --- annotation list ------------------------------------------
        self._list_box = QGroupBox("标注区域（点击定位）")
        list_layout = QVBoxLayout(self._list_box)
        self._region_list = QListWidget()
        self._region_list.setAlternatingRowColors(True)
        self._region_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._region_list.itemClicked.connect(self._on_region_clicked)
        list_layout.addWidget(self._region_list)
        self._layout.addWidget(self._list_box)

        self._layout.addStretch(1)
        self._copy_button.clicked.connect(self._copy_content)

    # -- helpers ----------------------------------------------------------
    def _on_symbol_clicked(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None:
            return
        self.symbolActivated.emit(int(index))

    def _on_region_clicked(self, item: QListWidgetItem) -> None:
        key = item.data(Qt.ItemDataRole.UserRole)
        if key:
            self.regionActivated.emit(str(key))

    def _copy_content(self) -> None:
        from PySide6.QtWidgets import QApplication

        text = self._content.toPlainText()
        if text:
            QApplication.clipboard().setText(text)

    def clear(self) -> None:
        self._headline.setText("尚未加载图像")
        self._headline.setStyleSheet("padding:6px; border-radius:6px; background:#2c2f36;")
        for row in (self._row_symbology, self._row_capability,
                    self._row_decoded, self._row_engine, self._row_agree,
                    self._row_variant, self._row_geometry, self._row_stage,
                    self._row_reason, self._row_ctype, self._row_finders,
                    self._row_align, self._row_format, self._row_verinfo,
                    self._row_data, self._row_corrected):
            row.set("—")
        self._content.clear()
        self._region_list.clear()
        self._symbol_list.clear()
        self._symbol_box.setVisible(False)
        self._row_symbology.set("—")
        self._row_capability.set("—")
        self._clear_param_rows()
        self._copy_button.setEnabled(False)

    # -- populate ---------------------------------------------------------
    def show_image_report(self, report: ImageReport) -> None:
        """Populate from the all-symbology report (possibly several codes)."""
        self.clear()
        symbols = report.symbols
        if not symbols:
            self._headline.setText("%s　（%.2f 秒）" % (report.headline, report.elapsed))
            self._headline.setStyleSheet(
                "padding:6px; border-radius:6px; background:#3d1e1e; color:#ffffff;")
            self._row_decoded.set("否", UNREADABLE)
            self._row_stage.set("检测", UNREADABLE)
            self._row_reason.set("未检测到任何码", UNREADABLE)
            for note in report.notes:
                self._region_list.addItem(QListWidgetItem("备注：%s" % note))
            return

        if len(symbols) > 1:
            self._symbol_box.setVisible(True)
            for symbol in symbols:
                item = QListWidgetItem("%d. %s　%s" % (
                    symbol.index + 1, symbol.symbology_name, symbol.headline))
                item.setData(Qt.ItemDataRole.UserRole, symbol.index)
                item.setToolTip(symbol.text or symbol.failure_reason or "")
                if symbol.decoded:
                    item.setIcon(QIcon(_dot_pixmap("#2e9e5b", 10)))
                elif symbol.partial:
                    item.setIcon(QIcon(_dot_pixmap("#d0a020", 10)))
                else:
                    item.setIcon(QIcon(_dot_pixmap("#c0392b", 10)))
                self._symbol_list.addItem(item)
            self._symbol_list.setCurrentRow(0)

        primary = symbols[0]
        bg = "#1e3d2a" if primary.decoded else ("#3d3618" if primary.partial
                                                else "#3d1e1e")
        self._headline.setText("%s　（%.2f 秒）" % (report.headline, report.elapsed))
        self._headline.setStyleSheet(
            "padding:6px; border-radius:6px; background:%s; color:#ffffff;" % bg)

        self._populate_symbol(primary)
        if len(symbols) > 1:
            header = QListWidgetItem("全部标注区域（%d 个码）" % len(symbols))
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            self._region_list.addItem(header)
            for symbol in symbols:
                group = QListWidgetItem("— %s —" % symbol.symbology_name)
                group.setFlags(Qt.ItemFlag.NoItemFlags)
                self._region_list.addItem(group)
                self._append_annotations(symbol)

    def _append_annotations(self, symbol: SymbolReport) -> None:
        for annotation in symbol.annotations:
            item = QListWidgetItem("%s　—　%s" % (annotation.label,
                                                  annotation.status_label))
            item.setData(Qt.ItemDataRole.UserRole, annotation.key)
            fill, border, _ = STATUS_COLORS.get(annotation.status,
                                                STATUS_COLORS["unreadable"])
            item.setIcon(QIcon(_dot_pixmap(border.name(), 10)))
            if annotation.detail:
                item.setToolTip(annotation.detail)
            self._region_list.addItem(item)

    def _clear_param_rows(self) -> None:
        while self._param_layout.count():
            item = self._param_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._param_rows.clear()

    def _param_row(self, key: str) -> _FieldRow:
        row = self._param_rows.get(key)
        if row is None:
            row = _FieldRow(symbologies.field_label(key))
            self._param_rows[key] = row
            self._param_layout.addWidget(row)
        return row

    def _populate_params(self, symbol: SymbolReport) -> None:
        """Show exactly the fields this symbology defines; never leave a gap."""
        self._clear_param_rows()
        for key in (symbologies.fields_of(symbol.symbology) or ("size",)):
            row = self._param_row(key)
            value = {
                "version": symbol.version,
                "size": symbol.size,
                "level": symbol.level,
                "mask": symbol.mask,
                "mode": symbol.mode,
            }.get(key)
            if value is not None:
                self._set_value_row(row, value)
            elif key == "module_size":
                if symbol.module_size_px:
                    row.set("%.2f px" % symbol.module_size_px, CERTAIN,
                            "由符号外框尺寸与模块数反推")
                else:
                    row.set("不可用", UNREADABLE, "未能测量模块尺寸")
            elif key == "symbol_number":
                if symbol.symbol_number is not None:
                    row.set(str(symbol.symbol_number), CERTAIN, "Micro QR 格式信息")
                else:
                    row.set("未实现", UNREADABLE, "Micro QR 格式信息解析未成功")
            else:
                text = (symbol.structural or {}).get(key)
                if text:
                    row.set(str(text), ESTIMATED, "由引擎报告或几何估计")
                else:
                    row.set("未实现", UNREADABLE,
                            "该参数解析为后续增强，当前仅提供类型与定位框")

    def _populate_symbol(self, symbol: SymbolReport) -> None:
        self._row_symbology.set(symbol.symbology_name, CERTAIN,
                                "码制由解码引擎报告的格式标识确定")
        self._row_capability.set(
            symbol.capability_label,
            CERTAIN if symbol.capability != symbologies.CAP_UNSUPPORTED else UNREADABLE,
            "本工具对该码制的识别深度")
        self._row_decoded.set("是" if symbol.decoded else "否",
                              CERTAIN if symbol.decoded else UNREADABLE)
        self._row_engine.set(symbol.engine or "—")
        if symbol.decoded:
            self._row_agree.set("一致" if symbol.engines_agree else "不一致",
                                CERTAIN if symbol.engines_agree else ESTIMATED,
                                symbol.conflict or "")
        else:
            self._row_agree.set("—")
        self._row_variant.set(symbol.variant or "—")
        if symbol.quad:
            self._row_geometry.set("定位框已获得",
                                   CERTAIN if symbol.decoded else ESTIMATED,
                                   "四角点：%s" % (symbol.quad,))
        else:
            self._row_geometry.set("未定位", UNREADABLE)
        self._row_stage.set(symbol.failure_stage or "—",
                            UNREADABLE if symbol.failure_stage else "")
        self._row_reason.set(symbol.failure_reason or "—",
                             UNREADABLE if symbol.failure_reason else "")

        self._populate_params(symbol)

        if symbol.decoded:
            self._row_ctype.set(symbol.content_type_name, CERTAIN)
            self._content.setPlainText(symbol.text)
            self._copy_button.setEnabled(True)
        else:
            self._row_ctype.set("—")
            self._content.setPlainText("")
            self._copy_button.setEnabled(False)

        qr_family = symbologies.group_of(symbol.symbology) == symbologies.QR_FAMILY
        if qr_family:
            self._row_finders.set(
                "%d / %d" % (symbol.finder_count, symbol.finder_expected),
                CERTAIN if symbol.finder_count == symbol.finder_expected else UNREADABLE)
            self._row_align.set(
                "%d / %d" % (symbol.alignment_found, symbol.alignment_expected),
                CERTAIN if (symbol.alignment_expected == 0
                            or symbol.alignment_found == symbol.alignment_expected)
                else ESTIMATED)
            self._row_format.set("副本1 %s ／ 副本2 %s" % (
                STATUS_LABELS.get(symbol.format_status[0], symbol.format_status[0]),
                STATUS_LABELS.get(symbol.format_status[1], symbol.format_status[1])),
                symbol.format_status[0] if symbol.format_status[0] == symbol.format_status[1]
                else ESTIMATED)
            version_value = symbol.version.value
            if (version_value is not None and str(version_value).isdigit()
                    and int(version_value) >= 7):
                self._row_verinfo.set("副本1 %s ／ 副本2 %s" % (
                    STATUS_LABELS.get(symbol.version_status[0], symbol.version_status[0]),
                    STATUS_LABELS.get(symbol.version_status[1], symbol.version_status[1])),
                    symbol.version_status[0] if symbol.version_status[0] == symbol.version_status[1]
                    else ESTIMATED)
            else:
                self._row_verinfo.set("不适用（版本 < 7）")
            self._row_data.set(symbol.data_status_text, symbol.data_status)
        else:
            self._row_finders.set("不适用（非 QR 家族）")
            self._row_align.set("不适用（非 QR 家族）")
            self._row_format.set("未实现", UNREADABLE,
                                 "该码制的格式信息解析为后续增强")
            self._row_verinfo.set("不适用（非 QR 家族）")
            self._row_data.set(symbol.data_status_text, symbol.data_status)
        self._row_corrected.set(str(symbol.corrected_errors)
                                if (symbol.decoded or symbol.corrected_errors) else "—")

        self._append_annotations(symbol)

    def show_report(self, report) -> None:
        """Accept either an ImageReport or a single-symbol AnalysisReport."""
        if isinstance(report, ImageReport):
            self.show_image_report(report)
            return
        from ..core.symbols import symbol_from_analysis
        self.show_image_report(ImageReport(symbols=[symbol_from_analysis(report)],
                                           scope="qr", elapsed=report.elapsed))

    def _set_value_row(self, row: _FieldRow, value, tooltip: str = "") -> None:
        if value is None:
            row.set("—")
            return
        text = value.text if hasattr(value, "text") else str(value)
        status = getattr(value, "status", "")
        basis = getattr(value, "basis", "")
        tip = basis
        if tooltip:
            tip = (tip + "\n" + tooltip) if tip else tooltip
        row.set(text, status, tip)
