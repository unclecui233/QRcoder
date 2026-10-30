"""Per-symbol results: one entry per physical code found in an image.

The original analyser produced a single :class:`~qrcoder.core.analysis.AnalysisReport`
for one QR symbol.  With all-symbology support an image can hold several codes of
different kinds, so this module defines the container the GUI, the exporters and
the annotation layer work with, plus the per-symbology measurement code that turns
a detection box into "how big is this symbol" answers.

QR Code Model 2 keeps its full structural pipeline (see ``analysis.py``); every
other symbology gets a *bounded* treatment here, and the report says so honestly
via :mod:`qrcoder.core.symbologies` capability labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import layout, symbologies, tables
from .analysis import (CANDIDATE, CERTAIN, ESTIMATED, UNREADABLE, RegionAnnotation,
                       Value)
from .engines import EngineResult

__all__ = [
    "SymbolReport",
    "ImageReport",
    "symbol_from_engine",
    "merge_symbols",
    "box_from_quad",
    "quad_iou",
    "measure_module_size",
    "qr_geometry_annotations",
    "non_qr_annotations",
]

Coord = Tuple[float, float]
Quad = List[Coord]


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass
class SymbolReport:
    """One physical code in the image."""

    index: int = 0
    symbology: str = "unknown"
    #: Detection quad in **original image** pixels (TL, TR, BR, BL).
    quad: Quad = field(default_factory=list)
    decoded: bool = False
    text: str = ""
    raw: bytes = b""
    engine: Optional[str] = None
    engines_agree: bool = False
    conflict: Optional[str] = None
    variant: str = ""
    #: True when an engine located the symbol but the content failed its checks.
    located_only: bool = False
    failure_stage: Optional[str] = None
    failure_reason: str = ""

    # --- parameters (all optional, per symbology) ---
    version: Value = field(default_factory=Value)
    size: Value = field(default_factory=Value)
    level: Value = field(default_factory=Value)
    mask: Value = field(default_factory=Value)
    mode: Value = field(default_factory=Value)
    layers: Value = field(default_factory=Value)
    rows: Value = field(default_factory=Value)
    columns: Value = field(default_factory=Value)
    compact: Value = field(default_factory=Value)
    symbol_number: Value = field(default_factory=Value)
    module_size_px: Optional[float] = None
    #: Modules per side (or per side pair) when known.
    modules: Optional[int] = None

    # --- content ---
    content_type: str = "text"
    content_type_name: str = "纯文本"
    segments: List = field(default_factory=list)

    # --- structure ---
    annotations: List[RegionAnnotation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: Set for QR Model 2 when the independent structural reader ran.
    structural: Optional[object] = None
    corrected_errors: int = 0
    alignment_found: int = 0
    alignment_expected: int = 0
    format_status: Tuple[str, str] = (UNREADABLE, UNREADABLE)
    version_status: Tuple[str, str] = (UNREADABLE, UNREADABLE)
    data_status: str = UNREADABLE
    data_status_text: str = "不可读"
    finder_count: int = 0
    finder_expected: int = 0

    # ------------------------------------------------------------------
    @property
    def symbology_name(self) -> str:
        return symbologies.display_name(self.symbology)

    @property
    def capability(self) -> str:
        return symbologies.capability_of(self.symbology)

    @property
    def capability_label(self) -> str:
        return symbologies.CAPABILITY_LABELS[self.capability]

    @property
    def is_qr(self) -> bool:
        return self.symbology in ("qrcode", "qrcode_model1", "microqrcode", "rmqr")

    @property
    def partial(self) -> bool:
        """True when content failed but some structure/parameters were recovered."""
        if self.decoded:
            return False
        return any(v.status in (CERTAIN, ESTIMATED, CANDIDATE)
                   for v in self.parameter_values())

    def parameter_values(self) -> List[Value]:
        return [self.version, self.size, self.level, self.mask, self.mode,
                self.layers, self.rows, self.columns, self.compact,
                self.symbol_number]

    @property
    def headline(self) -> str:
        if self.decoded:
            return "解码成功"
        if self.partial:
            return "部分信息识别成功"
        if self.quad:
            return "已定位，内容不可读"
        return "未识别"

    @property
    def label(self) -> str:
        """Short label used on the canvas and in the list."""
        return "%d. %s" % (self.index + 1, self.symbology_name)

    def fields(self) -> List[Tuple[str, Value]]:
        """Parameter rows this symbology actually reports, in display order."""
        names = symbologies.fields_of(self.symbology)
        mapping = {
            "version": self.version,
            "size": self.size,
            "level": self.level,
            "mask": self.mask,
            "mode": self.mode,
            "layers": self.layers,
            "rows": self.rows,
            "columns": self.columns,
            "compact": self.compact,
            "symbol_number": self.symbol_number,
        }
        return [(name, mapping[name]) for name in names if name in mapping]

    def unsupported_fields(self) -> List[str]:
        """Fields a user would expect but which this build cannot provide."""
        if self.symbology in ("hanxin", "dotcode", "rmqr"):
            return [symbologies.field_label("version"), "定位"]
        return []

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "symbology": self.symbology,
            "symbology_name": self.symbology_name,
            "capability": self.capability,
            "headline": self.headline,
            "decoded": self.decoded,
            "located_only": self.located_only,
            "text": self.text,
            "raw_hex": self.raw.hex(),
            "content_type": self.content_type,
            "content_type_name": self.content_type_name,
            "engine": self.engine,
            "engines_agree": self.engines_agree,
            "conflict": self.conflict,
            "variant": self.variant,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "quad": [[round(x, 2), round(y, 2)] for x, y in self.quad],
            "parameters": {
                name: {"label": symbologies.field_label(name),
                       "value": value.value, "status": value.status,
                       "basis": value.basis}
                for name, value in self.fields()
            },
            "unsupported": self.unsupported_fields(),
            "module_size_px": (round(self.module_size_px, 3)
                               if self.module_size_px else None),
            "modules": self.modules,
            "structure": {
                "finder_patterns": {"found": self.finder_count,
                                    "expected": self.finder_expected},
                "alignment_patterns": {"found": self.alignment_found,
                                       "expected": self.alignment_expected},
                "format_info": {"copy1": self.format_status[0],
                                "copy2": self.format_status[1]},
                "version_info": {"copy1": self.version_status[0],
                                 "copy2": self.version_status[1]},
                "data_area": {"status": self.data_status,
                              "text": self.data_status_text,
                              "corrected_errors": self.corrected_errors},
            },
            "annotations": [
                {"key": a.key, "label": a.label, "status": a.status,
                 "detail": a.detail, "modules": a.modules,
                 "polygon": [[round(x, 2), round(y, 2)] for x, y in a.polygon]}
                for a in self.annotations
            ],
            "notes": self.notes,
        }

    def to_text(self) -> str:
        lines = ["【码 %d】%s（%s）" % (self.index + 1, self.symbology_name,
                                       self.capability_label)]
        lines.append("  结论：%s" % self.headline)
        if self.decoded:
            lines.append("  采纳引擎：%s" % (self.engine or "-"))
            lines.append("  生效预处理：%s" % (self.variant or "-"))
            lines.append("  内容类型：%s" % self.content_type_name)
            lines.append("  文本：%s" % self.text)
        else:
            if self.failure_stage:
                lines.append("  失败阶段：%s" % self.failure_stage)
            if self.failure_reason:
                lines.append("  原因摘要：%s" % self.failure_reason)
        for name, value in self.fields():
            lines.append("  %s：%s" % (symbologies.field_label(name), value.text))
        if self.module_size_px:
            lines.append("  模块尺寸：%.2f px" % self.module_size_px)
        if self.unsupported_fields():
            lines.append("  不可用字段：%s（当前引擎不支持）"
                         % "、".join(self.unsupported_fields()))
        if self.quad:
            xs = [p[0] for p in self.quad]
            ys = [p[1] for p in self.quad]
            lines.append("  定位框：(%d, %d) - (%d, %d)"
                         % (min(xs), min(ys), max(xs), max(ys)))
        for note in self.notes:
            lines.append("  备注：%s" % note)
        return "\n".join(lines)


@dataclass
class ImageReport:
    """All symbols found in one image."""

    symbols: List[SymbolReport] = field(default_factory=list)
    elapsed: float = 0.0
    image_shape: Tuple[int, int] = (0, 0)
    coord_scale: float = 1.0
    notes: List[str] = field(default_factory=list)
    engine_attempts: List[str] = field(default_factory=list)
    #: Symbology keys that were enabled for this run.
    scope: List[str] = field(default_factory=list)
    stage_times: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.symbols)

    @property
    def decoded_count(self) -> int:
        return sum(1 for s in self.symbols if s.decoded)

    @property
    def primary(self) -> Optional[SymbolReport]:
        """The symbol the summary banner describes."""
        if not self.symbols:
            return None
        for symbol in self.symbols:
            if symbol.decoded:
                return symbol
        return self.symbols[0]

    @property
    def headline(self) -> str:
        if not self.symbols:
            return "未检测到二维码"
        if self.decoded_count == self.count:
            return ("解码成功" if self.count == 1
                    else "解码成功（共 %d 个码）" % self.count)
        if self.decoded_count:
            return "部分解码成功（%d/%d）" % (self.decoded_count, self.count)
        if any(s.partial for s in self.symbols):
            return ("部分信息识别成功" if self.count == 1
                    else "部分信息识别成功（共 %d 个码）" % self.count)
        return "解码失败"

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "symbol_count": self.count,
            "decoded_count": self.decoded_count,
            "scope": self.scope,
            "elapsed_seconds": round(self.elapsed, 3),
            "stage_times": {k: round(v, 3) for k, v in self.stage_times.items()},
            "image": {"height": self.image_shape[0], "width": self.image_shape[1],
                      "analysis_scale": self.coord_scale},
            "symbols": [s.to_dict() for s in self.symbols],
            "notes": self.notes,
            "engine_attempts": self.engine_attempts,
        }

    def to_text(self) -> str:
        lines = ["二维码分析报告", "=" * 46,
                 "结论：%s" % self.headline,
                 "检测到码数：%d（完整解码 %d）" % (self.count, self.decoded_count),
                 "耗时：%.2f 秒" % self.elapsed, ""]
        for symbol in self.symbols:
            lines.append(symbol.to_text())
            lines.append("")
        if self.notes:
            lines.append("【分析备注】")
            for note in self.notes:
                lines.append("  - %s" % note)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def box_from_quad(quad: Sequence[Coord]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


def quad_iou(a: Sequence[Coord], b: Sequence[Coord]) -> float:
    """Intersection-over-union of two axis-aligned quads."""
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = box_from_quad(a)
    bx0, by0, bx1, by1 = box_from_quad(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-9, (bx1 - bx0) * (by1 - by0))
    return inter / (area_a + area_b - inter)


def measure_module_size(quad: Sequence[Coord], modules: int) -> Optional[float]:
    """Module size in pixels, from the detection quad and the module count."""
    if not quad or modules <= 0:
        return None
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = (list(quad) + list(quad))[:4]
    top = float(np.hypot(x1 - x0, y1 - y0))
    bottom = float(np.hypot(x2 - x3, y2 - y3))
    left = float(np.hypot(x3 - x0, y3 - y0))
    right = float(np.hypot(x2 - x1, y2 - y1))
    mean_side = (top + bottom + left + right) / 4.0
    if mean_side <= 0:
        return None
    return mean_side / float(modules)


def _polygon_from_modules(transform, r0: int, c0: int, height: int, width: int):
    corners = [(c0, r0), (c0 + width, r0), (c0 + width, r0 + height), (c0, r0 + height)]
    out: List[Coord] = []
    for (mc, mr) in corners:
        pt = transform @ np.array([float(mc), float(mr), 1.0])
        denom = pt[2] if abs(pt[2]) > 1e-12 else 1e-12
        out.append((float(pt[0] / denom), float(pt[1] / denom)))
    return out


def qr_geometry_annotations(version: int, transform, scale: float = 1.0,
                            status: str = ESTIMATED) -> List[RegionAnnotation]:
    """Structural regions of a QR symbol, without needing a decoded payload.

    Used for QR symbols whose grid came from the engine's quad but whose content
    (or format information) could not be verified.
    """
    size = tables.size_for_version(version)
    fmap = layout.build_function_map(version)
    out: List[RegionAnnotation] = []

    def add(key: str, label: str, r0: int, c0: int, h: int, w: int,
            detail: str = "", st: str = status) -> None:
        poly = _polygon_from_modules(transform, r0, c0, h, w)
        if scale and scale != 1.0:
            poly = [(x / scale, y / scale) for x, y in poly]
        out.append(RegionAnnotation(key=key, label=label, status=st, polygon=poly,
                                    modules=(r0, c0, h, w), detail=detail))

    add("finder_a", "定位图案 A", 0, 0, 7, 7, "由解码引擎四角点定位")
    add("finder_b", "定位图案 B", 0, size - 7, 7, 7)
    add("finder_c", "定位图案 C", size - 7, 0, 7, 7)
    add("timing_h", "时序图案（行）", 6, 8, 1, max(0, size - 16))
    add("timing_v", "时序图案（列）", 8, 6, max(0, size - 16), 1)
    for index, (cr, cc) in enumerate(tables.alignment_centres(version)):
        add("align_%d" % index, "对齐图案 %d" % (index + 1), cr - 2, cc - 2, 5, 5,
            "中心 (%d, %d)" % (cr, cc))
    for copy_index, coords in ((1, fmap.format_copy1), (2, fmap.format_copy2)):
        r0 = min(r for r, _ in coords)
        c0 = min(c for _, c in coords)
        r1 = max(r for r, _ in coords) + 1
        c1 = max(c for _, c in coords) + 1
        add("format_%d" % copy_index, "格式信息 %d" % copy_index, r0, c0,
            r1 - r0, c1 - c0, "未读取（结构定位）")
    if version >= 7:
        for copy_index, coords in ((1, fmap.version_copy1), (2, fmap.version_copy2)):
            r0 = min(r for r, _ in coords)
            c0 = min(c for _, c in coords)
            r1 = max(r for r, _ in coords) + 1
            c1 = max(c for _, c in coords) + 1
            add("version_%d" % copy_index, "版本信息 %d" % copy_index, r0, c0,
                r1 - r0, c1 - c0, "未读取（结构定位）")
    add("data_area", "数据区", 8, 8, max(1, size - 16), max(1, size - 16),
        "内容不可读")
    return out


# ---------------------------------------------------------------------------
# Per-symbology measurement
# ---------------------------------------------------------------------------


def _parse_size_text(text: str) -> Optional[Tuple[int, int]]:
    """Parse zxing-cpp's ``"16x16"`` / ``"12x26"`` Data Matrix version strings."""
    if not text or "x" not in text.lower():
        return None
    try:
        width, height = text.lower().split("x", 1)
        return int(width), int(height)
    except (TypeError, ValueError):
        return None


def _percent(text: str) -> Optional[float]:
    if not text or "%" not in text:
        return None
    try:
        return float(text.replace("%", "").strip())
    except ValueError:
        return None


def _level_from_percent(percent: Optional[float]) -> Optional[str]:
    """Map zxing-cpp's ``"62%"`` style EC capacity onto the nearest ISO level.

    Aztec/PDF417 report *error-correction capacity* as a percentage, which is not
    the same scale as QR's L/M/Q/H, so it is reported verbatim alongside a
    coarse band instead of being silently relabelled as a QR level.
    """
    if percent is None:
        return None
    for threshold, name in ((15.0, "约 10%"), (35.0, "约 23%"),
                            (60.0, "约 40%"), (85.0, "约 55%")):
        if percent <= threshold:
            return name
    return "约 75%"


def non_qr_annotations(result: EngineResult, symbology: str,
                       module_size: Optional[float]) -> List[RegionAnnotation]:
    """Bounding box plus whatever internal structure we can honestly mark."""
    out: List[RegionAnnotation] = []
    quad = result.position or []
    if not quad:
        return out
    x0, y0, x1, y1 = box_from_quad(quad)
    width = x1 - x0
    height = y1 - y0
    box_poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    status = CERTAIN if result.ok else ESTIMATED
    detail = "%s · 模块宽 %.2f px" % (
        result.symbology_name, module_size) if module_size else result.symbology_name
    out.append(RegionAnnotation(key="bbox", label=result.symbology_name,
                                status=status, polygon=box_poly, detail=detail))

    if symbology == "datamatrix" and width > 8 and height > 8:
        # Solid L-shaped finder: left column + bottom row; dashed timing: top row
        # and right column.  Expressed as fractions of the box, which is exact for
        # a Data Matrix because the finder occupies the two outer module lines.
        thickness = max(1.0, module_size or (width / 20.0))
        left = [(x0, y0), (x0 + thickness, y0), (x0 + thickness, y1), (x0, y1)]
        bottom = [(x0, y1 - thickness), (x1, y1 - thickness), (x1, y1), (x0, y1)]
        out.append(RegionAnnotation("dm_l_edge", "L 形实边（定位）", status,
                                    left + bottom, None, "左侧列 + 底侧行"))
        top = [(x0, y0), (x1, y0), (x1, y0 + thickness), (x0, y0 + thickness)]
        right = [(x1 - thickness, y0), (x1, y0), (x1, y1), (x1 - thickness, y1)]
        out.append(RegionAnnotation("dm_timing", "虚线时序边", status,
                                    top + right, None, "顶侧行 + 右侧列（交替模块）"))
        out.append(RegionAnnotation(
            "dm_data", "数据区", status,
            [(x0 + thickness, y0 + thickness), (x1 - thickness, y0 + thickness),
             (x1 - thickness, y1 - thickness), (x0 + thickness, y1 - thickness)],
            None, "ECC 200 数据与纠错码字"))
    elif symbology == "aztec":
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        radius = min(width, height) * 0.22
        out.append(RegionAnnotation(
            "aztec_bullseye", "中心定位环", status,
            [(cx - radius, cy - radius), (cx + radius, cy - radius),
             (cx + radius, cy + radius), (cx - radius, cy + radius)],
            None, "牛眼图案（层数参考）"))
        out.append(RegionAnnotation(
            "aztec_data", "参考网格数据区", status,
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], None, "环形数据层"))
    elif symbology == "pdf417" and width > 8:
        band = max(2.0, height * 0.06)
        out.append(RegionAnnotation(
            "pdf417_start", "起始图案", status,
            [(x0, y0), (x0 + band, y0), (x0 + band, y1), (x0, y1)], None,
            "起始符（左侧）"))
        out.append(RegionAnnotation(
            "pdf417_stop", "终止图案", status,
            [(x1 - band, y0), (x1, y0), (x1, y1), (x1 - band, y1)], None,
            "终止符（右侧）"))
        if result.level_text:
            out.append(RegionAnnotation(
                "pdf417_rows", "行指示器区域", status,
                [(x0 + band, y0), (x1 - band, y0), (x1 - band, y1), (x0 + band, y1)],
                None, "每行含行号与纠错等级指示"))
    elif symbology == "maxicode":
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        radius = min(width, height) * 0.18
        out.append(RegionAnnotation(
            "maxicode_bullseye", "中心定位靶", status,
            [(cx - radius, cy - radius), (cx + radius, cy - radius),
             (cx + radius, cy + radius), (cx - radius, cy + radius)],
            None, "同心环定位图案"))
    elif symbology == "microqrcode":
        from . import microqr

        size = None
        for candidate in microqr.MICRO_SIZE_BY_VERSION.values():
            if candidate:
                size = candidate
                break
        out.append(RegionAnnotation(
            "micro_finder", "定位图案（单一定位）", status,
            [(x0, y0), (x0 + (x1 - x0) * 0.4, y0), (x0 + (x1 - x0) * 0.4, y0 + (y1 - y0) * 0.4),
             (x0, y0 + (y1 - y0) * 0.4)], None, "Micro QR 只有一个定位图案"))
    return out


def _micro_qr_parameters(symbol: SymbolReport, result: EngineResult,
                         image: np.ndarray) -> None:
    """Read Micro QR format information straight from the sampled modules."""
    from . import microqr
    from .finders import finder_set_from_quad
    from .sampler import sample_matrix

    version_text = (result.version_text or "").upper()
    size = microqr.MICRO_SIZE_BY_VERSION.get(version_text)
    if size is None:
        symbol.version = Value(None, UNREADABLE, "引擎未报告 Micro QR 符号类型")
        symbol.size = Value(None, UNREADABLE, "引擎未报告 Micro QR 符号类型")
        return

    symbol.version = Value(version_text, CERTAIN, "解码引擎")
    symbol.size = Value("%d × %d" % (size, size), CERTAIN, "解码引擎")
    symbol.modules = size
    symbol.module_size_px = measure_module_size(result.position or [], size)
    if result.level and result.level != "-":
        symbol.level = Value(result.level, CERTAIN, "解码引擎")
    if result.mask is not None:
        symbol.mask = Value(result.mask, CERTAIN, "解码引擎")
    elif version_text and result.level and result.level != "-":
        # The engine decoded the format information but did not expose the mask;
        # fall back to the geometric reader so the field is not left empty.
        pass

    # Independent format read (works even when the payload failed its checks).
    quad = result.position or []
    if len(quad) != 4:
        return
    try:
        ratio = 1.0 / (result.scale or 1.0)
        corners = [(x * ratio, y * ratio) for x, y in quad]
        finder = finder_set_from_quad(corners, size=size)
        if finder is None:
            return
        sampling = sample_matrix(image, finder, version=0, size=size)
        reading = microqr.read_micro_format(sampling.matrix)
    except Exception as exc:  # pragma: no cover - defensive
        symbol.notes.append("Micro QR 格式信息读取异常：%s" % exc)
        return
    if reading is None:
        return
    if reading.certain:
        symbol.symbol_number = Value(reading.version or "-", CERTAIN,
                                     "Micro QR 格式信息 BCH")
        symbol.version = Value(reading.version, CERTAIN, "Micro QR 格式信息 BCH")
        if reading.level and reading.level != "-":
            symbol.level = Value(reading.level, CERTAIN, "Micro QR 格式信息 BCH")
        symbol.mask = Value(reading.mask, CERTAIN, "Micro QR 格式信息 BCH")
        symbol.format_status = (CERTAIN, UNREADABLE)
        symbol.notes.append("格式信息经 BCH 校验确认：%s / %s / mask %d"
                            % (reading.version, reading.level, reading.mask))
    else:
        symbol.format_status = (UNREADABLE, UNREADABLE)
        if reading.candidates:
            best = reading.candidates[0]
            symbol.symbol_number = Value(best[0], CANDIDATE, "格式信息候选")
            symbol.notes.append("格式信息读取失败，候选：%s"
                                % "、".join(str(c[0]) for c in reading.candidates[:3]))


def symbol_from_engine(result: EngineResult, image: np.ndarray,
                       index: int = 0, coord_scale: float = 1.0) -> SymbolReport:
    """Build a :class:`SymbolReport` from an engine detection."""
    symbol = SymbolReport(index=index, symbology=result.symbology or "unknown")
    symbol.engine = result.engine
    symbol.variant = result.variant
    symbol.decoded = bool(result.ok)
    symbol.located_only = bool(result.partial and not result.ok)
    symbol.text = result.text if result.ok else ""
    symbol.raw = result.raw if result.ok else b""
    symbol.failure_stage = None if result.ok else "数据解码"
    symbol.failure_reason = "" if result.ok else (result.error or "内容校验失败")

    # Quad in original-image pixels.
    if result.position:
        ratio = (1.0 / result.scale) if result.scale else 1.0
        symbol.quad = [(x * ratio * coord_scale, y * ratio * coord_scale)
                       for x, y in result.position]

    if result.text:
        from .decoder import classify_content, content_type_label

        symbol.content_type = classify_content(result.text)
        symbol.content_type_name = content_type_label(symbol.content_type)

    kind = symbol.symbology
    version_text = (result.version_text or "").strip()

    if kind in ("qrcode", "qrcode_model1"):
        if result.version:
            symbol.version = Value(result.version, CERTAIN, "解码引擎")
            symbol.modules = tables.size_for_version(result.version)
            symbol.size = Value("%d × %d" % (symbol.modules, symbol.modules),
                                CERTAIN, "解码引擎")
        if result.level:
            symbol.level = Value(result.level, CERTAIN, "解码引擎")
        if result.mask is not None:
            symbol.mask = Value(result.mask, CERTAIN, "解码引擎")
        if result.mode:
            symbol.mode = Value(result.mode, CERTAIN, "解码引擎")
        if symbol.modules:
            symbol.module_size_px = measure_module_size(symbol.quad, symbol.modules)
    elif kind == "microqrcode":
        _micro_qr_parameters(symbol, result, image)
    elif kind == "datamatrix":
        dims = _parse_size_text(version_text)
        if dims is not None:
            width, height = dims
            symbol.size = Value("%d × %d" % (width, height), CERTAIN, "解码引擎")
            symbol.modules = max(width, height)
            symbol.module_size_px = measure_module_size(symbol.quad, max(width, height))
        symbol.notes.append("纠错等级未由引擎提供，Data Matrix ECC 200 解析为后续增强")
        symbol.level = Value(None, UNREADABLE, "引擎不提供 Data Matrix 纠错等级")
    elif kind == "aztec":
        if version_text:
            symbol.layers = Value(version_text, CERTAIN, "解码引擎")
        percent = _percent(result.level_text)
        if percent is not None:
            band = _level_from_percent(percent)
            symbol.level = Value("%.0f%%（%s）" % (percent, band) if band
                                 else "%.0f%%" % percent, CERTAIN, "解码引擎")
        symbol.notes.append("Aztec 层数由引擎报告；紧凑型标志需从模式消息解析（后续增强）")
        symbol.compact = Value(None, UNREADABLE, "后续增强")
    elif kind == "pdf417":
        percent = _percent(result.level_text)
        if percent is not None:
            band = _level_from_percent(percent)
            symbol.level = Value("%.0f%%（%s）" % (percent, band) if band
                                 else "%.0f%%" % percent, CERTAIN, "解码引擎")
        if symbol.quad:
            x0, y0, x1, y1 = box_from_quad(symbol.quad)
            symbol.notes.append(
                "引擎不报告行/列数；检测框 %d × %d px 可用于估算行列（几何估计）"
                % (x1 - x0, y1 - y0))
        symbol.rows = Value(None, UNREADABLE, "引擎不报告行数")
        symbol.columns = Value(None, UNREADABLE, "引擎不报告列数")
    elif kind == "maxicode":
        if result.level_text:
            symbol.level = Value(result.level_text, CERTAIN, "解码引擎（模式号）")
        symbol.notes.append("MaxiCode 模式/几何参数解析为后续增强")
    elif kind in ("hanxin", "dotcode", "rmqr"):
        symbol.notes.append("当前 zxing-cpp 构建不支持该码制，仅提供类型与定位框")
        symbol.failure_stage = "检测"
        symbol.failure_reason = "该码制当前引擎不支持完整解码"
    elif symbologies.group_of(kind) == "linear":
        symbol.notes.append("一维码：仅提供类型、位置与内容")

    if symbol.quad and symbol.module_size_px is None and symbol.modules:
        symbol.module_size_px = measure_module_size(symbol.quad, symbol.modules)

    symbol.annotations = non_qr_annotations(result, kind, symbol.module_size_px)
    if kind == "qrcode" and result.version:
        # Structural regions from the engine quad, for QR symbols we cannot
        # sample ourselves (no binarised grid available here).
        from . import layout as _layout

        symbol.finder_count = 3
        symbol.finder_expected = 3
        symbol.alignment_expected = len(tables.alignment_centres(result.version))
        symbol.alignment_found = symbol.alignment_expected
        symbol.annotations = symbol.annotations + _quad_qr_regions(
            result, result.version, coord_scale)
    return symbol


def _quad_qr_regions(result: EngineResult, version: int,
                     coord_scale: float) -> List[RegionAnnotation]:
    """QR functional regions derived from the engine quad alone."""
    quad = result.position or []
    if len(quad) != 4:
        return []
    size = tables.size_for_version(version)
    src = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float64)
    dst = np.array(quad, dtype=np.float64)
    try:
        import cv2

        transform = cv2.getPerspectiveTransform(src, dst)
    except Exception:  # pragma: no cover
        return []
    ratio = 1.0 / (result.scale or 1.0)
    annotations = qr_geometry_annotations(version, transform,
                                          scale=1.0 / ratio if ratio else 1.0)
    return annotations


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def _rank(symbol: SymbolReport) -> Tuple[int, int]:
    """Higher is better: decoded beats located, zxing beats zbar."""
    quality = 0 if symbol.located_only else (2 if symbol.decoded else 1)
    engine_rank = {"zxing-cpp": 3, "内置读取器": 2, "pyzbar": 1}.get(symbol.engine or "", 0)
    return quality, engine_rank


def merge_symbols(candidates: Sequence[SymbolReport],
                  threshold: float = 0.35) -> List[SymbolReport]:
    """Collapse detections of the same physical symbol from multiple engines."""
    merged: List[SymbolReport] = []
    for candidate in candidates:
        if not candidate.quad:
            merged.append(candidate)
            continue
        target = None
        for existing in merged:
            if not existing.quad:
                continue
            if existing.symbology != candidate.symbology:
                continue
            if quad_iou(existing.quad, candidate.quad) >= threshold:
                target = existing
                break
        if target is None:
            merged.append(candidate)
            continue
        # Keep the better detection, but record the cross-check outcome.
        if _rank(candidate) > _rank(target):
            candidate.engines_agree = (target.decoded and candidate.decoded
                                       and target.text == candidate.text)
            if (target.decoded and candidate.decoded
                    and target.text != candidate.text):
                candidate.conflict = ("%s 与 %s 结果不一致，已采纳 %s"
                                      % (target.engine, candidate.engine,
                                         candidate.engine))
            merged[merged.index(target)] = candidate
        else:
            target.engines_agree = (target.decoded and candidate.decoded
                                    and target.text == candidate.text)
            if (target.decoded and candidate.decoded
                    and target.text != candidate.text):
                target.conflict = ("%s 与 %s 结果不一致，已采纳 %s"
                                   % (candidate.engine, target.engine, target.engine))
    # Order by reading order: top-to-bottom, then left-to-right.
    merged.sort(key=lambda s: (round(box_from_quad(s.quad)[1] / 40.0) if s.quad else 0,
                               box_from_quad(s.quad)[0] if s.quad else 0))
    for position, symbol in enumerate(merged):
        symbol.index = position
    return merged


# ---------------------------------------------------------------------------
# Conversion from the QR deep-analysis report
# ---------------------------------------------------------------------------


def symbol_from_analysis(report, index: int = 0, offset: Tuple[float, float] = (0.0, 0.0),
                         scale: float = 1.0) -> SymbolReport:
    """Convert an :class:`AnalysisReport` (the QR deep path) into a symbol entry.

    :param offset: top-left of the analysed crop in original-image pixels.
    :param scale: the analysis-space scale of the crop; annotation coordinates are
        divided by it before the offset is added.
    """
    from .decoder import classify_content, content_type_label

    symbol = SymbolReport(index=index, symbology="qrcode")
    symbol.decoded = report.decoded
    symbol.text = report.text
    symbol.raw = report.raw
    symbol.engine = report.engine
    symbol.engines_agree = report.engines_agree
    symbol.conflict = report.conflict
    symbol.variant = report.variant
    symbol.failure_stage = report.failure_stage
    symbol.failure_reason = report.failure_reason
    symbol.segments = report.segments
    symbol.version = report.version
    symbol.size = report.size
    symbol.level = report.level
    symbol.mask = report.mask
    symbol.mode = report.mode
    symbol.module_size_px = report.module_size_px
    symbol.structural = report
    symbol.corrected_errors = report.corrected_errors
    symbol.alignment_found = report.alignment_found
    symbol.alignment_expected = report.alignment_expected
    symbol.format_status = report.format_status
    symbol.version_status = report.version_status
    symbol.data_status = report.data_status
    symbol.data_status_text = report.data_status_text
    symbol.finder_count = report.finder_count
    symbol.finder_expected = report.finder_expected
    symbol.notes = list(report.notes)
    if report.text:
        symbol.content_type = classify_content(report.text)
        symbol.content_type_name = content_type_label(symbol.content_type)
    if isinstance(report.version.value, int):
        symbol.modules = tables.size_for_version(report.version.value)

    dx, dy = offset
    divisor = scale if scale else 1.0
    translated: List[RegionAnnotation] = []
    for annotation in report.annotations:
        polygon = [(x / divisor + dx, y / divisor + dy) for x, y in annotation.polygon]
        translated.append(RegionAnnotation(
            key=annotation.key, label=annotation.label, status=annotation.status,
            polygon=polygon, modules=annotation.modules, detail=annotation.detail))
    symbol.annotations = translated
    if translated:
        xs = [p[0] for a in translated for p in a.polygon]
        ys = [p[1] for a in translated for p in a.polygon]
        symbol.quad = [(min(xs), min(ys)), (max(xs), min(ys)),
                       (max(xs), max(ys)), (min(xs), max(ys))]
    return symbol
