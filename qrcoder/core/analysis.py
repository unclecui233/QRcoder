"""The analysis pipeline: engines, independent reader, merged report.

This is the only module the GUI needs.  It runs in the order the spec lays out:

1. multi-engine / multi-variant decoding,
2. independent structural reading (finder geometry, format info, version info,
   codewords) so partial information survives even when content does not,
3. merging into one :class:`AnalysisReport` where every value carries an explicit
   confidence label.

Nothing here writes to disk, opens a socket, or logs decoded content.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import bch, layout, symbologies, tables
from .decoder import (
    CONTENT_TYPES,
    DecodeOutcome,
    FormatReading,
    VersionReading,
    classify_content,
    content_type_label,
    decode_symbol,
    extract_codewords,
    decode_blocks,
    parse_segments,
    read_format_info,
    read_version_info,
)
from .engines import (
    ENGINE_OPENCV,
    ENGINE_ZBAR,
    ENGINE_ZXING,
    EngineResult,
    available_engines,
    read_all_pyzbar,
    read_all_zxingcpp,
    run_opencv,
    run_pyzbar,
    run_zxingcpp,
)
from .finders import (FinderSet, detect_finders, finder_set_from_quad,
                      order_triple, refine_version_by_timing)
from .preprocess import build_variants, to_gray
from .sampler import SamplingResult, module_to_pixel_matrix, sample_matrix

__all__ = [
    "CERTAIN",
    "ESTIMATED",
    "UNREADABLE",
    "CANDIDATE",
    "Value",
    "RegionAnnotation",
    "AnalysisReport",
    "analyze_image",
    "analyze_symbols",
]

CERTAIN = "certain"
ESTIMATED = "estimated"
UNREADABLE = "unreadable"
CANDIDATE = "candidate"

STATUS_LABELS = {
    CERTAIN: "确定",
    ESTIMATED: "几何估计",
    UNREADABLE: "无法读取",
    CANDIDATE: "候选",
}


@dataclass
class Value:
    """One reported parameter plus how it was obtained."""

    value: object = None
    status: str = UNREADABLE
    basis: str = ""

    @property
    def text(self) -> str:
        if self.value is None:
            return "无法读取"
        suffix = ""
        if self.status == ESTIMATED:
            suffix = "（几何估计）"
        elif self.status == CANDIDATE:
            suffix = "（候选）"
        return "%s%s" % (self.value, suffix)


@dataclass
class RegionAnnotation:
    """A functional region to draw on the image."""

    key: str
    label: str
    status: str
    #: ``[(x, y), ...]`` pixel polygon in original-image coordinates.
    polygon: List[Tuple[float, float]] = field(default_factory=list)
    #: ``(row, col, height, width)`` in module units, when applicable.
    modules: Optional[Tuple[int, int, int, int]] = None
    detail: str = ""

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)


@dataclass
class AnalysisReport:
    """Everything the GUI and the exporters need."""

    # --- decode ---
    decoded: bool = False
    text: str = ""
    raw: bytes = b""
    segments: List = field(default_factory=list)
    engine: Optional[str] = None
    engines_agree: bool = False
    conflict: Optional[str] = None
    variant: str = ""
    failure_stage: Optional[str] = None
    failure_reason: str = ""

    # --- parameters ---
    version: Value = field(default_factory=Value)
    size: Value = field(default_factory=Value)
    level: Value = field(default_factory=Value)
    mask: Value = field(default_factory=Value)
    mode: Value = field(default_factory=Value)
    module_size_px: Optional[float] = None
    level_candidates: List[str] = field(default_factory=list)
    mask_candidates: List[int] = field(default_factory=list)

    # --- content ---
    content_type: str = "text"
    content_type_name: str = "纯文本"

    # --- structure ---
    finder_count: int = 0
    finder_expected: int = 3
    alignment_found: int = 0
    alignment_expected: int = 0
    format_status: Tuple[str, str] = (UNREADABLE, UNREADABLE)
    version_status: Tuple[str, str] = (UNREADABLE, UNREADABLE)
    data_status: str = UNREADABLE
    data_status_text: str = "不可读"
    corrected_errors: int = 0
    contrast: float = 0.0
    aligned: bool = False

    # --- annotations ---
    annotations: List[RegionAnnotation] = field(default_factory=list)
    #: Affine/homography mapping module space -> pixel space (3x3).
    transform: Optional[np.ndarray] = None
    matrix: Optional[np.ndarray] = None
    image_shape: Tuple[int, int] = (0, 0)
    notes: List[str] = field(default_factory=list)

    # --- timing ---
    elapsed: float = 0.0
    engine_attempts: List[str] = field(default_factory=list)
    #: A result where an engine located the symbol but content failed its
    #: checksum; carries trustworthy version/mask/EC level hints.
    partial_engine: Optional[EngineResult] = None
    #: How the module grid was located: ``"engine-quad"`` or ``"contour"``.
    geometry_source: str = ""
    #: Scale factor of the analysed image relative to the original; multiply
    #: annotation coordinates by ``1/coord_scale`` to get original pixels.
    coord_scale: float = 1.0

    # ------------------------------------------------------------------
    @property
    def partial(self) -> bool:
        """True when the content failed but structure was recovered."""
        return (not self.decoded) and (self.version.status != UNREADABLE
                                      or self.level.status != UNREADABLE)

    @property
    def headline(self) -> str:
        if self.decoded:
            return "解码成功"
        if self.partial:
            return "部分信息识别成功"
        return "解码失败"

    def level_letter(self) -> Optional[str]:
        if isinstance(self.level.value, str):
            return self.level.value
        return None

    def mask_number(self) -> Optional[int]:
        if isinstance(self.mask.value, int):
            return self.mask.value
        return None

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialise for the JSON export (content included on purpose)."""
        return {
            "headline": self.headline,
            "decoded": self.decoded,
            "content": {
                "text": self.text,
                "raw_hex": self.raw.hex(),
                "content_type": self.content_type,
                "content_type_name": self.content_type_name,
                "segments": [
                    {"mode": seg.mode, "chars": seg.char_count, "eci": seg.eci}
                    for seg in self.segments
                ],
            },
            "decode": {
                "engine": self.engine,
                "engines_agree": self.engines_agree,
                "conflict": self.conflict,
                "variant": self.variant,
                "failure_stage": self.failure_stage,
                "failure_reason": self.failure_reason,
                "attempts": self.engine_attempts,
            },
            "parameters": {
                "version": {"value": self.version.value, "status": self.version.status,
                            "basis": self.version.basis},
                "size": {"value": self.size.value, "status": self.size.status,
                         "basis": self.size.basis},
                "error_correction": {"value": self.level.value, "status": self.level.status,
                                     "basis": self.level.basis,
                                     "candidates": self.level_candidates},
                "mask": {"value": self.mask.value, "status": self.mask.status,
                         "basis": self.mask.basis, "candidates": self.mask_candidates},
                "encoding_mode": {"value": self.mode.value, "status": self.mode.status,
                                  "basis": self.mode.basis},
                "module_size_px": self.module_size_px,
            },
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
                "contrast": round(self.contrast, 2),
                "alignment_refined": self.aligned,
            },
            "annotations": [
                {"key": a.key, "label": a.label, "status": a.status,
                 "detail": a.detail, "modules": a.modules,
                 "polygon": [[round(x, 2), round(y, 2)] for x, y in a.polygon]}
                for a in self.annotations
            ],
            "geometry_source": self.geometry_source,
            "elapsed_seconds": round(self.elapsed, 3),
            "notes": self.notes,
        }

    def to_text(self) -> str:
        """Plain-text report (Chinese), suitable for copying."""
        lines: List[str] = []
        lines.append("QR 码分析报告")
        lines.append("=" * 46)
        lines.append("结论：%s" % self.headline)
        lines.append("耗时：%.2f 秒" % self.elapsed)
        lines.append("")

        lines.append("【解码状态】")
        lines.append("  完整解码：%s" % ("是" if self.decoded else "否"))
        if self.decoded:
            lines.append("  采纳引擎：%s" % (self.engine or "-"))
            lines.append("  引擎一致性：%s" % ("一致" if self.engines_agree else "不一致"))
            if self.conflict:
                lines.append("  冲突说明：%s" % self.conflict)
            lines.append("  生效预处理：%s" % (self.variant or "-"))
            lines.append("  网格定位：%s" % (
                "解码引擎四角点" if self.geometry_source == "engine-quad"
                else ("定位图案几何" if self.geometry_source == "contour" else "未定位")))
        else:
            lines.append("  失败阶段：%s" % (self.failure_stage or "-"))
            lines.append("  原因摘要：%s" % (self.failure_reason or "-"))
            lines.append("  网格定位：%s" % (
                "解码引擎四角点" if self.geometry_source == "engine-quad"
                else ("定位图案几何" if self.geometry_source == "contour" else "未定位")))
        lines.append("")

        lines.append("【QR 参数】")
        lines.append("  版本：%s" % self.version.text)
        lines.append("  尺寸：%s" % self.size.text)
        lines.append("  纠错等级：%s" % self.level.text)
        lines.append("  掩码：%s" % self.mask.text)
        lines.append("  编码模式：%s" % self.mode.text)
        if self.module_size_px:
            lines.append("  模块尺寸：%.2f px" % self.module_size_px)
        lines.append("")

        lines.append("【内容】")
        if self.decoded:
            lines.append("  类型：%s" % self.content_type_name)
            lines.append("  文本：%s" % self.text)
            if self.raw and self.raw != self.text.encode("utf-8", "replace"):
                lines.append("  原始字节：%s" % self.raw.hex())
        else:
            lines.append("  未能解出内容")
        lines.append("")

        lines.append("【区域识别汇总】")
        lines.append("  定位图案：%d / %d" % (self.finder_count, self.finder_expected))
        lines.append("  对齐图案：%d / %d" % (self.alignment_found, self.alignment_expected))
        lines.append("  格式信息：副本1=%s，副本2=%s"
                     % (STATUS_LABELS.get(self.format_status[0], self.format_status[0]),
                        STATUS_LABELS.get(self.format_status[1], self.format_status[1])))
        if self.version.value is not None and int(self.version.value) >= 7:
            lines.append("  版本信息：副本1=%s，副本2=%s"
                         % (STATUS_LABELS.get(self.version_status[0], self.version_status[0]),
                            STATUS_LABELS.get(self.version_status[1], self.version_status[1])))
        lines.append("  数据区：%s" % self.data_status_text)
        lines.append("")

        if self.annotations:
            lines.append("【标注区域】")
            for ann in self.annotations:
                lines.append("  %-14s %-6s %s" % (ann.label, ann.status_label, ann.detail))
            lines.append("")

        if self.notes:
            lines.append("【分析备注】")
            for note in self.notes:
                lines.append("  - %s" % note)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine stage
# ---------------------------------------------------------------------------


def _pick_engine_result(results: Sequence[EngineResult]) -> Tuple[Optional[EngineResult], bool, Optional[str]]:
    """Apply the adoption policy: agreement first, then zxing-cpp."""
    successful = [r for r in results if r.ok]
    if not successful:
        return None, False, None

    by_engine: Dict[str, EngineResult] = {}
    for result in successful:
        by_engine.setdefault(result.engine, result)

    zxing = by_engine.get(ENGINE_ZXING)
    zbar = by_engine.get(ENGINE_ZBAR)
    if zxing and zbar:
        if zxing.text == zbar.text:
            return zxing, True, None
        return zxing, False, "zxing-cpp 与 pyzbar 结果不一致，已采纳 zxing-cpp"

    if zxing:
        return zxing, False, None
    if zbar:
        return zbar, False, None
    return successful[0], False, None


def _run_engines(gray: np.ndarray, variants, deadline: float,
                 attempts: List[str], keys: Optional[Sequence[str]] = None
                 ) -> Tuple[Optional[EngineResult], bool, Optional[str], Optional[EngineResult]]:
    """Try engines over variants, stopping as soon as two engines agree.

    Returns ``(best, agree, conflict, partial)``.  ``partial`` is a result where
    the engine located the symbol but could not verify its content; it still
    carries a trustworthy version/mask/EC level and is used as a hint when the
    independent reader cannot locate the grid at all.
    """
    results: List[EngineResult] = []
    partial: Optional[EngineResult] = None
    engines = available_engines()

    for variant in variants:
        if time.monotonic() > deadline:
            attempts.append("已到时间上限，停止尝试剩余预处理")
            break
        for engine in engines:
            if engine == ENGINE_ZXING:
                result = run_zxingcpp(variant.image, variant.label, variant.scale,
                                      keys=keys)
            elif engine == ENGINE_ZBAR:
                result = run_pyzbar(variant.image, variant.label, variant.scale,
                                    keys=keys)
            else:
                result = run_opencv(variant.image, variant.label, variant.scale,
                                    keys=keys)
            attempts.append("%s / %s：%s" % (
                variant.label, engine,
                "成功" if result.ok else (result.error or "失败")))
            if result.ok:
                results.append(result)
                # Agreement check lets us stop immediately.
                texts = {r.text for r in results if r.ok}
                engines_seen = {r.engine for r in results if r.ok}
                if len(engines_seen) >= 2 and len(texts) == 1:
                    best, agree, conflict = _pick_engine_result(results)
                    return best, agree, conflict, partial
            elif result.partial and partial is None:
                partial = result
        if results:
            # One engine already succeeded; a second opinion is still useful but
            # only from the same variant, so finish this variant then stop.
            best, agree, conflict = _pick_engine_result(results)
            if best is not None:
                return best, agree, conflict, partial

    best, agree, conflict = _pick_engine_result(results)
    return best, agree, conflict, partial


# ---------------------------------------------------------------------------
# Structural stage
# ---------------------------------------------------------------------------


def _finder_geometry(gray: np.ndarray, notes: List[str]) -> Optional[FinderSet]:
    try:
        patterns = detect_finders(gray)
    except Exception as exc:  # pragma: no cover - defensive
        notes.append("定位图案检测异常：%s" % exc)
        return None
    finder_set = order_triple(patterns)
    if finder_set is None:
        notes.append("未能组成三个定位图案（检出 %d 个候选）" % len(patterns))
        return None
    finder_set = refine_version_by_timing(gray, finder_set)
    return finder_set


def _format_copy_status(reading: FormatReading) -> Tuple[str, str]:
    statuses: List[str] = []
    for (_, _, level, mask, dist) in reading.copies:
        if level is not None:
            statuses.append(CERTAIN)
        elif dist >= 0:
            statuses.append(CANDIDATE)
        else:
            statuses.append(UNREADABLE)
    while len(statuses) < 2:
        statuses.append(UNREADABLE)
    return statuses[0], statuses[1]


def _brute_force_format(sampling: SamplingResult, reading: Optional[FormatReading],
                        notes: List[str],
                        hints: Sequence[Tuple[str, int]] = ()) -> Tuple[List[str], List[int], bool]:
    """Try every plausible (level, mask) against the data area.

    Returns candidate level letters, candidate masks, and whether one
    combination made *every* block's Reed-Solomon check pass.  A combination that
    fully corrects is not a guess: the codeword checksums verify it, which is
    stronger evidence than the format information itself.

    :param hints: engine-reported ``(level, mask)`` pairs to try first.
    """
    combos: List[Tuple[str, int]] = list(hints)
    if reading is not None:
        for level, mask, _ in reading.candidates:
            if (level, mask) not in combos:
                combos.append((level, mask))
    if len(combos) < 8:
        # Fall back to the full 4x8 matrix; it costs 32 short RS runs.
        for level in tables.LEVELS:
            for mask in range(8):
                if (level, mask) not in combos:
                    combos.append((level, mask))
    combos = combos[:32]

    ranked: List[Tuple[int, int, str, int]] = []
    for order, (level, mask) in enumerate(combos):
        try:
            codewords = extract_codewords(sampling, level, mask)
            decoded = decode_blocks(codewords)
        except Exception:
            continue
        if decoded.ok:
            notes.append("格式信息经数据区校验确认：%s / mask %d（全部码字纠错成功）"
                         % (level, mask))
            return [level], [mask], True
        ranked.append((len(decoded.failed_blocks), order, level, mask))

    if not ranked:
        return [], [], False

    ranked.sort()
    levels: List[str] = []
    masks: List[int] = []
    for _, _, level, mask in ranked[:3]:
        if level not in levels:
            levels.append(level)
        if mask not in masks:
            masks.append(mask)
    notes.append("格式信息不可读，已按块失败数排序 32 种（等级, 掩码）组合")
    return levels, masks, False


def _build_annotations(report: AnalysisReport, sampling: SamplingResult,
                       finder_set: Optional[FinderSet], outcome: DecodeOutcome,
                       version: int) -> None:
    """Create the functional-region overlay geometry."""
    size = sampling.size
    fmap = layout.build_function_map(version)
    transform = sampling.transform

    def polygon_of(r0: int, c0: int, height: int, width: int) -> List[Tuple[float, float]]:
        corners = [(c0, r0), (c0 + width, r0), (c0 + width, r0 + height), (c0, r0 + height)]
        out: List[Tuple[float, float]] = []
        for (mc, mr) in corners:
            pt = transform @ np.array([float(mc), float(mr), 1.0])
            denom = pt[2] if abs(pt[2]) > 1e-12 else 1e-12
            out.append((float(pt[0] / denom), float(pt[1] / denom)))
        return out

    def add(key: str, label: str, status: str, r0: int, c0: int, h: int, w: int,
            detail: str = "") -> None:
        report.annotations.append(RegionAnnotation(
            key=key, label=label, status=status,
            polygon=polygon_of(r0, c0, h, w),
            modules=(r0, c0, h, w), detail=detail))

    # Finder patterns and separators.
    finder_specs = [
        ("finder_a", "定位图案 A", 0, 0),
        ("finder_b", "定位图案 B", 0, size - 7),
        ("finder_c", "定位图案 C", size - 7, 0),
    ]
    finder_status = CERTAIN if finder_set is not None else UNREADABLE
    module_note = ("模块宽 %.2f px" % finder_set.module_size) if finder_set else ""
    for key, label, r0, c0 in finder_specs:
        add(key, label, finder_status, r0, c0, 7, 7, module_note)
        add(key + "_sep", label + " 分隔符", finder_status,
            max(0, r0 - 1), max(0, c0 - 1), 9 if r0 == 0 else 8, 9 if c0 == 0 else 8,
            "1 模块宽白条")

    # Timing patterns.
    add("timing_h", "时序图案（行）", CERTAIN, 6, 8, 1, max(0, size - 16), "第 6 行交替模块")
    add("timing_v", "时序图案（列）", CERTAIN, 8, 6, max(0, size - 16), 1, "第 6 列交替模块")

    # Alignment patterns.
    centres = tables.alignment_centres(version)
    align_status = CERTAIN if sampling.aligned else (ESTIMATED if centres else CERTAIN)
    for index, (cr, cc) in enumerate(centres):
        add("align_%d" % index, "对齐图案 %d" % (index + 1), align_status,
            cr - 2, cc - 2, 5, 5, "中心 (%d, %d)" % (cr, cc))

    # Format information, two copies.
    for copy_index, coords in ((1, fmap.format_copy1), (2, fmap.format_copy2)):
        if copy_index == 1:
            r0 = min(r for r, _ in coords)
            c0 = min(c for _, c in coords)
            r1 = max(r for r, _ in coords) + 1
            c1 = max(c for _, c in coords) + 1
        else:
            r0 = min(r for r, _ in coords)
            c0 = min(c for _, c in coords)
            r1 = max(r for r, _ in coords) + 1
            c1 = max(c for _, c in coords) + 1
        status = report.format_status[copy_index - 1]
        reading = outcome.format
        if reading is not None and reading.certain and reading.source == copy_index:
            detail = "%s / mask %d（纠正 %d bit）" % (reading.level, reading.mask, reading.errors)
        elif reading is not None and reading.certain:
            detail = "副本未采用，主结果来自副本 %d" % reading.source
        elif report.level_candidates:
            detail = "读取失败，候选：%s" % "、".join(report.level_candidates[:3])
        else:
            detail = "读取失败"
        add("format_%d" % copy_index, "格式信息 %d" % copy_index, status,
            r0, c0, r1 - r0, c1 - c0, detail)

    # Version information, two copies (v >= 7 only).
    if version >= 7:
        for copy_index, coords in ((1, fmap.version_copy1), (2, fmap.version_copy2)):
            r0 = min(r for r, _ in coords)
            c0 = min(c for _, c in coords)
            r1 = max(r for r, _ in coords) + 1
            c1 = max(c for _, c in coords) + 1
            status = report.version_status[copy_index - 1]
            reading = outcome.version_info
            if reading is not None and reading.certain:
                detail = "版本 %d（纠正 %d bit）" % (reading.version, reading.errors)
            elif reading is not None and reading.candidates:
                detail = "读取失败，候选：%s" % "、".join(
                    str(v) for v, _ in reading.candidates[:3])
            else:
                detail = "读取失败"
            add("version_%d" % copy_index, "版本信息 %d" % copy_index, status,
                r0, c0, r1 - r0, c1 - c0, detail)

    # Data area: one block per codeword region would be far too many polygons, so
    # annotate the four data quadrants for the summary and let the grid layer
    # handle module-level detail.
    data_status = report.data_status
    quadrants = [
        ("data_q1", "数据区 左上", 8, 9, max(1, (size - 8) // 2 - 4), max(1, (size - 17) // 2)),
        ("data_q2", "数据区 右上", 8, size - 8 - max(1, (size - 17) // 2),
         max(1, (size - 8) // 2 - 4), max(1, (size - 17) // 2)),
        ("data_q3", "数据区 左下", size - 8 - max(1, (size - 8) // 2 - 4), 9,
         max(1, (size - 8) // 2 - 4), max(1, (size - 17) // 2)),
        ("data_q4", "数据区 右下", size - 8 - max(1, (size - 8) // 2 - 4),
         size - 8 - max(1, (size - 17) // 2),
         max(1, (size - 8) // 2 - 4), max(1, (size - 17) // 2)),
    ]
    for key, label, r0, c0, h, w in quadrants:
        add(key, label, data_status, r0, c0, h, w, report.data_status_text)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze_image(image: np.ndarray, time_budget: float = 12.0,
                  keys: Optional[Sequence[str]] = None) -> AnalysisReport:
    """Analyse one image (BGR or grayscale numpy array).

    :param keys: symbology scope for the engine stage; ``None`` means every
        symbology the engine supports.  The structural reader is QR-only, so a
        scope without the QR family simply produces no structural data.
    """
    started = time.monotonic()
    report = AnalysisReport()
    notes = report.notes

    gray = to_gray(image)
    report.image_shape = (gray.shape[0], gray.shape[1])

    # --- stage 1: engines ---------------------------------------------------
    variants = build_variants(image)
    deadline = started + time_budget
    best, agree, conflict, partial = _run_engines(gray, variants, deadline,
                                                 report.engine_attempts, keys=keys)
    report.partial_engine = partial

    # --- stage 2: independent structure ------------------------------------
    # Use the same downscaled grayscale the decoders saw so pixel coordinates
    # stay consistent with the variant that succeeded.
    analysis_gray = variants[0].image
    report.coord_scale = variants[0].scale

    finder_set: Optional[FinderSet] = None
    sampling: Optional[SamplingResult] = None
    outcome: Optional[DecodeOutcome] = None
    version: Optional[int] = None

    # An engine that located the symbol - even one whose content failed its
    # checksum - reports the exact symbol quad.  That is the most reliable grid
    # source available, so try it before falling back to contour detection.
    quad_source = best if (best is not None and best.position) else (
        partial if (partial is not None and partial.position) else None)
    if quad_source is not None and quad_source.version:
        ratio = (variants[0].scale / quad_source.scale) if quad_source.scale else 1.0
        corners = [(x * ratio, y * ratio) for (x, y) in quad_source.position]
        finder_set = finder_set_from_quad(corners, quad_source.version)
        if finder_set is not None:
            report.geometry_source = "engine-quad"
            notes.append("使用 %s 给出的符号四角点定位网格（版本 %d）"
                         % (quad_source.engine, quad_source.version))

    if finder_set is None:
        finder_set = _finder_geometry(analysis_gray, notes)
        if finder_set is not None:
            report.geometry_source = "contour"

    if finder_set is not None:
        report.finder_count = 3
        report.module_size_px = finder_set.module_size
        version = finder_set.version
        sampling = sample_matrix(analysis_gray, finder_set, version)
        outcome = decode_symbol(sampling, "M", 0)
        report.contrast = sampling.contrast
        report.aligned = sampling.aligned
        notes.extend(sampling.notes)
        if outcome.version_info is not None and outcome.version_info.certain:
            # Version info is authoritative; re-sample if it disagrees.
            if outcome.version_info.version != version:
                notes.append("版本信息区读出 v%d，与几何估计 v%d 不符，按版本信息重新采样"
                             % (outcome.version_info.version, version))
                version = outcome.version_info.version
                finder_set.version = version
                sampling = sample_matrix(analysis_gray, finder_set, version)
                outcome = decode_symbol(sampling, "M", 0)
        report.matrix = sampling.matrix
        report.transform = sampling.transform
        report.alignment_expected = len(tables.alignment_centres(version))
        if sampling.from_engine_quad:
            # The engine's quad spans the whole symbol, so every alignment
            # centre is located by construction rather than by template match.
            report.alignment_found = report.alignment_expected
        else:
            report.alignment_found = report.alignment_expected if sampling.aligned else 0
    else:
        report.failure_stage = "检测"

    # --- stage 3: merge ----------------------------------------------------
    if best is not None:
        report.decoded = True
        report.text = best.text
        report.raw = best.raw
        report.engine = best.engine
        report.engines_agree = agree
        report.conflict = conflict
        report.variant = best.variant
        report.mode = Value(best.mode, CERTAIN if best.mode else UNREADABLE, "解码引擎")
        if best.level:
            report.level = Value(best.level, CERTAIN, "解码引擎")
        if best.version:
            report.version = Value(best.version, CERTAIN, "解码引擎")
            if sampling is None or version != best.version:
                # Engine gave a version we did not sample; trust the engine and
                # re-sample so annotations land on the right grid.
                if finder_set is not None:
                    version = best.version
                    finder_set.version = version
                    sampling = sample_matrix(analysis_gray, finder_set, version)
                    outcome = decode_symbol(sampling, best.level or "M",
                                            best.mask if best.mask is not None else 0)
                    report.matrix = sampling.matrix
                    report.transform = sampling.transform
                    report.alignment_expected = len(tables.alignment_centres(version))
                    report.alignment_found = (report.alignment_expected
                                              if sampling.aligned else 0)
        if best.mask is not None:
            report.mask = Value(best.mask, CERTAIN, "解码引擎")
        if best.position and sampling is None:
            notes.append("引擎给出角点但几何检测失败，无法绘制模块网格")
    elif outcome is not None and outcome.ok:
        # The independent reader succeeded where the engines did not.
        report.decoded = True
        report.text = outcome.text
        report.raw = outcome.raw
        report.segments = outcome.segments
        report.engine = "内置读取器"
        report.engines_agree = True
        report.variant = "几何采样"
        report.mode = Value(
            "+".join(sorted({seg.mode for seg in outcome.segments})), CERTAIN, "内置读取器")

    # --- stage 4: partial parameters ---------------------------------------
    if sampling is not None and outcome is not None:
        fmt: Optional[FormatReading] = outcome.format
        ver: Optional[VersionReading] = outcome.version_info

        if fmt is not None:
            report.format_status = _format_copy_status(fmt)

        # Error correction level.
        if report.level.status != CERTAIN:
            if fmt is not None and fmt.certain:
                report.level = Value(fmt.level, CERTAIN, "格式信息 BCH 纠错")
            else:
                levels, masks, _ = _brute_force_format(sampling, fmt, notes)
                report.level_candidates = levels
                report.mask_candidates = masks
                if levels:
                    report.level = Value(levels[0], CANDIDATE, "格式信息穷举候选")
                    report.mask = Value(masks[0], CANDIDATE, "格式信息穷举候选")
                else:
                    report.level = Value(None, UNREADABLE, "格式信息不可读")
                    report.mask = Value(None, UNREADABLE, "格式信息不可读")

        # Mask.
        if report.mask.status != CERTAIN and fmt is not None and fmt.certain:
            report.mask = Value(fmt.mask, CERTAIN, "格式信息 BCH 纠错")

        # Version.
        if report.version.status != CERTAIN:
            if ver is not None and ver.certain:
                report.version = Value(ver.version, CERTAIN, "版本信息 Golay 纠错")
            else:
                report.version = Value(version, ESTIMATED,
                                       "定位图案间距 + 时序图案" if version else "几何估计")
        if report.version.value is not None:
            size = tables.size_for_version(int(report.version.value))
            report.size = Value("%d × %d" % (size, size), report.version.status,
                                report.version.basis)

        if version is not None and version >= 7:
            if ver is not None and ver.certain:
                report.version_status = (CERTAIN, CERTAIN)
            else:
                report.version_status = (UNREADABLE, UNREADABLE)

        # Data area status.
        if report.decoded:
            report.data_status = CERTAIN
            report.data_status_text = "可读"
        elif outcome.block_decode is not None:
            if outcome.block_decode.failed_blocks:
                total_blocks = len(outcome.block_decode.block_errors)
                good = total_blocks - len(outcome.block_decode.failed_blocks)
                report.data_status = ESTIMATED
                report.data_status_text = "部分可读（%d/%d 块纠错成功）" % (good, total_blocks)
            else:
                report.data_status = ESTIMATED
                report.data_status_text = "码字可纠错但内容解析失败"
            report.corrected_errors = outcome.block_decode.total_errors
        else:
            report.data_status = UNREADABLE
            report.data_status_text = "不可读"

        if not report.decoded:
            report.failure_stage = report.failure_stage or "数据解码"
            if outcome.block_decode is not None and outcome.block_decode.notes:
                report.failure_reason = "；".join(outcome.block_decode.notes[:2])
            elif outcome.notes:
                report.failure_reason = "；".join(outcome.notes[:2])
            else:
                report.failure_reason = "数据区无法通过纠错"
        else:
            report.corrected_errors = (outcome.block_decode.total_errors
                                       if outcome.block_decode else 0)

        # Segments for the report (engine path has none).
        if report.decoded and not report.segments and outcome.ok:
            report.segments = outcome.segments

        # Encoding mode fallback from our own segment parsing.
        if report.mode.status != CERTAIN and report.segments:
            report.mode = Value("+".join(sorted({seg.mode for seg in report.segments})),
                                CERTAIN, "段解析")
        elif report.mode.status != CERTAIN:
            # Infer from the raw bytes when nothing else worked.
            if report.raw:
                if all(48 <= b <= 57 for b in report.raw):
                    report.mode = Value("numeric", CANDIDATE, "按内容推测")
                elif all(chr(b) in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
                         for b in report.raw):
                    report.mode = Value("alnum", CANDIDATE, "按内容推测")
                else:
                    report.mode = Value("byte", CANDIDATE, "按内容推测")
    else:
        # No geometry at all: everything is unknown.
        if not report.decoded:
            report.failure_stage = report.failure_stage or "检测"
            report.failure_reason = report.failure_reason or "未检测到二维码定位图案"
        report.level = Value(None, UNREADABLE, "无法定位符号")
        report.mask = Value(None, UNREADABLE, "无法定位符号")
        if report.version.status != CERTAIN:
            report.version = Value(None, UNREADABLE, "无法定位符号")
            report.size = Value(None, UNREADABLE, "无法定位符号")

    # --- stage 5: content type ---------------------------------------------
    if report.text:
        report.content_type = classify_content(report.text)
        report.content_type_name = content_type_label(report.content_type)

    # --- stage 6: annotations ----------------------------------------------
    if sampling is not None and version is not None and outcome is not None:
        _build_annotations(report, sampling, finder_set, outcome, version)

    report.elapsed = time.monotonic() - started
    return report

# ---------------------------------------------------------------------------
# All-symbology entry point
# ---------------------------------------------------------------------------


def _symbols_module():
    """Import :mod:`qrcoder.core.symbols` lazily (it imports this module)."""
    from . import symbols as module
    return module


def _group_results(results: Sequence[EngineResult]) -> List[List[EngineResult]]:
    """Cluster engine detections that describe the same physical symbol."""
    clusters: List[List[EngineResult]] = []
    for result in results:
        if not (result.ok or result.partial):
            continue
        placed = False
        if result.position:
            for cluster in clusters:
                reference = cluster[0]
                if not reference.position:
                    continue
                if reference.symbology != result.symbology:
                    continue
                if _symbols_module().quad_iou(reference.position, result.position) >= 0.35:
                    cluster.append(result)
                    placed = True
                    break
        if not placed:
            clusters.append([result])
    return clusters


def _cluster_engine_results(results: Sequence[EngineResult]) -> EngineResult:
    """Apply the adoption policy inside one cluster."""
    ranking = {"zxing-cpp": 3, "pyzbar": 2, "opencv": 1}
    ordered = sorted(
        results,
        key=lambda r: (1 if r.ok else 0, ranking.get(r.engine, 0)),
        reverse=True)
    best = ordered[0]
    for other in ordered[1:]:
        if best.ok and other.ok and other.engine != best.engine:
            if other.text == best.text:
                best.engines_agree = True
            elif not best.conflict:
                best.conflict = ("%s 与 %s 结果不一致，已采纳 %s"
                                 % (other.engine, best.engine, best.engine))
            break
    return best


def _crop(image: np.ndarray, quad: Sequence[Tuple[float, float]],
          margin: float = 0.18) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """Cut a padded, upright crop of the quad out of *image*.

    The crop is deskewed to a square so the QR structural reader sees an upright
    symbol, which is what its module-space assumptions require.

    :returns: ``(crop, offset, scale)`` where *offset* is the crop's top-left in
        image pixels and *scale* the crop/analysis resolution factor.
    """
    import cv2

    height, width = image.shape[:2]
    points = [(float(a), float(b)) for a, b in list(quad)[:4]]
    if len(points) != 4:
        return image, (0.0, 0.0), 1.0
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
    side = float(max(
        np.hypot(x1 - x0, y1 - y0), np.hypot(x2 - x1, y2 - y1),
        np.hypot(x3 - x2, y3 - y2), np.hypot(x0 - x3, y0 - y3)))
    if side < 8:
        side = 8.0
    pad = side * margin
    xs = [x0, x1, x2, x3]
    ys = [y0, y1, y2, y3]
    bx0 = int(max(0, min(xs) - pad))
    by0 = int(max(0, min(ys) - pad))
    bx1 = int(min(width, max(xs) + pad))
    by1 = int(min(height, max(ys) + pad))
    if bx1 - bx0 < 8 or by1 - by0 < 8:
        return image, (0.0, 0.0), 1.0

    target = int(round(max(bx1 - bx0, by1 - by0)))
    src = np.array([[x0 - bx0, y0 - by0], [x1 - bx0, y1 - by0],
                    [x2 - bx0, y2 - by0], [x3 - bx0, y3 - by0]], dtype=np.float32)
    dst = np.array([[0, 0], [target, 0], [target, target], [0, target]],
                   dtype=np.float32)
    patch = image[by0:by1, bx0:bx1]
    try:
        transform = cv2.getPerspectiveTransform(src, dst)
        crop = cv2.warpPerspective(patch, transform, (target, target),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    except Exception:  # pragma: no cover - defensive
        return patch, (float(bx0), float(by0)), 1.0
    return crop, (float(bx0), float(by0)), 1.0


def _qr_symbol_from_quad(image: np.ndarray, result: EngineResult, index: int,
                         coord_scale: float) -> object:
    """Deep-analyse a QR symbol located by an engine, using a deskewed crop."""
    symbols = _symbols_module()
    base = symbols.symbol_from_engine(result, image, index=index,
                                      coord_scale=coord_scale)
    quad = list(result.position or ())
    if len(quad) != 4 or not result.version:
        return base

    crop, offset, crop_scale = _crop(image, quad)
    crop_gray = to_gray(crop) if crop.ndim == 3 else crop

    report = analyze_image(crop_gray, time_budget=4.0, keys=("qrcode",))
    deep = symbols.symbol_from_analysis(report, index=index, offset=offset,
                                        scale=crop_scale)
    # The engine's quad is authoritative for the outer box in image pixels.
    ratio = 1.0 / (result.scale or 1.0)
    deep.quad = [(x * ratio * coord_scale, y * ratio * coord_scale)
                 for x, y in quad]

    # Merge: the engine knows the payload, the structural reader knows the grid.
    if base.decoded:
        deep.decoded = True
        deep.text = base.text
        deep.raw = base.raw
        deep.engine = base.engine
        deep.engines_agree = base.engines_agree
        deep.conflict = base.conflict
        deep.variant = base.variant
        deep.content_type = base.content_type
        deep.content_type_name = base.content_type_name
        deep.failure_stage = None
        deep.failure_reason = ""
    if base.version.status == CERTAIN:
        deep.version = base.version
        deep.size = base.size
    if base.level.status == CERTAIN and deep.level.status != CERTAIN:
        deep.level = base.level
    if base.mask.status == CERTAIN and deep.mask.status != CERTAIN:
        deep.mask = base.mask
    if deep.modules is None and isinstance(deep.version.value, int):
        deep.modules = tables.size_for_version(deep.version.value)
    if not deep.annotations:
        deep.annotations = base.annotations
    if deep.module_size_px is None and base.module_size_px:
        deep.module_size_px = base.module_size_px
    for note in base.notes:
        if note not in deep.notes:
            deep.notes.append(note)
    return deep


def _geometric_fallback(gray: np.ndarray, scope: Sequence[str]) -> Optional[object]:
    """Detect a symbol without any engine help and read its structure.

    Only Micro QR is covered: its single finder pattern is enough to rebuild the
    grid, and its format information is a self-contained 15-bit BCH codeword, so
    a destroyed data area still leaves version / EC level / mask readable.
    """
    from . import microqr

    if "microqrcode" not in set(scope):
        return None
    try:
        reading = microqr.detect_micro_format(gray)
    except Exception:  # pragma: no cover - defensive
        return None
    if reading is None or not reading.certain:
        return None

    symbols = _symbols_module()
    symbol = symbols.SymbolReport(index=0, symbology="microqrcode")
    symbol.decoded = False
    symbol.failure_stage = "数据解码"
    symbol.failure_reason = "解码引擎未能定位符号，仅读取到格式信息"
    symbol.version = Value(reading.version, CERTAIN, "Micro QR 格式信息 BCH（几何定位）")
    symbol.size = Value("%d × %d" % (reading.size, reading.size), CERTAIN,
                        "Micro QR 格式信息 BCH（几何定位）")
    symbol.modules = reading.size
    symbol.level = Value(reading.level if reading.level != "-" else None,
                         CERTAIN if reading.level != "-" else UNREADABLE,
                         "Micro QR 格式信息 BCH（几何定位）")
    symbol.mask = Value(reading.mask, CERTAIN, "Micro QR 格式信息 BCH（几何定位）")
    symbol.format_status = (CERTAIN, UNREADABLE)
    symbol.notes.append("几何定位成功，格式信息 BCH 校验通过：%s / %s / mask %d"
                        % (reading.version, reading.level, reading.mask))
    symbol.notes.append("数据区无法解码，内容不可恢复")
    quad = getattr(reading, "quad", None)
    if quad:
        symbol.quad = list(quad)
        symbol.annotations = symbols.qr_geometry_annotations(
            symbol, quad, reading.size, "Micro QR")
    if not symbol.annotations:
        symbol.annotations = symbols.non_qr_annotations(
            symbol, "Micro QR", [], reading.size, None, reading.size)
    return symbol


def analyze_symbols(image: np.ndarray, keys: Optional[Sequence[str]] = None,
                    time_budget: float = 14.0) -> object:
    """Detect and analyse every code in *image*, of every enabled symbology.

    ``keys`` restricts the symbology scope; ``None`` means "everything the engine
    supports".  A scope containing only the QR family keeps the pre-existing
    behaviour: each QR symbol goes through the full structural pipeline.
    """
    symbols = _symbols_module()
    started = time.monotonic()
    report = symbols.ImageReport()
    if keys is None:
        report.scope = symbologies.all_keys(include_unsupported=False)
    else:
        report.scope = list(keys)
    qr_only = symbologies.qr_family_only(report.scope)
    qr_enabled = "qrcode" in set(report.scope)

    gray = to_gray(image)
    report.image_shape = (int(gray.shape[0]), int(gray.shape[1]))

    # --- stage 1: detection -------------------------------------------------
    detect_started = time.monotonic()
    variants = build_variants(image, extended=not qr_only)
    report.notes.append("预处理变体：%d 种" % len(variants))
    attempts: List[str] = []
    engine_results: List[EngineResult] = []
    deadline = started + time_budget
    for variant in variants:
        if time.monotonic() > deadline:
            attempts.append("已到时间上限，停止尝试剩余预处理")
            break
        found = read_all_zxingcpp(variant.image, variant.label, variant.scale,
                                  keys=report.scope)
        attempts.append("%s / zxing-cpp：%s" % (
            variant.label, ("%d 个结果" % len(found)) if found else "未检测到"))
        engine_results.extend(found)
        if not found:
            found = read_all_pyzbar(variant.image, variant.label, variant.scale,
                                    keys=report.scope)
            attempts.append("%s / pyzbar：%s" % (
                variant.label, ("%d 个结果" % len(found)) if found else "未检测到"))
            engine_results.extend(found)
        if found:
            break
    report.engine_attempts = attempts
    report.stage_times["检测"] = time.monotonic() - detect_started

    # --- stage 2: per-symbol analysis --------------------------------------
    detail_started = time.monotonic()
    clusters = _group_results(engine_results)
    coord_scale = variants[0].scale if variants else 1.0
    candidates: List[object] = []
    for index, cluster in enumerate(clusters):
        best = _cluster_engine_results(cluster)
        if best.symbology == "qrcode" and qr_enabled:
            try:
                symbol = _qr_symbol_from_quad(image, best, index, coord_scale)
            except Exception as exc:  # pragma: no cover - defensive
                report.notes.append("码 %d 深度分析异常：%s" % (index + 1, exc))
                symbol = symbols.symbol_from_engine(best, image, index=index,
                                                    coord_scale=coord_scale)
        else:
            symbol = symbols.symbol_from_engine(best, image, index=index,
                                                coord_scale=coord_scale)
        candidates.append(symbol)

    # --- stage 3: nothing detected -----------------------------------------
    # Last resort: locate the symbol geometrically and read whatever structure
    # survives.  This covers symbols whose data area is destroyed -- engines then
    # either report nothing at all or mis-classify the wreckage as another format.
    if not any(getattr(c, "decoded", False) for c in candidates):
        geometric = _geometric_fallback(gray, report.scope)
        if geometric is not None:
            candidates = [geometric] + [c for c in candidates if c.symbology == "qrcode"]
            report.notes.append("改由几何检测 + 结构读取获得部分信息")

    if not candidates and qr_enabled:
        # The structural reader is QR-only, so this fallback is meaningless for
        # a scope that excludes the QR family.
        fallback = analyze_image(image, time_budget=time_budget,
                                 keys=report.scope)
        if fallback.annotations or fallback.decoded or fallback.partial:
            candidates.append(symbols.symbol_from_analysis(fallback, index=0))
            report.notes.append("解码引擎未报告结果，改由内置读取器分析")
        for note in fallback.notes:
            if note not in report.notes:
                report.notes.append(note)
        report.engine_attempts.extend(fallback.engine_attempts)

    report.symbols = symbols.merge_symbols(candidates)
    report.stage_times["结构分析"] = time.monotonic() - detail_started
    report.elapsed = time.monotonic() - started
    return report
