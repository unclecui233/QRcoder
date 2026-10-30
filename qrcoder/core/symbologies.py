"""Symbology registry: what we can decode, and how deeply.

The scope of this application is bounded by what the installed zxing-cpp build
can actually read.  This module is the single place that records that boundary,
so the UI can say "engine does not support this" instead of failing silently,
and so the support matrix in the README is generated from the same data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "QR_FAMILY", "MATRIX", "LINEAR",
    "CAP_DECODE", "CAP_PARTIAL", "CAP_GEOMETRY", "CAP_UNSUPPORTED",
    "Symbology",
    "SYMBOLOGIES", "SYMBOLOGY_BY_KEY",
    "by_group", "keys_for_group", "resolve_key", "display_name",
    "group_of", "capability_of", "fields_of", "field_label", "engine_formats",
    "all_keys", "CAPABILITY_LABELS",
    "SUPPORT_MATRIX",
]

# Groups ---------------------------------------------------------------------
QR_FAMILY = "qr"
MATRIX = "matrix"
LINEAR = "linear"

# Capability tiers -----------------------------------------------------------
CAP_DECODE = "decode"            # full content decode
CAP_PARTIAL = "partial"          # structure-level parameters (format/version/EC)
CAP_GEOMETRY = "geometry"        # type + bounding box + size estimate only
CAP_UNSUPPORTED = "unsupported"  # engine cannot read it at all

CAPABILITY_LABELS = {
    CAP_DECODE: "完整解码",
    CAP_PARTIAL: "部分信息识别",
    CAP_GEOMETRY: "仅类型 + 几何",
    CAP_UNSUPPORTED: "引擎不支持",
}


@dataclass(frozen=True)
class Symbology:
    """One symbology we know about."""

    key: str                     # canonical short key, e.g. "qrcode"
    name: str                    # display name, e.g. "QR Code Model 2"
    group: str                   # QR_FAMILY / MATRIX / LINEAR
    priority: str                # "P0" / "P1" / "P2"
    #: zxing-cpp ``BarcodeFormat`` attribute names that map to this key.
    formats: Tuple[str, ...] = ()
    #: Highest capability we implement *ourselves* (independent of the engine).
    capability: str = CAP_DECODE
    #: Which extra parameters we can report, in display order.
    fields: Tuple[str, ...] = ()
    #: True when the current engine build reports this format at all.
    engine_supported: bool = True
    note: str = ""
    default_enabled: bool = True

    @property
    def capability_label(self) -> str:
        return CAPABILITY_LABELS.get(self.capability, self.capability)


# Field keys understood by the info panel --------------------------------
FIELD_LABELS = {
    "version": "版本",
    "size": "尺寸",
    "level": "纠错等级",
    "mask": "掩码",
    "mode": "编码模式",
    "layers": "层数",
    "rows": "行数",
    "columns": "列数",
    "compact": "紧凑型",
    "module_size": "模块尺寸",
    "symbol_number": "符号类型",
    "dimension": "矩阵尺寸",
}

_DEFS: Tuple[Symbology, ...] = (
    # --- QR family -------------------------------------------------------
    Symbology(
        key="qrcode", name="QR Code Model 2", group=QR_FAMILY, priority="P0",
        formats=("QRCode", "QRCodeModel2"),
        capability=CAP_PARTIAL,
        fields=("version", "size", "level", "mask", "mode", "module_size"),
        note="回归基线：格式信息 BCH(15,5) + 版本信息 Golay(18,6) + 几何估计",
    ),
    Symbology(
        key="microqrcode", name="Micro QR", group=QR_FAMILY, priority="P0",
        formats=("MicroQRCode",),
        capability=CAP_PARTIAL,
        fields=("symbol_number", "version", "size", "level", "mask", "mode"),
        note="Micro QR 格式信息（15 bit BCH，含符号类型指示，掩码仅 0–3）",
    ),
    Symbology(
        key="rmqr", name="rMQR (Rectangular Micro QR)", group=QR_FAMILY,
        priority="P2", formats=("RMQRCode",),
        capability=CAP_GEOMETRY, engine_supported=False,
        fields=("size",),
        note="引擎枚举存在但无解码实现，降级为类型 + 定位框",
    ),
    Symbology(
        key="qrcode_model1", name="QR Code Model 1", group=QR_FAMILY, priority="P1",
        formats=("QRCodeModel1",), capability=CAP_GEOMETRY,
        fields=("version", "size", "module_size"),
        note="引擎可读；无 Model 1 专属格式信息解析，仅几何估计",
    ),
    # --- 2-D matrix ------------------------------------------------------
    Symbology(
        key="datamatrix", name="Data Matrix (ECC 200)", group=MATRIX, priority="P0",
        formats=("DataMatrix",),
        capability=CAP_GEOMETRY,
        fields=("dimension", "level", "module_size"),
        note="引擎给出矩阵尺寸；纠错等级本阶段不独立解析",
    ),
    Symbology(
        key="aztec", name="Aztec", group=MATRIX, priority="P0",
        formats=("Aztec", "AztecCode"),
        capability=CAP_GEOMETRY, fields=("layers", "level", "compact"),
        note="引擎给出版本即层数；紧凑/全尺寸本阶段不独立判定",
    ),
    Symbology(
        key="aztec_rune", name="Aztec Rune", group=MATRIX, priority="P2",
        formats=("AztecRune",), capability=CAP_GEOMETRY, fields=("layers",),
        note="引擎可读，无参数解析",
    ),
    Symbology(
        key="pdf417", name="PDF417", group=MATRIX, priority="P0",
        formats=("PDF417", "CompactPDF417", "MicroPDF417"),
        capability=CAP_GEOMETRY, fields=("rows", "columns", "level", "compact"),
        note="引擎只给纠错百分比，行/列数本阶段不解析",
    ),
    Symbology(
        key="maxicode", name="MaxiCode", group=MATRIX, priority="P1",
        formats=("MaxiCode",), capability=CAP_GEOMETRY,
        fields=("level", "module_size"), note="引擎给出纠错等级编号，模式本阶段不解析",
    ),
    Symbology(
        key="hanxin", name="Han Xin", group=MATRIX, priority="P1",
        formats=("HanXin",), capability=CAP_GEOMETRY, engine_supported=False,
        fields=("version", "size"),
        note="当前 zxing-cpp 版本无 Han Xin 编解码器，降级为类型 + 定位框",
    ),
    Symbology(
        key="dotcode", name="DotCode", group=MATRIX, priority="P1",
        formats=("DotCode",), capability=CAP_GEOMETRY, engine_supported=False,
        fields=("size",),
        note="当前 zxing-cpp 版本无 DotCode 解码器，降级为类型 + 定位框",
    ),
    # --- 1-D (default on per user decision) ------------------------------
    Symbology(
        key="code128", name="Code 128", group=LINEAR, priority="P2",
        formats=("Code128",), capability=CAP_GEOMETRY, fields=(),
        note="仅类型 + 定位框；一维码无二维参数",
    ),
    Symbology(
        key="code39", name="Code 39", group=LINEAR, priority="P2",
        formats=("Code39", "Code39Std", "Code39Ext"), capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="code93", name="Code 93", group=LINEAR, priority="P2",
        formats=("Code93",), capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="codabar", name="Codabar", group=LINEAR, priority="P2",
        formats=("Codabar",), capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="ean13", name="EAN-13", group=LINEAR, priority="P2",
        formats=("EAN13", "ISBN"), capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="ean8", name="EAN-8", group=LINEAR, priority="P2",
        formats=("EAN8",), capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="upca", name="UPC-A", group=LINEAR, priority="P2",
        formats=("UPCA",), capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="upce", name="UPC-E", group=LINEAR, priority="P2",
        formats=("UPCE",), capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="itf", name="ITF", group=LINEAR, priority="P2",
        formats=("ITF", "ITF14"), capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="databar", name="GS1 DataBar", group=LINEAR, priority="P2",
        formats=("DataBar", "DataBarOmni", "DataBarStk", "DataBarLtd",
                 "DataBarExp", "DataBarExpStk", "DataBarStkOmni"),
        capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="pzn", name="PZN (Pharmazentralnummer)", group=LINEAR, priority="P2",
        formats=("PZN",), capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="code32", name="Code 32 (Italian Pharmacode)", group=LINEAR,
        priority="P2", formats=("Code32",), capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="dxfilmedge", name="DX Film Edge", group=LINEAR, priority="P2",
        formats=("DXFilmEdge",), capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="ean_addon", name="EAN-2 / EAN-5 附加码", group=LINEAR, priority="P2",
        formats=("EAN2", "EAN5"), capability=CAP_GEOMETRY,
    ),
    Symbology(
        key="telepen", name="Telepen", group=LINEAR, priority="P2",
        formats=("Telepen", "TelepenAlpha", "TelepenNumeric"),
        capability=CAP_GEOMETRY,
    ),
)

SYMBOLOGIES: Dict[str, Symbology] = {s.key: s for s in _DEFS}
SYMBOLOGY_BY_KEY = SYMBOLOGIES


def all(include_unsupported: bool = True) -> List[Symbology]:
    """Every registered symbology, in priority order."""
    if include_unsupported:
        return list(_DEFS)
    return [item for item in _DEFS if item.engine_supported]


def by_group(group: str) -> List[Symbology]:
    return [s for s in SYMBOLOGIES.values() if s.group == group]


def keys_for_group(group: str) -> List[str]:
    return [s.key for s in by_group(group)]


def all_keys(include_unsupported: bool = True) -> List[str]:
    return [s.key for s in SYMBOLOGIES.values()
            if include_unsupported or s.engine_supported]


def display_name(key: Optional[str]) -> str:
    if not key:
        return "未识别"
    symbology = SYMBOLOGIES.get(key)
    return symbology.name if symbology else str(key)


def group_of(key: Optional[str]) -> str:
    symbology = SYMBOLOGIES.get(key or "")
    return symbology.group if symbology else MATRIX


def capability_of(key: Optional[str]) -> str:
    symbology = SYMBOLOGIES.get(key or "")
    return symbology.capability if symbology else CAP_GEOMETRY


def fields_of(key: Optional[str]) -> Tuple[str, ...]:
    symbology = SYMBOLOGIES.get(key or "")
    return symbology.fields if symbology else ()


def field_label(name: str) -> str:
    return FIELD_LABELS.get(name, name)


#: zxing-cpp format attribute name -> our symbology key.
_FORMAT_TO_KEY: Dict[str, str] = {}


def _normalise(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _engine_format_names() -> Dict[str, str]:
    """Map *both* enum attribute names and reported strings to our keys.

    zxing-cpp reports human names (``"EAN-13"``, ``"Code 39 Extended"``) that
    differ from the enum attribute names (``EAN13``, ``Code39Ext``), so both
    spellings are registered.
    """
    lookup: Dict[str, str] = {}
    try:
        import zxingcpp as _z  # noqa: F401
    except Exception:
        _z = None
    for _symbology in SYMBOLOGIES.values():
        for _format in _symbology.formats:
            lookup[_normalise(_format)] = _symbology.key
            if _z is not None:
                member = getattr(_z.BarcodeFormat, _format, None)
                if member is not None:
                    lookup[_normalise(str(member))] = _symbology.key
    return lookup


_FORMAT_TO_KEY.update(_engine_format_names())


def resolve_key(format_name: Optional[str], text_hint: str = "") -> str:
    """Map a zxing-cpp format name onto a symbology key.

    Accepts either the enum attribute name or the human-readable string the
    engine reports.  ``text_hint`` is unused today but kept so callers do not
    need to change if we later disambiguate compact PDF417 by geometry.
    """
    if not format_name:
        return "unknown"
    cleaned = _normalise(format_name)
    if not cleaned:
        return "unknown"
    if cleaned in _FORMAT_TO_KEY:
        return _FORMAT_TO_KEY[cleaned]
    for alias, key in _FORMAT_TO_KEY.items():
        if alias.startswith(cleaned) or cleaned.startswith(alias):
            return key
    return "unknown"


def engine_formats(keys: Optional[Sequence[str]] = None) -> List[str]:
    """Return the zxing-cpp ``BarcodeFormat`` names for *keys*."""
    if keys is None:
        keys = all_keys(include_unsupported=False)
    names: List[str] = []
    for key in keys:
        symbology = SYMBOLOGIES.get(key)
        if symbology is None or not symbology.engine_supported:
            continue
        for name in symbology.formats:
            if name not in names:
                names.append(name)
    return names


def SUPPORT_MATRIX() -> List[Dict[str, str]]:   # noqa: N802 - matrix is data
    """Rows for the README support table."""
    rows = []
    for symbology in SYMBOLOGIES.values():
        rows.append({
            "key": symbology.key,
            "name": symbology.name,
            "group": symbology.group,
            "priority": symbology.priority,
            "capability": symbology.capability_label,
            "engine": "支持" if symbology.engine_supported else "不支持",
            "note": symbology.note,
        })
    return rows


# ---------------------------------------------------------------------------
# Scope presets used by the settings UI
# ---------------------------------------------------------------------------

SCOPE_ALL = "all"
SCOPE_QR = "qr"
SCOPE_CUSTOM = "custom"

SCOPE_LABELS = {
    SCOPE_ALL: "全部码制",
    SCOPE_QR: "仅 QR 家族",
    SCOPE_CUSTOM: "自定义",
}


def keys_for_scope(scope: str,
                   custom: Optional[Sequence[str]] = None) -> List[str]:
    """Resolve a scope preset to the symbology keys that should be enabled.

    ``SCOPE_QR`` reproduces the pre-change behaviour exactly (QR Code Model 1/2,
    Micro QR, rMQR) so users who relied on the old partial-information pipeline
    keep it.
    """
    if scope == SCOPE_QR:
        return [s.key for s in by_group(QR_FAMILY) if s.engine_supported]
    if scope == SCOPE_CUSTOM:
        chosen = [k for k in (custom or []) if k in SYMBOLOGIES]
        return chosen or [s.key for s in by_group(QR_FAMILY) if s.engine_supported]
    return all_keys(include_unsupported=False)


def qr_family_only(keys: Sequence[str]) -> bool:
    """True when *keys* covers the QR family and nothing else."""
    family = {s.key for s in by_group(QR_FAMILY)}
    return bool(keys) and set(keys).issubset(family)
