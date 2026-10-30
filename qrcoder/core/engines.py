"""Decoder engine adapters: zxing-cpp, ZBar and OpenCV.

Each adapter normalises its library's result into :class:`EngineResult` so the
pipeline can compare them, prefer agreement, and record conflicts.  A missing
library is not an error: the adapter simply reports itself as unavailable, which
keeps the app working when a wheel is absent on some platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import symbologies

__all__ = [
    "EngineResult",
    "available_engines",
    "run_zxingcpp",
    "run_pyzbar",
    "run_opencv",
    "read_all_zxingcpp",
    "read_all_pyzbar",
    "ENGINE_ZXING",
    "ENGINE_ZBAR",
    "ENGINE_OPENCV",
    "build_format_mask",
]

ENGINE_ZXING = "zxing-cpp"
ENGINE_ZBAR = "pyzbar"
ENGINE_OPENCV = "opencv"

#: EC level letters as reported by zxing-cpp -> our canonical letters.
_ZXING_LEVELS = {"L": "L", "M": "M", "Q": "Q", "H": "H"}


@dataclass
class EngineResult:
    """One engine's verdict for one image variant.

    ``ok`` means the *content* was decoded and its checksum verified.  A symbol
    that the engine detected but whose Reed-Solomon failed still carries useful
    structure (version, mask, EC level), so those fields stay populated while
    ``ok`` remains False.
    """

    engine: str
    ok: bool = False
    text: str = ""
    raw: bytes = b""
    level: Optional[str] = None
    version: Optional[int] = None
    mask: Optional[int] = None
    mode: Optional[str] = None
    #: ``[(x, y), ...]`` corners in the coordinates of the tested image.
    position: Optional[List[Tuple[float, float]]] = None
    #: Scale factor of the variant relative to the original image.
    scale: float = 1.0
    #: Variant label that produced the result.
    variant: str = ""
    error: Optional[str] = None
    #: True when the engine located the symbol but could not verify content.
    partial: bool = False
    #: Extra engine-specific detail (never contains decoded content).
    detail: dict = field(default_factory=dict)
    #: Symbology key from :mod:`qrcoder.core.symbologies`.
    symbology: str = "unknown"
    #: Human-readable symbology name as reported by the engine.
    format_name: str = ""
    #: Version/level strings the engine reported verbatim (e.g. ``"M4"``,
    #: ``"16x16"``, ``"62%"``) for symbologies without numeric versions.
    version_text: str = ""
    level_text: str = ""

    def key(self) -> Tuple[str, bytes]:
        return self.text, self.raw

    @property
    def symbology_name(self) -> str:
        return symbologies.display_name(self.symbology)


# ---------------------------------------------------------------------------
# zxing-cpp
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import guard
    import zxingcpp as _zxing

    _ZXING_OK = hasattr(_zxing, "read_barcode")
except Exception:  # pragma: no cover
    _zxing = None
    _ZXING_OK = False


def _zxing_barcode_to_result(barcode, engine: str, variant: str,
                             scale: float) -> EngineResult:
    result = EngineResult(engine=engine, variant=variant, scale=scale)
    try:
        result.text = barcode.text or ""
    except Exception:
        result.text = ""
    try:
        result.raw = bytes(barcode.bytes)
    except Exception:
        result.raw = result.text.encode("utf-8", "replace")

    # zxing-cpp returns a Barcode even when Reed-Solomon fails: ``valid`` is False
    # and ``error`` carries e.g. "ChecksumError".  Structure is still reported,
    # so keep it but never treat the content as decoded.
    valid = bool(getattr(barcode, "valid", True))
    engine_error = getattr(barcode, "error", None)
    result.ok = valid and bool(result.text)
    if not result.ok:
        result.partial = True
        result.text = ""
        result.raw = b""
        result.error = str(engine_error) if engine_error else "校验失败"
        # Keep only the part before the source location for the UI.
        if " @ " in result.error:
            result.error = result.error.split(" @ ")[0]

    level = getattr(barcode, "ec_level", None)
    if level:
        level_str = str(level)
        result.level_text = level_str
        result.level = _ZXING_LEVELS.get(level_str.upper()[:1])

    extra = getattr(barcode, "extra", None) or {}
    version = extra.get("Version")
    if version is not None:
        result.version_text = str(version)
        try:
            result.version = int(str(version).strip("vV"))
        except (TypeError, ValueError):
            # Data Matrix reports "16x16", Micro QR "M4", Aztec a layer count.
            result.version = None
    mask = extra.get("DataMask")
    if mask is not None:
        try:
            result.mask = int(mask)
        except (TypeError, ValueError):
            result.mask = None

    format_name = ""
    try:
        format_name = str(barcode.format)
    except Exception:
        format_name = ""
    result.format_name = format_name
    result.symbology = symbologies.resolve_key(format_name)
    if result.symbology == "unknown" and not format_name:
        # Very old bindings expose ``symbology_identifier`` only.
        result.symbology = symbologies.resolve_key(
            getattr(barcode, "symbology_identifier", ""))

    position = getattr(barcode, "position", None)
    if position is not None:
        try:
            # zxing-cpp exposes the symbol quad as a Position object with four
            # corner attributes; older builds allowed iteration.
            if hasattr(position, "top_left"):
                corners = [position.top_left, position.top_right,
                           position.bottom_right, position.bottom_left]
                result.position = [(float(p.x), float(p.y)) for p in corners]
            else:
                result.position = [(float(p.x), float(p.y)) for p in position]
        except Exception:
            result.position = None

    try:
        result.mode = str(barcode.content_type)
    except Exception:
        result.mode = None
    return result


#: zxing-cpp format sets we may ask for.
_FORMAT_MASK_CACHE: Dict[str, object] = {}


def build_format_mask(keys: Optional[Sequence[str]] = None):
    """Build the ``formats`` argument for the requested symbology keys.

    ``BarcodeFormat`` values are **not** powers of two in zxing-cpp 3.x, so
    OR-ing their integer values silently produces ``AllReadable``.  The bindings
    accept a list/tuple of members and honour it, so we hand back a list.
    ``None`` means "everything the engine can read".
    """
    if not _ZXING_OK:  # pragma: no cover
        return None
    if keys is None:
        return _zxing.BarcodeFormat.AllReadable
    names = symbologies.engine_formats(keys)
    formats: List[object] = []
    seen = set()
    for name in names:
        member = getattr(_zxing.BarcodeFormat, name, None)
        if member is None or name in seen:
            continue
        seen.add(name)
        formats.append(member)
    if not formats:
        return _zxing.BarcodeFormat.AllReadable
    return formats


def _read_zxing(image: np.ndarray, formats, many: bool):
    """Call zxing-cpp, degrading gracefully across binding versions."""
    reader = _zxing.read_barcodes if many else _zxing.read_barcode
    kwargs = dict(
        formats=formats,
        try_rotate=True,
        try_downscale=True,
        try_invert=True,
        is_pure=False,
        return_errors=True,
    )
    try:
        return reader(image, **kwargs)
    except TypeError:
        # Bindings differ between versions; drop the optional extras one by one.
        for key in ("return_errors", "is_pure"):
            kwargs.pop(key, None)
            try:
                return reader(image, **kwargs)
            except TypeError:
                continue
        try:
            return reader(image)
        except TypeError:  # pragma: no cover
            raise


def run_zxingcpp(image: np.ndarray, variant: str = "", scale: float = 1.0,
                 binarizer: Optional[str] = None,
                 keys: Optional[Sequence[str]] = None) -> EngineResult:
    """Run zxing-cpp over every enabled symbology and keep the first result."""
    results = read_all_zxingcpp(image, variant=variant, scale=scale,
                               binarizer=binarizer, keys=keys)
    if not results:
        return EngineResult(ENGINE_ZXING, variant=variant, scale=scale,
                            error="未检测到二维码")
    for result in results:
        if result.ok:
            return result
    return results[0]


def read_all_zxingcpp(image: np.ndarray, variant: str = "", scale: float = 1.0,
                      binarizer: Optional[str] = None,
                      keys: Optional[Sequence[str]] = None) -> List[EngineResult]:
    """Return **all** symbols zxing-cpp finds, one result per symbol."""
    if not _ZXING_OK:
        return [EngineResult(ENGINE_ZXING, variant=variant, scale=scale,
                             error="zxing-cpp 未安装")]
    formats = build_format_mask(keys)
    kwargs = {}
    if binarizer:
        kwargs["binarizer"] = getattr(_zxing.Binarizer, binarizer)
    try:
        barcodes = _read_zxing(image, formats, many=True)
    except Exception as exc:
        return [EngineResult(ENGINE_ZXING, variant=variant, scale=scale,
                             error="%s: %s" % (type(exc).__name__, exc))]
    if not barcodes:
        return []
    results = [_zxing_barcode_to_result(barcode, ENGINE_ZXING, variant, scale)
               for barcode in barcodes]
    if keys is not None:
        # The engine's format sets overlap (asking for QRCode also yields Micro
        # QR), so enforce the user's selection ourselves.
        allowed = set(keys)
        results = [item for item in results if item.symbology in allowed]
    return results


# ---------------------------------------------------------------------------
# ZBar
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import guard
    from pyzbar import pyzbar as _pyzbar

    _ZBAR_OK = True
except Exception:  # pragma: no cover
    _pyzbar = None
    _ZBAR_OK = False


#: ZBar symbol types -> our symbology keys.
_ZBAR_KEYS = {
    "QRCODE": "qrcode", "EAN13": "ean13", "EAN8": "ean8",
    "UPCA": "upca", "UPCE": "upce", "CODE128": "code128",
    "CODE39": "code39", "CODE93": "code93", "CODABAR": "codabar",
    "I25": "itf", "DATABAR": "databar", "DATABAR_EXP": "databar",
    "PDF417": "pdf417", "DATAMATRIX": "datamatrix", "COMPOSITE": "unknown",
}


def _symbol_to_result(symbol, variant: str, scale: float) -> EngineResult:
    result = EngineResult(engine=ENGINE_ZBAR, ok=True, variant=variant, scale=scale)
    try:
        result.text = symbol.data.decode("utf-8")
        result.raw = symbol.data
    except UnicodeDecodeError:
        result.raw = symbol.data
        result.text = symbol.data.decode("utf-8", "replace")
    polygon = getattr(symbol, "polygon", None)
    if polygon:
        result.position = [(float(p.x), float(p.y)) for p in polygon]
    type_name = str(getattr(symbol, "type", ""))
    result.format_name = type_name
    result.symbology = _ZBAR_KEYS.get(type_name, symbologies.resolve_key(type_name))
    result.detail["quality"] = getattr(symbol, "quality", None)
    return result


def run_pyzbar(image: np.ndarray, variant: str = "", scale: float = 1.0,
               keys: Optional[Sequence[str]] = None) -> EngineResult:
    """Run ZBar and keep the first result (see :func:`read_all_pyzbar`)."""
    results = read_all_pyzbar(image, variant=variant, scale=scale, keys=keys)
    if not results:
        return EngineResult(ENGINE_ZBAR, variant=variant, scale=scale,
                            error="未检测到二维码")
    return results[0]


def read_all_pyzbar(image: np.ndarray, variant: str = "", scale: float = 1.0,
                    keys: Optional[Sequence[str]] = None) -> List[EngineResult]:
    """Return every symbol ZBar finds.  ZBar has no version/mask reporting."""
    if not _ZBAR_OK:
        return [EngineResult(ENGINE_ZBAR, variant=variant, scale=scale,
                             error="pyzbar 未安装")]
    try:
        symbols = _pyzbar.decode(image)
    except Exception as exc:
        return [EngineResult(ENGINE_ZBAR, variant=variant, scale=scale,
                             error="%s: %s" % (type(exc).__name__, exc))]
    allowed = None
    if keys is not None:
        allowed = {key for key in keys}
    out: List[EngineResult] = []
    for symbol in symbols or []:
        result = _symbol_to_result(symbol, variant, scale)
        if allowed is not None and result.symbology not in allowed:
            continue
        out.append(result)
    return out


# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------


def run_opencv(image: np.ndarray, variant: str = "", scale: float = 1.0,
               keys: Optional[Sequence[str]] = None) -> EngineResult:
    """Third opinion from OpenCV's built-in detector (no version info).

    OpenCV only reads QR codes, so it is skipped unless the QR family is enabled.
    """
    if keys is not None and "qrcode" not in keys and "microqrcode" not in keys:
        return EngineResult(ENGINE_OPENCV, variant=variant, scale=scale,
                            error="已跳过（当前码制范围不含 QR 家族）")
    try:
        detector = cv2_detector()
        data, points, _ = detector.detectAndDecode(image)
    except Exception as exc:
        return EngineResult(ENGINE_OPENCV, variant=variant, scale=scale,
                            error="%s: %s" % (type(exc).__name__, exc))
    if not data:
        return EngineResult(ENGINE_OPENCV, variant=variant, scale=scale,
                            error="未检测到二维码")
    result = EngineResult(engine=ENGINE_OPENCV, ok=True, variant=variant, scale=scale,
                          symbology="qrcode", format_name="QR Code")
    result.text = data
    result.raw = data.encode("utf-8", "replace")
    if points is not None:
        result.position = [(float(p[0]), float(p[1])) for p in points.reshape(-1, 2)]
    return result


_DETECTOR = None


def cv2_detector():
    global _DETECTOR
    if _DETECTOR is None:
        import cv2

        _DETECTOR = cv2.QRCodeDetector()
    return _DETECTOR


# ---------------------------------------------------------------------------


def available_engines() -> List[str]:
    engines: List[str] = []
    if _ZXING_OK:
        engines.append(ENGINE_ZXING)
    if _ZBAR_OK:
        engines.append(ENGINE_ZBAR)
    engines.append(ENGINE_OPENCV)
    return engines
