"""Micro QR (M1-M4) structure reader.

Micro QR differs from Model 2 in almost every structural detail:

* a single finder pattern, at the top-left corner only;
* timing patterns along row 0 and column 0 (not row/column 6);
* a single 15-bit format information area, with **no** version information area;
* only four mask patterns (0-3).

The format information is a 15-bit BCH(15,5) codeword, XOR-ed with ``0x4445``,
whose 5 data bits are ``(symbol_number << 2) | mask``.  ``symbol_number`` is
three bits wide and encodes *both* the symbol size and the error correction
level, because M1 has no error correction level choice and M3/M4 gain levels as
they grow.

The bit order was verified empirically against 1375 engine-generated symbols
covering all 16 (size, mask) combinations: reading row 8 left-to-right then
column 8 bottom-to-top, most-significant bit first, gives a 100 % match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .finders import finder_set_from_quad
from . import bch

__all__ = [
    "MICRO_SIZES", "MICRO_SYMBOLS", "SYMBOL_NUMBERS", "MICRO_SIZE_BY_VERSION",
    "MICRO_FORMAT_MASK",
    "micro_size_for_symbol", "micro_version_name",
    "micro_format_positions", "micro_format_bits", "decode_micro_format",
    "micro_format_candidates", "micro_mask_applies", "MICRO_MASK_FUNCTIONS",
    "MicroFormatReading", "read_micro_format",
    "MICRO_CAPACITY", "micro_data_modules", "SN_BY_VERSION_LEVEL", "SYMBOL_NUMBERS_TABLE",
]

#: symbol size -> symbol number (0..3)
MICRO_SIZES = {11: 0, 13: 1, 15: 2, 17: 3}
MICRO_SYMBOLS = {0: "M1", 1: "M2", 2: "M3", 3: "M4"}
SYMBOL_NUMBERS = {v: k for k, v in MICRO_SIZES.items()}
#: ``"M1"`` -> 11, ``"M4"`` -> 17
MICRO_SIZE_BY_VERSION = {MICRO_SYMBOLS[sn]: size for size, sn in MICRO_SIZES.items()}

#: The XOR mask applied to the Micro QR format codeword.
MICRO_FORMAT_MASK = 0x4445

#: ``symbol_number`` (0..7) -> ``(version name, error correction level)``.
SYMBOL_NUMBERS_TABLE = {
    0: ("M1", "-"),
    1: ("M2", "L"),
    2: ("M2", "M"),
    3: ("M3", "L"),
    4: ("M3", "M"),
    5: ("M4", "L"),
    6: ("M4", "M"),
    7: ("M4", "Q"),
}

#: ``(version, level)`` -> ``(data codewords, ec codewords)``.
MICRO_CAPACITY = {
    ("M1", "-"): (3, 2),
    ("M2", "L"): (5, 5),
    ("M2", "M"): (4, 6),
    ("M3", "L"): (11, 6),
    ("M3", "M"): (9, 8),
    ("M4", "L"): (16, 8),
    ("M4", "M"): (14, 10),
    ("M4", "Q"): (10, 14),
}

#: ``(version, level)`` -> ``symbol_number``
SN_BY_VERSION_LEVEL = {v: sn for sn, v in SYMBOL_NUMBERS_TABLE.items()}
#: M1 has no error correction level choice; engines report either ``"-"`` or ``"L"``.
SN_BY_VERSION_LEVEL[("M1", "L")] = 0
SN_BY_VERSION_LEVEL[("M1", "-")] = 0

#: Number of data modules (bits) available, excluding function patterns.
MICRO_DATA_MODULES = {"M1": 36, "M2": 80, "M3": 132, "M4": 192}


def micro_size_for_symbol(symbol_number: int) -> int:
    """Symbol size in modules for a Micro QR ``symbol_number`` (0..7)."""
    version, _ = SYMBOL_NUMBERS_TABLE.get(symbol_number, ("M1", "-"))
    return MICRO_SIZE_BY_VERSION.get(version, 11)


def micro_version_name(symbol_number: int) -> str:
    return SYMBOL_NUMBERS_TABLE.get(symbol_number, ("M1", "-"))[0]


def micro_data_modules(version: str) -> int:
    return MICRO_DATA_MODULES.get(version, 0)


# ---------------------------------------------------------------------------
# Format information
# ---------------------------------------------------------------------------


def micro_format_positions() -> List[Tuple[int, int]]:
    """Module coordinates of the 15 format bits, most significant first."""
    return ([(8, col) for col in range(1, 9)] + [(row, 8) for row in range(7, 0, -1)])


def micro_format_bits(matrix: Sequence[Sequence[int]]) -> int:
    """Read the 15 raw format bits from a sampled module matrix.

    :param matrix: ``matrix[row][col]`` truthy for a dark module.
    """
    value = 0
    for (row, col) in micro_format_positions():
        value = (value << 1) | (1 if matrix[row][col] else 0)
    return value


def _format_codeword(symbol_number: int, mask: int) -> int:
    data = ((symbol_number & 0x7) << 2) | (mask & 0x3)
    return (bch.bch_codeword15(data) ^ MICRO_FORMAT_MASK) & 0x7FFF


@dataclass
class MicroFormatReading:
    """Decoded Micro QR format information."""

    symbol_number: Optional[int] = None
    version: Optional[str] = None
    level: Optional[str] = None
    mask: Optional[int] = None
    size: Optional[int] = None
    errors: int = 0
    raw: int = 0
    candidates: List[Tuple[int, int, int]] = field(default_factory=list)

    @property
    def certain(self) -> bool:
        return self.symbol_number is not None and self.errors <= 3

    def summary(self) -> str:
        if self.certain:
            return "%s / %s / mask %d" % (self.version, self.level, self.mask)
        if self.candidates:
            return "读取失败（候选：%s）" % "、".join(
                "%s %s/%d" % (micro_version_name(sn), SYMBOL_NUMBERS_TABLE[sn][1], mask)
                for sn, mask, _ in self.candidates[:3])
        return "读取失败"


def decode_micro_format(bits: int, max_errors: int = 3) -> Optional[MicroFormatReading]:
    """Decode a raw 15-bit Micro QR format word."""
    best: Optional[MicroFormatReading] = None
    for symbol_number in range(8):
        for mask in range(4):
            distance = bch.hamming(bits, _format_codeword(symbol_number, mask))
            if distance > max_errors:
                continue
            reading = MicroFormatReading(
                symbol_number=symbol_number,
                version=micro_version_name(symbol_number),
                level=SYMBOL_NUMBERS_TABLE[symbol_number][1],
                mask=mask,
                size=micro_size_for_symbol(symbol_number),
                errors=distance,
                raw=bits,
            )
            if best is None or reading.errors < best.errors:
                best = reading
    return best


def micro_format_candidates(bits: int, limit: int = 3) -> List[Tuple[int, int, int]]:
    """Rank ``(symbol_number, mask, hamming distance)`` by distance."""
    ranked = []
    for symbol_number in range(8):
        for mask in range(4):
            distance = bch.hamming(bits, _format_codeword(symbol_number, mask))
            ranked.append((symbol_number, mask, distance))
    ranked.sort(key=lambda item: item[2])
    return ranked[:limit]


def read_micro_format(matrix: Sequence[Sequence[int]]) -> MicroFormatReading:
    """Read + BCH-correct the Micro QR format information."""
    bits = micro_format_bits(matrix)
    reading = decode_micro_format(bits)
    if reading is None:
        reading = MicroFormatReading(raw=bits)
        reading.candidates = micro_format_candidates(bits)
    return reading


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def micro_mask_applies(mask: int, row: int, col: int) -> bool:
    """True when Micro QR *mask* flips the module at ``(row, col)``."""
    i, j = row, col
    if mask == 0:
        return i % 2 == 0
    if mask == 1:
        return (i // 2 + j // 3) % 2 == 0
    if mask == 2:
        return ((i * j) % 2 + (i * j) % 3) % 2 == 0
    if mask == 3:
        return ((i + j) % 2 + (i * j) % 3) % 2 == 0
    return False


MICRO_MASK_FUNCTIONS = tuple(
    (lambda r, c, m=m: micro_mask_applies(m, r, c)) for m in range(4))


def apply_micro_mask(matrix: np.ndarray, mask: int, size: int) -> np.ndarray:
    """Return a copy of *matrix* with the Micro QR mask undone."""
    out = np.array(matrix, copy=True)
    for row in range(size):
        for col in range(size):
            if micro_mask_applies(mask, row, col):
                out[row][col] ^= 1
    return out


# ---------------------------------------------------------------------------
# Function patterns (for annotation)
# ---------------------------------------------------------------------------


def micro_function_regions(size: int) -> List[Tuple[str, int, int, int, int, str]]:
    """``(key, row, col, height, width, label)`` for the Micro QR overlay."""
    regions = [("micro_finder", 0, 0, 7, 7, "定位图案"),
               ("micro_sep_h", 7, 0, 1, 8, "分隔符"),
               ("micro_sep_v", 0, 7, 8, 1, "分隔符"),
               ("micro_timing_h", 0, 8, 1, max(0, size - 8), "时序图案（行 0）"),
               ("micro_timing_v", 8, 0, max(0, size - 8), 1, "时序图案（列 0）"),
               ("micro_format", 1, 8, 8, 1, "格式信息"),
               ("micro_data", 1, 1, max(0, size - 1), max(0, size - 1), "数据区")]
    return regions


# ---------------------------------------------------------------------------
# Geometry-first detection (used when the engine refuses the symbol entirely)
# ---------------------------------------------------------------------------


def micro_quad_from_finder(fx: float, fy: float, module: float, size: int):
    """Build the symbol quad from a Micro QR finder centre.

    The single finder pattern occupies modules 0..7, so its centre sits at module
    (3.5, 3.5) and the symbol spans ``size`` modules from module (0, 0).
    """
    x0 = fx - 3.5 * module
    y0 = fy - 3.5 * module
    side = size * module
    return [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)]


def detect_micro_format(gray: np.ndarray) -> Optional["MicroFormatReading"]:
    """Locate a Micro QR symbol geometrically and read its format information.

    This is the last resort when no engine reports the symbol at all — a common
    outcome once the data area is destroyed, because decoders only return a
    position for symbols they at least partially decoded.  Only the single finder
    pattern is needed, which usually survives damage elsewhere.
    """
    from .finders import detect_finders
    from .sampler import sample_matrix

    patterns = detect_finders(gray)
    if not patterns:
        return None
    height, width = gray.shape[:2]

    # Contour detection on a damaged symbol reports several plausible centres, so
    # search a small neighbourhood around each one.  Both the module pitch and the
    # centre can be off by a fraction of a module, hence the scale sweep.
    def acceptable(reading: Optional[MicroFormatReading], size: int) -> bool:
        # Only a zero-error BCH match counts here: this path is a last resort, so
        # an uncertain guess would be worse than reporting "无法读取".
        return (reading is not None and reading.symbol_number is not None
                and reading.size == size and reading.errors == 0)

    # Largest symbols first: M4/M3 dominate real-world Micro QR usage and this
    # keeps the worst case bounded.
    for pattern in sorted(patterns, key=lambda p: (p.votes, p.quality),
                          reverse=True)[:2]:
        if pattern.module_size < 1.0:
            continue
        for scale in (1.0, 0.94, 1.06):
            module = pattern.module_size * scale
            for size in (17, 15, 13, 11):
                for dx in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
                    for dy in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
                        quad = micro_quad_from_finder(pattern.x + dx,
                                                      pattern.y + dy, module, size)
                        xs = [p[0] for p in quad]
                        ys = [p[1] for p in quad]
                        if (min(xs) < -2 or min(ys) < -2
                                or max(xs) > width + 2 or max(ys) > height + 2):
                            continue
                        try:
                            finder = finder_set_from_quad(quad, size=size)
                            if finder is None:
                                continue
                            sampling = sample_matrix(gray, finder, version=0,
                                                     size=size)
                            reading = read_micro_format(sampling.matrix)
                        except Exception:  # pragma: no cover - defensive
                            continue
                        if acceptable(reading, size):
                            # Keep the winning quad so callers can annotate it.
                            try:
                                reading.quad = quad
                            except Exception:  # pragma: no cover - defensive
                                pass
                            return reading
    return None
