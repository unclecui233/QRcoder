"""Symbol layout: which modules are function patterns, and in what order the
data/error-correction codewords are laid out.

This module owns the *geometry of the symbol in module coordinates*.  It knows
nothing about pixels; the sampler (``sampler.py``) turns module coordinates into
pixel coordinates using the detected finder patterns.

Everything here follows ISO/IEC 18004:

* Function patterns: finder patterns + separators, timing patterns, alignment
  patterns, the dark module, format information and (v>=7) version information.
* Data placement: two-module-wide columns walked from the bottom-right corner
  upwards, skipping the vertical timing column (column 6) and all function
  patterns.
* Format information bit order, both copies.
* Version information bit order, both copies (v>=7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from . import tables

__all__ = [
    "FunctionMap",
    "build_function_map",
    "data_bit_positions",
    "format_bit_positions",
    "version_bit_positions",
    "alignment_centres",
    "codeword_regions",
    "MASK_FUNCTIONS",
    "mask_applies",
]

Coord = Tuple[int, int]  # (row, col)


# ---------------------------------------------------------------------------
# Function pattern map
# ---------------------------------------------------------------------------


@dataclass
class FunctionMap:
    """Bit map of function-pattern modules plus the region bookkeeping."""

    version: int
    size: int
    #: True where the module is part of a function pattern (not data).
    is_function: List[List[bool]]
    #: Region tag per module, used by the annotator.  Empty string for data.
    region: List[List[str]]
    #: Alignment pattern centres (row, col).
    alignment: List[Coord]
    #: Format information module coordinates, copy 1 and copy 2 (bit order).
    format_copy1: List[Coord]
    format_copy2: List[Coord]
    #: Version information module coordinates (v>=7), copies 1 and 2.
    version_copy1: List[Coord]
    version_copy2: List[Coord]

    def function_count(self) -> int:
        return sum(row.count(True) for row in self.is_function)


def _blank(size: int) -> List[List[bool]]:
    return [[False] * size for _ in range(size)]


def _blank_region(size: int) -> List[List[str]]:
    return [[""] * size for _ in range(size)]


def _mark_finder(fm: List[List[bool]], region: List[List[str]], size: int,
                 top: int, left: int, tag: str) -> None:
    """Mark a 7x7 finder pattern plus its 1-module separator."""
    for dr in range(-1, 8):
        for dc in range(-1, 8):
            r = top + dr
            c = left + dc
            if not (0 <= r < size and 0 <= c < size):
                continue
            fm[r][c] = True
            if 0 <= dr <= 6 and 0 <= dc <= 6:
                region[r][c] = tag
            else:
                region[r][c] = "separator"


def build_function_map(version: int) -> FunctionMap:
    """Build the complete function-pattern map for *version*."""
    size = tables.size_for_version(version)
    fm = _blank(size)
    region = _blank_region(size)

    # Finder patterns + separators (A top-left, B top-right, C bottom-left).
    _mark_finder(fm, region, size, 0, 0, "finder_a")
    _mark_finder(fm, region, size, 0, size - 7, "finder_b")
    _mark_finder(fm, region, size, size - 7, 0, "finder_c")

    # Timing patterns: row 6 and column 6.
    for i in range(8, size - 8):
        if not fm[6][i]:
            fm[6][i] = True
            region[6][i] = "timing_h"
        if not fm[i][6]:
            fm[i][6] = True
            region[i][6] = "timing_v"

    # Alignment patterns (5x5), skipping the three finder corners.
    alignment = tables.alignment_centres(version)
    for (r, c) in alignment:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size:
                    fm[rr][cc] = True
                    region[rr][cc] = "alignment"

    # Dark module (always dark, just above the bottom-left format copy).
    fm[size - 8][8] = True
    region[size - 8][8] = "dark_module"

    # Format information reservation: copy 1 hugs the top-left finder, copy 2 is
    # split between the bottom-left and top-right finders.
    fmt1 = format_bit_positions(size)[0]
    fmt2 = format_bit_positions(size)[1]
    for i, (r, c) in enumerate(fmt1):
        fm[r][c] = True
        region[r][c] = "format1"
    for i, (r, c) in enumerate(fmt2):
        fm[r][c] = True
        region[r][c] = "format2"

    # Version information (v >= 7).
    v1, v2 = version_bit_positions(version)
    for (r, c) in v1:
        fm[r][c] = True
        region[r][c] = "version1"
    for (r, c) in v2:
        fm[r][c] = True
        region[r][c] = "version2"

    return FunctionMap(
        version=version,
        size=size,
        is_function=fm,
        region=region,
        alignment=alignment,
        format_copy1=fmt1,
        format_copy2=fmt2,
        version_copy1=v1,
        version_copy2=v2,
    )


# ---------------------------------------------------------------------------
# Format / version information bit coordinates
# ---------------------------------------------------------------------------


def format_bit_positions(size: int) -> Tuple[List[Coord], List[Coord]]:
    """Return the two format information copies in bit order (LSB first).

    Copy 1 runs down the left side of the top-left finder then right along its
    top edge; copy 2 runs along the top-right finder's bottom edge then down the
    bottom-left finder's right edge.  Bit 0 is the least significant bit.
    """
    copy1: List[Coord] = []
    for i in range(6):
        copy1.append((i, 8))
    copy1.append((7, 8))
    copy1.append((8, 8))
    copy1.append((8, 7))
    for i in range(9, 15):
        copy1.append((8, 14 - i))

    copy2: List[Coord] = []
    for i in range(8):
        copy2.append((8, size - 1 - i))
    for i in range(8, 15):
        copy2.append((size - 15 + i, 8))

    return copy1, copy2


def version_bit_positions(version: int) -> Tuple[List[Coord], List[Coord]]:
    """Return the two version information copies in bit order (LSB first)."""
    if version < 7:
        return [], []
    size = tables.size_for_version(version)
    # Copy 1: 3x6 block to the left of the top-right finder.
    copy1: List[Coord] = []
    for i in range(18):
        copy1.append((i // 3, size - 11 + i % 3))
    # Copy 2: the transposed 6x3 block above the bottom-left finder.
    copy2: List[Coord] = []
    for i in range(18):
        copy2.append((size - 11 + i % 3, i // 3))
    return copy1, copy2


# ---------------------------------------------------------------------------
# Data placement
# ---------------------------------------------------------------------------


def data_bit_positions(version: int) -> List[Coord]:
    """Module coordinates of every data/EC bit, in placement order.

    The walk starts at the bottom-right corner and moves upwards in pairs of
    columns, right column first, skipping the vertical timing column (6) and all
    function patterns.  The returned order is exactly the order in which the
    codeword bits are laid out.
    """
    size = tables.size_for_version(version)
    fmap = build_function_map(version)
    is_func = fmap.is_function

    positions: List[Coord] = []
    col = size - 1
    upward = True
    while col > 0:
        if col == 6:  # vertical timing pattern column is skipped entirely
            col -= 1
            continue
        rows = range(size - 1, -1, -1) if upward else range(size)
        for row in rows:
            for c in (col, col - 1):
                if not is_func[row][c]:
                    positions.append((row, c))
        upward = not upward
        col -= 2
    return positions


def codeword_regions(version: int, level: str) -> Dict[str, object]:
    """Split the data bit positions into codeword-level regions.

    Returns a dict with:

    * ``positions``: the ordered list of module coordinates
    * ``blocks``: list of ``(block_index, kind, start_bit, end_bit, codeword_range)``
      where kind is ``"data"`` or ``"ec"``
    * ``data_bits`` / ``ec_bits``: totals
    """
    positions = data_bit_positions(version)
    groups = tables.ECC_BLOCKS[version][level]

    # Build the interleaved codeword -> (block, kind, index-in-block) table.
    block_specs: List[Tuple[int, int, int]] = []  # (total, data, ec)
    for (count, total, data) in groups:
        for _ in range(count):
            block_specs.append((total, data, total - data))

    n_blocks = len(block_specs)
    max_data = max(spec[1] for spec in block_specs)
    max_ec = max(spec[2] for spec in block_specs)

    # Interleaving order: all data codeword #0 across blocks, then #1, ...
    order: List[Tuple[int, str, int]] = []
    for i in range(max_data):
        for b, spec in enumerate(block_specs):
            if i < spec[1]:
                order.append((b, "data", i))
    for i in range(max_ec):
        for b, spec in enumerate(block_specs):
            if i < spec[2]:
                order.append((b, "ec", i))

    blocks: List[Tuple[int, str, int, int, Tuple[int, int]]] = []
    for cw_index, (b, kind, idx) in enumerate(order):
        start = cw_index * 8
        blocks.append((b, kind, start, start + 8, (cw_index, cw_index + 1)))

    return {
        "positions": positions,
        "blocks": blocks,
        "order": order,
        "block_specs": block_specs,
        "data_bits": sum(spec[1] for spec in block_specs) * 8,
        "ec_bits": sum(spec[2] for spec in block_specs) * 8,
        "total_codewords": len(order),
    }


# ---------------------------------------------------------------------------
# Data masking
# ---------------------------------------------------------------------------


def mask_applies(mask: int, row: int, col: int) -> bool:
    """True when data mask *mask* inverts the module at (row, col)."""
    if mask == 0:
        return (row + col) % 2 == 0
    if mask == 1:
        return row % 2 == 0
    if mask == 2:
        return col % 3 == 0
    if mask == 3:
        return (row + col) % 3 == 0
    if mask == 4:
        return (row // 2 + col // 3) % 2 == 0
    if mask == 5:
        return (row * col) % 2 + (row * col) % 3 == 0
    if mask == 6:
        return ((row * col) % 2 + (row * col) % 3) % 2 == 0
    if mask == 7:
        return ((row + col) % 2 + (row * col) % 3) % 2 == 0
    raise ValueError("mask must be 0..7")


#: Same predicates, exposed as callables for the mask-penalty scorer.
MASK_FUNCTIONS = tuple(
    (lambda r, c, m=m: mask_applies(m, r, c)) for m in range(8)
)


def apply_mask(matrix: List[List[int]], mask: int, version: int) -> List[List[int]]:
    """Return *matrix* with data mask *mask* applied (function patterns untouched)."""
    size = len(matrix)
    fmap = build_function_map(version)
    out = [row[:] for row in matrix]
    for r in range(size):
        for c in range(size):
            if fmap.is_function[r][c]:
                continue
            if mask_applies(mask, r, c):
                out[r][c] ^= 1
    return out


def region_labels(version: int) -> Dict[str, str]:
    """Human-readable names for the region tags used by :class:`FunctionMap`."""
    return {
        "finder_a": "定位图案 A",
        "finder_b": "定位图案 B",
        "finder_c": "定位图案 C",
        "separator": "分隔符",
        "timing_h": "时序图案",
        "timing_v": "时序图案",
        "alignment": "对齐图案",
        "format1": "格式信息 1",
        "format2": "格式信息 2",
        "version1": "版本信息 1",
        "version2": "版本信息 2",
        "dark_module": "固定暗模块",
    }


# ---------------------------------------------------------------------------
# Expected function-pattern values (used to score a sampling hypothesis)
# ---------------------------------------------------------------------------


def expected_function_matrix(version: int) -> List[List[int]]:
    """Module values that the standard fixes for every function pattern.

    Returns a ``size x size`` matrix where ``-1`` marks "unknown / data" and
    ``0``/``1`` are the light/dark values the standard mandates.  Comparing a
    sampled matrix against this map is how the sampler decides which geometric
    hypothesis (affine vs. homography, version N vs. N+1) is the right one.
    """
    size = tables.size_for_version(version)
    fmap = build_function_map(version)
    out = [[-1] * size for _ in range(size)]

    def finder(top: int, left: int) -> None:
        for dr in range(7):
            for dc in range(7):
                r, c = top + dr, left + dc
                outer = dr in (0, 6) or dc in (0, 6)
                core = 2 <= dr <= 4 and 2 <= dc <= 4
                out[r][c] = 1 if (outer or core) else 0

    finder(0, 0)
    finder(0, size - 7)
    finder(size - 7, 0)

    # Separators (light) - the ring around each finder.
    for (top, left) in ((0, 0), (0, size - 7), (size - 7, 0)):
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                r, c = top + dr, left + dc
                if not (0 <= r < size and 0 <= c < size):
                    continue
                if 0 <= dr <= 6 and 0 <= dc <= 6:
                    continue
                out[r][c] = 0

    # Timing patterns alternate, starting dark at the finder edge.
    for i in range(8, size - 8):
        out[6][i] = 1 if i % 2 == 0 else 0
        out[i][6] = 1 if i % 2 == 0 else 0

    # Alignment patterns: 5x5 with a dark border, light ring, dark core.
    for (cr, cc) in fmap.alignment:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = cr + dr, cc + dc
                if not (0 <= r < size and 0 <= c < size):
                    continue
                ring = max(abs(dr), abs(dc))
                out[r][c] = 1 if ring in (0, 2) else 0

    # Dark module.
    out[size - 8][8] = 1
    return out


def score_function_patterns(matrix, version: int) -> Tuple[int, int, float]:
    """Compare *matrix* against the mandated function patterns.

    :returns: ``(matches, total, ratio)``.
    """
    expected = expected_function_matrix(version)
    size = len(expected)
    matches = 0
    total = 0
    for r in range(size):
        row_e = expected[r]
        row_m = matrix[r]
        for c in range(size):
            want = row_e[c]
            if want < 0:
                continue
            total += 1
            if int(row_m[c]) == want:
                matches += 1
    ratio = (matches / float(total)) if total else 0.0
    return matches, total, ratio
