"""Finder pattern detection and geometric estimation.

The detector is built on OpenCV's contour hierarchy.  A QR finder pattern is a
7x7 dark square, a 5x5 light ring and a 3x3 dark core, so in the contour tree it
appears as a dark contour containing a light contour containing a dark contour.
That nesting plus an area-ratio test identifies finders reliably and cheaply,
without the slow per-row ratio scan that pure-Python implementations need.

When contour detection fails (heavy blur, extreme noise) the module falls back to
a scanline 1:1:3:1:1 ratio search on a downscaled copy, which is slower but more
tolerant of broken contours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import tables

__all__ = [
    "FinderPattern",
    "FinderSet",
    "detect_finders",
    "estimate_version",
    "finder_set_from_quad",
    "order_triple",
    "refine_version_by_timing",
]


@dataclass
class FinderPattern:
    """One detected finder pattern."""

    x: float
    y: float
    module_size: float
    #: Outer square side in pixels (approx. 7 modules).
    outer_size: float
    #: Number of independent detections that merged into this candidate.
    votes: int = 1
    #: Detection source: ``"contour"`` or ``"scanline"``.
    source: str = "contour"
    #: Area-ratio quality score, 0 (bad) .. 1 (perfect).
    quality: float = 1.0

    def distance_to(self, other: "FinderPattern") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class FinderSet:
    """The three finder patterns of one symbol, plus derived geometry."""

    top_left: FinderPattern
    top_right: FinderPattern
    bottom_left: FinderPattern
    #: Estimated module size in pixels (averaged over the three patterns).
    module_size: float
    #: Estimated version (0 when the estimate is not a valid QR version).
    version: int
    #: Estimated module count.
    size: int
    #: Horizontal and vertical module vectors in pixel space.
    ux: Tuple[float, float] = (0.0, 0.0)
    uy: Tuple[float, float] = (0.0, 0.0)
    #: Extra candidates that were rejected, for diagnostics.
    rejected: List[FinderPattern] = field(default_factory=list)

    def as_list(self) -> List[FinderPattern]:
        return [self.top_left, self.top_right, self.bottom_left]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binarize(gray: np.ndarray) -> List[np.ndarray]:
    """Return several binarisations to try (dark = 255 in each)."""
    out = []
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    out.append(otsu)
    block = max(15, (min(gray.shape[:2]) // 8) | 1)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 5
    )
    out.append(adaptive)
    return out


def _centroid(mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Centroid and area of the non-zero pixels of *mask*."""
    moments = cv2.moments(mask, binaryImage=True)
    area = moments["m00"]
    if area <= 0:
        return None
    return moments["m10"] / area, moments["m01"] / area, area


def _quad_centre(points: np.ndarray) -> Tuple[float, float]:
    """Centre of a quadrilateral given as 4 corner points."""
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def _contour_finders(binary: np.ndarray) -> List[FinderPattern]:
    """Find finder patterns via the dark/light/dark contour nesting."""
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not len(contours):
        return []
    hierarchy = hierarchy[0]
    image_area = float(binary.shape[0] * binary.shape[1])
    found: List[FinderPattern] = []

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 20 or area > image_area * 0.25:
            continue
        # depth 0 => outer dark ring; must have a child (light ring) with a child.
        child = hierarchy[idx][2]
        if child < 0:
            continue
        grandchild = hierarchy[child][2]
        if grandchild < 0:
            continue

        outer_area = area
        ring_area = cv2.contourArea(contours[child])
        core_area = cv2.contourArea(contours[grandchild])
        if outer_area <= 0 or ring_area <= 0 or core_area <= 0:
            continue

        # Expected ratios for a clean symbol: ring/outer ~ 0.51, core/outer ~ 0.18.
        ring_ratio = ring_area / outer_area
        core_ratio = core_area / outer_area
        if not (0.25 <= ring_ratio <= 0.85):
            continue
        if not (0.05 <= core_ratio <= 0.45):
            continue
        if core_ratio >= ring_ratio:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect = w / float(h)
        if not (0.62 <= aspect <= 1.62):
            continue
        # A finder is roughly square and its outer box is ~7 modules.
        extent = outer_area / float(w * h)
        if extent < 0.55:
            continue

        # Sub-pixel centre from the 3x3 core centroid (most stable point).
        core_mask = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(core_mask, contours, grandchild, 255, thickness=cv2.FILLED)
        centre = _centroid(core_mask)
        if centre is None:
            centre = (x + w / 2.0, y + h / 2.0, 0.0)
        cx, cy = centre[0], centre[1]

        outer_size = (w + h) / 2.0
        module_size = outer_size / 7.0
        if module_size < 0.7:
            continue

        # Quality: how close the ratios are to the ideal values.
        q = 1.0 - (abs(ring_ratio - 0.51) / 0.51 + abs(core_ratio - 0.184) / 0.184) / 2.0
        found.append(
            FinderPattern(
                x=cx,
                y=cy,
                module_size=module_size,
                outer_size=outer_size,
                source="contour",
                quality=max(0.0, min(1.0, q)),
            )
        )
    return found


def _scanline_finders(binary: np.ndarray, max_rows: int = 220) -> List[FinderPattern]:
    """Fallback 1:1:3:1:1 ratio scan (ZXing style), on a downscaled binary image."""
    height, width = binary.shape[:2]
    if height < 21 or width < 21:
        return []
    scale = 1.0
    work = binary
    if height > max_rows:
        scale = height / float(max_rows)
        work = cv2.resize(binary, (max(1, int(width / scale)), max_rows),
                          interpolation=cv2.INTER_AREA)
    h, w = work.shape[:2]
    rows = work.astype(np.int16)

    candidates: List[Tuple[float, float, float]] = []
    for y in range(h):
        row = rows[y]
        # Run-length encode the row.
        change = np.flatnonzero(np.diff(row)) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [w]))
        values = row[starts]
        # Slide a 5-run window starting with a dark run.
        n = len(starts)
        for i in range(n - 4):
            if values[i] != 255:
                continue
            counts = ends[i:i + 5] - starts[i:i + 5]
            total = int(counts.sum())
            if total < 7:
                continue
            m = total / 7.0
            tol = max(1.0, m * 0.6)
            if (abs(counts[0] - m) <= tol and abs(counts[1] - m) <= tol
                    and abs(counts[2] - 3 * m) <= 3 * tol and abs(counts[3] - m) <= tol
                    and abs(counts[4] - m) <= tol):
                cx = (starts[i + 2] + ends[i + 2]) / 2.0
                cy = float(y)
                # Vertical cross-check through (cx, cy).
                col = work[:, int(min(w - 1, max(0, cx)))]
                vcounts = _run_lengths_around(col, int(cy))
                if vcounts is None:
                    continue
                vtotal = sum(vcounts)
                vm = vtotal / 7.0
                vtol = max(1.0, vm * 0.6)
                if not (abs(vcounts[0] - vm) <= vtol and abs(vcounts[1] - vm) <= vtol
                        and abs(vcounts[2] - 3 * vm) <= 3 * vtol
                        and abs(vcounts[3] - vm) <= vtol and abs(vcounts[4] - vm) <= vtol):
                    continue
                vy = cy - vcounts[4] - vcounts[3] - vcounts[2] / 2.0
                candidates.append((cx * scale, (vy + cy) / 2.0 * scale, m * scale))

    return _merge_candidates(candidates, source="scanline")


def _run_lengths_around(column: np.ndarray, centre: int) -> Optional[List[int]]:
    """Count the 1:1:3:1:1 runs vertically around *centre* (a dark run centre)."""
    h = len(column)
    if not (0 <= centre < h) or column[centre] != 255:
        return None
    counts = [0, 0, 0, 0, 0]
    # Walk up from centre through the central dark run.
    idx = centre
    while idx >= 0 and column[idx] == 255:
        counts[2] += 1
        idx -= 1
    if idx < 0:
        return None
    while idx >= 0 and column[idx] == 0:
        counts[1] += 1
        idx -= 1
    if idx < 0:
        return None
    while idx >= 0 and column[idx] == 255:
        counts[0] += 1
        idx -= 1
    if counts[0] == 0:
        return None
    # Walk down.
    idx = centre + 1
    while idx < h and column[idx] == 255:
        counts[2] += 1
        idx += 1
    if idx >= h:
        return None
    while idx < h and column[idx] == 0:
        counts[3] += 1
        idx += 1
    if idx >= h:
        return None
    while idx < h and column[idx] == 255:
        counts[4] += 1
        idx += 1
    if counts[4] == 0:
        return None
    return counts


def _merge_candidates(candidates: Sequence[Tuple[float, float, float]],
                      source: str) -> List[FinderPattern]:
    merged: List[FinderPattern] = []
    for (x, y, m) in candidates:
        for existing in merged:
            if (abs(existing.x - x) <= max(1.5, m * 0.75)
                    and abs(existing.y - y) <= max(1.5, m * 0.75)
                    and abs(existing.module_size - m) <= max(1.0, m * 0.6)):
                n = existing.votes + 1
                existing.x = (existing.x * existing.votes + x) / n
                existing.y = (existing.y * existing.votes + y) / n
                existing.module_size = (existing.module_size * existing.votes + m) / n
                existing.votes = n
                break
        else:
            merged.append(FinderPattern(x=x, y=y, module_size=m,
                                        outer_size=m * 7.0, source=source))
    return merged


def _dedupe(patterns: Sequence[FinderPattern]) -> List[FinderPattern]:
    """Merge overlapping detections from different binarisations."""
    out: List[FinderPattern] = []
    for p in sorted(patterns, key=lambda q: (q.votes, q.quality), reverse=True):
        for existing in out:
            limit = max(2.0, max(p.module_size, existing.module_size) * 1.2)
            if (math.hypot(p.x - existing.x, p.y - existing.y) <= limit
                    and abs(p.module_size - existing.module_size)
                    <= max(1.0, existing.module_size * 0.7)):
                n = existing.votes + p.votes
                existing.x = (existing.x * existing.votes + p.x * p.votes) / n
                existing.y = (existing.y * existing.votes + p.y * p.votes) / n
                existing.module_size = (
                    existing.module_size * existing.votes + p.module_size * p.votes
                ) / n
                existing.outer_size = existing.module_size * 7.0
                existing.votes = n
                existing.quality = max(existing.quality, p.quality)
                break
        else:
            out.append(p)
    return out


def detect_finders(gray: np.ndarray) -> List[FinderPattern]:
    """Detect all finder-pattern candidates in a grayscale image."""
    if gray is None or gray.size == 0:
        return []
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    all_found: List[FinderPattern] = []
    for binary in _binarize(gray):
        all_found.extend(_contour_finders(binary))
        all_found.extend(_contour_finders(cv2.bitwise_not(binary)))
    patterns = _dedupe(all_found)
    patterns = [p for p in patterns if p.votes >= 1]

    # If the contour pass found fewer than three, try the scanline fallback.
    if len(patterns) < 3:
        for binary in _binarize(gray):
            extra = _scanline_finders(binary)
            extra += _scanline_finders(cv2.bitwise_not(binary))
            patterns = _dedupe(list(patterns) + list(extra))
            if len(patterns) >= 3:
                break
    return patterns


# ---------------------------------------------------------------------------
# Triple selection and version estimation
# ---------------------------------------------------------------------------


def _angle(a: FinderPattern, b: FinderPattern, c: FinderPattern) -> float:
    """Angle at *b* in degrees."""
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def order_triple(patterns: Sequence[FinderPattern]) -> Optional[FinderSet]:
    """Pick the three patterns that best form a QR finder triangle.

    The corner with the angle closest to 90 degrees becomes the top-left
    pattern; the other two are ordered by cross product so that
    ``top_right`` is clockwise from ``bottom_left`` in image space.
    """
    if len(patterns) < 3:
        return None

    best: Optional[Tuple[float, Tuple[FinderPattern, FinderPattern, FinderPattern]]] = None
    n = len(patterns)
    # Limit the combinatorial search on pathological images.
    pool = list(patterns)
    if n > 12:
        pool = sorted(patterns, key=lambda p: (p.votes, p.quality), reverse=True)[:12]
        n = len(pool)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                trio = (pool[i], pool[j], pool[k])
                # Try each vertex as the right-angle corner.
                for corner_idx in range(3):
                    a = trio[corner_idx]
                    b = trio[(corner_idx + 1) % 3]
                    c = trio[(corner_idx + 2) % 3]
                    ang = _angle(b, a, c)
                    if ang < 60 or ang > 120:
                        continue
                    # Legs must be similar in length for a square symbol.
                    l1 = a.distance_to(b)
                    l2 = a.distance_to(c)
                    if l1 <= 0 or l2 <= 0:
                        continue
                    ratio = max(l1, l2) / min(l1, l2)
                    if ratio > 2.2:
                        continue
                    # Module sizes should agree.
                    ms = [p.module_size for p in trio]
                    ms_ratio = max(ms) / max(1e-6, min(ms))
                    if ms_ratio > 2.2:
                        continue
                    score = abs(ang - 90.0) / 90.0 + (ratio - 1.0) + (ms_ratio - 1.0)
                    score -= 0.02 * sum(p.votes for p in trio)
                    if best is None or score < best[0]:
                        best = (score, (a, b, c))

    if best is None:
        return None

    corner, p1, p2 = best[1]
    # Orient: top-right should be clockwise from bottom-left around the corner.
    cross = ((p1.x - corner.x) * (p2.y - corner.y)
             - (p1.y - corner.y) * (p2.x - corner.x))
    if cross < 0:
        p1, p2 = p2, p1
    # In image coordinates (y down), a clockwise order means top_right has the
    # larger x.  Assign by geometry to be safe.
    top_right, bottom_left = (p1, p2) if p1.x >= p2.x else (p2, p1)

    return _build_finder_set(corner, top_right, bottom_left, patterns)


def _build_finder_set(top_left: FinderPattern, top_right: FinderPattern,
                      bottom_left: FinderPattern,
                      all_patterns: Sequence[FinderPattern]) -> FinderSet:
    module_size = (top_left.module_size + top_right.module_size
                   + bottom_left.module_size) / 3.0
    span_x = top_left.distance_to(top_right)
    span_y = top_left.distance_to(bottom_left)
    span = (span_x + span_y) / 2.0

    # Finder centres are (size - 7) modules apart.
    if module_size <= 0:
        version = 0
        size = 0
    else:
        modules_between = span / module_size
        size = int(round(modules_between)) + 7
        version = int(round((size - 17) / 4.0))
        if version < 1 or version > 40:
            # Snap to the nearest legal version so the caller still gets a value.
            version = max(1, min(40, version))
        size = 17 + 4 * version

    ux = ((top_right.x - top_left.x) / (size - 7), (top_right.y - top_left.y) / (size - 7)) \
        if size > 7 else (0.0, 0.0)
    uy = ((bottom_left.x - top_left.x) / (size - 7), (bottom_left.y - top_left.y) / (size - 7)) \
        if size > 7 else (0.0, 0.0)

    used = {id(top_left), id(top_right), id(bottom_left)}
    rejected = [p for p in all_patterns if id(p) not in used]

    return FinderSet(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        module_size=module_size,
        version=version,
        size=size,
        ux=ux,
        uy=uy,
        rejected=rejected,
    )


def estimate_version(size_px: float, module_size: float) -> int:
    """Version implied by a symbol of *size_px* pixels and a given module size."""
    if module_size <= 0:
        return 0
    modules = size_px / module_size
    version = int(round((modules - 17) / 4.0))
    return max(0, min(40, version))


# ---------------------------------------------------------------------------
# Version refinement using the timing pattern
# ---------------------------------------------------------------------------


def _affine_from_finders(fs: FinderSet, size: int) -> Tuple[float, float, float, float, float, float]:
    """Return (a, b, c, d, e, f) mapping module (col, row) -> pixel (x, y).

    The three finder centres sit at module coordinates (3.5, 3.5),
    (size-3.5, 3.5) and (3.5, size-3.5), which uniquely determines the affine
    transform (rotation + scale + shear are all absorbed).
    """
    span = float(size - 7)
    tl, tr, bl = fs.top_left, fs.top_right, fs.bottom_left
    ux = ((tr.x - tl.x) / span, (tr.y - tl.y) / span)
    uy = ((bl.x - tl.x) / span, (bl.y - tl.y) / span)
    # Module (col,row) = (3.5, 3.5) maps to the top-left finder centre.
    a, b = ux
    c, d = uy
    e = tl.x - (a * 3.5 + c * 3.5)
    f = tl.y - (b * 3.5 + d * 3.5)
    return a, b, c, d, e, f


def _sample(gray: np.ndarray, x: float, y: float) -> Optional[int]:
    """Bilinear sample of *gray* at (x, y); None when outside the image."""
    h, w = gray.shape[:2]
    if not (-1.0 <= x <= w and -1.0 <= y <= h):
        return None
    x = min(max(x, 0.0), w - 1.001)
    y = min(max(y, 0.0), h - 1.001)
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    v = (gray[y0, x0] * (1 - fx) * (1 - fy) + gray[y0, x1] * fx * (1 - fy)
         + gray[y1, x0] * (1 - fx) * fy + gray[y1, x1] * fx * fy)
    return int(v)


def _timing_score(gray: np.ndarray, fs: FinderSet, version: int) -> float:
    """Fraction of timing-pattern modules that match the expected alternation."""
    size = tables.size_for_version(version)
    a, b, c, d, e, f = _affine_from_finders(fs, size)
    values: List[int] = []
    expected: List[int] = []

    # Horizontal timing: module row 6, columns 8..size-9.
    for col in range(8, size - 8):
        x = a * (col + 0.5) + c * 6.5 + e
        y = b * (col + 0.5) + d * 6.5 + f
        v = _sample(gray, x, y)
        if v is None:
            return 0.0
        values.append(v)
        expected.append(1 if col % 2 == 0 else 0)  # dark at even columns

    # Vertical timing: module column 6, rows 8..size-9.
    for row in range(8, size - 8):
        x = a * 6.5 + c * (row + 0.5) + e
        y = b * 6.5 + d * (row + 0.5) + f
        v = _sample(gray, x, y)
        if v is None:
            return 0.0
        values.append(v)
        expected.append(1 if row % 2 == 0 else 0)

    if not values:
        return 0.0
    # Use a threshold halfway between the observed extremes so the score is
    # insensitive to global brightness and contrast.
    lo, hi = min(values), max(values)
    if hi - lo < 12:
        return 0.0
    mid = (lo + hi) / 2.0
    matches = sum(1 for v, want in zip(values, expected) if (1 if v < mid else 0) == want)
    return matches / float(len(values))


def refine_version_by_timing(gray: np.ndarray, fs: FinderSet,
                             window: int = 6) -> FinderSet:
    """Re-estimate the version by scoring the timing pattern.

    The finder-centre spacing gives a version that is usually right but drifts by
    one for large symbols, because the module size derived from a 7-module-wide
    bounding box carries a couple of percent of error.  The timing pattern spans
    the whole symbol and therefore pins the version down far more tightly: we
    score every plausible version and keep the best match.
    """
    if gray is None or gray.size == 0 or fs.module_size <= 0:
        return fs

    coarse = fs.version
    candidates = list(range(max(1, coarse - window), min(40, coarse + window) + 1))
    if coarse < 1:
        candidates = list(range(1, 41))

    best_version = coarse
    best_score = -1.0
    scores: List[Tuple[int, float]] = []
    for version in candidates:
        score = _timing_score(gray, fs, version)
        scores.append((version, score))
        if score > best_score + 1e-9:
            best_score = score
            best_version = version

    # A confident timing match is > 0.8; below that keep the geometric estimate.
    if best_score < 0.7:
        return fs

    size = tables.size_for_version(best_version)
    span = float(size - 7)
    module_size = ((fs.top_left.distance_to(fs.top_right)
                    + fs.top_left.distance_to(fs.bottom_left)) / 2.0) / span
    ux = ((fs.top_right.x - fs.top_left.x) / span, (fs.top_right.y - fs.top_left.y) / span)
    uy = ((fs.bottom_left.x - fs.top_left.x) / span, (fs.bottom_left.y - fs.top_left.y) / span)

    refined = FinderSet(
        top_left=fs.top_left,
        top_right=fs.top_right,
        bottom_left=fs.bottom_left,
        module_size=module_size,
        version=best_version,
        size=size,
        ux=ux,
        uy=uy,
        rejected=fs.rejected,
    )
    refined.timing_score = best_score  # type: ignore[attr-defined]
    refined.timing_scores = scores  # type: ignore[attr-defined]
    return refined


# ---------------------------------------------------------------------------
# Geometry from an engine-supplied symbol quad
# ---------------------------------------------------------------------------


def _finder_from_centre(x: float, y: float, module_size: float) -> FinderPattern:
    return FinderPattern(x=x, y=y, module_size=max(0.5, module_size),
                         outer_size=max(0.5, module_size) * 7.0,
                         votes=1, source="engine-quad", quality=1.0)


def finder_set_from_quad(corners: Sequence[Tuple[float, float]],
                         version: int = 0,
                         size: int = 0) -> Optional[FinderSet]:
    """Build a :class:`FinderSet` from the four symbol corners an engine reports.

    zxing-cpp returns the exact symbol quad (module 0 to module ``size``), which
    is far more reliable than contour detection on damaged symbols.  Deriving the
    three finder centres from it gives the sampler a correct grid even when the
    finder patterns themselves are unreadable.

    :param corners: ``[(x, y), ...]`` in the order top-left, top-right,
        bottom-right, bottom-left.
    :param version: the QR version the quad belongs to (ignored when ``size`` is
        given).
    :param size: module count per side; use this for non-QR grids such as Micro
        QR, which has no version number.
    """
    if len(corners) < 4:
        return None
    if size:
        size = int(size)
        version = max(1, min(40, (size - 17) // 4)) if size >= 21 else 1
    else:
        version = max(1, min(40, int(version)))
        size = tables.size_for_version(version)

    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = [(float(a), float(b)) for a, b in corners[:4]]
    # Module-space corners of the symbol, in the same order.
    mx = [0.0, size, size, 0.0]
    my = [0.0, 0.0, size, size]
    src = np.array([[mx[i], my[i]] for i in range(4)], dtype=np.float64)
    dst = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], dtype=np.float64)

    homography, _ = cv2.findHomography(src, dst, 0)
    if homography is None or not np.all(np.isfinite(homography)):
        return None

    def module_to_pixel(mc: float, mr: float) -> Tuple[float, float]:
        pt = homography @ np.array([mc, mr, 1.0])
        denom = pt[2] if abs(pt[2]) > 1e-12 else 1e-12
        return float(pt[0] / denom), float(pt[1] / denom)

    tl = module_to_pixel(3.5, 3.5)
    tr = module_to_pixel(size - 3.5, 3.5)
    bl = module_to_pixel(3.5, size - 3.5)

    # Module pitch from the finder centre spacing (size - 7 modules apart).
    span = float(size - 7)
    module_size = ((math.dist(tl, tr) + math.dist(tl, bl)) / 2.0) / span
    if module_size <= 0:
        return None

    fs = FinderSet(
        top_left=_finder_from_centre(*tl, module_size),
        top_right=_finder_from_centre(*tr, module_size),
        bottom_left=_finder_from_centre(*bl, module_size),
        module_size=module_size,
        version=version,
        size=size,
        ux=((tr[0] - tl[0]) / span, (tr[1] - tl[1]) / span),
        uy=((bl[0] - tl[0]) / span, (bl[1] - tl[1]) / span),
        rejected=[],
    )
    fs.quad = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]  # type: ignore[attr-defined]
    return fs
