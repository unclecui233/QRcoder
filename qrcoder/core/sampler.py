"""Module sampling: turn pixels into a module matrix.

Two stages:

1. Build an affine transform from the three finder centres (they sit at known
   module coordinates, so three points are enough).
2. If an alignment pattern is expected, refine the transform into a full
   homography using the bottom-right alignment pattern.  That is what makes
   perspective-distorted symbols sample correctly instead of drifting.

The sampler also returns the transform and the sampled coordinates so the GUI can
draw the grid exactly where the analyser looked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import layout, tables
from .finders import FinderSet

__all__ = [
    "SamplingResult",
    "module_to_pixel_matrix",
    "sample_matrix",
    "refine_with_alignment",
    "matrix_to_binary",
]

Coord = Tuple[int, int]


@dataclass
class SamplingResult:
    """Outcome of sampling one symbol."""

    version: int
    size: int
    #: ``size x size`` array of module values in 0..255 (dark is low).
    values: np.ndarray
    #: ``size x size`` array of module bits, 1 = dark.
    matrix: np.ndarray
    #: 3x3 homography mapping module (col, row) -> pixel (x, y).
    transform: np.ndarray
    #: Module size in pixels (average of the two axes).
    module_size: float
    #: Whether an alignment pattern refined the transform.
    aligned: bool = False
    #: True when the transform came from an engine-supplied symbol quad.
    from_engine_quad: bool = False
    #: Mean absolute deviation of the sampled values (a contrast proxy).
    contrast: float = 0.0
    #: Diagnostics.
    notes: List[str] = field(default_factory=list)

    def pixel_of(self, row: int, col: int) -> Tuple[float, float]:
        """Pixel coordinate of the centre of module (row, col)."""
        pt = self.transform @ np.array([col + 0.5, row + 0.5, 1.0])
        return float(pt[0] / pt[2]), float(pt[1] / pt[2])


def module_to_pixel_matrix(fs: FinderSet, size: int) -> np.ndarray:
    """Affine homography from module coordinates to pixels using finder centres."""
    span = float(size - 7)
    tl, tr, bl = fs.top_left, fs.top_right, fs.bottom_left
    ux = ((tr.x - tl.x) / span, (tr.y - tl.y) / span)
    uy = ((bl.x - tl.x) / span, (bl.y - tl.y) / span)
    # Module (3.5, 3.5) is the top-left finder centre.
    h = np.array(
        [
            [ux[0], uy[0], tl.x - (ux[0] * 3.5 + uy[0] * 3.5)],
            [ux[1], uy[1], tl.y - (ux[1] * 3.5 + uy[1] * 3.5)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return h


def _sample_patch(image: np.ndarray, x: float, y: float, radius: float) -> Optional[float]:
    """Average of a small square patch centred on (x, y)."""
    h, w = image.shape[:2]
    if not (-radius <= x < w + radius and -radius <= y < h + radius):
        return None
    r = max(0, int(round(radius)))
    x0 = int(math.floor(x)) - r
    x1 = int(math.floor(x)) + r + 1
    y0 = int(math.floor(y)) - r
    y1 = int(math.floor(y)) + r + 1
    x0c, x1c = max(0, x0), min(w, x1)
    y0c, y1c = max(0, y0), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return None
    patch = image[y0c:y1c, x0c:x1c]
    return float(patch.mean())


def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu threshold over the module values themselves."""
    flat = values.reshape(-1)
    lo, hi = float(flat.min()), float(flat.max())
    if hi - lo < 1e-6:
        return (lo + hi) / 2.0
    hist, edges = np.histogram(flat, bins=64, range=(lo, hi))
    hist = hist.astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total <= 0:
        return (lo + hi) / 2.0
    w0 = np.cumsum(hist)
    w1 = total - w0
    sum0 = np.cumsum(hist * centres)
    sum_total = sum0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.where(w0 > 0, sum0 / w0, 0.0)
        mu1 = np.where(w1 > 0, (sum_total - sum0) / w1, 0.0)
        between = w0 * w1 * (mu0 - mu1) ** 2
    idx = int(np.nanargmax(between))
    return float(centres[idx])


def _alignment_template_score(gray: np.ndarray, cx: float, cy: float,
                             module: float) -> float:
    """Correlate the 5x5 alignment pattern template centred at (cx, cy).

    The template is the ISO 18004 alignment pattern::

        dark  dark  dark  dark  dark
        dark  light light light dark
        dark  light dark  light dark
        dark  light light light dark
        dark  dark  dark  dark  dark

    The score is ``(mean_dark - mean_light) / 255``: positive and large when the
    pattern is present, near zero or negative otherwise.
    """
    dark_sum = 0.0
    dark_n = 0
    light_sum = 0.0
    light_n = 0
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            x = cx + dc * module
            y = cy + dr * module
            v = _sample_patch(gray, x, y, max(0.0, module * 0.2))
            if v is None:
                return -1.0
            is_light = (abs(dr) == 1 and abs(dc) <= 1) or (abs(dc) == 1 and abs(dr) <= 1) or (dr == 0 and dc == 0)
            if is_light:
                light_sum += v
                light_n += 1
            else:
                dark_sum += v
                dark_n += 1
    if dark_n == 0 or light_n == 0:
        return -1.0
    return ((light_sum / light_n) - (dark_sum / dark_n)) / 255.0


def _refine_alignment_centroid(gray: np.ndarray, ax: float, ay: float,
                               module: float) -> Optional[Tuple[float, float]]:
    """Snap an alignment-pattern estimate onto the dark centre module.

    The centre of a 5x5 alignment pattern is a single dark module ringed by one
    light module, so isolating the dark blob whose area is closest to one module
    square gives a sub-pixel centre that does not suffer from the correlation
    plateau of the coarse template score.
    """
    half = int(round(module * 1.6))
    x0 = int(round(ax)) - half
    y0 = int(round(ay)) - half
    x1 = x0 + 2 * half + 1
    y1 = y0 + 2 * half + 1
    h, w = gray.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None

    patch = gray[y0:y1, x0:x1]
    _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count < 2:
        return None

    target_area = module * module
    best: Optional[Tuple[float, float, float]] = None
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[index][0]), float(centroids[index][1])
        gx, gy = x0 + cx, y0 + cy
        # Area must be close to a single module, and the blob must sit near the
        # seed so the outer dark ring can never win.
        if not (0.4 * target_area <= area <= 1.8 * target_area):
            continue
        offset = math.hypot(gx - ax, gy - ay)
        if offset > module:
            continue
        cost = abs(area - target_area) / target_area + offset / module
        if best is None or cost < best[0]:
            best = (cost, gx, gy)
    if best is None:
        return None
    return best[1], best[2]


def refine_with_alignment(gray: np.ndarray, fs: FinderSet, size: int,
                          transform: np.ndarray) -> Tuple[np.ndarray, bool, List[str]]:
    """Refine the transform with the bottom-right alignment pattern.

    The alignment pattern nearest the bottom-right corner sits at module
    coordinate ``(size-6.5, size-6.5)``.  Its predicted pixel position seeds a
    local template search: we correlate the 5x5 dark/light/dark template over a
    small neighbourhood, then fit a parabola to the score peak for sub-pixel
    precision.  A homography fitted to four correspondences removes the
    perspective drift that a pure affine transform cannot express.
    """
    notes: List[str] = []
    version = tables.version_for_size(size)
    if version < 2:
        return transform, False, ["version 1 has no alignment pattern"]

    seed = transform @ np.array([size - 6.5, size - 6.5, 1.0])
    sx, sy = float(seed[0] / seed[2]), float(seed[1] / seed[2])

    h, w = gray.shape[:2]
    module = max(1.0, fs.module_size)
    if not (-module <= sx < w + module and -module <= sy < h + module):
        return transform, False, ["predicted alignment centre outside the image"]

    # Coarse search on a 0.5-module grid, then a fine pass around the winner.
    span = int(max(2, round(module * 2.5)))
    step = max(1.0, module / 2.0)

    def search(cx: float, cy: float, half: float, delta: float) -> Tuple[float, float, float]:
        best = (-2.0, cx, cy)
        n = int(half / delta)
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                x = cx + i * delta
                y = cy + j * delta
                score = _alignment_template_score(gray, x, y, module)
                if score > best[0]:
                    best = (score, x, y)
        return best

    score, ax, ay = search(sx, sy, span, step)
    if score < 0.25:
        return transform, False, notes + [
            "alignment pattern not confirmed (score %.3f); using affine transform" % score
        ]
    # Fine pass with quarter-module steps.
    fine = search(ax, ay, step, max(0.25, module / 8.0))
    if fine[0] >= score:
        score, ax, ay = fine

    # The 5x5 template is one module wide per ring, so its correlation score
    # plateaus over a ~2 px neighbourhood and the grid search can settle half a
    # module off centre.  Snap to the centroid of the dark centre module.
    refined = _refine_alignment_centroid(gray, ax, ay, module)
    if refined is not None:
        ax, ay = refined

    notes.append("alignment pattern at (%.2f, %.2f), score %.2f, offset %.1f px"
                 % (ax, ay, score, math.hypot(ax - sx, ay - sy)))

    src = np.array(
        [[3.5, 3.5], [size - 3.5, 3.5], [3.5, size - 3.5], [size - 6.5, size - 6.5]],
        dtype=np.float64,
    )
    dst = np.array(
        [[fs.top_left.x, fs.top_left.y],
         [fs.top_right.x, fs.top_right.y],
         [fs.bottom_left.x, fs.bottom_left.y],
         [ax, ay]],
        dtype=np.float64,
    )
    homography, _ = cv2.findHomography(src, dst, 0)
    if homography is None or not np.all(np.isfinite(homography)):
        return transform, False, notes + ["homography solve failed"]
    return homography, True, notes


def sample_matrix(image: np.ndarray, fs: FinderSet, version: int = 0,
                  use_alignment: bool = True, size: int = 0) -> SamplingResult:
    """Sample the module matrix of the symbol described by *fs*.

    :param image: grayscale image (uint8).
    :param fs: detected finder patterns with a version estimate.
    :param version: the version to sample (may differ from ``fs.version``).
    :param use_alignment: refine the transform with the alignment pattern.
    :param size: module count per side; overrides ``version`` for grids that have
        no version number, such as Micro QR.
    """
    if size:
        size = int(size)
    else:
        size = tables.size_for_version(version)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    quad = getattr(fs, "quad", None)
    if quad:
        # An engine reported the exact symbol quad; that homography is more
        # reliable than anything derived from detected finder centres, so use it
        # directly and skip both the affine fit and the alignment refinement.
        src = np.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]],
                       dtype=np.float64)
        dst = np.array(quad, dtype=np.float64)
        transform, _ = cv2.findHomography(src, dst, 0)
        if transform is not None and np.all(np.isfinite(transform)):
            values = _sample_values(image, transform, size, fs.module_size)
            threshold = _otsu_threshold(values)
            return SamplingResult(
                version=version,
                size=size,
                values=values,
                matrix=(values < threshold).astype(np.uint8),
                transform=transform,
                module_size=fs.module_size,
                aligned=True,
                from_engine_quad=True,
                contrast=float(np.std(values)),
                notes=["使用解码引擎给出的符号四角点构建网格"],
            )

    affine = module_to_pixel_matrix(fs, size)
    transform = affine
    aligned = False
    notes: List[str] = []
    if use_alignment:
        candidate, aligned, notes = refine_with_alignment(image, fs, size, affine)
        if aligned:
            # A homography fitted to a mis-detected alignment centre can be
            # worse than the plain affine transform, so sample both and keep
            # whichever explains more of the mandated function patterns.
            homography = _sample_values(image, candidate, size, fs.module_size)
            base = _sample_values(image, affine, size, fs.module_size)
            h_score = layout.score_function_patterns(_binarize_values(homography), version)[2]
            a_score = layout.score_function_patterns(_binarize_values(base), version)[2]
            if a_score > h_score:
                notes.append("affine transform scores better (%.3f vs %.3f); "
                             "alignment refinement discarded" % (a_score, h_score))
                aligned = False
            else:
                transform = candidate

    values = _sample_values(image, transform, size, fs.module_size)

    if np.isnan(values).any():
        values = np.nan_to_num(values, nan=255.0)

    threshold = _otsu_threshold(values)
    matrix = (values < threshold).astype(np.uint8)
    contrast = float(np.std(values))

    return SamplingResult(
        version=version,
        size=size,
        values=values,
        matrix=matrix,
        transform=transform,
        module_size=fs.module_size,
        aligned=aligned,
        contrast=contrast,
        notes=notes,
    )


def _binarize_values(values: np.ndarray) -> np.ndarray:
    return (values < _otsu_threshold(values)).astype(np.uint8)


def _sample_values(image: np.ndarray, transform: np.ndarray, size: int,
                   module_size: float) -> np.ndarray:
    """Sample every module centre through *transform*.

    Fully vectorised: the module grid is pushed through the transform in one
    matrix multiply, then the small averaging patch is applied with a box blur
    so a 177x177 symbol costs one remap instead of 31k Python-level lookups.
    """
    coords = np.arange(size, dtype=np.float64) + 0.5
    grid_c, grid_r = np.meshgrid(coords, coords)          # (size, size)
    ones = np.ones_like(grid_c)
    pts = np.stack([grid_c, grid_r, ones], axis=-1)        # (size, size, 3)
    mapped = pts @ transform.T                             # (size, size, 3)
    denom = mapped[..., 2]
    denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    xs = (mapped[..., 0] / denom).astype(np.float32)
    ys = (mapped[..., 1] / denom).astype(np.float32)

    h, w = image.shape[:2]
    radius = max(0.0, module_size * 0.22)
    if radius >= 1.0:
        k = int(round(radius * 2)) | 1
        source = cv2.blur(image, (k, k))
    else:
        source = image

    values = cv2.remap(source, xs, ys, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return values.astype(np.float64)


def matrix_to_binary(sampling: SamplingResult) -> np.ndarray:
    """Render a sampled matrix as a scaled black/white image (for debugging/GUI)."""
    size = sampling.size
    scale = 8
    img = np.where(sampling.matrix > 0, 0, 255).astype(np.uint8)
    return cv2.resize(img, (size * scale, size * scale), interpolation=cv2.INTER_NEAREST)
