"""Image preprocessing variants fed to the decoders.

The spec is explicit that robustness comes from trying many renderings of the
same image, not from one clever algorithm.  Every variant here is cheap, keeps
the original resolution unless stated, and is paired with a label so the report
can say which one produced the hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "Variant",
    "load_image",
    "to_gray",
    "build_variants",
    "binarize_otsu",
    "binarize_adaptive",
    "invert",
    "upscale",
    "sharpen",
    "deskew_by_finders",
]


@dataclass
class Variant:
    """One rendering of the source image handed to a decoder."""

    label: str
    image: np.ndarray
    #: Multiplier applied to the original resolution (1.0 = unchanged).
    scale: float = 1.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Variant(%s, %s, x%.2f)" % (self.label, self.image.shape, self.scale)


MAX_SIDE = 2400


def load_image(path: str) -> np.ndarray:
    """Load an image file as BGR, honouring EXIF orientation."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取图像文件：%s" % path)
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _limit_resolution(gray: np.ndarray, max_side: int = MAX_SIDE) -> Tuple[np.ndarray, float]:
    """Downscale very large photos; decoding speed matters more than pixels."""
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return gray, 1.0
    factor = max_side / float(longest)
    resized = cv2.resize(gray, (int(round(w * factor)), int(round(h * factor))),
                         interpolation=cv2.INTER_AREA)
    return resized, factor


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def binarize_adaptive(gray: np.ndarray, block: Optional[int] = None,
                      c: int = 5) -> np.ndarray:
    h, w = gray.shape[:2]
    if block is None:
        block = max(15, (min(h, w) // 16) | 1)
    block = min(block, min(h, w) if min(h, w) % 2 else min(h, w) - 1)
    block = max(3, block | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, c)


def invert(image: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(image)


def upscale(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * factor), int(h * factor)),
                      interpolation=cv2.INTER_CUBIC)


def sharpen(gray: np.ndarray, amount: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)


def deskew_by_finders(gray: np.ndarray) -> Optional[np.ndarray]:
    """Warp the symbol's finder-pattern quad onto a square.

    Only attempted when three finder patterns are found; the result is a pure
    square rendering at roughly the original module pitch, which the decoders
    like a lot when the photo is strongly tilted.
    """
    from .finders import detect_finders, order_triple

    try:
        finder_set = order_triple(detect_finders(gray))
    except Exception:  # pragma: no cover - detection must never break decoding
        return None
    if finder_set is None:
        return None

    module = max(1.0, finder_set.module_size)
    span = max(finder_set.top_right.x - finder_set.top_left.x,
               finder_set.bottom_left.y - finder_set.top_left.y)
    side = int(round(span + module * 14))  # symbol + quiet zone
    if side < 40:
        return None
    side = min(side, 2000)

    src = np.array([[finder_set.top_left.x, finder_set.top_left.y],
                    [finder_set.top_right.x, finder_set.top_right.y],
                    [finder_set.bottom_left.x, finder_set.bottom_left.y]], dtype=np.float32)
    offset = 3.5 * module
    dst = np.array([[offset, offset],
                    [side - offset, offset],
                    [offset, side - offset]], dtype=np.float32)
    matrix = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(gray, matrix, (side, side), flags=cv2.INTER_LINEAR,
                          borderValue=255)


def channel_variants(image: np.ndarray) -> List[Variant]:
    """Colour-channel renderings for coloured codes and low-contrast codes.

    A red-on-white QR is unreadable in plain grayscale but perfectly readable in
    the green or blue channel, so the individual channels are worth trying.  The
    HSV saturation channel helps when foreground and background share luminance
    but differ in hue.
    """
    if image.ndim == 2:
        return []
    bgr = image[:, :, :3]
    variants = [
        Variant("R 通道", bgr[:, :, 2], 1.0),
        Variant("G 通道", bgr[:, :, 1], 1.0),
        Variant("B 通道", bgr[:, :, 0], 1.0),
    ]
    try:
        import cv2

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        variants.append(Variant("S 通道", hsv[:, :, 1], 1.0))
        variants.append(Variant("V 通道", hsv[:, :, 2], 1.0))
    except Exception:  # pragma: no cover - cv2 is a hard dependency
        pass
    return variants


def close_gaps(gray: np.ndarray, size: Optional[int] = None) -> np.ndarray:
    """Morphological closing, to bridge broken finder patterns / dotted modules."""
    import cv2

    if size is None:
        size = max(3, (min(gray.shape[:2]) // 220) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)


def build_variants(image: np.ndarray, extended: bool = False) -> List[Variant]:
    """Produce the ordered list of renderings to try.

    The order matters: the first hit wins for cheap cases, and the expensive
    variants (upscale, deskew) only run when the cheap ones fail.  Callers that
    want everything can just iterate; the analyser stops early.

    ``extended=True`` appends the colour-channel, morphological and multi-scale
    variants that all-format scanning needs (coloured codes, logo-damaged codes,
    very small or very large symbols).
    """
    gray, scale = _limit_resolution(to_gray(image))
    variants: List[Variant] = [
        Variant("原图", gray, scale),
        Variant("Otsu 二值化", binarize_otsu(gray), scale),
        Variant("自适应阈值", binarize_adaptive(gray), scale),
        Variant("反色", invert(gray), scale),
    ]

    if min(gray.shape[:2]) < 600:
        variants.append(Variant("2x 上采样", upscale(gray, 2.0), scale * 2.0))

    variants.append(Variant("锐化", sharpen(gray), scale))

    deskewed = deskew_by_finders(gray)
    if deskewed is not None:
        variants.append(Variant("透视校正", deskewed, 1.0))

    variants.append(Variant("自适应阈值(反色)", invert(binarize_adaptive(gray)), scale))

    if extended:
        for variant in channel_variants(image):
            variants.append(Variant(variant.label, variant.image, scale))
        variants.append(Variant("形态学闭运算", close_gaps(gray), scale))
        variants.append(Variant("形态学闭运算(反色)", invert(close_gaps(gray)), scale))
        if min(gray.shape[:2]) > 900:
            variants.append(Variant("0.5x 缩放", _resize(gray, 0.5), scale * 0.5))
        if min(gray.shape[:2]) < 400:
            variants.append(Variant("4x 上采样", upscale(gray, 4.0), scale * 4.0))
        variants.append(Variant("自适应阈值(大核)", binarize_adaptive(gray, c=10), scale))
    return variants


def _resize(gray: np.ndarray, factor: float) -> np.ndarray:
    import cv2

    height = max(1, int(round(gray.shape[0] * factor)))
    width = max(1, int(round(gray.shape[1] * factor)))
    return cv2.resize(gray, (width, height),
                      interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)
