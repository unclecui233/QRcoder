"""BCH(15,5) format information and Golay(18,6) version information.

Both codes are implemented from scratch (as required by the project spec): no
third-party BCH/Golay helper is used.

Format information
    * 15 bits: 5 data bits (2 EC-level bits + 3 mask bits) + 10 BCH bits.
    * Generator polynomial 0x537 (x^10 + x^8 + x^5 + x^4 + x^2 + x + 1).
    * The final codeword is XORed with the mask 0x5412 so an all-zero data word
      never produces an all-zero symbol.
    * Corrects up to 3 bit errors.

Version information (versions 7..40)
    * 18 bits: 6 data bits (version number) + 12 Golay bits.
    * Generator polynomial 0x1F25 (x^12 + x^11 + x^10 + x^9 + x^8 + x^5 + x^2 + 1).
    * Corrects up to 3 bit errors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .tables import BITS_TO_LEVEL, LEVEL_TO_BITS

__all__ = [
    "FORMAT_GENERATOR",
    "FORMAT_MASK",
    "VERSION_GENERATOR",
    "encode_format",
    "decode_format",
    "format_candidates",
    "encode_version",
    "decode_version",
    "version_candidates",
    "format_bits_from_level_mask",
    "hamming",
]

FORMAT_GENERATOR = 0x537
FORMAT_MASK = 0x5412
VERSION_GENERATOR = 0x1F25

FORMAT_DATA_BITS = 5
FORMAT_TOTAL_BITS = 15
VERSION_DATA_BITS = 6
VERSION_TOTAL_BITS = 18


def hamming(a: int, b: int) -> int:
    """Population count of ``a XOR b``."""
    return bin(a ^ b).count("1")


def _bch_remainder(value: int, generator: int, gen_degree: int) -> int:
    """Polynomial remainder of *value* modulo *generator* (bitwise long division)."""
    rem = value
    for shift in range(value.bit_length() - 1, gen_degree - 1, -1):
        if rem >> shift & 1:
            rem ^= generator << (shift - gen_degree)
    return rem


# ---------------------------------------------------------------------------
# Format information
# ---------------------------------------------------------------------------


def format_bits_from_level_mask(level: str, mask: int) -> int:
    """5-bit data word carried by the format information."""
    if level not in LEVEL_TO_BITS:
        raise ValueError("unknown error correction level: %r" % (level,))
    if not 0 <= mask <= 7:
        raise ValueError("mask must be in 0..7")
    return (LEVEL_TO_BITS[level] << 3) | mask


def bch_codeword15(data: int) -> int:
    """Unmasked 15-bit BCH(15,5) codeword for a 5-bit data word.

    Micro QR uses the same generator polynomial as Model 2 format information
    but a different final XOR mask, so the raw codeword is exposed separately.
    """
    data &= 0x1F
    rem = _bch_remainder(data << 10, FORMAT_GENERATOR, 10)
    return ((data << 10) | rem) & 0x7FFF


def encode_format(level: str, mask: int) -> int:
    """Return the 15-bit masked format information codeword."""
    data = format_bits_from_level_mask(level, mask)
    return (bch_codeword15(data) ^ FORMAT_MASK) & 0x7FFF


#: All 32 valid (masked) format codewords -> (level, mask).
_FORMAT_TABLE: Dict[int, Tuple[str, int]] = {}
for _level in LEVEL_TO_BITS:
    for _mask in range(8):
        _FORMAT_TABLE[encode_format(_level, _mask)] = (_level, _mask)


def decode_format(bits: int, max_errors: int = 3) -> Optional[Tuple[str, int, int]]:
    """Decode a 15-bit format information word.

    :returns: ``(level, mask, errors_corrected)`` or ``None`` when the word is
        more than *max_errors* bits away from every valid codeword.
    """
    bits &= 0x7FFF
    best: Optional[Tuple[str, int, int]] = None
    best_dist = max_errors + 1
    for codeword, (level, mask) in _FORMAT_TABLE.items():
        dist = hamming(bits, codeword)
        if dist < best_dist:
            best_dist = dist
            best = (level, mask, dist)
            if dist == 0:
                break
    return best


def format_candidates(bits: int, limit: int = 3) -> List[Tuple[str, int, int]]:
    """Ranked list of ``(level, mask, distance)`` nearest to *bits*.

    Used when the format information cannot be corrected with confidence: the
    caller tries these candidates against the actual data region.
    """
    ranked = sorted(
        ((hamming(bits & 0x7FFF, cw), lvl, msk) for cw, (lvl, msk) in _FORMAT_TABLE.items()),
        key=lambda item: item[0],
    )
    return [(lvl, msk, dist) for dist, lvl, msk in ranked[:limit]]


# ---------------------------------------------------------------------------
# Version information
# ---------------------------------------------------------------------------


def encode_version(version: int) -> int:
    """Return the 18-bit version information codeword."""
    if not 7 <= version <= 40:
        raise ValueError("version information only exists for versions 7..40")
    rem = _bch_remainder(version << 12, VERSION_GENERATOR, 12)
    return ((version << 12) | rem) & 0x3FFFF


_VERSION_TABLE: Dict[int, int] = {encode_version(v): v for v in range(7, 41)}


def decode_version(bits: int, max_errors: int = 3) -> Optional[Tuple[int, int]]:
    """Decode an 18-bit version information word.

    :returns: ``(version, errors_corrected)`` or ``None``.
    """
    bits &= 0x3FFFF
    best: Optional[Tuple[int, int]] = None
    best_dist = max_errors + 1
    for codeword, version in _VERSION_TABLE.items():
        dist = hamming(bits, codeword)
        if dist < best_dist:
            best_dist = dist
            best = (version, dist)
            if dist == 0:
                break
    return best


def version_candidates(bits: int, limit: int = 3) -> List[Tuple[int, int]]:
    """Ranked list of ``(version, distance)`` nearest to *bits*."""
    ranked = sorted(
        ((hamming(bits & 0x3FFFF), v) for cw, v in _VERSION_TABLE.items()),
        key=lambda item: item[0],
    )
    return [(v, dist) for dist, v in ranked[:limit]]


def level_from_bits(bits: int) -> Optional[str]:
    return BITS_TO_LEVEL.get(bits & 0b11)
