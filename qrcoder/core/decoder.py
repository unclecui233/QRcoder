"""Reading a symbol: format info, version info, codewords, segments.

This module is the "independent reader" the spec asks for.  It never needs a
decoder library: it samples the format information with BCH error correction,
extracts and de-interleaves the codewords, runs Reed-Solomon over every block and
parses the segment stream.  That means a symbol whose *content* is unreadable can
still yield a certain EC level and mask, and a symbol whose format info is
destroyed can still be brute-forced against the data area.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import bch, tables
from .gf import ReedSolomonError, rs_decode
from .layout import build_function_map, data_bit_positions, mask_applies
from .sampler import SamplingResult

__all__ = [
    "FormatReading",
    "VersionReading",
    "CodewordResult",
    "Segment",
    "DecodeOutcome",
    "read_format_info",
    "read_version_info",
    "extract_codewords",
    "decode_blocks",
    "parse_segments",
    "decode_symbol",
    "ALNUMERIC_CHARS",
    "CONTENT_TYPES",
    "classify_content",
]

ALNUMERIC_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"


# ---------------------------------------------------------------------------
# Format information
# ---------------------------------------------------------------------------


@dataclass
class FormatReading:
    """Result of reading one or both format information copies."""

    #: ``(level, mask)`` when a copy was decoded with <= 3 bit errors.
    level: Optional[str] = None
    mask: Optional[int] = None
    #: Bit errors corrected in the winning copy.
    errors: int = -1
    #: Which copy produced the result (1, 2, or 0 when neither worked).
    source: int = 0
    #: Raw 15-bit words sampled from each copy.
    raw: Tuple[int, int] = (0, 0)
    #: Ranked ``(level, mask, distance)`` candidates when decoding failed.
    candidates: List[Tuple[str, int, int]] = field(default_factory=list)
    #: Per-copy detail for the GUI: (copy_index, bits, level, mask, distance).
    copies: List[Tuple[int, int, Optional[str], Optional[int], int]] = field(default_factory=list)

    @property
    def certain(self) -> bool:
        """True when the format information was read within BCH capability."""
        return self.level is not None and self.mask is not None

    def summary(self) -> str:
        if self.certain:
            return "%s / mask %d（副本 %d，纠正 %d bit）" % (
                self.level, self.mask, self.source, self.errors)
        if self.candidates:
            best = self.candidates[0]
            return "读取失败（最近候选 %s / mask %d，距离 %d bit）" % (best[0], best[1], best[2])
        return "读取失败"


def _read_bits(sampling: SamplingResult, coords: Sequence[Tuple[int, int]]) -> int:
    """Assemble a bit word from module coordinates, LSB first."""
    value = 0
    for i, (r, c) in enumerate(coords):
        if 0 <= r < sampling.size and 0 <= c < sampling.size:
            if int(sampling.matrix[r, c]):
                value |= 1 << i
    return value


def read_format_info(sampling: SamplingResult) -> FormatReading:
    """Sample and BCH-correct both format information copies."""
    fmap = build_function_map(sampling.version)
    raw1 = _read_bits(sampling, fmap.format_copy1)
    raw2 = _read_bits(sampling, fmap.format_copy2)

    reading = FormatReading(raw=(raw1, raw2))
    best: Optional[Tuple[int, str, int, int]] = None  # (errors, level, mask, copy)
    for copy_index, raw in ((1, raw1), (2, raw2)):
        decoded = bch.decode_format(raw)
        if decoded is None:
            reading.copies.append((copy_index, raw, None, None, -1))
            continue
        level, mask, dist = decoded
        reading.copies.append((copy_index, raw, level, mask, dist))
        if best is None or dist < best[0]:
            best = (dist, level, mask, copy_index)

    if best is not None:
        reading.errors, reading.level, reading.mask, reading.source = best
        return reading

    # Neither copy was within the 3-bit correction radius.  Rank the closest
    # valid codewords; the caller may test them against the data region.
    candidates: Dict[Tuple[str, int], int] = {}
    for raw in (raw1, raw2):
        for level, mask, dist in bch.format_candidates(raw, limit=8):
            key = (level, mask)
            if key not in candidates or dist < candidates[key]:
                candidates[key] = dist
    reading.candidates = sorted(
        ((level, mask, dist) for (level, mask), dist in candidates.items()),
        key=lambda item: item[2],
    )[:5]
    return reading


# ---------------------------------------------------------------------------
# Version information
# ---------------------------------------------------------------------------


@dataclass
class VersionReading:
    version: Optional[int] = None
    errors: int = -1
    source: int = 0
    raw: Tuple[int, int] = (0, 0)
    candidates: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def certain(self) -> bool:
        return self.version is not None

    def summary(self) -> str:
        if self.certain:
            return "版本 %d（副本 %d，纠正 %d bit）" % (self.version, self.source, self.errors)
        return "无法读取"


def read_version_info(sampling: SamplingResult) -> VersionReading:
    """Sample and Golay-correct both version information copies (v >= 7)."""
    reading = VersionReading()
    if sampling.version < 7:
        return reading

    fmap = build_function_map(sampling.version)
    raw1 = _read_bits(sampling, fmap.version_copy1)
    raw2 = _read_bits(sampling, fmap.version_copy2)
    reading.raw = (raw1, raw2)

    best: Optional[Tuple[int, int, int]] = None  # (errors, version, copy)
    for copy_index, raw in ((1, raw1), (2, raw2)):
        decoded = bch.decode_version(raw)
        if decoded is None:
            continue
        version, dist = decoded
        if best is None or dist < best[0]:
            best = (dist, version, copy_index)

    if best is not None:
        reading.errors, reading.version, reading.source = best
        return reading

    merged: Dict[int, int] = {}
    for raw in (raw1, raw2):
        for version, dist in bch.version_candidates(raw, limit=5):
            if version not in merged or dist < merged[version]:
                merged[version] = dist
    reading.candidates = sorted(merged.items(), key=lambda item: item[1])[:5]
    return reading


# ---------------------------------------------------------------------------
# Codeword extraction
# ---------------------------------------------------------------------------


@dataclass
class CodewordResult:
    """De-interleaved codeword blocks."""

    version: int
    level: str
    mask: int
    #: One list of codewords per block, data symbols first then EC symbols.
    blocks: List[List[int]]
    #: ``(total, data, ec)`` per block.
    specs: List[Tuple[int, int, int]]
    #: Raw bit stream in placement order (for diagnostics).
    bits: List[int] = field(default_factory=list)

    def data_bytes(self) -> bytes:
        out = bytearray()
        for block, (total, data, ec) in zip(self.blocks, self.specs):
            out.extend(block[:data])
        return bytes(out)


def _unmask(sampling: SamplingResult, mask: int) -> List[int]:
    """Read the data region in placement order, undoing *mask*."""
    fmap = build_function_map(sampling.version)
    bits: List[int] = []
    for (r, c) in data_bit_positions(sampling.version):
        bit = 1 if sampling.matrix[r, c] else 0
        if mask_applies(mask, r, c):
            bit ^= 1
        bits.append(bit)
    return bits


def _pack(bits: Sequence[int]) -> List[int]:
    words: List[int] = []
    for i in range(0, len(bits) - 7, 8):
        value = 0
        for bit in bits[i:i + 8]:
            value = (value << 1) | int(bit)
        words.append(value)
    return words


def extract_codewords(sampling: SamplingResult, level: str, mask: int) -> CodewordResult:
    """Undo the mask, pack bits into codewords and de-interleave the blocks."""
    version = sampling.version
    groups = tables.ECC_BLOCKS[version][level]

    specs: List[Tuple[int, int, int]] = []
    for (count, total, data) in groups:
        for _ in range(count):
            specs.append((total, data, total - data))

    bits = _unmask(sampling, mask)
    words = _pack(bits)

    max_data = max(spec[1] for spec in specs)
    max_ec = max(spec[2] for spec in specs)
    blocks: List[List[int]] = [[] for _ in specs]
    pos = 0
    total_words = len(words)
    for i in range(max_data):
        for b, spec in enumerate(specs):
            if i < spec[1] and pos < total_words:
                blocks[b].append(words[pos])
                pos += 1
    for i in range(max_ec):
        for b, spec in enumerate(specs):
            if i < spec[2] and pos < total_words:
                blocks[b].append(words[pos])
                pos += 1

    return CodewordResult(version=version, level=level, mask=mask,
                          blocks=blocks, specs=specs, bits=bits)


# ---------------------------------------------------------------------------
# Block decoding
# ---------------------------------------------------------------------------


@dataclass
class CodewordDecode:
    """Result of Reed-Solomon over all blocks."""

    ok: bool
    data: bytes = b""
    #: Errors corrected per block.
    block_errors: List[int] = field(default_factory=list)
    #: Blocks that could not be corrected.
    failed_blocks: List[int] = field(default_factory=list)
    #: Human-readable failure reasons.
    notes: List[str] = field(default_factory=list)

    @property
    def total_errors(self) -> int:
        return sum(e for e in self.block_errors if e > 0)


def decode_blocks(codewords: CodewordResult) -> CodewordDecode:
    """Run Reed-Solomon over every block and concatenate the data symbols."""
    out = bytearray()
    errors: List[int] = []
    failed: List[int] = []
    notes: List[str] = []

    for index, (block, spec) in enumerate(zip(codewords.blocks, codewords.specs)):
        total, data, ec = spec
        if len(block) < total:
            failed.append(index)
            notes.append("块 %d 码字不足（%d/%d）" % (index, len(block), total))
            errors.append(-1)
            continue
        try:
            fixed, nerr = rs_decode(block[:total], ec)
        except ReedSolomonError as exc:
            failed.append(index)
            errors.append(-1)
            notes.append("块 %d 纠错失败：%s" % (index, exc))
            continue
        errors.append(nerr)
        out.extend(fixed[:data])

    ok = not failed
    return CodewordDecode(ok=ok, data=bytes(out), block_errors=errors,
                          failed_blocks=failed, notes=notes)


# ---------------------------------------------------------------------------
# Segment parsing
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    mode: str
    text: str
    raw: bytes = b""
    char_count: int = 0
    eci: Optional[int] = None


class _BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.bit = 0
        self.total = len(data) * 8

    def remaining(self) -> int:
        return self.total - self.bit

    def read(self, count: int) -> int:
        if count == 0:
            return 0
        if self.bit + count > self.total:
            raise ValueError("segment stream ended unexpectedly")
        value = 0
        for _ in range(count):
            byte = self.data[self.bit >> 3]
            value = (value << 1) | ((byte >> (7 - (self.bit & 7))) & 1)
            self.bit += 1
        return value


def _char_count_bits(mode: str, version: int) -> int:
    group = 0 if version < 10 else (1 if version < 27 else 2)
    table = {
        "numeric": (10, 12, 14),
        "alnum": (9, 11, 13),
        "byte": (8, 16, 16),
        "kanji": (8, 10, 12),
    }
    return table[mode][group]


def _decode_kanji(value: int) -> str:
    """Decode one 13-bit Kanji mode value to its Shift-JIS character."""
    if value < 0x1F00:
        sjis = value + 0x8140
    else:
        sjis = value + 0xC140
    high = (sjis >> 8) & 0xFF
    low = sjis & 0xFF
    try:
        return bytes([high, low]).decode("shift_jis")
    except UnicodeDecodeError:
        return "\ufffd"


_ECI_ENCODINGS = {
    3: "iso-8859-1",
    4: "iso-8859-2",
    5: "iso-8859-3",
    6: "iso-8859-4",
    7: "iso-8859-5",
    8: "iso-8859-6",
    9: "iso-8859-7",
    10: "iso-8859-8",
    11: "iso-8859-9",
    12: "iso-8859-10",
    13: "iso-8859-11",
    15: "iso-8859-13",
    16: "iso-8859-14",
    17: "iso-8859-15",
    18: "iso-8859-16",
    20: "shift_jis",
    21: "windows-1250",
    22: "windows-1251",
    23: "windows-1252",
    24: "windows-1256",
    25: "utf-16-be",
    26: "utf-8",
    27: "us-ascii",
    28: "big5",
    29: "gb18030",
    30: "euc-kr",
    170: "us-ascii",
}


def parse_segments(data: bytes, version: int) -> List[Segment]:
    """Parse the segment stream of a corrected data region."""
    reader = _BitReader(data)
    segments: List[Segment] = []
    eci: Optional[int] = None

    while reader.remaining() >= 4:
        mode_bits = reader.read(4)
        if mode_bits == 0:  # terminator
            break
        if mode_bits == 0b0111:  # ECI
            first = reader.read(8)
            if first & 0x80 == 0:
                eci = first & 0x7F
            elif first & 0xC0 == 0x80:
                eci = ((first & 0x3F) << 8) | reader.read(8)
            else:
                eci = ((first & 0x1F) << 16) | reader.read(16)
            continue
        if mode_bits == 0b0011:  # structured append
            reader.read(16)
            continue
        if mode_bits == 0b0101:  # FNC1 first position
            continue
        if mode_bits == 0b1001:  # FNC1 second position
            reader.read(8)
            continue

        if mode_bits == 0b0001:  # numeric
            count = reader.read(_char_count_bits("numeric", version))
            text = []
            while count >= 3:
                text.append("%03d" % reader.read(10))
                count -= 3
            if count == 2:
                text.append("%02d" % reader.read(7))
            elif count == 1:
                text.append("%d" % reader.read(4))
            value = "".join(text)
            segments.append(Segment("numeric", value, value.encode("ascii"), len(value), eci))
        elif mode_bits == 0b0010:  # alphanumeric
            count = reader.read(_char_count_bits("alnum", version))
            text = []
            while count >= 2:
                pair = reader.read(11)
                text.append(ALNUMERIC_CHARS[pair // 45])
                text.append(ALNUMERIC_CHARS[pair % 45])
                count -= 2
            if count == 1:
                text.append(ALNUMERIC_CHARS[reader.read(6)])
            value = "".join(text)
            segments.append(Segment("alnum", value, value.encode("ascii"), len(value), eci))
        elif mode_bits == 0b0100:  # byte
            count = reader.read(_char_count_bits("byte", version))
            raw = bytes(reader.read(8) for _ in range(count))
            encoding = _ECI_ENCODINGS.get(eci or 26, "utf-8")
            try:
                value = raw.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                value = raw.decode("utf-8", "replace")
            segments.append(Segment("byte", value, raw, count, eci))
        elif mode_bits == 0b1000:  # kanji
            count = reader.read(_char_count_bits("kanji", version))
            text = "".join(_decode_kanji(reader.read(13)) for _ in range(count))
            segments.append(Segment("kanji", text, text.encode("shift_jis", "replace"),
                                    count, eci))
        else:
            raise ValueError("unsupported segment mode 0b%s" % format(mode_bits, "04b"))

    return segments


# ---------------------------------------------------------------------------
# Whole-symbol decoding
# ---------------------------------------------------------------------------


@dataclass
class DecodeOutcome:
    """Everything the independent reader managed to learn about a symbol."""

    ok: bool = False
    text: str = ""
    raw: bytes = b""
    segments: List[Segment] = field(default_factory=list)
    format: Optional[FormatReading] = None
    version_info: Optional[VersionReading] = None
    codewords: Optional[CodewordResult] = None
    block_decode: Optional[CodewordDecode] = None
    notes: List[str] = field(default_factory=list)
    #: ``(level, mask, distance)`` when the format info had to be brute-forced.
    format_guess: Optional[Tuple[str, int, int]] = None


def decode_symbol(sampling: SamplingResult, level: str, mask: int,
                  trust_format: bool = True) -> DecodeOutcome:
    """Decode a sampled symbol using an explicit EC level and mask."""
    outcome = DecodeOutcome()
    outcome.format = read_format_info(sampling)
    outcome.version_info = read_version_info(sampling)

    if trust_format and outcome.format.certain:
        level = outcome.format.level or level
        mask = outcome.format.mask if outcome.format.mask is not None else mask

    codewords = extract_codewords(sampling, level, mask)
    outcome.codewords = codewords
    decoded = decode_blocks(codewords)
    outcome.block_decode = decoded

    if not decoded.ok:
        outcome.notes.extend(decoded.notes)
        return outcome

    try:
        segments = parse_segments(decoded.data, sampling.version)
    except ValueError as exc:
        outcome.notes.append("段解析失败：%s" % exc)
        return outcome

    if not segments:
        outcome.notes.append("数据区为空或只有终止符")
        return outcome

    outcome.ok = True
    outcome.segments = segments
    outcome.text = "".join(seg.text for seg in segments)
    outcome.raw = b"".join(seg.raw for seg in segments)
    return outcome


# ---------------------------------------------------------------------------
# Content classification
# ---------------------------------------------------------------------------

CONTENT_TYPES: Tuple[Tuple[str, str], ...] = (
    ("wifi", "WiFi 配置"),
    ("vcard", "vCard 名片"),
    ("mecard", "MECARD 名片"),
    ("mailto", "邮件"),
    ("sms", "短信"),
    ("tel", "电话"),
    ("geo", "地理坐标"),
    ("otpauth", "OTP 令牌"),
    ("url", "URL 链接"),
    ("text", "纯文本"),
)


def classify_content(text: str) -> str:
    """Best-effort content-type label (matches the generator's payload types)."""
    if not text:
        return "text"
    lowered = text.strip().lower()
    if lowered.startswith("wifi:"):
        return "wifi"
    if lowered.startswith("begin:vcard"):
        return "vcard"
    if lowered.startswith("mecard:"):
        return "mecard"
    if lowered.startswith("mailto:"):
        return "mailto"
    if lowered.startswith(("smtp:", "smtpto:", "smsto:")):
        return "sms"
    if lowered.startswith(("tel:", "facetime:")):
        return "tel"
    if lowered.startswith("geo:"):
        return "geo"
    if lowered.startswith("otpauth:"):
        return "otpauth"
    if lowered.startswith(("http://", "https://", "ftp://", "www.")):
        return "url"
    # A bare domain is still a URL for practical purposes.
    first = lowered.split()[0] if lowered.split() else ""
    if "." in first and " " not in first and len(first) < 100:
        return "url"
    return "text"


def content_type_label(kind: str) -> str:
    for key, label in CONTENT_TYPES:
        if key == kind:
            return label
    return kind
