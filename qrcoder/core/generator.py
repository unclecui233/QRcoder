"""Payload builders and QR generation.

Generation goes through :mod:`segno` when available because it implements the
standard penalty rules for automatic mask selection and knows the exact version
capacities.  A minimal fallback encoder is provided so the app still works if
segno is missing, but segno is the documented dependency.

Every generated symbol reports the parameters that were actually used, so the GUI
can display version / EC level / mask / mode alongside the matrix.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PayloadType",
    "PAYLOAD_TYPES",
    "build_payload",
    "parse_payload",
    "GenerationRequest",
    "GenerationResult",
    "generate",
    "GenerationError",
    "segno_available",
]

try:  # pragma: no cover - import guard
    import segno as _segno

    _SEGNO_OK = True
except Exception:  # pragma: no cover
    _segno = None
    _SEGNO_OK = False


class GenerationError(ValueError):
    """Raised when the requested parameters cannot produce a symbol."""


def segno_available() -> bool:
    return _SEGNO_OK


# ---------------------------------------------------------------------------
# Payload types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PayloadType:
    key: str
    label: str
    fields: Tuple[Tuple[str, str, str], ...]  # (name, label, placeholder)
    help_text: str = ""


PAYLOAD_TYPES: Tuple[PayloadType, ...] = (
    PayloadType(
        "text", "纯文本",
        (("text", "文本内容", "任意文本，支持多行"),),
        "原样编码为二维码内容。",
    ),
    PayloadType(
        "url", "URL 链接",
        (("url", "网址", "https://example.com"),
         ("scheme", "补全协议", "https://")),
        "缺少协议时自动补全。",
    ),
    PayloadType(
        "wifi", "WiFi 配置",
        (("ssid", "网络名称 SSID", "MyNetwork"),
         ("password", "密码", "password"),
         ("security", "加密方式", "WPA"),
         ("hidden", "隐藏网络", "false")),
        "格式：WIFI:T:<WPA|WEP|nopass>;S:<SSID>;P:<密码>;H:<true|false>;;",
    ),
    PayloadType(
        "vcard", "vCard 名片",
        (("name", "姓名", "张三"),
         ("org", "组织", "示例公司"),
         ("title", "职位", "工程师"),
         ("phone", "电话", "+86 138 0000 0000"),
         ("email", "邮箱", "zhangsan@example.com"),
         ("url", "网址", "https://example.com"),
         ("address", "地址", "北京市朝阳区")),
        "生成 vCard 3.0 文本。",
    ),
    PayloadType(
        "email", "邮件",
        (("to", "收件人", "someone@example.com"),
         ("subject", "主题", "主题"),
         ("body", "正文", "正文内容")),
        "生成 mailto: 链接。",
    ),
    PayloadType(
        "sms", "短信",
        (("phone", "手机号", "+8613800000000"),
         ("body", "短信内容", "你好")),
        "生成 SMSTO: 链接。",
    ),
    PayloadType(
        "tel", "电话",
        (("phone", "电话号码", "+8613800000000"),),
        "生成 tel: 链接。",
    ),
    PayloadType(
        "geo", "地理坐标",
        (("lat", "纬度", "39.9042"),
         ("lon", "经度", "116.4074")),
        "生成 geo: 链接。",
    ),
)

_PAYLOAD_BY_KEY = {p.key: p for p in PAYLOAD_TYPES}


def payload_type(key: str) -> PayloadType:
    return _PAYLOAD_BY_KEY.get(key, PAYLOAD_TYPES[0])


# ---------------------------------------------------------------------------
# Escaping helpers
# ---------------------------------------------------------------------------

_WIFI_SPECIALS = set("\\;,:\"")


def _wifi_escape(value: str) -> str:
    out: List[str] = []
    for ch in value:
        if ch in _WIFI_SPECIALS:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _vcard_escape(value: str) -> str:
    return (value.replace("\\", "\\\\")
                 .replace(";", "\\;")
                 .replace(",", "\\,")
                 .replace("\n", "\\n"))


def build_payload(kind: str, values: dict) -> str:
    """Build the text to encode from structured field values."""
    kind = kind or "text"
    get = lambda name, default="": str(values.get(name, default) or "").strip()

    if kind == "text":
        text = values.get("text", "")
        if not str(text).strip():
            raise GenerationError("文本内容不能为空")
        return str(text)

    if kind == "url":
        url = get("url")
        if not url:
            raise GenerationError("网址不能为空")
        if "://" not in url and not url.lower().startswith(("mailto:", "tel:")):
            scheme = get("scheme", "https://") or "https://"
            if not scheme.endswith("//") and scheme.endswith(":"):
                scheme = scheme + "//"
            url = scheme + url
        return url

    if kind == "wifi":
        ssid = get("ssid")
        if not ssid:
            raise GenerationError("WiFi 网络名称（SSID）不能为空")
        security = get("security", "WPA").upper()
        if security not in ("WPA", "WEP", "NOPASS", "NONE"):
            security = "WPA"
        if security in ("NOPASS", "NONE"):
            security = "nopass"
            password = ""
        else:
            password = get("password")
            if not password:
                raise GenerationError("加密方式为 %s 时必须填写密码（无密码请选择 nopass）"
                                      % security)
        hidden = get("hidden", "false").lower() in ("1", "true", "yes", "是")
        parts = ["WIFI:", "T:%s;" % security, "S:%s;" % _wifi_escape(ssid)]
        if password:
            parts.append("P:%s;" % _wifi_escape(password))
        if hidden:
            parts.append("H:true;")
        parts.append(";")
        return "".join(parts)

    if kind == "vcard":
        name = get("name")
        if not name:
            raise GenerationError("姓名不能为空")
        lines = ["BEGIN:VCARD", "VERSION:3.0", "N:%s;;;;" % _vcard_escape(name),
                 "FN:%s" % _vcard_escape(name)]
        if get("org"):
            lines.append("ORG:%s" % _vcard_escape(get("org")))
        if get("title"):
            lines.append("TITLE:%s" % _vcard_escape(get("title")))
        if get("phone"):
            lines.append("TEL;TYPE=CELL:%s" % _vcard_escape(get("phone")))
        if get("email"):
            lines.append("EMAIL;TYPE=INTERNET:%s" % _vcard_escape(get("email")))
        if get("url"):
            lines.append("URL:%s" % _vcard_escape(get("url")))
        if get("address"):
            lines.append("ADR;TYPE=WORK:;;%s;;;;" % _vcard_escape(get("address")))
        lines.append("END:VCARD")
        return "\r\n".join(lines)

    if kind == "email":
        to = get("to")
        if not to:
            raise GenerationError("收件人不能为空")
        params = []
        if get("subject"):
            params.append("subject=%s" % _url_quote(get("subject")))
        if get("body"):
            params.append("body=%s" % _url_quote(get("body")))
        suffix = ("?" + "&".join(params)) if params else ""
        return "mailto:%s%s" % (to, suffix)

    if kind == "sms":
        phone = get("phone")
        if not phone:
            raise GenerationError("手机号不能为空")
        body = get("body")
        return "SMSTO:%s:%s" % (phone, body) if body else "SMSTO:%s" % phone

    if kind == "tel":
        phone = get("phone")
        if not phone:
            raise GenerationError("电话号码不能为空")
        return "tel:%s" % phone

    if kind == "geo":
        lat, lon = get("lat"), get("lon")
        if not lat or not lon:
            raise GenerationError("纬度和经度都不能为空")
        try:
            float(lat), float(lon)
        except ValueError:
            raise GenerationError("纬度和经度必须是数字")
        return "geo:%s,%s" % (lat, lon)

    raise GenerationError("未知的内容类型：%s" % kind)


def _url_quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text, safe="")


def parse_payload(text: str) -> Tuple[str, dict]:
    """Best-effort inverse of :func:`build_payload` for the "load into editor" flow."""
    from .decoder import classify_content

    kind = classify_content(text)
    values: dict = {}
    stripped = text.strip()
    try:
        if kind == "wifi":
            body = stripped[5:]
            if body.endswith(";;"):
                body = body[:-2]
            elif body.endswith(";"):
                body = body[:-1]
            current = ""
            pairs: List[Tuple[str, str]] = []
            i = 0
            while i < len(body):
                ch = body[i]
                if ch == "\\" and i + 1 < len(body):
                    current += body[i + 1]
                    i += 2
                    continue
                if ch == ":" and not current.startswith("_value"):
                    key = current.strip().upper()
                    current = "_value"
                    pairs.append((key, ""))
                    i += 1
                    continue
                if ch == ";":
                    if pairs:
                        key, _ = pairs[-1]
                        pairs[-1] = (key, current.replace("_value", ""))
                    current = ""
                    i += 1
                    continue
                current += ch
                i += 1
            if pairs:
                key, value = pairs[-1]
                pairs[-1] = (key, current.replace("_value", ""))
            mapping = {"S": "ssid", "P": "password", "T": "security", "H": "hidden"}
            for key, value in pairs:
                if key in mapping:
                    values[mapping[key]] = value
        elif kind == "url":
            values["url"] = stripped
        elif kind in ("tel",):
            values["phone"] = stripped.split(":", 1)[1]
        elif kind == "geo":
            parts = stripped[4:].split(",")
            if len(parts) >= 2:
                values["lat"], values["lon"] = parts[0], parts[1]
        elif kind == "vcard":
            for line in stripped.replace("\r\n", "\n").split("\n"):
                if line.startswith("FN:"):
                    values["name"] = line[3:]
                elif line.startswith("ORG:"):
                    values["org"] = line[4:]
                elif line.startswith("TITLE:"):
                    values["title"] = line[6:]
                elif line.startswith("TEL"):
                    values["phone"] = line.split(":", 1)[-1]
                elif line.startswith("EMAIL"):
                    values["email"] = line.split(":", 1)[-1]
        elif kind == "mailto":
            rest = stripped[7:]
            to, _, query = rest.partition("?")
            values["to"] = to
            for pair in query.split("&"):
                if "=" in pair:
                    key, _, value = pair.partition("=")
                    from urllib.parse import unquote

                    values[key] = unquote(value)
        elif kind == "sms":
            rest = stripped[6:] if stripped.lower().startswith("smsto:") else ""
            phone, _, body = rest.partition(":")
            values["phone"] = phone
            if body:
                values["body"] = body
    except Exception:
        pass
    # classify_content() speaks the decoder's vocabulary; map it to the
    # generator's payload keys.
    kind = {"mailto": "email", "mecard": "vcard"}.get(kind, kind)
    if kind not in _PAYLOAD_BY_KEY:
        kind = "text"
    if kind == "text":
        values["text"] = text
    return kind, values


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

LEVELS = ("L", "M", "Q", "H")


@dataclass
class GenerationRequest:
    content: str
    error: str = "M"
    version: Optional[int] = None          # None = smallest that fits
    mask: Optional[int] = None             # None = automatic (penalty rules)
    boost_error: bool = False
    micro: bool = False
    scale: int = 8
    border: int = 4
    dark: str = "#000000"
    light: str = "#ffffff"


@dataclass
class GenerationResult:
    request: GenerationRequest
    version: int
    size: int
    error: str
    mask: int
    mode: str
    char_count: int
    byte_count: int
    matrix: np.ndarray
    qr: object = None
    notes: List[str] = field(default_factory=list)

    def png_bytes(self, scale: Optional[int] = None,
                  border: Optional[int] = None) -> bytes:
        if self.qr is not None:
            buffer = io.BytesIO()
            self.qr.save(buffer, kind="png",
                         scale=scale or self.request.scale,
                         border=border if border is not None else self.request.border,
                         dark=self.request.dark, light=self.request.light)
            return buffer.getvalue()
        return _matrix_png(self.matrix, scale or self.request.scale,
                           border if border is not None else self.request.border,
                           self.request.dark, self.request.light)

    def svg_bytes(self, scale: Optional[int] = None,
                  border: Optional[int] = None) -> bytes:
        if self.qr is not None:
            buffer = io.BytesIO()
            self.qr.save(buffer, kind="svg",
                         scale=scale or self.request.scale,
                         border=border if border is not None else self.request.border,
                         dark=self.request.dark, light=self.request.light,
                         xmldecl=True)
            return buffer.getvalue()
        raise GenerationError("SVG 导出需要 segno 库")


def _matrix_png(matrix: np.ndarray, scale: int, border: int,
                dark: str, light: str) -> bytes:
    import cv2

    size = matrix.shape[0]
    img = np.where(matrix > 0, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (size * scale, size * scale),
                     interpolation=cv2.INTER_NEAREST)
    pad = border * scale
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                             cv2.BORDER_CONSTANT, value=255)
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise GenerationError("PNG 编码失败")
    return buffer.tobytes()


def _mode_name(mode: object) -> str:
    """Map a segno mode string to the label shown in the UI.

    segno reports ``'numeric'``, ``'alphanumeric'``, ``'byte'``, ``'kanji'`` or
    ``'hanzi'``; the UI shows the standard mode names.
    """
    name = str(mode).lower()
    # "alphanumeric" contains "numeric", so the longer name must be tested first.
    if "alphanumeric" in name or "alnum" in name:
        return "Alphanumeric"
    if "numeric" in name:
        return "Numeric"
    if "kanji" in name:
        return "Kanji"
    if "hanzi" in name:
        return "Hanzi"
    return "Byte"


def generate(request: GenerationRequest) -> GenerationResult:
    """Generate a QR symbol, raising :class:`GenerationError` with a clear reason."""
    content = request.content
    if not content:
        raise GenerationError("内容不能为空")

    error = (request.error or "M").upper()
    if error not in LEVELS:
        raise GenerationError("纠错等级必须是 L / M / Q / H 之一")

    version = request.version
    if version is not None and not (1 <= int(version) <= 40):
        raise GenerationError("版本必须在 1–40 之间")

    mask = request.mask
    if mask is not None and not (0 <= int(mask) <= 7):
        raise GenerationError("掩码必须在 0–7 之间")

    if not _SEGNO_OK:
        return _generate_fallback(request, error, version, mask)

    kwargs = dict(
        error=error,
        version=version,
        mask=mask,
        boost_error=request.boost_error,
        micro=request.micro,
    )
    try:
        qr = _segno.make(content, **kwargs)
    except Exception as exc:
        message = str(exc)
        suggestion = _suggest_version(content, error)
        if version is not None:
            raise GenerationError(
                "版本 %d 无法容纳该内容（纠错等级 %s）。%s原始错误：%s"
                % (version, error,
                   ("建议最小可行版本：%d。" % suggestion) if suggestion else "",
                   message))
        raise GenerationError("生成失败：%s" % message)

    matrix = np.array([[1 if cell else 0 for cell in row] for row in qr.matrix],
                      dtype=np.uint8)
    notes: List[str] = []
    if qr.version != (version or qr.version):
        notes.append("版本由 %s 调整为 %d" % (version, qr.version))
    if qr.error != error:
        notes.append("纠错等级由 %s 调整为 %s" % (error, qr.error))

    return GenerationResult(
        request=request,
        version=int(qr.version),
        size=int(qr.symbol_size(border=0)[0]),
        error=str(qr.error),
        mask=int(qr.mask),
        mode=_mode_name(qr.mode),
        char_count=len(content),
        byte_count=len(content.encode("utf-8")),
        matrix=matrix,
        qr=qr,
        notes=notes,
    )


def _suggest_version(content: str, error: str) -> Optional[int]:
    """Smallest version that fits *content* at *error*, or None."""
    if not _SEGNO_OK:
        return None
    for candidate in range(1, 41):
        try:
            _segno.make(content, error=error, version=candidate,
                        boost_error=False, micro=False)
        except Exception:
            continue
        return candidate
    return None


def _generate_fallback(request: GenerationRequest, error: str,
                       version: Optional[int], mask: Optional[int]) -> GenerationResult:
    """Minimal encoder used only when segno is unavailable."""
    try:
        import qrcode
    except Exception:
        raise GenerationError("缺少 segno 与 qrcode 库，无法生成二维码")

    qr = qrcode.QRCode(version=version, error_correction={
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }[error], box_size=1, border=0, mask_pattern=mask)
    qr.add_data(request.content)
    try:
        qr.make(fit=version is None)
    except Exception as exc:
        suggestion = _suggest_version(request.content, error)
        raise GenerationError("生成失败：%s%s" % (
            exc, ("（建议最小可行版本：%d）" % suggestion) if suggestion else ""))

    matrix = np.array(qr.get_matrix(), dtype=np.uint8)
    size = matrix.shape[0]
    return GenerationResult(
        request=request,
        version=int(qr.version),
        size=size,
        error=error,
        mask=int(getattr(qr, "mask_pattern", 0) or 0),
        mode="byte",
        char_count=len(request.content),
        byte_count=len(request.content.encode("utf-8")),
        matrix=matrix,
        qr=None,
        notes=["使用 qrcode 回退编码器（未安装 segno）"],
    )
