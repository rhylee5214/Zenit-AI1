"""Text cleaning and quality filters."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t  -​]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@dataclass
class CleanOptions:
    normalize_unicode: bool = True
    strip_control: bool = True
    collapse_whitespace: bool = True
    scrub_pii: bool = True
    min_chars: int = 20
    max_chars: int = 200_000
    min_alpha_ratio: float = 0.4       # fraction of letters among non-space chars
    max_repeated_line_ratio: float = 0.3
    keep_code: bool = True             # do not collapse indentation in code


def scrub_pii(text: str) -> str:
    text = _EMAIL.sub("<EMAIL>", text)
    text = _PHONE.sub("<PHONE>", text)
    text = _IPV4.sub("<IP>", text)
    return text


def clean_text(text: str, opt: CleanOptions | None = None) -> str:
    opt = opt or CleanOptions()
    if opt.normalize_unicode:
        text = unicodedata.normalize("NFC", text)
    if opt.strip_control:
        text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if opt.collapse_whitespace:
        lines = []
        for line in text.split("\n"):
            if opt.keep_code and line.startswith((" ", "\t")):
                lines.append(line.rstrip())
            else:
                lines.append(_SPACES.sub(" ", line).strip())
        text = _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()
    if opt.scrub_pii:
        text = scrub_pii(text)
    return text


def quality_reason(text: str, opt: CleanOptions | None = None) -> str | None:
    """Returns why a document fails the filters, or None if it passes."""
    opt = opt or CleanOptions()
    if len(text) < opt.min_chars:
        return "too_short"
    if len(text) > opt.max_chars:
        return "too_long"
    visible = [c for c in text if not c.isspace()]
    if visible:
        alpha = sum(c.isalpha() for c in visible) / len(visible)
        if alpha < opt.min_alpha_ratio and not _looks_like_code(text):
            return "low_alpha"
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) >= 5:
        repeated = 1 - len(set(lines)) / len(lines)
        if repeated > opt.max_repeated_line_ratio:
            return "repetitive"
    return None


def _looks_like_code(text: str) -> bool:
    markers = ("def ", "return", "{", "}", "=>", "function", "import ", ";")
    return sum(text.count(m) for m in markers) >= 3
