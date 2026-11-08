"""Document text extraction and chunking."""

from __future__ import annotations

import io
import re
from html.parser import HTMLParser

TEXT_EXT = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".log", ".rst", ".c", ".cpp", ".h", ".cs", ".java",
    ".go", ".rs", ".rb", ".php", ".sh", ".sql", ".lua", ".xml", ".css",
}

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class UnsupportedFile(ValueError):
    pass


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("p", "br", "div", "li", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def extract_text(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    """Returns [(page_number_or_None, text)]."""

    if len(data) > MAX_UPLOAD_BYTES:
        raise UnsupportedFile("file larger than 25 MB")

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return [
            (i + 1, page.extract_text() or "")
            for i, page in enumerate(reader.pages)
        ]

    if ext == ".docx":
        import docx

        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]

        for table in d.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))

        return [(None, "\n".join(parts))]

    if ext in (".html", ".htm"):
        p = _HTMLText()
        p.feed(data.decode("utf-8", errors="replace"))
        return [(None, "".join(p.parts))]

    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
        return [(None, f"Image uploaded: {filename}")]

    if ext in TEXT_EXT or not ext:
        return [(None, data.decode("utf-8", errors="replace"))]

    raise UnsupportedFile(f"unsupported file type: {ext or 'unknown'}")


_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    pages: list[tuple[int | None, str]],
    size: int = 900,
    overlap: int = 150,
) -> list[dict]:
    chunks: list[dict] = []

    for page, text in pages:
        text = re.sub(r"[ \t]+", " ", text.replace("\r", "")).strip()

        if not text:
            continue

        chunks.append({
            "page": page,
            "idx": len(chunks),
            "text": text
        })

    return chunks