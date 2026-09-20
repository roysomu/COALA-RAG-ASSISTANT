from dataclasses import dataclass
from hashlib import sha256


def content_hash(content):
    return sha256(content if isinstance(content, bytes) else content.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    page: int | None = None
    start: int = 0
    end: int = 0

    @property
    def content_hash(self):
        return content_hash(self.text)


def split_text(text, size=900, overlap=150, page=None, start_index=1):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError('Chunk size must be positive with 0 <= overlap < size.')
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind('\n\n', start + max(overlap + 1, size // 2), end)
            if boundary != -1:
                end = boundary + 2
        part = text[start:end]
        if part.strip():
            chunks.append(Chunk(part, start_index + len(chunks), page, start, end))
        if end == len(text):
            break
        start = end - overlap
    return chunks
