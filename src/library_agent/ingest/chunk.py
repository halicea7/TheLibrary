"""Structure-aware chunking.

Chunks never cross a section boundary, so a chunk always has one true section path for
its context prefix. Within a section, split on paragraph boundaries and only fall back to
sentence or hard splits when a paragraph is itself oversized."""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from library_agent.config import settings
from library_agent.ingest.extract import Extracted
from library_agent.ingest.structure import Section


@lru_cache(maxsize=1)
def _encoder():
    # A proxy for bge-m3's tokenizer. Chunk sizing only needs to be approximately right.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


@dataclass
class Chunk:
    order_index: int
    text: str
    context_prefix: str
    token_count: int
    char_start: int
    char_end: int
    page_start: int | None
    section_index: int | None
    # True when the span was found exactly: text[char_start:char_end] holds this chunk's
    # characters, differing at most in whitespace. False means the location is approximate.
    span_exact: bool = True


_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _segments(text: str, max_tokens: int) -> list[str]:
    """Paragraphs, subdivided by sentence then hard-split only where necessary."""
    out: list[str] = []
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if count_tokens(para) <= max_tokens:
            out.append(para)
            continue
        buf = ""
        for sent in _SENT_SPLIT.split(para):
            candidate = f"{buf} {sent}".strip()
            if count_tokens(candidate) > max_tokens and buf:
                out.append(buf)
                buf = sent
            else:
                buf = candidate
        if buf:
            # A single sentence longer than the window (tables, math dumps): hard split.
            if count_tokens(buf) > max_tokens * 1.5:
                enc = _encoder()
                ids = enc.encode(buf, disallowed_special=())
                for i in range(0, len(ids), max_tokens):
                    out.append(enc.decode(ids[i : i + max_tokens]))
            else:
                out.append(buf)
    return out


def build_context_prefix(doc_title: str, section: Section | None, orientation: str = "") -> str:
    """Deterministic contextual prefix -- no LLM call.

    The corpus is structured (papers, docs, markdown), so title plus section breadcrumb
    recovers most of what Anthropic's generated contextual prefix provides, at zero cost.
    `orientation` is filled in at Tier 1 once the document has been read."""
    parts = [doc_title.strip()]
    if section and section.path:
        parts.append(section.path)
    if orientation:
        parts.append(orientation.strip())
    return " › ".join(p for p in parts if p)


class _Aligner:
    """Finds a chunk in the extracted text, ignoring whitespace.

    Packing only ever changes whitespace -- paragraphs are stripped, sentences rejoined
    with one space, an overlap tail re-spaced -- so the chunk's non-whitespace characters
    appear in the text exactly, in order. Matching on those gives the true span; matching
    on the chunk's opening 120 characters, as before, missed whenever the spacing differed
    and fell back to a cursor, which put one chunk in six at the wrong place or length."""

    def __init__(self, text: str):
        self.pos = [i for i, ch in enumerate(text) if not ch.isspace()]
        self.dense = "".join(text[i] for i in self.pos)

    def _dense_index(self, char_offset: int) -> int:
        return bisect.bisect_left(self.pos, char_offset)

    def find(self, piece: str, start: int, end: int) -> tuple[int, int] | None:
        key = "".join(piece.split())
        if not key:
            return None
        lo, hi = self._dense_index(start), self._dense_index(end)
        j = self.dense.find(key, lo, hi)
        if j == -1:
            return None
        return self.pos[j], self.pos[j + len(key) - 1] + 1


def chunk_document(
    ex: Extracted, sections: list[Section], doc_title: str, orientation: str = ""
) -> list[Chunk]:
    cfg = settings()
    target = cfg.chunk_target_tokens
    overlap_tokens = int(target * cfg.chunk_overlap_ratio)
    chunks: list[Chunk] = []
    aligner = _Aligner(ex.text)

    for section in sections:
        body = ex.text[section.char_start : section.char_end]
        if not body.strip():
            continue
        prefix = build_context_prefix(doc_title, section, orientation)
        segments = _segments(body, target)

        # Pack segments up to the target, carrying a small overlap for continuity.
        packed: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for seg in segments:
            seg_tokens = count_tokens(seg)
            if buf and buf_tokens + seg_tokens > target:
                packed.append("\n\n".join(buf))
                if overlap_tokens > 0:
                    tail = buf[-1]
                    while count_tokens(tail) > overlap_tokens and "\n\n" not in tail:
                        tail = " ".join(tail.split()[-overlap_tokens:])
                        break
                    buf, buf_tokens = [tail], count_tokens(tail)
                else:
                    buf, buf_tokens = [], 0
            buf.append(seg)
            buf_tokens += seg_tokens
        if buf:
            packed.append("\n\n".join(buf))

        # Merge a trailing runt rather than emitting a near-empty chunk.
        if len(packed) > 1 and count_tokens(packed[-1]) < cfg.chunk_min_tokens:
            runt = packed.pop()
            packed[-1] = packed[-1] + "\n\n" + runt

        cursor = section.char_start
        for piece in packed:
            span = aligner.find(piece, cursor, section.char_end)
            exact = span is not None
            if span is None:  # not expected; keep the old estimate, marked as such
                found = ex.text.find(piece[:120], cursor, section.char_end + 200)
                start = found if found != -1 else cursor
                span = (start, min(start + len(piece), len(ex.text)))
            start, end = span
            chunks.append(
                Chunk(
                    order_index=len(chunks),
                    text=piece,
                    context_prefix=prefix,
                    token_count=count_tokens(piece),
                    char_start=start,
                    char_end=end,
                    page_start=ex.page_for_offset(start),
                    section_index=section.order_index,
                    span_exact=exact,
                )
            )
            # The next chunk starts at or after this one's start (overlap allows before its end).
            cursor = max(cursor, start + 1)

    return chunks
