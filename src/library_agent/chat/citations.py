"""Citation numbering and validation.

The spec proposed letting the model self-tag which parts of an answer are grounded. Local
models are unreliable at that: they cite passages they did not use and omit citations for
ones they did. Here the retrieved set is numbered, the model emits [n], and a post-pass
resolves every marker against the real sources -- dropping any that do not resolve.

Same UX as the spec intended, but verifiable rather than self-reported."""

from __future__ import annotations

import re
from dataclasses import dataclass

from library_agent.retrieval.hybrid import SearchHit

_MARKER = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")


@dataclass
class Source:
    n: int
    chunk_id: str
    document_id: str
    document_title: str
    section_path: str
    page: int | None
    # Provenance: which cartridge the document arrived in (None = the local shelf), and
    # whether what we hold is the owner's reading rather than the text.
    cartridge: dict | None = None
    readings_only: bool = False
    # Chosen by the person (held from Find or Threads) rather than found by retrieval.
    held: bool = False
    # "reading": the library's summary of the section, not a passage of its text.
    kind: str = "passage"

    def label(self) -> str:
        loc = f", p.{self.page}" if self.page else ""
        if self.readings_only and self.cartridge:
            return f"{self.cartridge['name']}'s reading of {self.document_title}{loc}"
        return f"{self.document_title}{loc}"


def build_sources(hits: list[SearchHit], held: set | None = None) -> list[Source]:
    held = held or set()
    return [
        Source(
            n=i,
            chunk_id=str(h.chunk_id),
            document_id=str(h.document_id),
            document_title=h.document_title,
            section_path=h.section_path,
            page=h.page,
            held=h.chunk_id in held,
            kind=getattr(h, "kind", "passage"),
        )
        for i, h in enumerate(hits, start=1)
    ]


def render_context(
    hits: list[SearchHit],
    sources: list[Source],
    *,
    max_chars: int | None = None,
    reflections: dict[str, str] | None = None,
) -> str:
    """Numbered passages, in the order the model is told to cite them.

    Passages are truncated because every character here is prefill latency the user waits
    through before seeing a word. `reflections` (chunk id -> the library's Tier 2 note on
    that passage) are attached beneath their passage when given; whether that helps
    answers is what the A/B in scripts/reflections_ab.py measures."""
    blocks = []
    for src, hit in zip(sources, hits, strict=True):
        loc = f", p.{src.page}" if src.page else ""
        body = hit.text.strip()
        if max_chars and len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + " …"
        note = (reflections or {}).get(str(hit.chunk_id))
        what = "the library's reading of " if src.kind == "reading" else ""
        blocks.append(
            f"[{src.n}] {what}{src.document_title}{loc}"
            f"{f' — {src.section_path}' if src.section_path else ''}\n{body}"
            + (f"\n(the library's note on this passage: {note.strip()[:500]})" if note else "")
        )
    return "\n\n".join(blocks)


def validate(answer: str, sources: list[Source]) -> tuple[str, list[Source]]:
    """Strip markers that do not resolve, and report which sources were actually cited.

    A model that invents [7] when six passages were supplied would otherwise render a
    citation the user cannot check -- exactly the failure the numbering exists to prevent."""
    valid = {s.n for s in sources}
    used: set[int] = set()

    def replace(match: re.Match[str]) -> str:
        kept = []
        for part in match.group(1).split(","):
            try:
                n = int(part.strip())
            except ValueError:
                continue
            if n in valid:
                kept.append(n)
                used.add(n)
        return f"[{', '.join(str(n) for n in kept)}]" if kept else ""

    cleaned = _MARKER.sub(replace, answer)
    # Collapse whitespace left behind by a removed marker.
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), [s for s in sources if s.n in used]


def citation_validity(answer: str, sources: list[Source]) -> dict[str, int | float]:
    """Mechanical groundedness metric: what fraction of emitted markers resolve."""
    emitted = 0
    resolved = 0
    valid = {s.n for s in sources}
    for match in _MARKER.finditer(answer):
        for part in match.group(1).split(","):
            try:
                n = int(part.strip())
            except ValueError:
                continue
            emitted += 1
            if n in valid:
                resolved += 1
    return {
        "markers_emitted": emitted,
        "markers_resolved": resolved,
        "validity": round(resolved / emitted, 4) if emitted else 1.0,
    }
