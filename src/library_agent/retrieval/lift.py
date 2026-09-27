"""The sentence in each hit that answers the query.

A passage found by meaning may not contain the query's words at all, so a highlighter
that only knows terms lights nothing on the hits that matter most. This embeds the
sentences of the hits (a few dozen short strings, one call) and picks the one nearest
the query, which is the "why did this come up" answer at a glance."""

from __future__ import annotations

import re

import numpy as np

from library_agent.llm.embed import embed_texts
from library_agent.llm.ollama import Ollama

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])|\n{2,}")
MAX_SENTENCES_PER_HIT = 12
MIN_CHARS = 25


def sentences(text: str) -> list[str]:
    out = []
    for s in _SPLIT.split(" ".join(text.split())):
        s = s.strip()
        if len(s) >= MIN_CHARS:
            out.append(s[:400])
    return out[:MAX_SENTENCES_PER_HIT]


async def lift(query: str, texts: list[str], client: Ollama | None = None) -> list[str | None]:
    """For each text, its sentence closest to the query (None when it has none)."""
    per_text = [sentences(t) for t in texts]
    flat = [s for ss in per_text for s in ss]
    if not flat:
        return [None] * len(texts)
    vecs = np.array(await embed_texts([query, *flat], client), dtype=np.float32)
    vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
    sims = vecs[1:] @ vecs[0]
    out: list[str | None] = []
    i = 0
    for ss in per_text:
        if not ss:
            out.append(None)
            continue
        block = sims[i : i + len(ss)]
        out.append(ss[int(np.argmax(block))])
        i += len(ss)
    return out


_STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "how",
        "what",
        "why",
        "when",
        "which",
        "who",
        "does",
        "do",
        "did",
        "can",
        "could",
        "should",
        "would",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "at",
        "about",
        "into",
        "than",
        "then",
        "there",
        "their",
        "them",
        "they",
        "not",
        "no",
    ]
)


def lift_kind(query: str, sentence: str | None) -> str | None:
    """ "words" when the lifted sentence shares a content word with the query; "meaning"
    when it was chosen by embedding alone and shares none."""
    if not sentence:
        return None
    words = {w for w in re.findall(r"[a-z0-9]{3,}", query.lower()) if w not in _STOP}
    low = sentence.lower()
    return "words" if any(w in low for w in words) else "meaning"
