"""Exact matches first, for queries that name something exactly.

A query like `O_DIRECT`, `ECONNRESET`, `--no-verify`, `RFC 7231` or `"write amplification"`
names a thing, not a topic. Dense retrieval treats it as a topic (a passage about disk
flags in general ranks beside the one that says O_DIRECT) and the lexical index splits it
at its punctuation. So Find looks for those terms literally, in the same scope, and puts
the passages that contain every one of them first; the ranked results follow.

Nothing here embeds or generates: it is a substring scan in Postgres."""

from __future__ import annotations

import re
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.classification import default_level as _default_level
from library_agent.retrieval.hybrid import SearchHit

_QUOTED = re.compile(r'"([^"]{3,80})"|“([^”]{3,80})”')
# A token that is an identifier, not a word: it carries a digit, an inner _ . / : or -,
# a leading --flag, or capitals after its first letter (camelCase, ALLCAPS of 3+).
_TOKEN = re.compile(r"[\w./:+#-]+")


def _is_identifier(tok: str, *, short_query: bool = False) -> bool:
    t = tok.strip(".,;:!?()[]{}'\"")
    if len(t) < 3:
        return False
    if t.startswith("--") or t.startswith("-") and len(t) > 2 and t[1].isalpha():
        return True
    if any(ch.isdigit() for ch in t) and any(ch.isalpha() for ch in t):
        return True
    if re.search(r"\w[_./:#]\w", t):
        return True
    if re.search(r"[a-z][A-Z]", t) or (
        any(c.islower() for c in t) and any(c.isupper() for c in t[1:])
    ):
        return True  # camelCase, SSTable
    # A bare capitalised word (HTTP, TLS) is a name only when the query is a lookup; in a
    # sentence it is just a word, and "http" is in half the shelf.
    return short_query and t.isupper() and t.isalpha()


def literal_terms(query: str) -> list[str]:
    """The quoted phrases and identifier-like tokens in a query, in order, deduplicated.
    Numbers standing next to a word are kept with it ("RFC 7231", "HTTP 429")."""
    out: list[str] = []
    rest = query
    for m in _QUOTED.finditer(query):
        out.append((m.group(1) or m.group(2)).strip())
        rest = rest.replace(m.group(0), " ")
    toks = _TOKEN.findall(rest)
    short = len(_TOKEN.findall(query)) <= 3
    i = 0
    while i < len(toks):
        tok = toks[i].strip(".,;:!?()[]{}'\"")
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if tok.isalpha() and tok.isupper() and len(tok) >= 2 and nxt.isdigit():
            out.append(f"{tok} {nxt}")  # "RFC 7231": the pair is the name
            i += 2
            continue
        if _is_identifier(tok, short_query=short):
            out.append(tok)
        i += 1
    words = [w.strip(".,;:!?()[]{}'\"") for w in toks]
    if not out and len(words) == 1 and len(words[0]) >= 3:
        out.append(words[0])  # a one-word query names the thing it wants
    return list(dict.fromkeys(t for t in out if t))


_SQL = """
select c.id, c.document_id, d.title, coalesce(s.path, '') as path, c.page_start, c.text,
       coalesce(c.context_prefix, '') as prefix
from chunk c
join document d on d.id = c.document_id
left join section s on s.id = c.section_id
where {conds}
  and (cast(:docs as uuid[]) is null or c.document_id = any(cast(:docs as uuid[])))
  and (cast(:cats as uuid[]) is null or exists (
      select 1 from document_category dc
      where dc.document_id = c.document_id and dc.category_id = any(cast(:cats as uuid[]))
  ) or exists (
      select 1 from chunk_category cc
      where cc.chunk_id = c.id and cc.category_id = any(cast(:cats as uuid[]))
  ))
  and (cast(:carts as uuid[]) is null or exists (
      select 1 from cartridge_document cd
      where cd.document_id = c.document_id and cd.cartridge_id = any(cast(:carts as uuid[]))
  ))
      and (cast(:levels as text[]) is null or coalesce(c.portion, (select d0.classification from document d0 where d0.id = c.document_id), cast(:deflt as text)) = any(cast(:levels as text[])))
order by length(c.text)
limit :lim
"""


def _like(term: str) -> str:
    return "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


async def literal_hits(
    db: AsyncSession,
    terms: list[str],
    *,
    limit: int = 10,
    category_ids: list[uuid.UUID] | None = None,
    cartridge_ids: list[uuid.UUID] | None = None,
    levels: list[str] | None = None,
    document_ids: list[uuid.UUID] | None = None,
) -> list[SearchHit]:
    """Passages containing every term, case-insensitively, shortest first (the term is a
    larger share of a short passage). Empty when there are no terms."""
    if not terms:
        return []
    params: dict = {
        "docs": [str(x) for x in document_ids] if document_ids else None,
        "cats": [str(x) for x in category_ids] if category_ids else None,
        "carts": [str(x) for x in cartridge_ids] if cartridge_ids else None,
        "levels": levels,
        "deflt": _default_level() if levels else None,
        "lim": limit,
    }
    conds = []
    for i, t in enumerate(terms[:6]):
        params[f"t{i}"] = _like(t)
        conds.append(f"c.text ilike :t{i} escape '\\'")
    rows = (await db.execute(text(_SQL.format(conds=" and ".join(conds))), params)).all()
    return [
        SearchHit(
            chunk_id=r.id,
            document_id=r.document_id,
            document_title=r.title,
            section_path=r.path,
            page=r.page_start,
            text=r.text,
            score=1.0,
            dense_rank=None,
            lexical_rank=None,
            context_prefix=r.prefix,
        )
        for r in rows
    ]


def contains_all(text_: str, terms: list[str]) -> bool:
    low = text_.lower()
    return all(t.lower() in low for t in terms)


def exact_first(
    ranked: list[SearchHit], literal: list[SearchHit], terms: list[str], limit: int
) -> tuple[list[SearchHit], set]:
    """Passages that contain every literal term lead, the ranked ones among them keeping
    their rank order and the literal-only ones after; then the rest of the ranking.
    Returns the list and the ids of the exact matches."""
    if not terms:
        return ranked[:limit], set()
    exact_ranked = [h for h in ranked if contains_all(h.text, terms)]
    seen = {h.chunk_id for h in exact_ranked}
    exact_extra = [h for h in literal if h.chunk_id not in seen]
    exact = exact_ranked + exact_extra
    ids = {h.chunk_id for h in exact}
    rest = [h for h in ranked if h.chunk_id not in ids]
    return (exact + rest)[:limit], ids


# --------------------------------------------------------------------------- rare words

# A word the question uses that only a handful of passages contain is almost always the
# thing being asked about: a person ("what has Hector said"), a project ("RaaSP"), a host.
# Meaning-based search can't place a rare name, and the lexical half drowns it in common
# words -- measured on Astra's judgements: six questions about people whose names appear in
# 9-14 passages each, and search surfaced none of those passages. So such a word is looked
# up exactly, the way an identifier is, and what holds it joins the candidates.
# Rare: held by at most 0.2% of passages (and never fewer than 20), so the bar grows with
# the library -- a project named in 33 of 23,673 passages is still the thing asked about.
RARE_SHARE = 0.002
RARE_FLOOR = 20
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{3,}")
_COMMON = {
    "what",
    "which",
    "where",
    "when",
    "with",
    "would",
    "could",
    "should",
    "about",
    "there",
    "their",
    "these",
    "those",
    "this",
    "that",
    "have",
    "does",
    "your",
    "from",
    "into",
    "they",
    "them",
    "been",
    "were",
    "will",
    "just",
    "like",
    "some",
    "tell",
    "give",
    "show",
    "list",
    "please",
    "library",
    "know",
    "anything",
    "information",
    "said",
    "written",
    "wrote",
}


async def rare_terms(db: AsyncSession, query: str, *, max_df: int | None = None) -> list[str]:
    """Words of the query found in few passages (see RARE_SHARE), or among the authors."""
    if max_df is None:
        total = (await db.execute(text("select count(*) from chunk"))).scalar_one()
        max_df = max(RARE_FLOOR, int(total * RARE_SHARE))
    words = []
    for w in _WORD.findall(query or ""):
        w = w.strip(".-_").lower()
        if len(w) >= 4 and w not in _COMMON and w not in words:
            words.append(w)
    out = []
    for w in words[:8]:
        n = (
            await db.execute(
                text("select count(*) from chunk where fts @@ plainto_tsquery('english', :w)"),
                {"w": w},
            )
        ).scalar_one()
        authored = (
            await db.execute(
                text(
                    "select count(*) from document d, jsonb_array_elements_text("
                    "coalesce(d.authors, '[]'::jsonb)) a where a ilike :p"
                ),
                {"p": f"%{w}%"},
            )
        ).scalar_one()
        if authored or (1 <= n <= max_df and await _written_as_a_name(db, w)):
            out.append(w)
    return out


async def _written_as_a_name(db: AsyncSession, w: str) -> bool:
    """Does the library write this word as a name -- capitalised (Rubeel, RaaSP, RCEs) in
    most places it appears? A rare *ordinary* word ("summarise", "calibration") is
    lowercase in running text, and looking it up exactly only crowds out better results:
    measured, it cost one of Astra's questions its first place."""
    rows = (
        await db.execute(
            text(
                "select text from chunk where fts @@ plainto_tsquery('english', :w) limit 30"
            ),
            {"w": w},
        )
    ).scalars().all()
    rx = re.compile(rf"\b({re.escape(w)})", re.IGNORECASE)
    seen = [m.group(1) for body in rows for m in rx.finditer(body)]
    if not seen:
        return False
    named = sum(1 for s in seen if s[0].isupper())
    return named / len(seen) >= 0.6


async def rare_hits(
    db: AsyncSession,
    terms: list[str],
    *,
    per_term: int = 4,
    category_ids: list[uuid.UUID] | None = None,
    cartridge_ids: list[uuid.UUID] | None = None,
    levels: list[str] | None = None,
    document_ids: list[uuid.UUID] | None = None,
) -> list[SearchHit]:
    """For each rare word: the passages that hold it (by the full-text index, so "hector"
    is not found inside "vector"), and the opening passages of volumes it authored."""
    base: dict = {
        "docs": [str(x) for x in document_ids] if document_ids else None,
        "cats": [str(x) for x in category_ids] if category_ids else None,
        "carts": [str(x) for x in cartridge_ids] if cartridge_ids else None,
        "levels": levels,
        "deflt": _default_level() if levels else None,
        "lim": per_term,
    }
    seen: set = set()
    out: list[SearchHit] = []
    for w in terms[:4]:
        for conds in (
            "c.fts @@ plainto_tsquery('english', :t0)",
            (
                "c.kind = 'text' and c.order_index < 2 and exists (select 1 from "
                "jsonb_array_elements_text(coalesce(d.authors, '[]'::jsonb)) a where a ilike :a0)"
            ),
        ):
            rows = (
                await db.execute(text(_SQL.format(conds=conds)), {**base, "t0": w, "a0": f"%{w}%"})
            ).all()
            for r in rows:
                if r.id in seen:
                    continue
                seen.add(r.id)
                out.append(
                    SearchHit(
                        chunk_id=r.id,
                        document_id=r.document_id,
                        document_title=r.title,
                        section_path=r.path,
                        page=r.page_start,
                        text=r.text,
                        score=1.0,
                        dense_rank=None,
                        lexical_rank=None,
                        context_prefix=r.prefix,
                    )
                )
    return out
