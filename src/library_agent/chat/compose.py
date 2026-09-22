"""Composing a document from the library.

A brief goes in; a document comes out, written the way the librarian answers: every claim
drawn from the shelf carries a `[n]` that resolves to a volume, page and section, and
anything unmarked is the librarian's own synthesis. The difference from a chat turn is
structure and scale. The librarian first plans an outline (thinking on), then writes
each section *retrieving for that section*, so a six-section document draws on six
retrievals rather than one. Citation numbers run across the whole document -- a passage
used in two sections keeps its number -- and each section is verified against the
passages it was actually given, so an invented marker is stripped before anyone sees it.
A references list closes the document.

The result is markdown. It can be saved, printed, or shelved back into the library as a
volume, at which point the collection can read what it wrote."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from library_agent.chat.citations import Source, citation_validity, render_context, validate
from library_agent.config import settings
from library_agent.db.models import Document
from library_agent.db.session import session_scope
from library_agent.library.cartridge import cartridge_provenance
from library_agent.llm import providers
from library_agent.llm.client import LLM
from library_agent.llm.lease import mark_chat_active, mark_chat_done, redis_client
from library_agent.llm.liveness import Busy, gate, liveness
from library_agent.reading.prompts import SYSTEM_LIBRARIAN
from library_agent.retrieval.hybrid import SearchHit
from library_agent.retrieval.pipeline import RetrievalConfig, retrieve
from library_agent.retrieval.readings import named_documents, retrieve_readings
from library_agent.retrieval.threads import relevant_threads, render_threads

log = logging.getLogger(__name__)

LENGTHS = {
    "short": ("3 to 4 sections", "150 to 250 words"),
    "medium": ("4 to 6 sections", "250 to 400 words"),
    "long": ("6 to 9 sections", "400 to 600 words"),
    "report": ("8 to 12 sections", "500 to 800 words"),
    "thesis": ("4 to 7 chapters, each of 2 to 4 sections", "500 to 800 words"),
}
# Lengths that plan in two levels: chapters, each with its own sections. The document
# then carries a running thesis per chapter as well as a per-section ledger, so it holds
# an argument over a scale that would otherwise drift.
NESTED = {"thesis"}

# Per section, compose retrieves harder than a chat turn: it is a background job, and the
# document is only as good as what each section is given. Passages carry the exact words;
# readings (the library's Tier 1 summary of a section) carry a whole work's argument, so a
# section that surveys a theme is not built from five stray paragraphs.
COMPOSE_PASSAGES = 10
COMPOSE_READINGS = 6
COMPOSE_CONFIG = RetrievalConfig(name="compose", use_reranker=True, rerank_depth=40, per_document=3)
# The thread block plus a section's passages and readings are a large prompt; the compose
# model is chosen to have the room. A section still writes into a modest reply.
COMPOSE_NUM_CTX = 65536
COMPOSE_THREADS = 12

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 120},
        "notes": {"type": "string", "maxLength": 800},
        "sections": {
            "type": "array",
            "minItems": 2,
            "maxItems": 32,
            "items": {
                "type": "object",
                "properties": {
                    "chapter": {"type": "string", "maxLength": 90},
                    "heading": {"type": "string", "maxLength": 90},
                    "covers": {"type": "string", "maxLength": 300},
                    "retrieve": {"type": "string", "maxLength": 200},
                },
                "required": ["heading", "covers", "retrieve"],
            },
        },
    },
    "required": ["title", "notes", "sections"],
}

PLAN_PROMPT = """You are planning a document to be written from a research library, for its owner.

Brief:
{brief}

{scope}
{threads}
Plan it: a title, and {n_sections}. For each section give the heading, what it should
cover (one or two sentences), and the search query that would find the right passages in
the library for it (concrete terms, not a question). Sections should not overlap; order
them so the document reads front to back.
{nesting}

The threads above are connections the library has already found across several volumes —
use them as the backbone of the plan, not an afterthought: a section should usually
develop a thread, drawing the works it spans together. Where the library marked a
disagreement (⚡), give it its due — a section, or a clearly argued paragraph within one —
rather than smoothing it over. In `notes`, say briefly what shape you chose and why.
Do not write the document.
"""

WRITE_SYSTEM = """You are the librarian of a personal research library, writing a document for its
owner from what the library holds.

You will be given numbered passages retrieved for THIS section. Use them as material:

- When a claim comes from a passage, cite it inline as [n] or [n, m], using the numbers
  exactly as given. Never invent a number; never cite a passage you did not use.
- You may synthesise across passages and add your own reasoning; leave that uncited so the
  reader can tell it apart from what the shelf says.
- If the passages do not cover something the section needs, say so in a sentence rather
  than inventing it.
- Some numbered items are marked "the library's reading of": its own summary of a section,
  not a quotation. Cite them like passages. Lean on them for a work's overall argument and
  on the passages for its exact words and figures.

This section is part of a longer document. You are told what earlier sections have already
established. Build on it: do not restate what is settled, and where you rely on an earlier
point, refer to it briefly ("as established earlier") rather than re-arguing it. Do not
contradict it without saying you are doing so and why.

Write only this section's body in markdown: no title, no heading (it is added for you),
no preamble, no summary of other sections. Use lists, code and tables where they belong.
Be concrete and specific."""

WRITE_PROMPT = """Document: {title}
Brief: {brief}

What the document has established so far:
{established}

Now write the section "{heading}", which should cover: {covers}
Aim for {length}.

Passages for this section:

{context}
"""

# One or two sentences distilling what a finished section actually established -- not its
# topic, its claim -- so a later section builds on the argument rather than the outline.
TAKEAWAY_PROMPT = """A section titled "{heading}" was just written for the document "{title}".

{body}

State, as `claim`, the specific claim, result or distinction this section establishes that
later sections should build on — one sentence, not what it is "about" but what it settles.
No preamble, no reference to "this section"."""

CHAPTER_PROMPT = """A chapter titled "{chapter}" of the document "{title}" is complete. Its
sections established, in order:
{takeaways}

State, as `claim`, what the chapter as a whole establishes in one sentence — the
through-line later chapters can rely on. No preamble."""


@dataclass
class Composition:
    title: str = ""
    brief: str = ""
    sections: list[dict[str, Any]] = field(default_factory=list)  # heading, covers, body
    sources: list[Source] = field(default_factory=list)  # global numbering
    hits_by_n: dict[int, SearchHit] = field(default_factory=dict)
    emitted: int = 0
    resolved: int = 0

    def markdown(self) -> str:
        out = [f"# {self.title}", ""]
        nested = any(s.get("chapter") for s in self.sections)
        last_ch = None
        for s in self.sections:
            ch = s.get("chapter")
            if nested and ch and ch != last_ch:
                out += [f"## {ch}", ""]
                last_ch = ch
            depth = "###" if nested else "##"
            out += [f"{depth} {s['heading']}", "", s.get("body", "").strip(), ""]
        used = sorted({n for s in self.sections for n in s.get("cited", [])})
        if used:
            out += ["## References", ""]
            for n in used:
                src = next((x for x in self.sources if x.n == n), None)
                if not src:
                    continue
                loc = f", p.{src.page}" if src.page else ""
                sec = f" — {src.section_path}" if src.section_path else ""
                who = f" ({src.cartridge['name']})" if src.cartridge else ""
                out.append(f"[{n}] {src.document_title}{loc}{sec}{who}")
        stamp = datetime.now(UTC).strftime("%Y-%m-%d")
        out += [
            "",
            f"*Composed by The Library on {stamp} from {len(used)} passages; unmarked claims are the librarian's own synthesis.*",
            "",
        ]
        return "\n".join(out)


LEDGER_CAP = 3500  # the established-so-far block, before a section's own material


_TAKEAWAY_SCHEMA = {
    "type": "object",
    "properties": {"claim": {"type": "string", "maxLength": 400}},
    "required": ["claim"],
}


async def _takeaway(client, model, title: str, heading: str, body: str) -> str:
    """One sentence on what a finished section established. A structured call, so a chatty
    model cannot leak its preamble into the running memory; the memory is only as good as
    this is faithful."""
    body = (body or "").strip()
    if len(body) < 80:
        return ""
    try:
        out = await client.structured(
            model,
            TAKEAWAY_PROMPT.format(title=title, heading=heading, body=body[:4000]),
            _TAKEAWAY_SCHEMA,
            temperature=0.2,
            think=False,
            num_predict=200,
        )
        return " ".join(str(out.get("claim") or "").split())[:400]
    except Exception:
        log.warning("takeaway failed for %r", heading, exc_info=True)
        return ""


async def _chapter_thesis(client, model, title: str, chapter: str, takeaways: list[str]) -> str:
    kept = [t for t in takeaways if t]
    if not kept:
        return ""
    try:
        out = await client.structured(
            model,
            CHAPTER_PROMPT.format(
                title=title,
                chapter=chapter,
                takeaways="\n".join(f"- {t}" for t in kept),
            ),
            _TAKEAWAY_SCHEMA,
            temperature=0.2,
            think=False,
            num_predict=200,
        )
        return " ".join(str(out.get("claim") or "").split())[:400] or kept[-1]
    except Exception:
        log.warning("chapter thesis failed for %r", chapter, exc_info=True)
        return kept[-1]


def _established(sections: list[dict], chapter_theses: dict[str, str], current_chapter) -> str:
    """The running argument, two levels: the through-line of every chapter already closed,
    then each finished section of the chapter in progress. Flat documents have one implicit
    chapter, so this is simply their section ledger."""
    parts: list[str] = []
    for ch, thesis in chapter_theses.items():
        if ch != current_chapter and thesis:
            parts.append(f"Chapter «{ch}» established: {thesis}")
    for s in sections:
        if s.get("chapter") == current_chapter and s.get("takeaway"):
            parts.append(f"§ {s['heading']}: {s['takeaway']}")
    if not parts:
        return "Nothing yet — this is the opening section."
    block = "\n".join(parts)
    if len(block) > LEDGER_CAP:  # keep the newest, drop the oldest section lines
        block = "…\n" + block[-LEDGER_CAP:]
    return block


def slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return (s or "document")[:60]


def compositions_dir() -> Path:
    d = settings().storage_dir.parent / "compositions"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def compose(
    brief: str,
    *,
    model: str | None = None,
    length: str = "medium",
    category_ids: list[uuid.UUID] | None = None,
    cartridge_ids: list[uuid.UUID] | None = None,
    scope_label: str = "",
    caller: str = "ui",
) -> AsyncIterator[dict]:
    """Yields SSE-shaped events: plan → for each section (section, sources, token*,
    section_done) → done. The lease is held throughout: this is one long piece of work
    and background reading yields to it like it would to a chat."""
    cfg = settings()
    model = model or providers.model_for("compose")
    n_sections, words = LENGTHS.get(length, LENGTHS["medium"])
    redis = redis_client()
    client = LLM()
    comp = Composition(brief=brief)
    held = False
    try:
        if not liveness.alive:
            yield {
                "event": "error",
                "data": f"the model is not answering: {liveness.detail}",
                "code": 503,
            }
            return
        try:
            await gate.acquire(caller)
            held = True
        except Busy as b:
            yield {"event": "error", "data": f"{b}; try again in {b.retry_after}s", "code": 429}
            return
        await mark_chat_active(redis)
        scope = f"Scope: only {scope_label}." if scope_label else "Scope: the whole library."
        # What the library already connects across volumes, nearest this brief. The plan
        # is built on these; a volume the brief names is favoured throughout.
        named = []
        async with session_scope() as db:
            named = await named_documents(db, brief)
            threads = await relevant_threads(
                db,
                brief,
                client=client,
                limit=COMPOSE_THREADS,
                category_ids=category_ids,
                cartridge_ids=cartridge_ids,
            )
        thread_block = render_threads(threads)
        thread_text = (
            f"\nThe library has already connected these across several volumes:\n{thread_block}\n"
            if thread_block
            else ""
        )
        yield {
            "event": "threads",
            "data": {
                "count": len(threads),
                "disputed": sum(1 for t in threads if t.get("contradiction")),
                "labels": [t["label"] for t in threads],
            },
        }
        nested = length in NESTED
        nesting = (
            "Group the sections into chapters: give every section a `chapter` (the title of "
            "the part it belongs to), with the sections of one chapter consecutive and in "
            "reading order. A chapter is a stage of the argument, not a bin."
            if nested
            else ""
        )
        plan = await client.structured(
            model,
            PLAN_PROMPT.format(
                brief=brief,
                scope=scope,
                threads=thread_text,
                n_sections=n_sections,
                nesting=nesting,
            ),
            PLAN_SCHEMA,
            system=SYSTEM_LIBRARIAN,
            instructions=PLAN_PROMPT,
            temperature=0.3,
            think=True,
            num_ctx=COMPOSE_NUM_CTX,
            num_predict=2500,
        )
        comp.title = (plan.get("title") or brief[:80]).strip()
        outline = [
            {
                "heading": str(s.get("heading") or "").strip(),
                "covers": str(s.get("covers") or "").strip(),
                "retrieve": str(s.get("retrieve") or s.get("heading") or "").strip(),
                "chapter": (str(s.get("chapter") or "").strip() or None) if nested else None,
            }
            for s in plan.get("sections") or []
            if s.get("heading")
        ]
        yield {
            "event": "plan",
            "data": {
                "title": comp.title,
                "notes": plan.get("notes", ""),
                "sections": outline,
                "model": model,
            },
        }
        chapter_theses: dict[str, str] = {}

        by_chunk: dict[str, Source] = {}
        for i, sec in enumerate(outline):
            await mark_chat_active(redis)
            async with session_scope() as db:
                q = sec["retrieve"] or sec["heading"]
                passages = await retrieve(
                    db,
                    q,
                    config=COMPOSE_CONFIG,
                    client=client,
                    limit=COMPOSE_PASSAGES,
                    category_ids=category_ids,
                    cartridge_ids=cartridge_ids,
                    favour=named,
                )
                readings = await retrieve_readings(
                    db,
                    q,
                    client=client,
                    limit=COMPOSE_READINGS,
                    category_ids=category_ids,
                    cartridge_ids=cartridge_ids,
                    favour=named,
                )
                # Passages first (exact words), then readings (a work's argument), each
                # section a page of the shelf and a page of what the library made of it.
                seen_r = {(h.document_id, h.section_path) for h in passages}
                hits = passages + [
                    h for h in readings if (h.document_id, h.section_path) not in seen_r
                ]
                docs = {
                    d.id: d
                    for d in (
                        await db.execute(
                            select(Document).where(
                                Document.id.in_([h.document_id for h in hits] or [uuid.uuid4()])
                            )
                        )
                    ).scalars()
                }
                provenance = await cartridge_provenance(db, list(docs))
            # Global numbering: a passage seen before keeps its number.
            section_sources: list[Source] = []
            section_hits: list[SearchHit] = []
            for h in hits:
                key = str(h.chunk_id)
                src = by_chunk.get(key)
                if not src:
                    d = docs.get(h.document_id)
                    src = Source(
                        n=len(comp.sources) + 1,
                        chunk_id=key,
                        document_id=str(h.document_id),
                        document_title=d.title if d else h.document_title,
                        section_path=h.section_path,
                        page=h.page,
                        cartridge=provenance.get(h.document_id),
                        readings_only=bool(d and d.readings_only),
                        kind=getattr(h, "kind", "passage"),
                    )
                    by_chunk[key] = src
                    comp.sources.append(src)
                    comp.hits_by_n[src.n] = h
                section_sources.append(src)
                section_hits.append(h)
            yield {
                "event": "sources",
                "data": {
                    "index": i,
                    "sources": [
                        {
                            "n": s.n,
                            "title": s.document_title,
                            "page": s.page,
                            "section": s.section_path,
                            "document_id": s.document_id,
                            "cartridge": s.cartridge,
                            "readings_only": s.readings_only,
                            "kind": s.kind,
                        }
                        for s in section_sources
                    ],
                },
            }
            # A chapter that just ended is compressed to a through-line before the next one.
            if nested and i > 0 and sec["chapter"] != outline[i - 1]["chapter"]:
                prev = outline[i - 1]["chapter"]
                if prev and prev not in chapter_theses:
                    chapter_theses[prev] = await _chapter_thesis(
                        client,
                        model,
                        comp.title,
                        prev,
                        [s["takeaway"] for s in comp.sections if s.get("chapter") == prev],
                    )
            established = _established(comp.sections, chapter_theses, sec["chapter"])
            context = (
                render_context(section_hits, section_sources, max_chars=cfg.chat_passage_chars)
                if section_hits
                else "(nothing relevant was found in the library for this section)"
            )
            messages = [
                {"role": "system", "content": WRITE_SYSTEM},
                {
                    "role": "user",
                    "content": WRITE_PROMPT.format(
                        title=comp.title,
                        brief=brief,
                        established=established,
                        heading=sec["heading"],
                        covers=sec["covers"],
                        length=words,
                        context=context,
                    ),
                },
            ]
            yield {
                "event": "section",
                "data": {
                    "index": i,
                    "heading": sec["heading"],
                    "covers": sec["covers"],
                    "chapter": sec["chapter"],
                },
            }
            buf: list[str] = []
            async for kind, piece in client.chat_stream(
                model, messages, temperature=0.5, num_ctx=COMPOSE_NUM_CTX
            ):
                if kind == "thinking":
                    yield {"event": "thinking", "data": piece}
                    continue
                buf.append(piece)
                yield {"event": "token", "data": piece}
            raw = "".join(buf)
            # Told not to, the model still often opens with the heading. Drop it.
            first, _, rest = raw.lstrip().partition("\n")
            if first.strip("# *").strip().lower() == sec["heading"].strip().lower():
                raw = rest
            cleaned, used = validate(raw, section_sources)
            m = citation_validity(raw, section_sources)
            comp.emitted += m["markers_emitted"]
            comp.resolved += m["markers_resolved"]
            # Distil what this section settled, for the sections that follow.
            takeaway = await _takeaway(client, model, comp.title, sec["heading"], cleaned)
            comp.sections.append(
                {
                    "heading": sec["heading"],
                    "covers": sec["covers"],
                    "chapter": sec["chapter"],
                    "body": cleaned,
                    "cited": [s.n for s in used],
                    "takeaway": takeaway,
                }
            )
            yield {
                "event": "section_done",
                "data": {
                    "index": i,
                    "body": cleaned,
                    "cited": [s.n for s in used],
                    "chapter": sec["chapter"],
                    "takeaway": takeaway,
                    **m,
                },
            }

        md = comp.markdown()
        yield {
            "event": "done",
            "data": {
                "title": comp.title,
                "markdown": md,
                "sections": len(comp.sections),
                "sources": len(comp.sources),
                "markers_emitted": comp.emitted,
                "markers_resolved": comp.resolved,
                "references": [
                    {
                        "n": s.n,
                        "title": s.document_title,
                        "page": s.page,
                        "section": s.section_path,
                        "document_id": s.document_id,
                        "cartridge": s.cartridge,
                    }
                    for s in comp.sources
                    if any(s.n in sec["cited"] for sec in comp.sections)
                ],
            },
        }
    except Exception as exc:
        log.exception("composition failed")
        from library_agent.ops.incidents import record_exception

        await record_exception(exc, source="chat", context={"compose": brief[:200]})
        yield {"event": "error", "data": str(exc)[:500]}
    finally:
        if held:
            gate.release(caller)
        await client.aclose()
        await mark_chat_done(redis)
        await redis.aclose()


def save_markdown(title: str, markdown: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    path = compositions_dir() / f"{slugify(title)}-{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def list_saved() -> list[dict]:
    out = []
    for p in sorted(compositions_dir().glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)[
        :100
    ]:
        text = p.read_text(encoding="utf-8", errors="replace")
        title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), p.stem)
        out.append(
            {
                "file": p.name,
                "title": title,
                "bytes": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime, UTC).isoformat(),
            }
        )
    return out
