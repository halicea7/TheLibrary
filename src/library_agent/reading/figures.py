"""Figures read by the vision model into caption passages.

The reader shows a PDF's figures already (ingest/figures.py finds and renders them).
This pass lets the *model* see them too: each figure is sent to the vision model with
its caption, and the description comes back as a chunk of kind "figure" -- a passage
like any other, so Find lights it, Ask cites it by page, and a cartridge carries it.
The reader keeps figure chunks out of the running text and shows the description under
the figure instead.

One call per figure, ~5 s on the GH; a paper has one to six. Runs at the end of Tier 1
for PDFs when `vision_model` is set, and can be run alone for volumes read before it
existed."""

from __future__ import annotations

import base64
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import CHUNK_NAMESPACE, Chunk, Document, Embedding, OwnerKind, Section
from library_agent.ingest import figures as fig
from library_agent.llm.embed import embed_texts
from library_agent.llm.ollama import Ollama

log = logging.getLogger(__name__)

PROMPT = """You are describing a figure from the document "{title}" for a reader who cannot
see it, and for a search index. Caption from the page: {caption}

In one paragraph of at most 120 words, plain prose, no headings or bullet points: what
the figure shows, what its axes or parts are, the trend or structure, and the specific
numbers or labels that matter. Do not speculate beyond what is visible."""

FIGURE_ORDER_BASE = 1_000_000  # figure chunks sort after every text chunk


@dataclass
class FigureResult:
    figures: int = 0
    described: int = 0


def figure_chunk_id(content_hash: str, n: int) -> uuid.UUID:
    return uuid.uuid5(CHUNK_NAMESPACE, f"{content_hash}:figure:{n}")


async def describe_figures(
    db: AsyncSession, document_id: uuid.UUID, *, client: Ollama | None = None, gate=None
) -> FigureResult:
    cfg = settings()
    res = FigureResult()
    if not cfg.vision_model:
        return res
    doc = await db.get(Document, document_id)
    if not doc or doc.readings_only or not doc.source_path:
        return res
    path = Path(doc.source_path)
    if path.suffix.lower() != ".pdf" or not path.exists():
        return res
    figs = fig.index(doc.content_hash, path)
    res.figures = len(figs)
    if not figs:
        return res
    sections = list(
        (
            await db.execute(
                select(Section)
                .where(Section.document_id == document_id)
                .order_by(Section.order_index)
            )
        ).scalars()
    )

    def home(page: int) -> Section | None:
        # The last section that starts on or before the figure's page.
        best = None
        for s in sections:
            if s.page_start and s.page_start <= page:
                best = s
        return best or (sections[0] if sections else None)

    own = client is None
    c = client or Ollama()
    texts: list[tuple[fig.Figure, str]] = []
    try:
        for f in figs:
            if gate:
                await gate()
            png = fig.render(doc.content_hash, path, f.n)
            if not png:
                continue
            try:
                r = await c._client.post(
                    "/api/generate",
                    json={
                        "model": cfg.vision_model,
                        "prompt": PROMPT.format(title=doc.title, caption=f.caption or "(none)"),
                        "images": [base64.b64encode(png.read_bytes()).decode()],
                        "stream": False,
                        "keep_alive": cfg.keep_alive,
                        "options": {"temperature": 0.2, "num_predict": 260},
                    },
                    timeout=180,
                )
                r.raise_for_status()
                desc = " ".join((r.json().get("response") or "").split())
            except Exception:
                log.warning(
                    "figure %s of %s: vision call failed", f.n, doc.title[:40], exc_info=True
                )
                continue
            if len(desc) < 40:
                continue
            head = f"Figure {f.n}, p.{f.page}"
            if f.caption:
                # The caption usually starts with its own "Figure 3:"; say it once.
                cap = re.sub(r"^\s*(figure|fig\.?)\s*\d+\s*[:.\-–—]?\s*", "", f.caption, flags=re.IGNORECASE)
                head += f": {cap}" if cap else ""
            texts.append((f, f"{head}\n\n{desc}"))
    finally:
        if own:
            await c.aclose()
    # Replace whatever an earlier pass wrote, vectors included.
    old = list(
        (
            await db.execute(
                select(Chunk.id).where(Chunk.document_id == document_id, Chunk.kind == "figure")
            )
        ).scalars()
    )
    if old:
        await db.execute(
            delete(Embedding).where(
                Embedding.owner_kind == OwnerKind.CHUNK, Embedding.owner_id.in_(old)
            )
        )
        await db.execute(delete(Chunk).where(Chunk.id.in_(old)))
    if not texts:
        return res
    vecs = await embed_texts([t for _, t in texts], client if not own else None)
    for (f, text), vec in zip(texts, vecs, strict=True):
        sec = home(f.page)
        cid = figure_chunk_id(doc.content_hash, f.n)
        db.add(
            Chunk(
                id=cid,
                document_id=document_id,
                section_id=sec.id if sec else None,
                order_index=FIGURE_ORDER_BASE + f.n,
                text=text,
                kind="figure",
                context_prefix=doc.title,
                token_count=len(text) // 4,
                page_start=f.page,
                char_start=0,
                char_end=0,
            )
        )
        db.add(Embedding(owner_kind=OwnerKind.CHUNK, owner_id=cid, model=cfg.embed_model, vec=vec))
    await db.flush()
    res.described = len(texts)
    return res
