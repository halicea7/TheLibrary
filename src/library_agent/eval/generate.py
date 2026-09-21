"""Automatic generation of retrieval eval questions.

For a sampled chunk, ask the model for a question that chunk answers. The gold label is
then the chunk id itself -- no manual labelling, which is what makes recall@k cheap enough
to run after every retrieval change.

The prompt deliberately pushes for natural phrasing rather than quoting the passage. A
question that copies rare terms verbatim is trivially solved by lexical search, which
would bias the eval toward BM25 and make the hybrid weighting look better than it is."""

from __future__ import annotations

import logging
import random
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings
from library_agent.db.models import Chunk, Document, EvalQuestion
from library_agent.llm.client import LLM
from library_agent.llm.ollama import Ollama

log = logging.getLogger(__name__)

SUITE_RETRIEVAL = "retrieval"
SUITE_KEYWORD = "keyword"

_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answerable": {"type": "boolean"},
    },
    "required": ["question", "answerable"],
}

_SYSTEM = (
    "You write evaluation questions for a document retrieval system. "
    "You are precise and you never invent facts that are not in the passage."
)

_PROMPT = """Below is a passage from "{title}" (section: {section}).

---
{text}
---

Write ONE natural question that this passage answers -- the kind of question someone
would actually type when searching a research library.

Rules:
- Ask about the substance, not about "the passage" or "the document".
- Paraphrase. Do NOT copy distinctive phrases verbatim from the passage; a good question
  tests understanding, not string matching.
- It must be specific enough that this passage is clearly the right answer, not a generic
  question that a hundred other papers could answer.
- If the passage is boilerplate, references, a table of numbers, or otherwise has no real
  content to ask about, set answerable to false.

Return JSON with "question" and "answerable"."""

# The natural-language prompt above deliberately forbids verbatim phrases, which also
# strips out the exact-term signal that lexical search exists to capture. Evaluating the
# lexical half on that suite alone understates it badly, so this second suite models the
# other real query style: someone who already knows the term they want.
_KEYWORD_PROMPT = """Below is a passage from "{title}" (section: {section}).

---
{text}
---

Write ONE short keyword-style search query -- 2 to 6 words -- of the kind someone types
when they already know roughly what they are looking for.

Rules:
- Use the distinctive technical terms, method names, acronyms, or proper nouns that appear
  in this passage. These SHOULD be verbatim; that is the point of this query style.
- No question words, no punctuation, no full sentences. Just the terms.
- If the passage has no distinctive terminology worth searching for, set answerable false.

Return JSON with "question" and "answerable"."""


async def generate_questions(
    db: AsyncSession,
    *,
    n: int = 60,
    suite: str = SUITE_RETRIEVAL,
    min_tokens: int = 120,
    seed: int = 7,
    replace: bool = True,
    style: str = "natural",
    client: Ollama | None = None,
) -> list[EvalQuestion]:
    """Sample chunks across documents and derive one query from each.

    style="natural" produces paraphrased questions; style="keyword" produces term-style
    queries. Both are real usage patterns and they stress different halves of retrieval."""
    cfg = settings()
    rows = list(
        (
            await db.execute(
                select(
                    Chunk.id, Chunk.document_id, Chunk.text, Chunk.context_prefix, Document.title
                )
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.token_count >= min_tokens)
            )
        ).all()
    )
    if not rows:
        return []

    # Stratify by document so one long book cannot dominate the question set.
    by_doc: dict[uuid.UUID, list] = {}
    for r in rows:
        by_doc.setdefault(r.document_id, []).append(r)
    rng = random.Random(seed)
    for v in by_doc.values():
        rng.shuffle(v)

    picked: list = []
    while len(picked) < n and any(by_doc.values()):
        for docs in by_doc.values():
            if docs and len(picked) < n:
                picked.append(docs.pop())

    if replace:
        await db.execute(delete(EvalQuestion).where(EvalQuestion.suite == suite))

    own = client is None
    c = client or LLM()
    created: list[EvalQuestion] = []
    try:
        for r in picked:
            section = (r.context_prefix or "").split(" › ", 1)[-1] or "unknown"
            try:
                template = _KEYWORD_PROMPT if style == "keyword" else _PROMPT
                out = await c.structured(
                    cfg.reader_model,
                    template.format(title=r.title, section=section, text=r.text[:3000]),
                    _SCHEMA,
                    system=_SYSTEM,
                    temperature=0.4,
                )
            except Exception:  # one bad generation must not kill the run
                log.warning("question generation failed for chunk %s", r.id, exc_info=True)
                continue
            q = (out.get("question") or "").strip()
            min_len = 4 if style == "keyword" else 15
            if not out.get("answerable") or len(q) < min_len:
                continue
            row = EvalQuestion(
                suite=suite, question=q, gold_chunk_id=r.id, gold_document_id=r.document_id
            )
            db.add(row)
            created.append(row)
    finally:
        if own:
            await c.aclose()

    await db.flush()
    return created
