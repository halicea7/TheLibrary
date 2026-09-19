"""Effort: how much work the librarian does around the model.

The model itself has no effort dial worth the name on these builds (qwen3 thinks or it
does not, and off leaks the reasoning into the answer), so effort here is the work
around it -- and that is where answers actually change:

    quick    3 passages, no reranker, no rewrite of follow-ups, a small context.
             A lookup: what page says X.
    normal   what the desk does by default: 5 passages, reranked, follow-ups rewritten.
    deep     the question is first broken into two to four searches, each retrieved,
             the union reranked down to ten passages, a larger context. For comparative
             and multi-part questions, where one search pulls everything from one volume.

Each level is measurable in seconds, which is the point of a dial."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from library_agent.config import settings
from library_agent.llm.ollama import Ollama
from library_agent.reading.prompts import SYSTEM_LIBRARIAN
from library_agent.retrieval.pipeline import RetrievalConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Effort:
    name: str
    passages: int
    config: RetrievalConfig
    rewrite: bool
    multi_query: bool
    num_ctx: int
    passage_chars: int


EFFORTS: dict[str, Effort] = {
    "quick": Effort(
        "quick",
        passages=3,
        config=RetrievalConfig(name="quick", use_reranker=False),
        rewrite=False,
        multi_query=False,
        num_ctx=8192,
        passage_chars=900,
    ),
    "normal": Effort(
        "normal",
        passages=5,
        config=RetrievalConfig(name="chat", use_reranker=True, rerank_depth=20, per_document=2),
        rewrite=True,
        multi_query=False,
        num_ctx=16384,
        passage_chars=1200,
    ),
    "deep": Effort(
        "deep",
        passages=10,
        config=RetrievalConfig(name="deep", use_reranker=True, rerank_depth=40, per_document=3),
        rewrite=True,
        multi_query=True,
        num_ctx=32768,
        passage_chars=1400,
    ),
}
DEFAULT = "normal"


def get(name: str | None) -> Effort:
    return EFFORTS.get((name or DEFAULT).lower(), EFFORTS[DEFAULT])


SPLIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 120},
        }
    },
    "required": ["queries"],
}

SPLIT_PROMPT = """A reader asked the library:

{question}

Write the searches that together would find what this needs -- two to four short,
concrete queries (terms, not questions), each aimed at a different part of the
question. If it compares things, one query per thing. If it is a single simple lookup,
one query is enough.
"""


async def split_question(client: Ollama, model: str, question: str) -> list[str]:
    """Deep effort: the searches behind the question. Falls back to the question."""
    try:
        out = await client.structured(
            model,
            SPLIT_PROMPT.format(question=question),
            SPLIT_SCHEMA,
            system=SYSTEM_LIBRARIAN,
            instructions=SPLIT_PROMPT,
            temperature=0.2,
            num_predict=300,
        )
        qs = [str(q).strip() for q in out.get("queries") or [] if str(q).strip()]
        return list(dict.fromkeys(qs))[:4] or [question]
    except Exception:
        log.warning("question split failed; retrieving once", exc_info=True)
        return [question]


def passage_chars(effort: Effort) -> int:
    return min(effort.passage_chars, settings().chat_passage_chars * 2)
