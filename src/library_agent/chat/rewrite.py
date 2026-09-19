"""Follow-up query rewriting.

"What about the second one?" embeds into noise. In conversational mode the question is
rewritten into a standalone query before retrieval.

This deliberately runs on the *conversation's own chat model* rather than a smaller one:
both candidate chat models are ~19GB and cannot be co-resident, so using a different model
here would trigger a full model swap on every single turn."""

from __future__ import annotations

import logging
import re

from library_agent.llm.ollama import Ollama

log = logging.getLogger(__name__)

# A rewrite call costs ~5s on a 19GB local model -- a third of the whole turn. Most
# follow-ups are already self-contained, so only pay for it when the message actually
# looks like it depends on what came before.
_DEPENDENT = re.compile(
    r"\b(it|its|that|this|these|those|they|them|their|the (first|second|third|last|former|latter)"
    r"|he|she|his|her|there|above|instead|also|too|as well|same|other|another)\b",
    re.IGNORECASE,
)


def looks_self_contained(message: str) -> bool:
    """Long, pronoun-free questions need no rewriting."""
    words = message.split()
    if len(words) < 4:
        return False  # "why?", "and BM25?" -- almost certainly a follow-up
    return not _DEPENDENT.search(message)


_SCHEMA = {
    "type": "object",
    "properties": {
        "standalone_query": {"type": "string"},
        "needs_retrieval": {"type": "boolean"},
    },
    "required": ["standalone_query", "needs_retrieval"],
}

_PROMPT = """Conversation so far:
{history}

Latest user message: {message}

Rewrite the latest message as a standalone search query that would work with no
conversation context — resolve pronouns and references ("it", "that one", "the second
approach") into explicit terms from the conversation.

Set needs_retrieval to false only if the message needs no library lookup at all (a
greeting, a thank-you, or a question purely about the conversation itself).

Keep the query concise and in the user's own terminology."""


async def rewrite_query(
    client: Ollama,
    model: str,
    message: str,
    history: list[tuple[str, str]],
) -> tuple[str, bool]:
    """Returns (query_to_embed, needs_retrieval). Falls back to the raw message."""
    if not history or looks_self_contained(message):
        return message, True
    transcript = "\n".join(f"{role}: {text[:400]}" for role, text in history[-6:])
    try:
        out = await client.structured(
            model,
            _PROMPT.format(history=transcript, message=message),
            _SCHEMA,
            temperature=0.1,
        )
    except Exception:
        log.warning("query rewrite failed; using raw message", exc_info=True)
        return message, True
    query = (out.get("standalone_query") or "").strip()
    return (query or message), bool(out.get("needs_retrieval", True))
