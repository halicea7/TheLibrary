"""Chat orchestration: rewrite → retrieve → generate → validate → persist."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.chat import answer as answer_mod
from library_agent.chat import effort as effort_mod
from library_agent.chat.citations import (
    Source,
    build_sources,
    citation_validity,
    plain_label,
    validate,
)
from library_agent.chat.rewrite import rewrite_query
from library_agent.config import settings
from library_agent.db.models import Conversation, Document, Message
from library_agent.db.session import session_scope
from library_agent.library.cartridge import cartridge_provenance
from library_agent.llm import providers
from library_agent.llm.client import LLM
from library_agent.llm.lease import mark_chat_active, mark_chat_done, redis_client
from library_agent.llm.liveness import Busy, gate, liveness
from library_agent.ops.incidents import record_exception
from library_agent.retrieval.hybrid import SearchHit
from library_agent.retrieval.pipeline import hits_for_chunks, retrieve
from library_agent.retrieval.readings import named_documents, retrieve_readings

log = logging.getLogger(__name__)

_MODULE_NS = uuid.uuid5(uuid.NAMESPACE_URL, "library-agent:module")


def _live_hit(lr) -> SearchHit:
    """A module's live result as a passage, so it numbers, cites, holds and verifies like
    any other -- but marked live, in the module's own colour."""
    from datetime import UTC, datetime

    when_label = datetime.fromtimestamp(lr.when, UTC).strftime("%H:%M")
    arg = str(next(iter(lr.args.values()), "")) if lr.args else ""
    if lr.error:
        body = (
            f"{lr.module.name} could not be reached just now ({lr.error}). Its live answer is "
            "unavailable; say the source was unreachable rather than guessing."
        )
    else:
        body = f"{lr.module.name} · {lr.op.summary}\n{lr.text}"
    return SearchHit(
        chunk_id=lr.chunk_id,
        document_id=uuid.uuid5(_MODULE_NS, lr.module.id),
        document_title=lr.module.name,
        section_path=lr.op.id,
        page=None,
        text=body,
        score=1.0,
        dense_rank=None,
        lexical_rank=None,
        kind="live",
        live={
            "module": lr.module.name,
            "colour": lr.module.colour,
            "op": lr.op.id,
            "when": lr.when,
            "when_label": when_label,
            "arg": arg,
            "error": lr.error,
            "count": lr.count,
        },
    )


@dataclass
class TurnState:
    question: str
    rewritten: str | None = None
    hits: list[SearchHit] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    answer: str = ""


async def _history(db: AsyncSession, conversation_id: uuid.UUID) -> list[tuple[str, str]]:
    rows = (
        await db.execute(
            select(Message.role, Message.content)
            .where(Message.conversation_id == conversation_id)
            .order_by(*Message.in_order())
        )
    ).all()
    return [(r.role, r.content) for r in rows]


async def get_or_create_conversation(
    db: AsyncSession,
    conversation_id: uuid.UUID | None,
    *,
    conversational: bool = True,
    model: str | None = None,
) -> Conversation:
    if conversation_id:
        conv = (
            await db.execute(select(Conversation).where(Conversation.id == conversation_id))
        ).scalar_one_or_none()
        if conv:
            return conv
    conv = Conversation(conversational=conversational, model=model)
    db.add(conv)
    await db.flush()
    return conv


async def run_turn(
    conversation_id: uuid.UUID,
    question: str,
    *,
    document_ids: list[uuid.UUID] | None = None,
    stance: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    caller: str = "ui",
    effort: str | None = None,
    pinned_chunk_ids: list[uuid.UUID] | None = None,
) -> AsyncIterator[dict]:
    """Yields SSE-shaped events: meta → sources → thinking* → token* → done.

    `caller` identifies who is asking for the gate (loopback UI, or a token); `temperature`
    and `seed` let the JSON API ask for cooler or reproducible answers."""
    cfg = settings()
    redis = redis_client()
    client = LLM()
    state = TurnState(question=question)
    held = False
    lvl = effort_mod.get(effort)

    try:
        if not liveness.alive:
            yield {
                "event": "error",
                "data": f"the model is not answering: {liveness.detail}. See Settings › services.",
                "code": 503,
            }
            return
        try:
            await gate.acquire(caller)
            held = True
        except Busy as b:
            yield {
                "event": "error",
                "data": f"{b}; try again in {b.retry_after}s",
                "code": 429,
                "retry_after": b.retry_after,
            }
            return
        # The lease is taken for the whole turn -- retrieval and generation both compete
        # with background reading for the same models.
        await mark_chat_active(redis)

        async with session_scope() as db:
            conv = (
                await db.execute(select(Conversation).where(Conversation.id == conversation_id))
            ).scalar_one()
            model = conv.model or providers.model_for("chat_general")
            conversational = conv.conversational
            category_ids = [uuid.UUID(x) for x in conv.category_ids] if conv.category_ids else None
            cartridge_ids = (
                [uuid.UUID(x) for x in conv.cartridge_ids] if conv.cartridge_ids else None
            )
            history = await _history(db, conversation_id) if conversational else []

        query = question
        needs_retrieval = True
        if conversational and history and lvl.rewrite:
            query, needs_retrieval = await rewrite_query(client, model, question, history)
            state.rewritten = query if query != question else None
        # Deep: several searches behind one question, unioned before reranking.
        queries = [query]
        if needs_retrieval and lvl.multi_query:
            queries = await effort_mod.split_question(client, model, query)

        yield {
            "event": "meta",
            "data": {
                "model": model,
                "rewritten": state.rewritten,
                "retrieving": needs_retrieval,
                "categories": len(category_ids) if category_ids else 0,
                "cartridges": len(cartridge_ids) if cartridge_ids else 0,
                "stance": stance,
                "effort": lvl.name,
                "searches": queries if len(queries) > 1 else None,
            },
        }

        # The consult step: a seated module may answer part of this question live, its
        # result joining the passage pool to be cited and verified like any other. Refused
        # when chat is on a remote provider and the module is local-only; the desk says so.
        live_hits: list[SearchHit] = []
        if needs_retrieval:
            from library_agent.modules import store as mod_store
            from library_agent.modules.consult import consult as consult_modules

            remote_chat = providers.is_remote(model)
            usable, held_back = [], []
            for mod, mcfg in mod_store.seated_modules():
                (held_back if (mod.local_only and remote_chat) else usable).append((mod, mcfg))
            if usable:
                for lr in await consult_modules(client, model, query, usable):
                    live_hits.append(_live_hit(lr))
            if usable or held_back:
                yield {
                    "event": "consulted",
                    "data": {
                        "results": [h.live for h in live_hits],
                        "held_back": [
                            {
                                "module": m.name,
                                "reason": "local models only; chat is on a remote provider",
                            }
                            for m, _ in held_back
                        ],
                    },
                }

        pinned = list(pinned_chunk_ids or [])
        if needs_retrieval or pinned:
            async with session_scope() as db:
                # Passages the person held lead the context; retrieval fills the rest of
                # the budget around them rather than replacing them.
                held_hits = await hits_for_chunks(db, pinned) if pinned else []
                seen: set = {h.chunk_id for h in held_hits}
                merged: list[SearchHit] = []
                # A volume the question names is its subject, not one voice among many.
                named = await named_documents(db, question) if needs_retrieval else []
                for q in queries if needs_retrieval else []:
                    for h in await retrieve(
                        db,
                        q,
                        config=lvl.config,
                        client=client,
                        limit=lvl.passages
                        if len(queries) == 1
                        else max(3, lvl.passages // len(queries) + 2),
                        category_ids=category_ids,
                        cartridge_ids=cartridge_ids,
                        favour=named,
                    ):
                        if h.chunk_id not in seen:
                            seen.add(h.chunk_id)
                            merged.append(h)
                # Across several searches, keep the strongest; within one, retrieve() already did.
                merged.sort(key=lambda h: -h.score)
                room = max(lvl.passages - len(held_hits), 0)
                # The library's reading of the nearest sections, after the passages.
                readings: list[SearchHit] = []
                sections: set = set()
                for q in queries if needs_retrieval and lvl.readings else []:
                    for h in await retrieve_readings(
                        db,
                        q,
                        client=client,
                        limit=lvl.readings
                        if len(queries) == 1
                        else max(2, lvl.readings // len(queries) + 1),
                        category_ids=category_ids,
                        cartridge_ids=cartridge_ids,
                        document_ids=document_ids,
                        favour=named,
                    ):
                        if (h.document_id, h.section_path) not in sections:
                            sections.add((h.document_id, h.section_path))
                            readings.append(h)
                # Live results lead the pool: they apply or they do not, and are not ranked
                # against passages or counted against the passage budget.
                state.hits = live_hits + held_hits + merged[:room] + readings[: lvl.readings]
                if document_ids:
                    state.hits = [h for h in state.hits if h.document_id in set(document_ids)]
                state.sources = build_sources(state.hits, {h.chunk_id for h in held_hits})
                docs = {
                    d.id: d
                    for d in (
                        await db.execute(
                            select(Document).where(
                                Document.id.in_(
                                    [h.document_id for h in state.hits] or [uuid.uuid4()]
                                )
                            )
                        )
                    ).scalars()
                }
                provenance = await cartridge_provenance(db, list(docs))
            for s in state.sources:
                d = docs.get(uuid.UUID(s.document_id))
                if d:
                    s.document_title = d.title
                    s.readings_only = d.readings_only
                    s.cartridge = provenance.get(d.id)

        yield {
            "event": "sources",
            "data": [
                {
                    "n": s.n,
                    "title": plain_label(s.document_title),
                    "page": s.page,
                    "section": plain_label(s.section_path),
                    "document_id": s.document_id,
                    "readings_only": s.readings_only,
                    "cartridge": s.cartridge,
                    "held": s.held,
                    "kind": s.kind,
                    "live": s.live,
                }
                for s in state.sources
            ],
        }

        messages = answer_mod.build_messages(
            question,
            state.hits,
            state.sources,
            history,
            stance=stance,
            foreign=any(s.cartridge for s in state.sources),
            max_passage_chars=effort_mod.passage_chars(lvl),
        )
        # A stance is an invitation to interpret; give the sampler room to take it.
        if temperature is None:
            temperature = 0.85 if stance else 0.6
        buffer: list[str] = []

        # Prefill on a local model can exceed the lease TTL before the first token
        # arrives, and a per-token refresh does nothing during that window -- a reading
        # job could resume mid-generation. Refresh on a timer instead.
        async def keepalive() -> None:
            while True:
                await asyncio.sleep(cfg.chat_lease_ttl_seconds / 3)
                await mark_chat_active(redis)

        heartbeat = asyncio.create_task(keepalive())
        try:
            async for kind, piece in answer_mod.stream_answer(
                client, model, messages, temperature=temperature, seed=seed, num_ctx=lvl.num_ctx
            ):
                if kind == "thinking":
                    yield {"event": "thinking", "data": piece}
                    continue
                buffer.append(piece)
                yield {"event": "token", "data": piece}
        finally:
            heartbeat.cancel()

        raw = "".join(buffer)
        cleaned, used = validate(raw, state.sources)
        metrics = citation_validity(raw, state.sources)
        state.answer = cleaned

        async with session_scope() as db:
            db.add(
                Message(
                    conversation_id=conversation_id,
                    role="user",
                    content=question,
                    rewritten_query=state.rewritten,
                )
            )
            db.add(
                Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=cleaned,
                    sources={
                        "cited": [
                            {
                                "n": s.n,
                                "title": plain_label(s.document_title),
                                "page": s.page,
                                "chunk_id": s.chunk_id,
                                "document_id": s.document_id,
                                "section": plain_label(s.section_path),
                                "kind": s.kind,
                                "live": s.live,
                            }
                            for s in used
                        ],
                        "retrieved": len(state.sources),
                        "stance": stance,
                        **metrics,
                    },
                )
            )

        yield {
            "event": "done",
            "data": {
                "answer": cleaned,
                "cited": [s.n for s in used],
                **metrics,
            },
        }
    except Exception as exc:
        await record_exception(
            exc, source="chat", context={"model": locals().get("model"), "question": question[:200]}
        )
        log.exception("chat turn failed")
        yield {"event": "error", "data": str(exc)[:500]}
    finally:
        if held:
            gate.release(caller)
        await client.aclose()
        await mark_chat_done(redis)
        await redis.aclose()
