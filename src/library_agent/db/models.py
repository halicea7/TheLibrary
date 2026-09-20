"""Schema. Postgres + pgvector, single store — vectors commit in the same transaction
as the rows they describe, so there is no dual-write reconciliation to get wrong."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from library_agent.config import settings

# Stable namespace so chunk ids are reproducible across machines and re-ingests.
CHUNK_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def chunk_id(content_hash: str, char_start: int, char_end: int) -> uuid.UUID:
    """Content-addressed chunk id: re-ingesting the same bytes yields the same ids,
    which makes ingestion idempotent and crash recovery a replay rather than a cleanup."""
    return uuid.uuid5(CHUNK_NAMESPACE, f"{content_hash}:{char_start}:{char_end}")


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSONB, list[str]: JSONB}


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(primary_key=True, default=uuid.uuid4)


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


# --------------------------------------------------------------------------- enums


class DocumentKind(StrEnum):
    PAPER = "paper"
    BOOK = "book"
    DOC = "doc"
    NOTE = "note"


class DocumentStatus(StrEnum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    READY = "ready"  # Tier 0 complete: searchable
    ERROR = "error"
    DUPLICATE = "duplicate"


class ArtifactKind(StrEnum):
    ORIENTATION = "orientation"
    REFLECTION = "reflection"
    SECTION_SUMMARY = "section_summary"
    DOCUMENT_SUMMARY = "document_summary"
    CLUSTER_SUMMARY = "cluster_summary"
    CONTRADICTION = "contradiction"


class TargetKind(StrEnum):
    DOCUMENT = "document"
    SECTION = "section"
    CHUNK = "chunk"
    CLUSTER = "cluster"


class OwnerKind(StrEnum):
    """What an embedding vector describes."""

    CHUNK = "chunk"
    ARTIFACT = "artifact"
    # Document fingerprint: title + opening text, written at Tier 0. Doubles as the
    # router vector until Tier 1 produces a real document summary to replace it.
    DOCUMENT = "document"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    YIELDED = "yielded"  # stepped aside for an active chat session
    DONE = "done"
    ERROR = "error"


# --------------------------------------------------------------------------- documents


class Document(Base):
    __tablename__ = "document"

    id: Mapped[uuid.UUID] = _pk()
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    kind: Mapped[str] = mapped_column(String(16), default=DocumentKind.PAPER)
    source_path: Mapped[str] = mapped_column(Text)
    original_filename: Mapped[str] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer, default=None)
    doi: Mapped[str | None] = mapped_column(Text, default=None, index=True)

    # Quality level, not pipeline stage: 0 = embedded only, 1 = structural, 2 = deep read.
    tier: Mapped[int] = mapped_column(SmallInteger, default=0, index=True)
    status: Mapped[str] = mapped_column(String(16), default=DocumentStatus.PENDING)
    starred: Mapped[bool] = mapped_column(default=False)
    retrieval_hits: Mapped[int] = mapped_column(Integer, default=0)  # Tier 2 promotion signal
    # Arrived in a readings-only cartridge: its "passages" are the sharer's section
    # summaries, synthesised as chunks. The originals were never shared.
    readings_only: Mapped[bool] = mapped_column(default=False)
    # The one place the volume sits on the shelf: a second-level category. Tags
    # (document_category) remain many-to-many for filtering; the shelf is singular.
    shelf_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("category.id", ondelete="SET NULL"), default=None, index=True
    )

    near_dup_of: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document.id", ondelete="SET NULL"), default=None
    )
    error: Mapped[str | None] = mapped_column(Text, default=None)
    added_at: Mapped[datetime] = _now()

    sections: Mapped[list[Section]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Section(Base):
    __tablename__ = "section"

    id: Mapped[uuid.UUID] = _pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), index=True
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("section.id", ondelete="CASCADE"), default=None
    )
    order_index: Mapped[int] = mapped_column(Integer)
    # Breadcrumb like "Methods › Training setup"; also feeds the deterministic chunk prefix.
    path: Mapped[str] = mapped_column(Text, default="")
    title: Mapped[str | None] = mapped_column(Text, default=None)
    level: Mapped[int] = mapped_column(SmallInteger, default=1)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)
    page_start: Mapped[int | None] = mapped_column(Integer, default=None)
    page_end: Mapped[int | None] = mapped_column(Integer, default=None)

    document: Mapped[Document] = relationship(back_populates="sections")

    __table_args__ = (Index("ix_section_doc_order", "document_id", "order_index"),)


class Chunk(Base):
    __tablename__ = "chunk"

    # Deterministic: uuid5(content_hash, char span). See chunk_id() above.
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), index=True
    )
    section_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("section.id", ondelete="SET NULL"), default=None, index=True
    )
    order_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    # "text" is a passage of the document; "figure" is the vision model's reading of a
    # figure, kept as a passage so retrieval and citation treat it like one.
    kind: Mapped[str] = mapped_column(String(16), default="text", server_default="text")
    # "Title › section path › orientation line" — built without an LLM call.
    context_prefix: Mapped[str] = mapped_column(Text, default="")
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    page_start: Mapped[int | None] = mapped_column(Integer, default=None)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)

    document: Mapped[Document] = relationship(back_populates="chunks")

    # Lexical half of hybrid retrieval. Generated column so it can never drift from text.
    fts: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(context_prefix,'') || ' ' || text)", persisted=True
        ),
    )

    __table_args__ = (
        Index("ix_chunk_fts", "fts", postgresql_using="gin"),
        Index("ix_chunk_doc_order", "document_id", "order_index"),
    )


# --------------------------------------------------------------------------- artifacts


class Artifact(Base):
    """Every LLM-generated text in one table, stamped with the model and prompt version
    that produced it. That stamp is what makes prompt iteration cheap: bump a version and
    only the artifacts carrying the old one regenerate."""

    __tablename__ = "artifact"

    id: Mapped[uuid.UUID] = _pk()
    kind: Mapped[str] = mapped_column(String(24), index=True)
    target_kind: Mapped[str] = mapped_column(String(16))
    target_id: Mapped[uuid.UUID] = mapped_column(index=True)
    text: Mapped[str] = mapped_column(Text)
    # Structured extras from the same call (entities, claims, proposed categories).
    data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    model: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(16))
    tier: Mapped[int] = mapped_column(SmallInteger, default=1)
    # Who wrote this: null is you; otherwise the cartridge it arrived in. Lets the reader
    # show your marginalia beside theirs, and lets an update replace theirs but not yours.
    cartridge_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("cartridge.id", ondelete="SET NULL"), default=None, index=True
    )
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        # NULLS NOT DISTINCT: without it two local artifacts (cartridge_id null) of the
        # same kind on the same target would both be allowed.
        UniqueConstraint(
            "kind",
            "target_kind",
            "target_id",
            "cartridge_id",
            name="uq_artifact_target",
            postgresql_nulls_not_distinct=True,
        ),
    )


class Embedding(Base):
    """Polymorphic by (owner_kind, owner_id): chunks and artifacts share one index."""

    __tablename__ = "embedding"

    owner_kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column(String(64), primary_key=True)
    # halfvec halves storage versus float4 with no measurable recall cost at this scale.
    vec: Mapped[Any] = mapped_column(HALFVEC(settings().embed_dim))
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        Index(
            "ix_embedding_hnsw",
            "vec",
            postgresql_using="hnsw",
            postgresql_ops={"vec": "halfvec_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
        Index("ix_embedding_lookup", "owner_kind", "model"),
    )


# --------------------------------------------------------------------------- cartridges


class Cartridge(Base):
    """A portable slice of a library, inserted whole and ejectable whole.

    The id comes from the manifest and is stable across versions, so re-importing a newer
    version of the same cartridge replaces its documents in place."""

    __tablename__ = "cartridge"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    slug: Mapped[str] = mapped_column(String(80), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    colour: Mapped[str] = mapped_column(String(16))  # "#rrggbb"
    icon_svg: Mapped[str | None] = mapped_column(Text, default=None)
    made_by: Mapped[str | None] = mapped_column(Text, default=None)
    made_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    level: Mapped[str] = mapped_column(String(16))  # full | readings | catalogue
    embed_model: Mapped[str] = mapped_column(String(64))
    reader_model: Mapped[str | None] = mapped_column(String(64), default=None)
    prompt_versions: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    manifest: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    content_hash: Mapped[str] = mapped_column(String(64))
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    imported_at: Mapped[datetime] = _now()
    # The maker's design: material, dials, art. It travels in the manifest and is not
    # editable after insertion -- a cartridge looks the same on every rack.
    design: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    art_path: Mapped[str | None] = mapped_column(Text, default=None)


class CartridgeDocument(Base):
    """A document in no cartridge is local. The same paper in two cartridges is one
    document with two memberships -- content-hash dedup already makes that so."""

    __tablename__ = "cartridge_document"

    cartridge_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cartridge.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), primary_key=True
    )
    # Did this import create the document, or was it already on the shelf? Eject
    # removes what it brought and leaves what was here.
    introduced: Mapped[bool] = mapped_column(default=False)


class CartridgeLevel(StrEnum):
    FULL = "full"
    READINGS = "readings"
    CATALOGUE = "catalogue"


# --------------------------------------------------------------------------- taxonomy


class Category(Base):
    __tablename__ = "category"

    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str] = mapped_column(Text, unique=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    # Two levels, no more: a top shelf (parent_id null) holds sub-shelves, which hold
    # volumes. Cybersecurity › Network Scanning › the document.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("category.id", ondelete="SET NULL"), default=None, index=True
    )
    # Names folded into this one by the backstop cleanup job.
    merged_from: Mapped[list[str] | None] = mapped_column(JSONB, default=None)
    canonical: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = _now()


class DocumentCategory(Base):
    __tablename__ = "document_category"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), primary_key=True
    )
    category_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("category.id", ondelete="CASCADE"), primary_key=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=1.0)


class ChunkCategory(Base):
    """Finer-grained than document level: one chapter of a general book may be squarely
    about something the book as a whole is not."""

    __tablename__ = "chunk_category"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunk.id", ondelete="CASCADE"), primary_key=True
    )
    category_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("category.id", ondelete="CASCADE"), primary_key=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=1.0)


# --------------------------------------------------------------- cross-document layer


class Cluster(Base):
    """A theme spanning documents. Built by clustering section-summary embeddings, which
    is free; only the one summary call per cluster costs anything."""

    __tablename__ = "cluster"

    id: Mapped[uuid.UUID] = _pk()
    label: Mapped[str | None] = mapped_column(Text, default=None)
    method: Mapped[str] = mapped_column(String(32), default="hdbscan")
    size: Mapped[int] = mapped_column(Integer, default=0)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    has_contradiction: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = _now()


class ClusterMember(Base):
    __tablename__ = "cluster_member"

    cluster_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cluster.id", ondelete="CASCADE"), primary_key=True
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("artifact.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), index=True
    )
    distance: Mapped[float] = mapped_column(Float, default=0.0)


class Citation(Base):
    """Reference edges between documents. Free structure: no generation involved."""

    __tablename__ = "citation"

    id: Mapped[uuid.UUID] = _pk()
    citing_document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), index=True
    )
    matched_document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document.id", ondelete="SET NULL"), default=None, index=True
    )
    raw_reference: Mapped[str] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(Text, default=None)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)


# --------------------------------------------------------------------------- chat


class Conversation(Base):
    __tablename__ = "conversation"

    id: Mapped[uuid.UUID] = _pk()
    title: Mapped[str | None] = mapped_column(Text, default=None)
    # False = each message retrieves fresh with no history or query rewriting.
    conversational: Mapped[bool] = mapped_column(default=True)
    # Overrides settings().chat_model for this conversation; None = use the default.
    # Switching evicts the other model, so this is per-conversation, not per-message.
    model: Mapped[str | None] = mapped_column(Text, default=None)
    category_ids: Mapped[list[str] | None] = mapped_column(JSONB, default=None)
    # "Ask Security's shelf": scope every turn of this conversation to these cartridges.
    cartridge_ids: Mapped[list[str] | None] = mapped_column(JSONB, default=None)
    created_at: Mapped[datetime] = _now()

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "message"

    id: Mapped[uuid.UUID] = _pk()
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversation.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    # Numbered sources behind this answer, so [n] markers stay resolvable after the fact.
    sources: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    rewritten_query: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = _now()

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


# --------------------------------------------------------------------------- jobs


class Job(Base):
    """Mirrors ARQ state so the UI has something durable to poll."""

    __tablename__ = "job"

    id: Mapped[uuid.UUID] = _pk()
    kind: Mapped[str] = mapped_column(String(32), index=True)
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), default=None, index=True
    )
    tier: Mapped[int | None] = mapped_column(SmallInteger, default=None)
    state: Mapped[str] = mapped_column(String(16), default=JobState.QUEUED, index=True)
    progress_current: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    yielded_reason: Mapped[str | None] = mapped_column(Text, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# --------------------------------------------------------------------------- eval


class Incident(Base):
    """Something went wrong, recorded so the library can help fix it. Fed by the API's
    exception handler, failed jobs, failed chat turns and ERROR-level logs. The same
    error recurring within a window increments `count` rather than adding a row."""

    __tablename__ = "incident"

    id: Mapped[uuid.UUID] = _pk()
    source: Mapped[str] = mapped_column(String(16), index=True)  # api | worker | chat | ingest
    kind: Mapped[str] = mapped_column(String(80), index=True)  # exception class or job kind
    message: Mapped[str] = mapped_column(Text)
    detail: Mapped[str | None] = mapped_column(Text, default=None)  # traceback
    context: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    count: Mapped[int] = mapped_column(Integer, default=1)
    first_at: Mapped[datetime] = _now()
    last_at: Mapped[datetime] = _now()
    resolved: Mapped[bool] = mapped_column(default=False, index=True)
    # The model's troubleshooting, cached; regenerated on request.
    advice: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    advice_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class EvalQuestion(Base):
    """Retrieval questions are generated from a known chunk, so the gold label comes free
    and recall@k needs no manual labeling."""

    __tablename__ = "eval_question"

    id: Mapped[uuid.UUID] = _pk()
    suite: Mapped[str] = mapped_column(String(32), index=True)
    question: Mapped[str] = mapped_column(Text)
    gold_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("chunk.id", ondelete="CASCADE"), default=None
    )
    gold_document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), default=None
    )
    created_at: Mapped[datetime] = _now()


class EvalRun(Base):
    __tablename__ = "eval_run"

    id: Mapped[uuid.UUID] = _pk()
    suite: Mapped[str] = mapped_column(String(32), index=True)
    # Free-form: which tier, reranker on/off, rrf_k, router width...
    params: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = _now()


class EvalResult(Base):
    __tablename__ = "eval_result"

    id: Mapped[uuid.UUID] = _pk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_run.id", ondelete="CASCADE"), index=True
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_question.id", ondelete="CASCADE")
    )
    rank: Mapped[int | None] = mapped_column(Integer, default=None)  # None = miss
    score: Mapped[float | None] = mapped_column(Float, default=None)
    answer: Mapped[str | None] = mapped_column(Text, default=None)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)

    __table_args__ = (CheckConstraint("rank is null or rank >= 1", name="ck_eval_rank_positive"),)
