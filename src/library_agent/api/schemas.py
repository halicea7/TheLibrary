"""API response models."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: uuid.UUID
    title: str
    kind: str
    status: str
    tier: int
    page_count: int | None
    original_filename: str
    sections: int
    chunks: int
    starred: bool
    near_dup_of: uuid.UUID | None
    added_at: datetime
    categories: list[str] = []
    readings_only: bool = False
    cartridge: dict | None = None  # {id, name, colour} when it came in a cartridge
    shelf: dict | None = None  # {top, top_id, sub, sub_id}: the one place it sits


class SectionOut(BaseModel):
    id: uuid.UUID
    title: str | None
    path: str
    level: int
    page_start: int | None
    page_end: int | None
    chunks: int


class DocumentDetail(DocumentOut):
    sections_detail: list[SectionOut]


class UploadResult(BaseModel):
    document_id: uuid.UUID
    title: str
    status: str
    sections: int
    chunks: int
    pages: int
    duplicate_of: uuid.UUID | None = None
    near_duplicate_sim: float | None = None
    needs_ocr: bool = False
    elapsed_seconds: float


class SearchHitOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    section_path: str
    page: int | None
    text: str
    score: float
    dense_rank: int | None
    lexical_rank: int | None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHitOut]
    elapsed_seconds: float


class HealthOut(BaseModel):
    ok: bool
    model_answering: bool = True  # the liveness probe, not mere reachability
    model_liveness: dict = {}
    generations: dict = {}
    documents: int
    chunks: int
    embeddings: int
    orphan_vectors: dict[str, int]
    models_resident: list[str]
    embed_model: str
    chat_model: str
