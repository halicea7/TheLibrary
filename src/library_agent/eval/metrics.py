"""Retrieval metrics."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class RetrievalMetrics:
    n: int = 0
    hits_at: dict[int, int] = field(default_factory=dict)
    reciprocal_ranks: list[float] = field(default_factory=list)
    doc_hits: int = 0
    ranks: list[int | None] = field(default_factory=list)

    def add(
        self,
        ranked_chunks: list[uuid.UUID],
        gold_chunk: uuid.UUID,
        ranked_docs: list[uuid.UUID],
        gold_doc: uuid.UUID,
        ks: tuple[int, ...],
    ) -> int | None:
        self.n += 1
        rank: int | None = None
        for i, cid in enumerate(ranked_chunks, start=1):
            if cid == gold_chunk:
                rank = i
                break
        self.ranks.append(rank)
        for k in ks:
            self.hits_at.setdefault(k, 0)
            if rank is not None and rank <= k:
                self.hits_at[k] += 1
        self.reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        if gold_doc in ranked_docs:
            self.doc_hits += 1
        return rank

    def summary(self, ks: tuple[int, ...]) -> dict[str, float]:
        if not self.n:
            return {}
        out = {f"recall@{k}": round(self.hits_at.get(k, 0) / self.n, 4) for k in ks}
        out["mrr"] = round(sum(self.reciprocal_ranks) / self.n, 4)
        out["doc_recall"] = round(self.doc_hits / self.n, 4)
        out["n"] = self.n
        return out
