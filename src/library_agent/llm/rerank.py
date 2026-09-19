"""Cross-encoder reranking.

A bi-encoder embeds query and passage separately, so it can never model interaction
between them. A cross-encoder reads both together and scores the pair directly, which is
why reranking a wide candidate set usually beats widening the first-stage retrieval.

Loaded lazily and kept on MPS. This is the one component that does not go through Ollama,
so it holds its own ~1.2GB outside the Ollama budget."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from library_agent.config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    import torch
    from sentence_transformers import CrossEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("loading reranker %s on %s", settings().reranker_model, device)
    return CrossEncoder(settings().reranker_model, device=device, max_length=512)


def available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def rerank(query: str, passages: list[str], *, batch_size: int = 16) -> list[float]:
    """Relevance score per passage, higher is better. Order matches the input."""
    if not passages:
        return []
    scores = _model().predict(
        [(query, p) for p in passages], batch_size=batch_size, show_progress_bar=False
    )
    return [float(s) for s in scores]


def warm() -> None:
    """Force the model to load, so the first real query does not pay for it."""
    _model().predict([("warm", "up")], show_progress_bar=False)
