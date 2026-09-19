"""Retrieval-layer tests."""

from __future__ import annotations

import pytest

from library_agent.eval.metrics import RetrievalMetrics
from library_agent.retrieval.hybrid import _build_sql
from library_agent.retrieval.pipeline import LADDER, RetrievalConfig

KS = (1, 3, 5, 10)


def test_build_sql_ablates_each_half():
    both = _build_sql(True, True)
    assert "dense as (" in both and "lex as (" in both

    dense = _build_sql(True, False)
    assert "dense as (" in dense and "lex as (" not in dense
    assert ":w_lex" not in dense  # unused bind params must not be emitted

    lex = _build_sql(False, True)
    assert "lex as (" in lex and "dense as (" not in lex
    assert ":w_dense" not in lex


def test_build_sql_uses_or_tsquery_not_websearch():
    """websearch_to_tsquery ANDs every term, so a natural-language question matches zero
    rows. This regression killed the entire lexical half of hybrid search."""
    sql = _build_sql(False, True)
    assert "or_tsquery(:q)" in sql
    assert "websearch_to_tsquery" not in sql
    assert "plainto_tsquery" not in sql


def test_build_sql_weights_are_parameterised():
    sql = _build_sql(True, True)
    assert ":w_dense / (:k + dense.rk)" in sql
    assert ":w_lex / (:k + lex.rk)" in sql


def test_ladder_names_unique():
    names = [c.name for c in LADDER]
    assert len(names) == len(set(names))


def test_config_rejects_no_retriever():
    cfg = RetrievalConfig(name="broken", use_dense=False, use_lexical=False)
    assert not (cfg.use_dense or cfg.use_lexical)  # pipeline raises on this


def test_metrics_rank_and_mrr():
    import uuid

    gold, other = uuid.uuid4(), uuid.uuid4()
    gd, od = uuid.uuid4(), uuid.uuid4()
    m = RetrievalMetrics()

    assert m.add([gold, other], gold, [gd], gd, KS) == 1
    assert m.add([other, gold], gold, [gd], gd, KS) == 2
    assert m.add([other, other], gold, [od], gd, KS) is None

    s = m.summary(KS)
    assert s["n"] == 3
    assert s["recall@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["recall@3"] == pytest.approx(2 / 3, abs=1e-4)
    # MRR = (1 + 1/2 + 0) / 3
    assert s["mrr"] == pytest.approx(0.5, abs=1e-4)
    assert s["doc_recall"] == pytest.approx(2 / 3, abs=1e-4)


def test_metrics_empty():
    assert RetrievalMetrics().summary(KS) == {}


def test_reflection_sql_maps_back_to_chunks():
    """Tier 2 reflections are separate vectors pointing at their chunk. Retrieval must
    resolve them to the chunk so eval gold labels still apply and one chunk cannot
    occupy two candidate slots."""
    from library_agent.retrieval.hybrid import _build_sql

    off = _build_sql(True, False, False)
    on = _build_sql(True, False, True)

    assert "artifact" not in off
    assert "a.target_id as chunk_id" in on
    assert "union all" in on
    assert "group by chunk_id" in on  # best score per chunk, not one row per vector


class TestCategoryFiltering:
    """The spec's category chips were decorative until Phase 7 — they toggled UI state and
    filtered nothing. These pin the filter into the SQL."""

    def test_filter_present_in_every_variant(self):
        from library_agent.retrieval.hybrid import _build_sql

        for dense, lexical, refl in [
            (True, True, False),
            (True, False, True),
            (False, True, False),
        ]:
            sql = _build_sql(dense, lexical, refl)
            assert ":cats" in sql, f"no category filter for dense={dense} lex={lexical}"
            assert "document_category" in sql
            assert "chunk_category" in sql

    def test_filter_matches_document_or_chunk_level(self):
        """A chapter of a general book can be squarely about something the book is not, so
        a chunk-level tag must bring the chunk into scope on its own."""
        from library_agent.retrieval.hybrid import _build_sql

        sql = _build_sql(True, False, False)
        doc_at = sql.index("document_category")
        chunk_at = sql.index("chunk_category")
        between = sql[doc_at:chunk_at]
        assert " or exists" in between, "document and chunk predicates must be OR-ed"

    def test_null_categories_disables_the_filter(self):
        from library_agent.retrieval.hybrid import _build_sql

        assert "cast(:cats as uuid[]) is null or" in _build_sql(True, False, False)
