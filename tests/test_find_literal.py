"""Find puts exact matches first when the query names something exactly, and says when
its highlighted sentence was chosen by meaning alone."""

from __future__ import annotations

import uuid

from conftest import make_document

from library_agent.retrieval.hybrid import SearchHit
from library_agent.retrieval.lift import lift_kind
from library_agent.retrieval.literal import exact_first, literal_hits, literal_terms


def test_names_are_found_and_questions_are_not():
    assert literal_terms("what does O_DIRECT do") == ["O_DIRECT"]
    assert literal_terms("--no-verify flag") == ["--no-verify"]
    assert literal_terms("RFC 7231 caching") == ["RFC 7231"]
    assert literal_terms("getElementById usage") == ["getElementById"]
    assert literal_terms("SSTable compaction") == ["SSTable"]
    assert literal_terms('"write amplification" in LSM trees') == ["write amplification"]
    assert literal_terms("fsync") == ["fsync"]  # a one-word query names its thing
    assert literal_terms("TLS") == ["TLS"]
    # A capitalised word inside a question is a word, not a name.
    assert literal_terms("how does HTTP caching work") == []
    assert literal_terms("how do leaders get elected") == []


def _h(text, rank=None):
    return SearchHit(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="T",
        section_path="S",
        page=1,
        text=text,
        score=1.0,
        dense_rank=rank,
        lexical_rank=None,
    )


def test_exact_matches_lead_in_rank_order():
    ranked = [_h("about flags in general", 1), _h("opens with O_DIRECT set", 2), _h("other", 3)]
    extra = _h("o_direct bypasses the page cache")
    out, ids = exact_first(ranked, [extra, ranked[1]], ["O_DIRECT"], 10)
    assert [h.text for h in out] == [
        "opens with O_DIRECT set",  # ranked and exact: first
        "o_direct bypasses the page cache",  # literal-only: next
        "about flags in general",
        "other",
    ]
    assert ids == {ranked[1].chunk_id, extra.chunk_id}
    same, none = exact_first(ranked, [], [], 2)
    assert same == ranked[:2] and none == set()


async def test_the_literal_scan_respects_scope(scratch_db):
    db = scratch_db
    a = await make_document(db, title="Storage", body="open the file with O_DIRECT set " * 8)
    b = await make_document(db, title="Other", body="O_DIRECT appears here too " * 8)
    hits = await literal_hits(db, ["o_direct"], limit=10)
    assert {h.document_id for h in hits} == {a.id, b.id}
    scoped = await literal_hits(db, ["O_DIRECT"], limit=10, document_ids=[a.id])
    assert {h.document_id for h in scoped} == {a.id}
    # "_" is literal, not a LIKE wildcard
    assert await literal_hits(db, ["OXDIRECT"], limit=10) == []
    assert await literal_hits(db, [], limit=10) == []


def test_a_sentence_chosen_by_meaning_says_so():
    assert lift_kind("leader election timeout", "Raft randomises election timeouts.") == "words"
    assert lift_kind("leader election timeout", "A node waits, then asks for votes.") == "meaning"
    assert lift_kind("q", None) is None


def test_an_exact_hit_lights_the_sentence_holding_the_name_wherever_it_is():
    from library_agent.retrieval.lift import holding_sentence

    text = " ".join(f"Sentence number {i} says nothing much at all." for i in range(20))
    text += " Opening with O_DIRECT bypasses the page cache."
    assert holding_sentence(text, ["O_DIRECT"]) == "Opening with O_DIRECT bypasses the page cache."
    assert holding_sentence("nothing here at all.", ["O_DIRECT"]) is None


async def test_a_rare_name_is_looked_up_but_a_rare_ordinary_word_is_not(db):
    from conftest import make_document

    from library_agent.retrieval.literal import rare_hits, rare_terms

    await make_document(
        db,
        title="Team notes",
        body="Zorbleck reviewed the storage plan. Zorbleck approved it. " * 6
        + "We will recalibrate the quernish gauges next week. " * 6,
    )
    words = await rare_terms(db, "what did zorbleck say about the quernish gauges?")
    assert "zorbleck" in words and "quernish" not in words  # a name, not a lowercase word
    hits = await rare_hits(db, ["zorbleck"])
    assert hits and all("Zorbleck" in h.text for h in hits)
