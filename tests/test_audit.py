"""The shared audit layer, deep Write retrieval, and the back-matter filter."""

from __future__ import annotations

import uuid

from library_agent.retrieval.hybrid import SearchHit
from library_agent.retrieval.pipeline import is_apparatus


def _hit(doc, text="x", score=1.0, kind="passage", path="Ch 1", chunk=None):
    return SearchHit(
        chunk_id=chunk or uuid.uuid4(),
        document_id=doc,
        document_title="T",
        section_path=path,
        page=1,
        text=text,
        score=score,
        dense_rank=None,
        lexical_rank=None,
        kind=kind,
    )


def test_back_matter_is_recognised_and_body_is_not():
    for p in [
        "Part I › Appendix A. Bibliography (cont. 2)",
        "Index (cont. 4)",
        "Chapter 3 › References",
        "Acknowledgments",
        "Table of Contents",
    ]:
        assert is_apparatus(p), p
    for p in [
        "3 What's wrong with Paxos? (cont. 2) › 9.3 Performance",
        "Indexing strategies",
        "Chapter 5. Replication",
        "Front matter",
        None,
    ]:
        assert not is_apparatus(p), p


def test_evidence_rules_reach_both_writers():
    from library_agent.chat.answer import SYSTEM
    from library_agent.chat.audit import EVIDENCE_RULES
    from library_agent.chat.compose import WRITE_SYSTEM

    assert EVIDENCE_RULES in SYSTEM and EVIDENCE_RULES in WRITE_SYSTEM
    assert "its authors' claims" in EVIDENCE_RULES


def test_deep_ask_is_audited_and_quick_is_not():
    from library_agent.chat import effort

    assert effort.get("deep").audit and not effort.get("quick").audit


async def test_facets_fall_back_to_the_brief():
    from library_agent.chat.compose import _facets

    class Broken:
        async def structured(self, *a, **k):
            raise RuntimeError("down")

    class Placeholder:
        async def structured(self, *a, **k):
            return {"facets": [{"facet": "x", "query": "…"}]}

    class Good:
        async def structured(self, *a, **k):
            return {
                "facets": [
                    {"facet": "LSM-tree write amplification", "query": "LSM compaction"},
                    {"facet": "Raft vs Paxos tail latency", "query": "Raft Paxos latency"},
                ]
            }

    assert (await _facets(Broken(), "m", "the brief"))[0]["query"] == "the brief"
    assert (await _facets(Placeholder(), "m", "the brief"))[0]["query"] == "the brief"
    assert [f["facet"] for f in await _facets(Good(), "m", "b")][1] == "Raft vs Paxos tail latency"


async def test_takeaway_is_told_what_the_review_flagged():
    from library_agent.chat.compose import _takeaway

    seen = {}

    class Client:
        async def structured(self, model, prompt, schema, **kw):
            seen["prompt"] = prompt
            return {"claim": "LSM trees trade read cost for sequential writes."}

    body = "LSM trees append; compaction rewrites. " * 5
    flags = [{"quote": "Raft always has lower tail latency", "issue": "authors' claim"}]
    await _takeaway(Client(), "m", "Doc", "H", body, flags=flags)
    assert "Raft always has lower tail latency" in seen["prompt"]
    await _takeaway(Client(), "m", "Doc", "H", body)
    assert "flagged" not in seen["prompt"].split("State, as")[0]


async def test_section_retrieval_splits_unions_and_caps(monkeypatch):
    """A section searches like a deep Ask: several queries, one union, a per-volume cap."""
    from library_agent.chat import compose

    a, b = uuid.uuid4(), uuid.uuid4()
    asked: list[str] = []

    async def fake_split(client, model, need):
        return ["raft election latency", "paxos multi leader"]

    async def fake_retrieve(db, q, **kw):
        asked.append(q)
        # every query finds six of volume a, and one of b only for the paxos search
        out = [_hit(a, f"{q} {i}", score=1 - i / 10) for i in range(6)]
        if "paxos" in q:
            out.append(_hit(b, "paxos", score=0.5))
        return out

    async def fake_readings(db, q, **kw):
        return [_hit(a, "reading", kind="reading", path=f"R {q}")]

    monkeypatch.setattr(compose, "split_question", fake_split)
    monkeypatch.setattr(compose, "retrieve", fake_retrieve)
    monkeypatch.setattr(compose, "retrieve_readings", fake_readings)
    sec = {"heading": "Consensus", "covers": "Raft vs Paxos", "retrieve": "consensus tail latency"}
    searches = await compose._section_queries(None, "m", "brief", sec)
    hits = await compose._section_hits(
        None, None, searches, category_ids=None, cartridge_ids=None, named=[]
    )
    assert searches == ["consensus tail latency", "raft election latency", "paxos multi leader"]
    assert asked[:3] == searches
    passages = [h for h in hits if h.kind == "passage"]
    readings = [h for h in hits if h.kind == "reading"]
    assert any(h.document_id == b for h in passages)  # the second volume made it in
    assert len(passages) <= compose.COMPOSE_PASSAGES
    assert len(readings) == 3  # one per search, distinct sections


def test_temp_and_hash_names_are_not_titles():
    from library_agent.ingest.extract import is_machine_title

    for t in [
        "tmpr6wwfid4",
        "tmp_ab12cd",
        "a3f9c2d1e4b5",
        "report.pdf",
        "4a8b0a0a-59a0-4b1e-89d1-a4ec4e2a65be",
    ]:
        assert is_machine_title(t), t
    for t in ["Database Internals", "Raft", "Paxos Made Simple", "tmux cheatsheet", None, ""]:
        assert not is_machine_title(t), t


async def test_primed_queries_are_not_embedded_again(monkeypatch):
    from library_agent.llm import embed

    calls: list[list[str]] = []

    async def fake(texts, client=None):
        calls.append(list(texts))
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(embed, "embed_texts", fake)
    await embed.prime_queries(["alpha", "beta", "alpha"])
    assert calls == [["alpha", "beta"]]  # one batch, deduplicated
    assert await embed.embed_query("beta") == [4.0]
    assert await embed.embed_query("gamma") == [5.0]
    assert calls == [["alpha", "beta"], ["gamma"]]


def _spaced_markdown(tmp_path):
    """Awkward spacing of the kind extraction produces: runs of spaces, lines broken mid-
    sentence, blank lines holding spaces -- everything packing normalises."""
    import random

    random.seed(3)
    words = [
        "the",
        "leader",
        "appends",
        "entries",
        "to",
        "its",
        "log",
        "and",
        "replicates",
        "them",
        "to",
        "followers",
    ]
    paras = []
    for p in range(60):
        sents = []
        for _s in range(random.randint(3, 9)):
            w = random.sample(words, random.randint(6, 12))
            sents.append(" ".join(w).capitalize() + ".")
        joiner = random.choice([" ", "  ", "\n", " \n  "])
        paras.append(joiner.join(sents))
    body = "\n \n".join(paras)
    text = "# Consensus\n\n" + body + "\n\n## Safety\n\n" + body[::-1].swapcase()
    f = tmp_path / "doc.md"
    f.write_text(text, encoding="utf-8")
    return f


def test_every_chunk_opens_on_its_own_text(tmp_path):
    """The span round-trips: text[char_start:char_end] is the chunk, whitespace aside."""
    import re

    from library_agent.ingest.chunk import chunk_document
    from library_agent.ingest.extract import extract
    from library_agent.ingest.structure import build_sections

    ex = extract(_spaced_markdown(tmp_path))
    chunks = chunk_document(ex, build_sections(ex), "Doc")
    assert len(chunks) > 5
    ws = re.compile(r"\s+")
    for c in chunks:
        assert c.span_exact
        assert ws.sub("", ex.text[c.char_start : c.char_end]) == ws.sub("", c.text)
    starts = [c.char_start for c in chunks]
    assert starts == sorted(starts)


def test_old_find_misplaced_chunks_on_the_same_text(tmp_path):
    """The defect the aligner replaced, measured on the same input: searching for the first
    120 characters misses whenever packing changed the spacing inside them."""
    import re

    from library_agent.ingest.chunk import chunk_document
    from library_agent.ingest.extract import extract
    from library_agent.ingest.structure import build_sections

    ex = extract(_spaced_markdown(tmp_path))
    chunks = chunk_document(ex, build_sections(ex), "Doc")
    misses = sum(ex.text.find(c.text[:120]) == -1 for c in chunks)
    assert misses > 0  # the input really does defeat a literal search
    ws = re.compile(r"\s+")
    assert all(ws.sub("", ex.text[c.char_start : c.char_end]) == ws.sub("", c.text) for c in chunks)


def test_a_reading_and_its_first_passage_are_two_sources():
    """Astra's defect: keyed by chunk id, a reading (cited through its section's first
    chunk) and that chunk cited as a passage shared one number, and the first mapping won."""
    from library_agent.chat.compose import Composition, _numbered

    doc, first = uuid.uuid4(), uuid.uuid4()
    art, other = uuid.uuid4(), uuid.uuid4()
    passage = _hit(doc, "the passage", chunk=first)
    reading = _hit(doc, "the reading", kind="reading", chunk=first)
    reading.artifact_id, reading.page_end, reading.span_chunk_ids = art, 9, [first, other]
    comp, by_key = Composition(), {}
    a = _numbered(passage, by_key, comp, {}, {})
    b = _numbered(reading, by_key, comp, {}, {})
    again = _numbered(reading, by_key, comp, {}, {})
    assert (a.n, b.n, again.n) == (1, 2, 2)
    assert b.kind == "reading" and b.artifact_id == str(art)
    assert b.span_chunk_ids == [str(first), str(other)]
    assert b.pages() == ", pp.1–9" and a.pages() == ", p.1"


def test_build_sources_carries_a_readings_span():
    from library_agent.chat.citations import build_sources

    doc, c1, art = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    r = _hit(doc, kind="reading", chunk=c1)
    r.artifact_id, r.page_end, r.span_chunk_ids = art, 4, [c1]
    p = _hit(doc, chunk=c1)
    srcs = build_sources([p, r])
    assert srcs[0].key() != srcs[1].key()
    assert srcs[1].artifact_id == str(art) and srcs[1].page_end == 4
    assert srcs[0].artifact_id is None


def test_a_long_section_is_read_in_parts_between_passages():
    from library_agent.reading.tier1 import section_parts

    pieces = [f"passage {i} " + "x" * 2500 for i in range(7)]  # ~17.5k chars
    parts = section_parts(pieces, 6000)
    assert len(parts) == 4
    assert all(len(p) <= 6000 for p in parts)
    assert "".join(parts).replace("\n\n", "") == "".join(pieces)  # nothing dropped
    assert section_parts(["short"], 6000) == ["short"]
    assert section_parts(["y" * 9000], 6000) == ["y" * 6000]  # one oversized passage: cut


async def test_parts_reconcile_into_one_reading():
    from library_agent.reading.tier1 import reconcile_parts

    seen = {}

    class Client:
        async def structured(self, model, prompt, schema, **kw):
            seen["prompt"] = prompt
            return {"summary": "Raft elects a leader, except under a partition, where it stalls."}

    outs = [
        {
            "summary": "Raft elects a leader.",
            "claims": ["Leaders are elected."],
            "entities": ["Raft"],
            "categories": ["Consensus", "Leader Election"],
        },
        {
            "summary": "Under a partition the minority stalls.",
            "claims": ["leaders are elected.", "A minority partition cannot commit."],
            "entities": ["Raft", "Partition"],
            "categories": ["Consensus"],
        },
    ]
    out = await reconcile_parts(Client(), "m", "Raft", "Elections", outs)
    assert "except under a partition" in out["summary"]
    assert "Under a partition the minority stalls." in seen["prompt"]
    assert out["claims"] == ["Leaders are elected.", "A minority partition cannot commit."]
    assert out["categories"][0] == "Consensus" and out["parts"] == 2

    class Down:
        async def structured(self, *a, **k):
            raise RuntimeError("down")

    fallback = await reconcile_parts(Down(), "m", "Raft", "Elections", outs)
    assert fallback["summary"] == "Raft elects a leader. Under a partition the minority stalls."
