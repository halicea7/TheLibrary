"""Regression tests for the invariants that were hardest to get right."""

from __future__ import annotations

import pytest

from library_agent.db.models import chunk_id
from library_agent.ingest.chunk import build_context_prefix, chunk_document, count_tokens
from library_agent.ingest.extract import Extracted, TocEntry, normalize
from library_agent.ingest.structure import build_sections, detect_headings


def test_chunk_ids_are_content_addressed():
    """Re-ingesting identical bytes must produce identical ids, or nothing is idempotent."""
    a = chunk_id("abc123", 0, 500)
    b = chunk_id("abc123", 0, 500)
    assert a == b
    assert chunk_id("abc123", 0, 501) != a
    assert chunk_id("different", 0, 500) != a


def test_normalize_repairs_hyphenation_and_ligatures():
    assert "contextual" in normalize("contex-\ntual retrieval")
    assert normalize("eﬃcient") == "efficient"
    # A single newline is a soft wrap; a blank line is a real paragraph break.
    assert normalize("one\ntwo") == "one two"
    assert "\n\n" in normalize("one\n\ntwo")


def test_detect_headings_finds_numbered_sections():
    text = (
        "intro text here\n\n1 Introduction\n\nbody\n\n"
        "2 Related Work\n\nbody\n\n3.1 Residual Learning\n\nbody\n"
    )
    found = detect_headings(text)
    titles = [t for _, _, t in found]
    assert "1 Introduction" in titles
    assert "3.1 Residual Learning" in titles
    # Sub-level derives from the numbering depth.
    assert {t: lvl for _, lvl, t in found}["3.1 Residual Learning"] == 2


def test_detect_headings_ignores_sentences():
    """A numbered list item that runs into prose is not a heading."""
    text = "1. This is a full sentence that happens to start with a number and keeps going on.\n"
    assert detect_headings(text) == []


def test_parent_heading_survives_pruning():
    """A parent whose own span is tiny must not be dropped -- doing so reparents its
    children under an unrelated section and corrupts every chunk's context prefix."""
    filler = "body text here. " * 20
    text = (
        f"front matter\n\n1 Introduction\n\n{filler}\n\n"
        f"3 Deep Residual Learning\n\n3.1 Residual Learning\n\n{filler}\n\n"
        f"4 Experiments\n\n{filler}"
    )
    ex = Extracted(text=text, page_offsets=[(0, 1)], toc=[], title="T", page_count=1)
    sections = build_sections(ex)
    paths = [s.path for s in sections]
    assert any("3 Deep Residual Learning › 3.1 Residual Learning" in p for p in paths), paths


def test_markdown_headings_use_exact_offsets():
    """Markdown gives exact positions; fuzzy text search would mis-locate '## Heading'."""
    filler_len = len("some body text goes here to give the section real length. " * 5)
    doc_at = 0
    alpha_at = len("# Doc\n\n") + filler_len + 2
    beta_at = alpha_at + len("## Alpha\n\n") + filler_len + 2
    toc = [
        TocEntry(level=1, title="Doc", page=1, char_start=doc_at),
        TocEntry(level=2, title="Alpha", page=1, char_start=alpha_at),
        TocEntry(level=2, title="Beta", page=1, char_start=beta_at),
    ]
    filler = "some body text goes here to give the section real length. " * 5
    text = f"# Doc\n\n{filler}\n\n## Alpha\n\n{filler}\n\n## Beta\n\n{filler}"
    ex = Extracted(text=text, page_offsets=[(0, 1)], toc=toc, title="Doc", page_count=1)
    sections = build_sections(ex)
    assert len(sections) == 3
    assert sections[1].path == "Doc › Alpha"
    assert sections[2].path == "Doc › Beta"


def test_context_prefix_composition():
    class S:
        path = "Methods › Training"

    assert build_context_prefix("Paper", S()) == "Paper › Methods › Training"
    assert build_context_prefix("Paper", S(), "about X") == "Paper › Methods › Training › about X"
    assert build_context_prefix("Paper", None) == "Paper"


def test_chunks_never_cross_section_boundaries():
    body = " ".join(f"word{i}" for i in range(4000))
    text = f"1 Alpha\n\n{body}\n\n2 Beta\n\n{body}"
    ex = Extracted(text=text, page_offsets=[(0, 1)], toc=[], title="T", page_count=1)
    sections = build_sections(ex)
    chunks = chunk_document(ex, sections, "T")
    assert chunks
    by_section: dict[int, set[str]] = {}
    for c in chunks:
        by_section.setdefault(c.section_index, set()).add(c.context_prefix)
    # One section => exactly one prefix, i.e. no chunk straddles two sections.
    for prefixes in by_section.values():
        assert len(prefixes) == 1


def test_chunks_respect_token_budget():
    body = " ".join(f"token{i}" for i in range(6000))
    ex = Extracted(text=body, page_offsets=[(0, 1)], toc=[], title="T", page_count=1)
    sections = build_sections(ex)
    chunks = chunk_document(ex, sections, "T")
    assert chunks
    # Allow headroom for the final merged runt, but nothing may be wildly oversized.
    assert max(c.token_count for c in chunks) < 1400
    assert count_tokens(chunks[0].text) == chunks[0].token_count


@pytest.mark.parametrize("bad", ["", "   \n\n  "])
def test_empty_documents_produce_no_chunks(bad):
    ex = Extracted(text=bad, page_offsets=[(0, 1)], toc=[], title="T", page_count=1)
    assert chunk_document(ex, build_sections(ex), "T") == []


def test_caps_heading_needs_a_real_word():
    """Figure labels ('BERT') and table headers ('E1 E2') must not become sections."""
    text = "intro\n\nBERT\n\nbody\n\nE1 E2\n\nbody\n\nRELATED WORK\n\nbody\n\nBERT BERT\n\nbody\n"
    titles = [t for _, _, t in detect_headings(text)]
    assert "Related Work" in titles
    for junk in ("Bert", "E1 E2", "Bert Bert"):
        assert junk not in titles, titles


def test_deoverlap_trims_repeated_tail():
    from library_agent.api.routes.documents import _deoverlap

    a = "The quick brown fox jumps over the lazy dog\nand keeps running through the field."
    # The carried-forward tail has had its whitespace normalised, as the chunker does.
    b = "over the lazy dog and keeps running through the field. Then it stopped to rest under a tree."
    out = _deoverlap([a, b])
    assert out[0] == a
    assert out[1] == "Then it stopped to rest under a tree."


def test_deoverlap_leaves_unrelated_chunks_alone():
    from library_agent.api.routes.documents import _deoverlap

    a, b = "Completely different opening sentence here.", "And an unrelated second passage."
    assert _deoverlap([a, b]) == [a, b]
