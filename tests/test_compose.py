"""Composition: the document assembles with global numbering and a references list, and
the parts that do not need a model."""

from __future__ import annotations

from library_agent.chat.citations import Source
from library_agent.chat.compose import LENGTHS, Composition, list_saved, save_markdown, slugify


def _src(n, title, page=None, cart=None):
    return Source(
        n=n,
        chunk_id=f"c{n}",
        document_id=f"d{n}",
        document_title=title,
        section_path="",
        page=page,
        cartridge=cart,
    )


class TestAssembly:
    def test_markdown_has_title_sections_and_only_cited_references(self):
        c = Composition(title="A Guide", brief="b")
        c.sources = [
            _src(1, "Alpha", 3),
            _src(2, "Beta"),
            _src(3, "Gamma", 9, {"id": "x", "name": "Ops", "colour": "#000"}),
        ]
        c.sections = [
            {"heading": "One", "covers": "", "body": "First [1].", "cited": [1]},
            {"heading": "Two", "covers": "", "body": "Second [3] and again [1].", "cited": [3, 1]},
        ]
        md = c.markdown()
        assert md.startswith("# A Guide\n")
        assert "## One" in md and "## Two" in md
        assert "## References" in md
        assert "[1] Alpha, p.3" in md and "[3] Gamma, p.9 (Ops)" in md
        assert "[2] Beta" not in md  # retrieved, never cited: not a reference
        assert "unmarked claims are the librarian's own synthesis" in md

    def test_slug_and_lengths(self):
        assert slugify("RAG Reading Guide: Core Concepts!") == "rag-reading-guide-core-concepts"
        assert set(LENGTHS) == {"short", "medium", "long"}

    def test_save_and_list(self, tmp_path, monkeypatch):
        from library_agent.config import settings

        monkeypatch.setattr(settings(), "storage_dir", tmp_path / "documents")
        p = save_markdown("Hello World", "# Hello World\n\nbody\n")
        assert p.exists() and p.name.startswith("hello-world-") and p.suffix == ".md"
        rows = list_saved()
        assert rows and rows[0]["title"] == "Hello World" and rows[0]["file"] == p.name
