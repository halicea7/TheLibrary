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
        # the composition carries a stable id, stamped in the colophon
        assert c.id and f"The Library · {c.id}" in md
        import uuid as _uuid

        _uuid.UUID(c.id)  # a real uuid

    def test_slug_and_lengths(self):
        assert slugify("RAG Reading Guide: Core Concepts!") == "rag-reading-guide-core-concepts"
        assert set(LENGTHS) == {"short", "medium", "long", "report", "thesis"}

    def test_save_and_list(self, tmp_path, monkeypatch):
        from library_agent.config import settings

        monkeypatch.setattr(settings(), "storage_dir", tmp_path / "documents")
        p = save_markdown("Hello World", "# Hello World\n\nbody\n")
        assert p.exists() and p.name.startswith("hello-world-") and p.suffix == ".md"
        rows = list_saved()
        assert rows and rows[0]["title"] == "Hello World" and rows[0]["file"] == p.name


class TestArgumentMemory:
    def test_established_flat_is_the_section_ledger(self):
        from library_agent.chat.compose import _established

        secs = [
            {"heading": "One", "chapter": None, "takeaway": "Recursion terminates on a base case."},
            {"heading": "Two", "chapter": None, "takeaway": "Tail calls need no stack frame."},
        ]
        # writing a third flat section: it sees both prior takeaways, none as a chapter
        block = _established(secs, {}, None)
        assert "§ One: Recursion terminates" in block
        assert "§ Two: Tail calls" in block
        assert "Chapter" not in block
        # the very first section has nothing yet
        assert "opening section" in _established([], {}, None)

    def test_established_nested_shows_closed_chapters_then_current(self):
        from library_agent.chat.compose import _established

        secs = [
            {"heading": "A1", "chapter": "Foundations", "takeaway": "Sets are primitive."},
            {"heading": "A2", "chapter": "Foundations", "takeaway": "Functions map sets."},
            {"heading": "B1", "chapter": "Structures", "takeaway": "Groups have inverses."},
        ]
        theses = {"Foundations": "The basic objects are sets and maps between them."}
        # writing a second section of chapter "Structures"
        block = _established(secs, theses, "Structures")
        assert "Chapter «Foundations» established: The basic objects" in block
        assert "§ B1: Groups have inverses" in block
        # the current chapter's own thesis, and the finished chapter's sections, are not repeated
        assert "§ A1:" not in block and "Chapter «Structures»" not in block

    def test_markdown_nests_chapters_as_subsections(self):
        from library_agent.chat.compose import Composition

        c = Composition(title="Book", brief="b")
        c.sections = [
            {"heading": "A1", "chapter": "Part I", "body": "x", "cited": []},
            {"heading": "A2", "chapter": "Part I", "body": "y", "cited": []},
            {"heading": "B1", "chapter": "Part II", "body": "z", "cited": []},
        ]
        md = c.markdown()
        assert "## Part I" in md and "## Part II" in md
        assert "### A1" in md and "### B1" in md
        # one chapter heading each, not one per section
        assert md.count("## Part I\n") == 1 and md.count("## Part II\n") == 1


class TestLabelSanitizing:
    """Section paths from HTML/wiki ingestion keep markup; it must not leak into a
    reference list, but code and paths must survive."""

    def test_strips_markup_keeps_code_and_paths(self):
        from library_agent.chat.citations import plain_label

        assert plain_label("🔧 **Learning Modules Overview**") == "🔧 Learning Modules Overview"
        assert plain_label("Strapi `where` smuggling") == "Strapi where smuggling"
        assert plain_label("<code>x</code> and <b>y</b>") == "x and y"
        assert plain_label("~~gone~~ kept") == "gone kept"
        # a lone glob, snake_case and a path are not markdown and stay intact
        assert plain_label("LFI via /proc/*/fd") == "LFI via /proc/*/fd"
        assert plain_label("Insecure Randomness › mt_rand") == "Insecure Randomness › mt_rand"
        assert plain_label("AF_UNIX MSG_OOB") == "AF_UNIX MSG_OOB"
        assert plain_label(None) == "" and plain_label("  a   b ") == "a b"

    def test_references_are_clean(self):
        from library_agent.chat.citations import Source
        from library_agent.chat.compose import Composition

        c = Composition(title="G", brief="b")
        c.sources = [
            Source(
                n=1,
                chunk_id="c1",
                document_id="d1",
                document_title="Wiki",
                section_path="Ops › **Overview** › `run`",
                page=4,
            ),
        ]
        c.sections = [{"heading": "One", "covers": "", "body": "x [1].", "cited": [1]}]
        md = c.markdown()
        assert "[1] Wiki, p.4 — Ops › Overview › run" in md
        assert "**" not in md and "`run`" not in md


class TestReviewFlags:
    def test_flags_render_as_caveats_under_their_section(self):
        from library_agent.chat.compose import Composition

        c = Composition(title="G", brief="b")
        c.sections = [
            {
                "heading": "One",
                "covers": "",
                "body": "Salting stops all attacks [1].",
                "cited": [1],
                "flags": [
                    {
                        "quote": "Salting stops all attacks",
                        "issue": "the passage covers storage, not the whole auth flow",
                        "condition": "the login path is itself injectable",
                    }
                ],
            },
            {
                "heading": "Two",
                "covers": "",
                "body": "A measured claim [1].",
                "cited": [1],
                "flags": [],
            },
        ]
        from library_agent.chat.citations import Source

        c.sources = [
            Source(
                n=1, chunk_id="c1", document_id="d1", document_title="T", section_path="", page=1
            )
        ]
        md = c.markdown()
        assert "> ⚠ **Review** — “Salting stops all attacks”: the passage covers storage" in md
        assert "It could fail if the login path is itself injectable" in md
        # the well-supported section gets no caveat
        assert md.count("⚠ **Review**") == 1

    async def test_review_flags_only_genuine_overreach(self, scratch_db):
        from library_agent.chat.compose import _review

        class Client:
            async def structured(self, model, prompt, schema, **kw):
                # a model that finds one overreach
                return {
                    "flags": [
                        {
                            "quote": "X is impossible",
                            "issue": "sources say hard, not impossible",
                            "condition": "a weak key is used",
                        },
                        {"quote": "", "issue": "dropped: no quote"},  # incomplete, filtered out
                    ]
                }

        flags = await _review(Client(), "m", "H", "A" * 200, "[1] passage text")
        assert len(flags) == 1 and flags[0]["quote"] == "X is impossible"
        assert flags[0]["condition"] == "a weak key is used"
        # too-short a body is not reviewed at all
        assert await _review(Client(), "m", "H", "tiny", "[1] x") == []
