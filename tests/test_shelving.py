"""The two-level shelf: the gate that rejects bad designs, and the pieces that need a database."""

from __future__ import annotations

from sqlalchemy import select

from library_agent.db.models import Category, Document, DocumentStatus
from library_agent.library import taxonomy
from library_agent.library.shelving import (
    Taxonomy,
    _taxonomy_is_sane,
    expand_category_ids,
    load_taxonomy,
    place_schema,
)


def design(*tops):
    return {"shelves": [{"name": n, "sub_shelves": list(subs)} for n, subs in tops]}


class TestGate:
    def test_accepts_a_good_shelf(self):
        d = design(
            ("Cybersecurity", ["Web Exploitation", "Network Scanning"]),
            ("Machine Learning", ["Language Models", "Retrieval"]),
        )
        assert _taxonomy_is_sane(d, 4)

    def test_rejects_one_field_sliced_in_two(self):
        d = design(("Security Vulnerabilities", ["a", "b"]), ("Security Tools", ["c", "d"]))
        assert not _taxonomy_is_sane(d, 4)
        d = design(("Security", ["a", "b"]), ("Cybersecurity", ["c", "d"]))
        assert not _taxonomy_is_sane(d, 4)

    def test_rejects_activities_and_filler(self):
        assert not _taxonomy_is_sane(design(("Research", ["a", "b"]), ("Biology", ["c", "d"])), 4)
        assert not _taxonomy_is_sane(
            design(("OSCP Preparation", ["a", "b"]), ("Biology", ["c", "d"])), 4
        )
        assert not _taxonomy_is_sane(
            design(("Top Shelf 1", ["Sub-Shelf 1", "Sub-Shelf 2"]), ("Biology", ["c", "d"])), 4
        )

    def test_rejects_a_bay_with_one_sub_shelf(self):
        assert not _taxonomy_is_sane(
            design(("Biology", ["Genetics"]), ("Economics", ["a", "b"])), 4
        )

    def test_machine_learning_is_not_an_activity(self):
        """'learning' was once on the banned list; Machine Learning was rejected every time."""
        assert _taxonomy_is_sane(
            design(("Machine Learning", ["a", "b"]), ("Biology", ["c", "d"])), 4
        )


class TestNames:
    def test_acronyms_survive_normalisation(self):
        assert taxonomy.normalize_name("NLP systems") == "NLP Systems"
        assert taxonomy.normalize_name("neural IR") == "Neural IR"
        assert taxonomy.normalize_name("machine learning") == "Machine Learning"

    def test_place_schema_is_finite(self):
        tx = Taxonomy(tops={"Cybersecurity": ["Web Exploitation", "Network Scanning"]})
        s = place_schema(tx)
        assert s["properties"]["sub_shelf"]["enum"][-1] == "(a new sub-shelf)"
        assert "new_sub_shelf_name" not in s["required"]
        assert place_schema(tx, allow_new=False)["properties"]["sub_shelf"]["enum"] == [
            "Network Scanning",
            "Web Exploitation",
        ]


class TestWithDatabase:
    async def test_load_and_expand(self, db):
        top = Category(name="Test Bay")
        db.add(top)
        await db.flush()
        a = Category(name="Test Run A", parent_id=top.id)
        b = Category(name="Test Run B", parent_id=top.id)
        db.add_all([a, b])
        await db.flush()
        tx = await load_taxonomy(db)
        assert tx.tops["Test Bay"] == ["Test Run A", "Test Run B"]
        # Scoping to the bay scopes to everything beneath it.
        assert set(await expand_category_ids(db, [top.id])) == {top.id, a.id, b.id}

    async def test_folded_name_resolves_to_survivor(self, db):
        winner = Category(name="Test Winner", merged_from=["Test Loser"])
        loser = Category(name="Test Loser", canonical=False)
        db.add_all([winner, loser])
        await db.flush()
        got = await taxonomy.get_or_create(db, "Test Loser")
        assert got is not None and got.id == winner.id

    async def test_a_volume_has_one_shelf(self, db):
        top = Category(name="Test Bay")
        db.add(top)
        await db.flush()
        sub = Category(name="Test Run", parent_id=top.id)
        db.add(sub)
        await db.flush()
        doc = Document(
            content_hash="a" * 64,
            title="T",
            source_path="",
            original_filename="t.md",
            status=DocumentStatus.READY,
            shelf_id=sub.id,
        )
        db.add(doc)
        await db.flush()
        got = (
            await db.execute(select(Document.shelf_id).where(Document.id == doc.id))
        ).scalar_one()
        assert got == sub.id
