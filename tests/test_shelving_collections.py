"""Each collection gets its own shelves, and a volume is placed only among them."""

from __future__ import annotations

import uuid

from conftest import make_document
from sqlalchemy import select

from library_agent.db.models import Cartridge, CartridgeDocument, Category, Document
from library_agent.library import shelving, taxonomy


class Designer:
    """A model that designs one shelf per collection by name, and always places a
    volume on the first sub-shelf it is offered."""

    def __init__(self):
        self.offered: list[list[str]] = []

    async def structured(self, model, prompt, schema, **kw):
        props = schema.get("properties", {})
        if "shelves" in props:
            if "Wiki" in prompt:
                return {
                    "notes": "ops",
                    "shelves": [{"name": "IT Operations", "sub_shelves": ["Storage", "Accounts"]}],
                }
            return {
                "notes": "sec",
                "shelves": [{"name": "Security", "sub_shelves": ["Web Attacks", "Storage"]}],
            }
        if "assignments" in props:
            return {"assignments": []}
        subs = [s for s in props["sub_shelf"]["enum"] if s != shelving.NEW]
        self.offered.append(subs)
        return {"why": "x", "sub_shelf": subs[0], "top_shelf": props["top_shelf"]["enum"][0]}

    async def aclose(self):
        pass


async def _cartridge(db, name, docs, description=None):
    c = Cartridge(
        id=uuid.uuid4(),
        name=name,
        slug=name.lower(),
        version=1,
        colour="#333",
        level="full",
        embed_model="bge-m3",
        content_hash="0" * 64,
        description=description,
    )
    db.add(c)
    await db.flush()
    for d in docs:
        db.add(CartridgeDocument(cartridge_id=c.id, document_id=d.id))
    await db.flush()
    return c


async def test_each_collection_shelved_in_its_own_terms(scratch_db, monkeypatch):
    db = scratch_db
    monkeypatch.setattr(shelving, "DESIGN_MIN", 2)
    wiki = [
        await make_document(db, title=f"Wiki page {i}", body=f"wiki {i} " * 30, tier=1)
        for i in range(3)
    ]
    sec = [
        await make_document(db, title=f"Pentest {i}", body=f"pentest {i} " * 30, tier=1)
        for i in range(3)
    ]
    await _cartridge(db, "Wiki", wiki, "The IT team's internal wiki.")
    await _cartridge(db, "Handbook", sec)
    d = Designer()
    tx = await shelving.build_taxonomy(db, client=d)
    assert set(tx.tops) == {"IT Operations", "Security"}
    # "Storage" was proposed by both; the second is dropped rather than made ambiguous
    assert sum("Storage" in subs for subs in tx.tops.values()) == 1
    assert "Accounts" in tx.tops["IT Operations"] and "Web Attacks" in tx.tops["Security"]
    # each top shelf knows whose it is
    assert len(tx.homes["IT Operations"]) == 1 and len(tx.homes["Security"]) == 1

    res = await shelving.reshelve(db, client=d, rebuild=False)
    assert res.unplaced == 0
    # every volume was offered only its own collection's shelves, and sits on one
    ops, secs = set(tx.tops["IT Operations"]), set(tx.tops["Security"])
    for w in wiki:
        shelf = await db.get(Category, (await db.get(Document, w.id)).shelf_id)
        assert shelf.name in ops
    for s in sec:
        shelf = await db.get(Category, (await db.get(Document, s.id)).shelf_id)
        assert shelf.name in secs
    assert all(set(o) <= ops or set(o) <= secs for o in d.offered)

    # the homes survive a reload of the taxonomy
    again = await shelving.load_taxonomy(db)
    assert again.homes == tx.homes


async def test_designate_takes_a_name_literally(scratch_db):
    db = scratch_db
    keep = await taxonomy.get_or_create(db, "Network Services")
    old = await taxonomy.get_or_create(db, "Security")
    await shelving._fold(db, old, keep)
    # a tag lookup follows the fold; a design does not
    assert (await taxonomy.get_or_create(db, "Security")).id == keep.id
    lit = await taxonomy.designate(db, "Security")
    assert lit.id == old.id and lit.canonical
    assert (await taxonomy.designate(db, "Brand New")).name == "Brand New"
    assert len(list((await db.execute(select(Category))).scalars())) == 3
