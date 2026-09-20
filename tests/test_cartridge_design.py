"""The cartridge's look: design clamping, art sanitising, the generated label, and the
round trip of both through a zip."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from conftest import BODY_A, make_document
from PIL import Image

from library_agent.library.cartridge import build_cartridge, import_cartridge, read_manifest
from library_agent.library.cartridge_design import (
    DEFAULT_DESIGN,
    clamp_design,
    decode_data_url,
    render_constellation,
    sanitize_art,
)


class TestDesign:
    def test_clamps_and_defaults(self):
        d = clamp_design(
            {"material": "glitter", "tint": 7, "opacity": -1, "sparkle": "x", "art": "upload"}
        )
        assert d["material"] == "glitter" and d["tint"] == 1.0 and d["opacity"] == 0.0
        assert d["sparkle"] == DEFAULT_DESIGN["sparkle"] and d["art"] == "upload"
        assert clamp_design(None) == DEFAULT_DESIGN
        assert clamp_design({"material": "lava"})["material"] == DEFAULT_DESIGN["material"]


    @pytest.mark.parametrize("finish", ["paper", "gloss", "holo", "prism", "gold", "chrome"])
    def test_label_finish_survives_serialization(self, finish):
        design = clamp_design({"material": "smoke", "labelFinish": finish, "labelFinishStrength": .37})
        assert clamp_design(json.loads(json.dumps(design))) == design
        assert design["labelFinish"] == finish
        assert design["labelFinishStrength"] == .37
        assert design["material"] == "smoke"

    def test_label_finish_defaults_and_limits(self):
        assert clamp_design({})["labelFinish"] == "paper"
        assert clamp_design({"labelFinish": "unknown"})["labelFinish"] == "paper"
        assert clamp_design({"labelFinish": "HOLO"})["labelFinish"] == "holo"
        for value, expected in [(-1, 0), (2, 1), ("0.4", .4), (None, .65), ("bad", .65), (float("nan"), .65), (float("inf"), .65)]:
            assert clamp_design({"labelFinishStrength": value})["labelFinishStrength"] == expected


class TestArt:
    def test_generated_label_is_a_png(self):
        png = render_constellation([(0.1, 0.2), (0.8, 0.7), (0.5, 0.5)], "#4f7a3a", size=256)
        im = Image.open(io.BytesIO(png))
        assert im.format == "PNG" and im.size == (256, 256)

    def test_upload_is_reencoded_and_capped(self):
        big = Image.new("RGB", (3000, 1500), (200, 30, 30))
        buf = io.BytesIO()
        big.save(buf, "JPEG")
        out = Image.open(io.BytesIO(sanitize_art(buf.getvalue())))
        assert out.format == "PNG" and max(out.size) <= 1024

    def test_non_image_is_refused(self):
        with pytest.raises(ValueError):
            sanitize_art(b"<svg onload=alert(1)></svg>")
        with pytest.raises(ValueError):
            sanitize_art(b"")

    def test_data_url(self):
        assert decode_data_url("data:image/png;base64,aGVsbG8=") == b"hello"
        with pytest.raises(ValueError):
            decode_data_url("javascript:alert(1)")


class TestRoundTrip:
    async def test_design_and_art_travel(self, db, tmp_path):
        a = await make_document(db, title="Alpha", body=BODY_A)
        png = Image.new("RGBA", (64, 64), (10, 200, 10, 255))
        buf = io.BytesIO()
        png.save(buf, "PNG")
        zip_path = await build_cartridge(
            db,
            document_ids=[a.id],
            level="readings",
            name="Looks",
            out_dir=tmp_path,
            design={"material": "smoke", "tint": 0.9, "opacity": 0.2, "art": "upload"},
            art_png=buf.getvalue(),
        )
        m = read_manifest(zip_path)
        assert m["design"]["material"] == "smoke" and m["design"]["tint"] == 0.9
        with zipfile.ZipFile(zip_path) as z:
            assert "art/label.png" in z.namelist()
        # A different library inserts it: the design is what the maker sealed in.
        from library_agent.db.purge import delete_document

        await delete_document(db, a.id)
        r = await import_cartridge(db, zip_path)
        from library_agent.db.models import Cartridge

        row = await db.get(Cartridge, r.cartridge_id)
        assert (
            row.design["material"] == "smoke"
            and row.art_path
            and row.art_path.endswith("label.png")
        )

    async def test_generated_art_by_default(self, db, tmp_path):
        a = await make_document(db, title="Alpha", body=BODY_A)
        zip_path = await build_cartridge(
            db, document_ids=[a.id], level="catalogue", name="Gen", out_dir=tmp_path
        )
        m = read_manifest(zip_path)
        assert m["design"]["art"] == "generated"
        with zipfile.ZipFile(zip_path) as z:
            im = Image.open(io.BytesIO(z.read("art/label.png")))
            assert im.format == "PNG"

    async def test_bad_shipped_art_is_dropped_not_fatal(self, db, tmp_path):
        a = await make_document(db, title="Alpha", body=BODY_A)
        zip_path = await build_cartridge(
            db, document_ids=[a.id], level="readings", name="Bad", out_dir=tmp_path
        )
        bad = tmp_path / "bad-art.zip"
        from library_agent.library import cartridge as cart

        parts = []
        with zipfile.ZipFile(zip_path) as src:
            for n in src.namelist():
                data = b"not a picture" if n == "art/label.png" else src.read(n)
                if n not in ("cartridge.json", "icon.svg"):
                    parts.append((n, data))
            m = json.loads(src.read("cartridge.json"))
        m["content_hash"] = cart._digest(parts)
        with zipfile.ZipFile(bad, "w") as dst:
            dst.writestr("cartridge.json", json.dumps(m))
            for n, data in parts:
                dst.writestr(n, data)
        from library_agent.db.purge import delete_document

        await delete_document(db, a.id)
        r = await import_cartridge(db, bad)
        from library_agent.db.models import Cartridge

        assert (await db.get(Cartridge, r.cartridge_id)).art_path is None


class TestClearance:
    def test_clamps(self):
        assert clamp_design({"clearance": "restricted"})["clearance"] == "restricted"
        assert clamp_design({"clearance": "top secret"})["clearance"] == "open"

    async def test_restricted_volumes_do_not_reexport(self, db, tmp_path):
        """A volume that arrived only inside a restricted cartridge is left out of any
        selection for a new cartridge on the receiving library."""
        from library_agent.db.purge import delete_document
        from library_agent.library.cartridge import resolve_selection

        a = await make_document(db, title="Alpha", body=BODY_A)
        zip_path = await build_cartridge(
            db,
            document_ids=[a.id],
            level="readings",
            name="R",
            out_dir=tmp_path,
            design={"clearance": "restricted"},
        )
        await delete_document(db, a.id)
        r = await import_cartridge(db, zip_path)
        from library_agent.db.models import Document

        back = (
            await db.execute(
                __import__("sqlalchemy").select(Document).where(Document.title == "Alpha")
            )
        ).scalar_one()
        assert await resolve_selection(db, document_ids=[back.id]) == []
        assert await resolve_selection(db, cartridge_ids=[r.cartridge_id]) == []


async def test_a_cartridge_made_here_is_editable_and_exports_as_itself(db, tmp_path, monkeypatch):
    """The maker may change a cartridge made on this machine; exporting it keeps its id
    and moves the version on, so a receiver upgrades in place."""
    import uuid
    from datetime import UTC, datetime

    from library_agent import config
    from library_agent.db.models import Cartridge, CartridgeDocument

    monkeypatch.setattr(config.settings(), "storage_dir", tmp_path / "documents")
    doc = await make_document(db, title="Runbook", body=BODY_A)
    cid = uuid.uuid4()
    db.add(
        Cartridge(
            id=cid,
            name="Ops",
            slug="ops",
            version=1,
            colour="#4f7a3a",
            made_by="import",
            made_at=datetime.now(UTC),
            level="full",
            embed_model=config.settings().embed_model,
            manifest={"origin": "import"},
            content_hash="0" * 64,
            document_count=1,
        )
    )
    db.add(CartridgeDocument(cartridge_id=cid, document_id=doc.id, introduced=True))
    await db.flush()

    from library_agent.library.cartridge import list_cartridges

    listed = next(c for c in await list_cartridges(db) if c["id"] == str(cid))
    assert listed["editable"] is True

    path = await build_cartridge(
        db,
        document_ids=[doc.id],
        level="full",
        name="Ops",
        colour="#4f7a3a",
        cartridge_id=cid,
        version=2,
        design={"material": "glitter", "clearance": "internal"},
    )
    m = read_manifest(path)
    assert m["id"] == str(cid) and m["version"] == 2
    assert m["design"]["material"] == "glitter" and m["design"]["clearance"] == "internal"


async def test_a_restricted_cartridge_made_here_still_exports(db, tmp_path, monkeypatch):
    """Restricted binds the receiver. The maker marked it, and may still ship it."""
    import uuid
    from datetime import UTC, datetime

    from library_agent import config
    from library_agent.db.models import Cartridge, CartridgeDocument
    from library_agent.library.cartridge import resolve_selection

    monkeypatch.setattr(config.settings(), "storage_dir", tmp_path / "documents")
    doc = await make_document(db, title="Runbook", body=BODY_A)
    cid = uuid.uuid4()
    db.add(
        Cartridge(
            id=cid,
            name="Ops",
            slug="ops",
            version=1,
            colour="#4f7a3a",
            made_by="import",
            made_at=datetime.now(UTC),
            level="full",
            embed_model=config.settings().embed_model,
            manifest={"origin": "import"},
            content_hash="0" * 64,
            document_count=1,
            design={"clearance": "restricted"},
        )
    )
    db.add(CartridgeDocument(cartridge_id=cid, document_id=doc.id, introduced=True))
    await db.flush()
    assert await resolve_selection(db, cartridge_ids=[cid]) == [doc.id]
