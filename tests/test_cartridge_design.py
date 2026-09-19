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
