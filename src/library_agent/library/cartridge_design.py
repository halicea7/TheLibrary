"""A cartridge's look: its design and its art.

The design is a small dict -- material preset plus dials -- that the maker sets and the
manifest carries, so a cartridge is the same object on every rack it lands on. It is
validated and clamped here on the way in and on the way out; nothing downstream trusts
raw values.

Art is the label on the front. Either the maker uploads an image, or the library draws
one: the cartridge's own constellation -- its volumes laid out by their document vectors
and knitted by nearest neighbours, in the cartridge's colour. Uploaded art crosses the
same trust boundary as everything else in a zip, so it is re-encoded through Pillow
(metadata gone, size capped, anything that does not decode refused)."""

from __future__ import annotations

import base64
import io
import math
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from library_agent.config import settings

MATERIALS = ("solid", "clear", "frosted", "smoke", "glitter", "metallic")
LABEL_FINISHES = ("paper", "gloss", "holo", "prism", "gold", "chrome")
# Clearance is a marking, sealed in like the rest of the design: it says how the maker
# meant the cartridge to travel. `restricted` is also enforced at one point -- the
# receiving library will not re-export a restricted cartridge's volumes into another.
CLEARANCES = ("open", "internal", "confidential", "restricted")
ART_MAX_PX = 1024
ART_MAX_BYTES = 2_000_000

DEFAULT_DESIGN: dict[str, Any] = {
    "material": "clear",
    "tint": 0.55,  # how much of the colour the plastic carries
    "opacity": 0.35,  # 0 clear .. 1 solid (matters for clear/smoke/glitter)
    "sparkle": 0.5,  # glitter density
    "roughness": 0.25,
    "labelFinish": "paper",
    "labelFinishStrength": 0.65,
    "foilMode": "artwork",
    "art": "generated",  # generated | upload
    "clearance": "open",
}


def clamp_design(raw: Any) -> dict[str, Any]:
    d = dict(DEFAULT_DESIGN)
    if isinstance(raw, dict):
        m = str(raw.get("material") or d["material"]).lower()
        d["material"] = m if m in MATERIALS else d["material"]
        for k in ("tint", "opacity", "sparkle", "roughness"):
            try:
                d[k] = round(min(1.0, max(0.0, float(raw.get(k, d[k])))), 3)
            except (TypeError, ValueError):
                pass
        finish = str(raw.get("labelFinish") or d["labelFinish"]).lower()
        d["labelFinish"] = finish if finish in LABEL_FINISHES else d["labelFinish"]
        try:
            strength = float(raw.get("labelFinishStrength", d["labelFinishStrength"]))
            if math.isfinite(strength):
                d["labelFinishStrength"] = round(min(1.0, max(0.0, strength)), 3)
        except (TypeError, ValueError, OverflowError):
            pass
        mode = raw.get("foilMode", "artwork")
        d["foilMode"] = mode if mode in ("artwork", "subject", "reverse") else "artwork"
        mask = raw.get("foilMask")
        if isinstance(mask, str) and len(mask) <= 1_500_000:
            try:
                image = Image.open(io.BytesIO(decode_data_url(mask)))
                if image.width * image.height <= 4_000_000:
                    image.load()
                    rgba = image.convert("RGBA")
                    grey = Image.new("L", image.size, 0)
                    grey.paste(rgba.convert("L"), mask=rgba.getchannel("A"))
                    grey.thumbnail((512, 512))
                    data = io.BytesIO()
                    grey.save(data, "PNG", optimize=True)
                    d["foilMask"] = "data:image/png;base64," + base64.b64encode(
                        data.getvalue()
                    ).decode("ascii")
            except Exception:  # noqa: BLE001, S110 - an unreadable mask is simply dropped
                pass
        d["art"] = "upload" if raw.get("art") == "upload" else "generated"
        c = str(raw.get("clearance") or d["clearance"]).lower()
        d["clearance"] = c if c in CLEARANCES else d["clearance"]
    return d


# --------------------------------------------------------------------------- art


def _hex(colour: str) -> tuple[int, int, int]:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", colour or "")
    v = int(m.group(1), 16) if m else 0x2F6F8F
    return (v >> 16) & 255, (v >> 8) & 255, v & 255


async def constellation_points(
    db: AsyncSession, document_ids: list[uuid.UUID]
) -> list[tuple[float, float]]:
    """Volumes laid out in 2D from their document vectors (PCA of whatever vector each has
    -- the Tier 1 summary if read, the Tier 0 fingerprint if not)."""
    if not document_ids:
        return []
    rows = (
        await db.execute(
            text(
                """
                with vec as (
                    select coalesce(a.target_id, e.owner_id) as document_id, e.vec
                    from embedding e
                    left join artifact a on a.id = e.owner_id and a.kind = 'document_summary'
                    where e.model = :m
                      and ((e.owner_kind = 'artifact' and a.id is not null) or e.owner_kind = 'document')
                )
                select distinct on (document_id) document_id, cast(vec as text) as v
                from vec where document_id = any(cast(:ids as uuid[]))
                order by document_id
                """
            ),
            {"m": settings().embed_model, "ids": [str(x) for x in document_ids]},
        )
    ).all()
    if len(rows) < 2:
        return [(0.5, 0.5)] * len(rows)
    import json

    X = np.asarray([json.loads(r.v) for r in rows], dtype=np.float32)
    X -= X.mean(axis=0)
    # Two principal directions are enough for a label; SVD on a few hundred rows is instant.
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    P = X @ vt[:2].T
    lo, hi = P.min(axis=0), P.max(axis=0)
    span = np.where(hi - lo > 1e-6, hi - lo, 1.0)
    P = (P - lo) / span
    return [(float(x), float(y)) for x, y in P]


def render_constellation(
    points: list[tuple[float, float]], colour: str, *, size: int = 768
) -> bytes:
    """The generated label: a dark field in the cartridge's colour, the volumes as points
    with a soft glow, the nearest few knitted. Deterministic for the same volumes."""
    r, g, b = _hex(colour)
    bg = (int(r * 0.13), int(g * 0.13), int(b * 0.13))
    img = Image.new("RGB", (size, size), bg)
    glow = Image.new("RGB", (size, size), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    pad = size * 0.12
    pts = [(pad + x * (size - 2 * pad), pad + y * (size - 2 * pad)) for x, y in points]
    # edges: three nearest per point, faint
    d = ImageDraw.Draw(img)
    if len(pts) > 1:
        arr = np.asarray(pts)
        for i, p in enumerate(arr):
            dist = np.linalg.norm(arr - p, axis=1)
            for j in np.argsort(dist)[1:4]:
                if dist[j] < size * 0.28:
                    d.line(
                        [tuple(p), tuple(arr[j])],
                        fill=(int(r * 0.35), int(g * 0.35), int(b * 0.35)),
                        width=1,
                    )
    for x, y in pts:
        rad = 9
        gd.ellipse(
            [x - rad * 3, y - rad * 3, x + rad * 3, y + rad * 3],
            fill=(int(r * 0.6), int(g * 0.6), int(b * 0.6)),
        )
    glow = glow.filter(ImageFilter.GaussianBlur(size * 0.03))
    img = Image.blend(
        img, Image.composite(glow, img, glow.convert("L").point(lambda v: min(255, v * 2))), 0.55
    )
    d = ImageDraw.Draw(img)
    for x, y in pts:
        d.ellipse(
            [x - 3.2, y - 3.2, x + 3.2, y + 3.2],
            fill=(min(255, r + 90), min(255, g + 90), min(255, b + 90)),
        )
    out = io.BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue()


def sanitize_art(data: bytes) -> bytes:
    """Re-encode uploaded or shipped art as a plain PNG: no metadata, no surprises, capped
    at ART_MAX_PX on the long side. Raises ValueError for anything that is not an image."""
    if not data or len(data) > ART_MAX_BYTES * 4:
        raise ValueError("art is empty or far too large")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:
        raise ValueError("art is not an image the library can read") from exc
    if im.width * im.height > 40_000_000:
        raise ValueError("art is too large")
    im = im.convert("RGBA")
    im.thumbnail((ART_MAX_PX, ART_MAX_PX))
    out = io.BytesIO()
    im.save(out, "PNG", optimize=True)
    png = out.getvalue()
    if len(png) > ART_MAX_BYTES:
        # Try again smaller rather than refuse a big but honest picture.
        im.thumbnail((512, 512))
        out = io.BytesIO()
        im.save(out, "PNG", optimize=True)
        png = out.getvalue()
    return png


def decode_data_url(s: str) -> bytes:
    """`data:image/png;base64,...` from the make panel."""
    m = re.match(r"^data:image/[a-z0-9.+-]+;base64,(.+)$", s or "", re.DOTALL)
    if not m:
        raise ValueError("art must be an image data URL")
    return base64.b64decode(m.group(1), validate=False)


def art_dir(cartridge_id: uuid.UUID) -> Path:
    return settings().storage_dir.parent / "cartridges" / "art" / str(cartridge_id)


def store_art(cartridge_id: uuid.UUID, png: bytes) -> Path:
    d = art_dir(cartridge_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "label.png"
    p.write_bytes(png)
    return p
