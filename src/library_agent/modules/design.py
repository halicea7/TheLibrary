"""A token's design: checked field by field before it is kept.

The browser sends the whole design; only known fields of the right kind are kept, and
the logo arrives as the traced mask (never the uploaded image), which is decoded and
re-encoded here so what is stored is a plain greyscale PNG whatever was sent."""

from __future__ import annotations

import base64
import binascii
import io
import re

MASK_MAX_BYTES = 400_000
MASK_MAX_SIDE = 1024

_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")
CHOICES = {
    "material": {"solid", "metal", "gold", "clear", "frosted", "glitter", "ceramic"},
    "logoFinish": {"match", "color", "holo", "gold", "chrome"},
    "faceFinish": {"match", "color", "holo", "gold", "chrome"},
    "solarFinish": {"etched", "color", "holo", "gold", "chrome"},
    "edgeFinish": {"satin", "polished", "none"},
}
COLOURS = {
    "color",
    "logoColor",
    "faceColor",
    "solarColor",
    "rimColor",
    "errorColor",
    "identityColor",
    "edgeColor",
    "boardColor",
    "circuitColor",
}
FLAGS = {"rim", "identity", "contacts", "dockingKey", "planets"}
NUMBERS = {"tint": (0.0, 1.0), "height": (0.025, 0.2), "brightness": (0.0, 3.0), "planet": (-1, 7)}
TEXTS = {"name": 24, "serial": 16}


class DesignError(ValueError):
    pass


def clean_mask(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    m = re.match(r"^data:image/(png|webp|jpeg);base64,([A-Za-z0-9+/=]+)$", value or "")
    if not m:
        raise DesignError("the logo must be an image data URL")
    try:
        raw = base64.b64decode(m.group(2), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DesignError("the logo is not valid base64") from exc
    if len(raw) > MASK_MAX_BYTES:
        raise DesignError("the logo mask is too large")
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(raw))
        if max(img.size) > MASK_MAX_SIDE:
            raise DesignError(f"the logo mask is over {MASK_MAX_SIDE}px")
        img = img.convert("L")  # a mask: brightness is all it carries
    except (UnidentifiedImageError, OSError) as exc:
        raise DesignError("the logo could not be decoded") from exc
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def clean(design: dict) -> dict:
    """The fields of a design worth keeping; anything unknown or malformed is dropped, a
    bad logo is refused."""
    if not isinstance(design, dict):
        raise DesignError("a design is an object")
    out: dict = {}
    for k, allowed in CHOICES.items():
        if design.get(k) in allowed:
            out[k] = design[k]
    for k in COLOURS:
        if isinstance(design.get(k), str) and _COLOUR.match(design[k]):
            out[k] = design[k].lower()
    for k in FLAGS:
        if isinstance(design.get(k), bool):
            out[k] = design[k]
    for k, (lo, hi) in NUMBERS.items():
        v = design.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = min(hi, max(lo, int(v) if k == "planet" else float(v)))
    for k, n in TEXTS.items():
        if isinstance(design.get(k), str):
            out[k] = "".join(ch for ch in design[k] if ch.isprintable())[:n]
    if "mask" in design:
        out["mask"] = clean_mask(design.get("mask"))
    return out
