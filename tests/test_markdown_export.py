"""Markdown as Google Docs and Pandoc export it: headings with anchors, images inside."""

from __future__ import annotations

import base64
import io

from PIL import Image

from library_agent.ingest import figures
from library_agent.ingest.markdown import (
    FIGURE_MARKER,
    clean_heading,
    embedded_images,
    is_generic_title,
    tidy,
)


def _png(colour: str) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (4, 3), colour).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


DOC = f"""![][image1]

**Table of Contents**

[**Overview**](#overview)\t**[1](#overview)**

# **Overview** {{#overview}}

Open the console. ![][image2]

#{" "}

## __Request access__ {{#request-access}}

Fill in the form ![the form](data:image/png;base64,{_png("blue")}) and submit.

[image1]: <data:image/png;base64,{_png("red")}>
[image2]: <data:image/png;base64,{_png("green")}>
"""


def test_headings_lose_their_anchors_and_emphasis():
    assert clean_heading("**Overview** {#overview}") == "Overview"
    assert clean_heading("__Request access__ {#request-access}") == "Request access"
    out = tidy(DOC)
    assert "# Overview\n" in out and "## Request access\n" in out
    assert "{#" not in out.split("Table of Contents")[1].split("[**Overview**]")[0]
    assert "\n#\n" not in out and "\n# \n" not in out  # the empty heading is gone
    assert is_generic_title("Overview") and not is_generic_title("Tufin How-To Guide")


def test_embedded_images_become_numbered_figures_where_they_sat():
    out = tidy(DOC)
    assert "base64" not in out and "data:image" not in out
    markers = [int(m.group(1)) for m in FIGURE_MARKER.finditer(out)]
    assert markers == [1, 2, 3]  # in the order the text uses them
    imgs = embedded_images(DOC)
    assert [(e.n, e.alt) for e in imgs] == [(1, ""), (2, ""), (3, "the form")]
    assert "(Figure 3: the form)" in out


def test_a_markdown_files_figures_are_decoded_into_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "figures_dir", lambda h: tmp_path / "figs" / h)
    md = tmp_path / "guide.md"
    md.write_text(DOC)
    figs = figures.index("abc", md)
    assert [f.n for f in figs] == [1, 2, 3] and figs[2].caption == "the form"
    png = figures.render("abc", md, 2)
    assert png and png.exists()
    with Image.open(png) as im:
        assert im.size == (4, 3) and im.getpixel((0, 0))[:3] == (0, 128, 0)
