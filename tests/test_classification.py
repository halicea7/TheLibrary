"""Classification: detection of markings, ceilings, and the marks on what the library writes."""

from __future__ import annotations

import pytest

from library_agent import classification as cls
from library_agent.chat.citations import Source
from library_agent.chat.compose import Composition


@pytest.fixture(autouse=True)
def scale_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBRARY_CLASSIFICATION_FILE", str(tmp_path / "classification.json"))


US = cls.preset("us")
CO = cls.Scale()


def test_banner_lines_mark_a_volume_and_words_mid_sentence_do_not():
    text = "SECRET//NOFORN\n\nThe plan.\n\nUNCLASSIFIED\n"
    assert cls.detect_document(US, text) == "S"
    assert cls.detect_document(US, "Classification: TOP SECRET//SI") == "TS"
    assert cls.detect_document(US, "This secret recipe is confidential.") is None
    assert cls.detect_document(CO, "== COMPANY CONFIDENTIAL ==\nQ3 numbers") == "confidential"
    assert cls.detect_document(CO, "Our confidential process") is None


def test_portion_marks_open_paragraphs():
    assert cls.detect_portion(US, "(U) Background.\n\n(S//NF) The source.") == "S"
    assert cls.detect_portion(US, "(C) One line") == "C"
    assert cls.detect_portion(US, "A sentence (S) with a mark inside.") is None
    assert cls.detect_portion(CO, "(S) no portion marks on this scale") is None


def test_ceilings():
    assert cls.allowed_levels(None, s=US) is None
    assert cls.allowed_levels("S", s=US) == ["U", "CUI", "C", "S"]
    # A remote model is held to the remote ceiling, whatever the question's own.
    assert cls.effective_ceiling("S", remote=True, s=US) == "U"
    assert cls.effective_ceiling(None, remote=True, s=CO) == "internal"
    assert cls.effective_ceiling("public", remote=True, s=CO) == "public"
    assert cls.allowed_levels("restricted", s=CO) is None  # the top of the scale: no limit


def test_output_takes_the_highest_level_it_cites():
    levels = {1: "U", 2: "S", 3: "C"}
    text = (
        "The plan is old [1].\n\n"
        "The source is named [2, 3].\n\n"
        "- a point [3]\n- another [1]\n\n"
        "## A heading\n\n"
        "An uncited synthesis."
    )
    md, banner = cls.marked_markdown(US, text, levels, context_level="S")
    assert "(U) The plan is old" in md and "(S) The source is named" in md
    assert "- (C) a point" in md and "- (U) another" in md
    assert "## A heading" in md and "(S) An uncited synthesis" in md  # strict
    assert banner == "S"
    cited = cls.Scale(**{**cls.preset("us").__dict__, "mode": "cited"})
    md, _ = cls.marked_markdown(cited, "An uncited synthesis.", levels, context_level="S")
    assert md.startswith("(U) ")


def test_the_scale_is_saved_and_checked():
    s = cls.preset("us")
    cls.save(s)
    assert cls.load().levels[-1].id == "TS" and cls.load().remote_ceiling == "U"
    with pytest.raises(ValueError):
        cls.save(cls.Scale(default="nope"))
    assert cls.load().scheme == "us"


def test_a_composition_carries_its_banner_and_marks():
    comp = Composition(title="T", scale=cls.preset("us"))
    comp.sources = [
        Source(
            n=1,
            chunk_id="a",
            document_id="d",
            document_title="A",
            section_path="",
            page=1,
            level="U",
        ),
        Source(
            n=2,
            chunk_id="b",
            document_id="d",
            document_title="B",
            section_path="",
            page=2,
            level="C",
        ),
    ]
    comp.sections = [
        {"heading": "One", "body": "Said [1].\n\nAlso [2].", "cited": [1, 2], "context_level": "C"}
    ]
    md = comp.markdown()
    assert md.startswith("**CONFIDENTIAL**") and md.rstrip().endswith("**CONFIDENTIAL**")
    assert "(U) Said [1]." in md and "(C) Also [2]." in md
    assert comp.banner()["short"] == "C"
    plain = Composition(title="T")
    plain.sections = comp.sections
    assert "**" not in plain.markdown().split("\n")[0]
