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


def _live(module_id):
    from types import SimpleNamespace

    return SimpleNamespace(kind="live", chunk_id="x", live={"module_id": module_id})


async def test_a_modules_live_results_carry_its_level(tmp_path, monkeypatch):
    from library_agent.modules import store

    monkeypatch.setenv("LIBRARY_MODULES_FILE", str(tmp_path / "modules.json"))
    store._cache = None
    s = cls.preset("us")
    store.save({"s1": store.ModuleConfig(id="s1", classification="C")})
    assert store.config_for("s1").classification == "C"
    assert cls.module_level("s1", s) == "C"
    assert cls.module_level("other", s) == "U"  # unset: the default
    store.save({"s1": store.ModuleConfig(id="s1", classification="gone")})
    assert cls.module_level("s1", s) == "U"  # a level no longer on the scale
    store.save({"s1": store.ModuleConfig(id="s1", classification="S")})
    # Only live results: no passages to look up, and they still get the module's level.
    assert await cls.hit_levels(None, [_live("s1"), _live("other")], s) == ["S", "U"]
    # A question held to Confidential may not consult a Secret module.
    assert cls.module_level("s1", s) not in cls.allowed_levels("C", s=s)


def test_a_provider_sets_its_own_ceiling_and_can_be_internal(tmp_path, monkeypatch):
    from library_agent.llm import providers

    monkeypatch.setenv("LIBRARY_PROVIDERS_FILE", str(tmp_path / "providers.json"))
    providers._cache = None
    cfg = providers.Config()
    cfg.providers["campus"] = providers.Provider(
        id="campus",
        name="Campus",
        base_url="http://gpu.test/v1",
        ceiling="confidential",
        internal=True,
    )
    cfg.providers["cloud"] = providers.Provider(
        id="cloud", name="Cloud", base_url="http://x.test/v1"
    )
    providers.save(cfg)
    providers._cache = None
    back = providers.load().providers["campus"]
    assert back.ceiling == "confidential" and back.internal  # saved and read back
    s = cls.Scale()  # company scale: remote ceiling Internal
    assert cls.model_limit("qwen3:30b-a3b", s) is None  # local: no limit
    assert cls.model_limit("campus:qwen3:30b-a3b", s) == "confidential"
    assert cls.model_limit("cloud:gpt", s) == "internal"  # unset: the remote ceiling
    # The lowest of the question's ceiling and each model's limit.
    assert cls.effective_ceiling(None, models=["campus:m"], s=s) == "confidential"
    assert cls.effective_ceiling("public", models=["campus:m"], s=s) == "public"
    assert cls.effective_ceiling(None, models=["campus:m", "cloud:m"], s=s) == "internal"
    assert cls.effective_ceiling(None, models=["qwen3:30b-a3b"], s=s) is None
    # Internal: may use modules cleared for local models only.
    assert providers.is_trusted("campus:m") and providers.is_trusted("qwen3:30b-a3b")
    assert not providers.is_trusted("cloud:m")
