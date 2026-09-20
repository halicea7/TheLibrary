"""Genre: what a claim is depends on what kind of writing the document is."""

from library_agent.reading import genre
from library_agent.reading.prompts import ORIENTATION_SCHEMA, SECTION_PROMPT


def test_every_genre_has_a_claims_rule_and_the_schema_offers_it():
    for g in genre.GENRES:
        assert g in genre.CLAIMS_BY_GENRE
        assert g in ORIENTATION_SCHEMA["properties"]["document_kind"]["enum"]


def test_normalise_and_dominant():
    assert genre.normalise("Runbook ") == "runbook"
    assert genre.normalise("essay") is None
    assert genre.dominant(["runbook", None, "runbook", "paper"]) == "runbook"
    assert genre.dominant([None, None]) is None


def test_section_prompt_reads_by_genre():
    p = SECTION_PROMPT.format(
        title="t",
        orientation="o",
        previous="",
        section_path="s",
        text="x",
        claims_guidance=genre.CLAIMS_BY_GENRE["runbook"],
        category_guidance="",
    )
    assert "naming the system or host it applies to" in p
    p2 = SECTION_PROMPT.format(
        title="t",
        orientation="o",
        previous="",
        section_path="s",
        text="x",
        claims_guidance=genre.CLAIMS_BY_GENRE["correspondence"],
        category_guidance="",
    )
    assert "who -> whom" in p2
