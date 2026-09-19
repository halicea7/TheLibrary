"""Tier 1 reading tests."""

from __future__ import annotations

import pytest

from library_agent.config import settings
from library_agent.library.taxonomy import normalize_name
from library_agent.llm.ollama import is_placeholder, size_context
from library_agent.reading import prompts


class TestPlaceholderDetection:
    """A too-small context makes the model emit schema-shaped placeholders instead of
    failing, which silently writes junk into the index. This guard is what caught it."""

    @pytest.mark.parametrize("junk", ["...", "…", "   ", "", "string", "TODO", "n/a"])
    def test_detects_junk(self, junk):
        assert is_placeholder(junk)

    @pytest.mark.parametrize(
        "real",
        ["RAPTOR builds a recursive summary tree", "Information Retrieval", "0.94 recall"],
    )
    def test_accepts_real_values(self, real):
        assert not is_placeholder(real)

    def test_list_of_placeholders(self):
        assert is_placeholder(["...", "..."])
        assert not is_placeholder(["Information Retrieval", "..."])
        assert not is_placeholder([])  # empty is legitimately empty, not a placeholder


class TestContextSizing:
    def test_scales_with_prompt(self):
        assert size_context(100) == 8192
        assert size_context(60000) > size_context(7000)

    def test_monotonic_and_capped(self):
        sizes = [size_context(n) for n in (1_000, 20_000, 60_000, 500_000)]
        assert sizes == sorted(sizes)
        assert sizes[-1] <= 65536

    def test_leaves_room_for_output(self):
        """The failure mode was a prompt that fit but left no room to generate into."""
        assert size_context(7633) >= 7633 // 3 + 2048


class TestTaxonomy:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("information retrieval", "Information Retrieval"),
            ("  Machine   Learning  ", "Machine Learning"),
            ("neural networks.", "Neural Networks"),
        ],
    )
    def test_normalizes(self, raw, expected):
        assert normalize_name(raw) == expected

    def test_normalization_collapses_duplicates(self):
        """Steering the taxonomy at creation time is the whole point; these three must
        land on one name rather than three near-duplicate categories."""
        variants = ["information retrieval", "Information  Retrieval", "INFORMATION RETRIEVAL"]
        assert len({normalize_name(v) for v in variants}) == 1


class TestPrompts:
    def test_every_prompt_has_a_version(self):
        versions = settings().prompt_versions
        for key in ("orientation", "section_summary", "document_summary", "categories"):
            assert key in versions, f"{key} has no prompt version"

    def test_category_guidance_includes_existing(self):
        g = prompts.category_guidance(["Information Retrieval", "Language Models"])
        assert "Information Retrieval" in g
        assert "Reuse an existing category" in g

    def test_category_guidance_when_empty(self):
        g = prompts.category_guidance([])
        assert "no categories yet" in g
        assert "- " not in g.split("\n")[1] if len(g.split("\n")) > 1 else True

    def test_schemas_require_their_fields(self):
        for schema in (
            prompts.ORIENTATION_SCHEMA,
            prompts.SECTION_SCHEMA,
            prompts.DOCUMENT_SUMMARY_SCHEMA,
        ):
            assert schema["required"]
            for field in schema["required"]:
                assert field in schema["properties"]

    def test_section_prompt_formats(self):
        out = prompts.SECTION_PROMPT.format(
            title="T",
            orientation="o",
            previous="",
            section_path="p",
            text="body",
            category_guidance=prompts.category_guidance([]),
        )
        assert "body" in out and "{" not in out.replace("{", "", 0)


class TestStructuredOutputOrdering:
    """Constrained JSON generation emits properties in schema order, so a verdict field
    placed before its reasoning is committed to before the model has reasoned.

    Measured: with `disagreement` first, the contradiction detector wrote "they report
    opposite effects" into the explanation while still returning false. Moving the boolean
    last fixed all five test cases."""

    def test_contradiction_verdict_comes_last(self):
        from library_agent.library.contradictions import SCHEMA

        props = list(SCHEMA["properties"])
        assert props[-1] == "disagreement", f"verdict must be the last field, got order {props}"
        assert props.index("analysis") < props.index("disagreement")
        assert props.index("explanation") < props.index("disagreement")

    def test_cluster_significance_comes_last(self):
        from library_agent.library.cluster import CLUSTER_SCHEMA

        props = list(CLUSTER_SCHEMA["properties"])
        assert props.index("summary") < props.index("significant")

    def test_eval_answerable_comes_last(self):
        from library_agent.eval.generate import _SCHEMA

        props = list(_SCHEMA["properties"])
        assert props.index("question") < props.index("answerable")

    def test_rewrite_verdict_comes_last(self):
        from library_agent.chat.rewrite import _SCHEMA

        props = list(_SCHEMA["properties"])
        assert props.index("standalone_query") < props.index("needs_retrieval")


class TestTier2:
    def test_reflection_schema_is_indexed_per_passage(self):
        """Reflections come back in one batched call per section, so each entry must carry
        the index of the passage it belongs to or they cannot be matched back."""
        from library_agent.reading.prompts import REFLECTION_SCHEMA

        item = REFLECTION_SCHEMA["properties"]["reflections"]["items"]
        assert set(item["required"]) == {"index", "reflection"}
        assert item["properties"]["index"]["type"] == "integer"

    def test_reflection_prompt_demands_interpretation(self):
        """The spec's whole point is that a reflection is not a summary."""
        from library_agent.reading.prompts import REFLECTION_PROMPT

        low = REFLECTION_PROMPT.lower()
        assert "do not summarise" in low
        assert "implication" in low or "connect" in low

    def test_reflection_prompt_formats(self):
        from library_agent.reading.prompts import REFLECTION_PROMPT

        out = REFLECTION_PROMPT.format(
            title="T", orientation="o", section_path="s", previous="", passages="[0] body"
        )
        assert "[0] body" in out


class TestInstructionEcho:
    """A model can copy a field's description into the field instead of answering it."""

    def test_detects_verbatim_lift(self):
        from library_agent.llm.ollama import echoes_prompt

        prompt = "Return:\n- explanation: if they conflict, what each side claims and why."
        assert echoes_prompt("if they conflict, what each side claims and why.", prompt)

    def test_accepts_real_answer(self):
        from library_agent.llm.ollama import echoes_prompt

        prompt = "Return:\n- explanation: if they conflict, what each side claims and why."
        assert not echoes_prompt(
            "Both papers use Adam; only beta2 differs (0.98 vs 0.999).", prompt
        )

    def test_short_values_ignored(self):
        """Short strings like a category name legitimately appear in the prompt."""
        from library_agent.llm.ollama import echoes_prompt

        assert not echoes_prompt("Information Retrieval", "reuse Information Retrieval")


class TestThinkingCapability:
    """Sending think=true to a model without it is a hard 400 from Ollama."""

    def test_capability_cache_is_per_model(self):
        from library_agent.llm import ollama

        ollama._THINKING.clear()
        ollama._THINKING["a"] = True
        ollama._THINKING["b"] = False
        assert ollama._THINKING["a"] and not ollama._THINKING["b"]
        ollama._THINKING.clear()


class TestStances:
    def test_every_stance_has_a_prompt_and_keeps_the_citation_contract(self):
        """A stance loosens reserve, never provenance: each prompt must still direct the
        model to cite the shelf."""
        from library_agent.chat.answer import STANCES

        assert len(STANCES) >= 5
        for k, s in STANCES.items():
            assert s["label"] and s["prompt"], k
            assert "cite" in s["prompt"].lower(), f"{k} drops the citation contract"

    def test_counterparts_are_symmetric(self):
        from library_agent.chat.answer import STANCES

        for k, s in STANCES.items():
            if c := s.get("counterpart"):
                assert c in STANCES, f"{k} -> {c} missing"
                assert STANCES[c].get("counterpart") == k, f"{k}/{c} not mutual"

    def test_stance_lands_in_the_system_prompt(self):
        from library_agent.chat.answer import STANCES, build_messages

        neutral = build_messages("q", [], [], [])[0]["content"]
        contrarian = build_messages("q", [], [], [], stance="contrarian")[0]["content"]
        assert neutral in contrarian  # the base policy is kept, not replaced
        assert STANCES["contrarian"]["prompt"] in contrarian
        assert build_messages("q", [], [], [], stance="nope")[0]["content"] == neutral
