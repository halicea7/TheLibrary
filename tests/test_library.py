"""Library-layer tests."""

from __future__ import annotations

import numpy as np
import pytest

from library_agent.library.citations import extract_references
from library_agent.library.cluster import _cluster


class TestReferenceExtraction:
    def test_extracts_numbered_entries(self):
        body = (
            "body text\n\nReferences\n\n"
            "[1] Vaswani et al. Attention is all you need. NIPS 2017.\n"
            "[2] Devlin et al. BERT: Pre-training of deep bidirectional transformers. 2019.\n"
            "[3] Karpukhin et al. Dense passage retrieval. EMNLP 2020.\n"
        )
        refs = extract_references(body)
        assert len(refs) == 3
        assert "Attention is all you need" in refs[0]
        assert not refs[0].startswith("[")  # numbering stripped

    def test_dotted_numbering(self):
        body = (
            "body\n\nBibliography\n\n"
            "1. Alpha et al. First paper title here. 2020.\n"
            "2. Beta et al. Second paper title here. 2021.\n"
            "3. Gamma et al. Third paper title here. 2022.\n"
        )
        assert len(extract_references(body)) == 3

    def test_no_reference_section(self):
        assert extract_references("just body text with no bibliography at all") == []

    def test_too_few_entries_is_not_a_reference_list(self):
        """'References' appearing in prose must not produce garbage entries."""
        assert extract_references("References are important.\n\n[1] only one\n") == []

    def test_uses_last_heading(self):
        """An early 'References' in the table of contents must not win over the real one."""
        body = (
            "Contents\n\nReferences\n\nchapter text\n\nReferences\n\n"
            "[1] Real entry one here. 2020.\n[2] Real entry two here. 2021.\n"
            "[3] Real entry three here. 2022.\n"
        )
        refs = extract_references(body)
        assert len(refs) == 3
        assert "Real entry one" in refs[0]


class TestClustering:
    def test_separates_distinct_groups(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 0.02, (12, 8)) + np.array([1, 0, 0, 0, 0, 0, 0, 0])
        b = rng.normal(0, 0.02, (12, 8)) + np.array([0, 1, 0, 0, 0, 0, 0, 0])
        vecs = np.vstack([a, b]).astype(np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
        labels = _cluster(vecs, 3)
        assert len(set(labels[labels >= 0])) >= 2

    @pytest.mark.parametrize("method", ["eom", "leaf"])
    def test_methods_run(self, method):
        rng = np.random.default_rng(1)
        vecs = rng.normal(0, 1, (40, 8)).astype(np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
        assert _cluster(vecs, 3, method).shape == (40,)


class TestContradictionConsistency:
    """A verdict of 'true' whose own explanation says the sources agree is a model
    contradicting itself; the reasoning wins."""

    @pytest.mark.parametrize(
        "text",
        [
            "The claims are not in conflict because they describe different methods.",
            "These are compatible: each measures a different quantity.",
            "There is no genuine contradiction here; the setups differ.",
            "The sources remain fully consistent with one another.",
        ],
    )
    def test_denials(self, text):
        from library_agent.library.contradictions import _explanation_denies

        assert _explanation_denies(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Paper A reports a 67% gain and Paper B a 12% loss for the same technique.",
            "They cannot both hold: one says use it, the other says never use it.",
        ],
    )
    def test_real_conflicts_pass(self, text):
        from library_agent.library.contradictions import _explanation_denies

        assert not _explanation_denies(text)


class TestInstructionShapedExplanation:
    @pytest.mark.parametrize(
        "text",
        ["Explain why they conflict or why they don't.", "State the conflict.", "", "too short"],
    )
    def test_junk(self, text):
        from library_agent.library.contradictions import _explanation_is_junk

        assert _explanation_is_junk(text)

    def test_real_explanation_passes(self):
        from library_agent.library.contradictions import _explanation_is_junk

        assert not _explanation_is_junk(
            "Claim 5 says the BERT-style objective beats prefix LM; claim 6 says they perform "
            "similarly on the same benchmark, so both cannot hold."
        )
