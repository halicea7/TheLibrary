"""Sentence splitting for the lift, and the rule that a quoted claim must be a quote."""

from library_agent.library.contradictions import _explanation_is_junk
from library_agent.retrieval.lift import sentences


def test_sentences_split_on_ends_and_keep_short_ones_out():
    text = (
        "Reranking. A cross-encoder reranker reads the query and candidate together. "
        "The cost is latency: reranking 100 candidates takes roughly 100ms on Apple Silicon."
    )
    out = sentences(text)
    assert out[0].startswith("A cross-encoder reranker")
    assert out[-1].startswith("The cost is latency")
    assert all(len(s) >= 25 for s in out)


def test_sentences_cap_per_hit():
    text = " ".join(f"Sentence number {i} says something worth reading here." for i in range(40))
    assert len(sentences(text)) == 12


def test_moved_content_is_not_a_disagreement():
    assert _explanation_is_junk(
        "All claims state the content has been moved, but to different URLs. This is a conflict."
    )
    assert not _explanation_is_junk(
        "One source recommends against DNS brute-forcing, while another presents it as standard."
    )
