"""Effort levels: what each one changes, and that deep can never be worse than normal."""

from __future__ import annotations

from library_agent.chat import effort


class TestLevels:
    def test_three_levels_ordered_by_work(self):
        q, n, d = effort.get("quick"), effort.get("normal"), effort.get("deep")
        assert q.passages < n.passages < d.passages
        assert not q.config.use_reranker and n.config.use_reranker and d.config.use_reranker
        assert not q.multi_query and not n.multi_query and d.multi_query
        assert q.num_ctx < n.num_ctx < d.num_ctx

    def test_unknown_or_missing_is_normal(self):
        assert effort.get(None).name == "normal"
        assert effort.get("maximum").name == "normal"
        assert effort.get("DEEP").name == "deep"

    async def test_split_falls_back_to_the_question(self):
        class Broken:
            async def structured(self, *a, **k):
                raise RuntimeError("no model")

        assert await effort.split_question(Broken(), "m", "compare a and b") == ["compare a and b"]

    async def test_split_dedupes_and_caps(self):
        class Fake:
            async def structured(self, *a, **k):
                return {
                    "queries": ["ssti detection", "sql injection", "ssti detection", "x", "y", "z"]
                }

        out = await effort.split_question(Fake(), "m", "q")
        assert out == ["ssti detection", "sql injection", "x", "y"]
