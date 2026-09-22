"""A book reads as a book: chapter-long sections are split into parts of about one read,
and the document entry is written from every stretch, not the opening chapters."""

from __future__ import annotations

from types import SimpleNamespace

from library_agent.ingest.extract import Extracted, TocEntry
from library_agent.ingest.structure import PART_CHARS, SPLIT_OVER, build_sections
from library_agent.reading import tier1


def _chapter(n: int, paras: int) -> str:
    body = "\n\n".join(f"Chapter {n} paragraph {i}. " + "words " * 60 for i in range(paras))
    return f"Chapter {n}\n\n{body}\n\n"


def test_chapter_bookmarks_split_into_parts():
    text = _chapter(1, 3) + _chapter(2, 120) + _chapter(3, 4)
    starts = [text.index(f"Chapter {n}\n") for n in (1, 2, 3)]
    ex = Extracted(
        text=text,
        page_offsets=[(0, 1)],
        toc=[
            TocEntry(level=1, title=f"Chapter {n}", page=1, char_start=s)
            for n, s in zip((1, 2, 3), starts, strict=True)
        ],
    )
    secs = build_sections(ex)
    two = [s for s in secs if (s.title or "").startswith("Chapter 2")]
    assert len(two) > 3, "the long chapter is read in parts"
    assert two[0].title == "Chapter 2" and two[1].title == "Chapter 2 (cont. 2)"
    assert all(s.char_end - s.char_start <= SPLIT_OVER for s in secs)
    assert all(s.char_end - s.char_start >= PART_CHARS // 3 for s in two)
    # parts tile the chapter exactly, in order, on paragraph boundaries
    assert [s.order_index for s in secs] == list(range(len(secs)))
    for a, b in zip(two, two[1:], strict=False):
        assert a.char_end == b.char_start and text[b.char_start - 2 : b.char_start] == "\n\n"
    assert [s.title for s in secs if "(cont." not in (s.title or "")] == [
        "Chapter 1",
        "Chapter 2",
        "Chapter 3",
    ]


def test_unbroken_block_is_still_cut():
    text = "Intro\n\n" + "x" * 30000
    secs = build_sections(Extracted(text=text, page_offsets=[(0, 1)]))
    assert len(secs) >= 4 and all(s.char_end - s.char_start <= SPLIT_OVER for s in secs)


class Digester:
    def __init__(self):
        self.prompts: list[str] = []

    async def generate(self, model, prompt, **kw):
        self.prompts.append(prompt)
        return f"digest {len(self.prompts)}"


async def test_document_entry_reads_every_stretch():
    sums = [
        (SimpleNamespace(path=f"Chapter {i}", title=f"Chapter {i}"), f"summary {i} " + "z" * 600)
        for i in range(120)
    ]
    c = Digester()
    joined = await tier1._digest_stretches(c, "m", "A Book", "a book", sums)
    assert len(c.prompts) >= 4
    assert "Chapter 119" in c.prompts[-1] and "summary 119" in c.prompts[-1]
    assert all(len(p) < tier1.SUMMARY_CHARS + 2000 for p in c.prompts)
    assert len(joined) < tier1.SUMMARY_CHARS and "[Chapter 0 … " in joined
