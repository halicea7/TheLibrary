"""Tier 1 prompts and their output schemas.

Each prompt has a version key in `settings().prompt_versions`. Bumping a version marks
every artifact of that kind stale, so only those regenerate rather than the whole corpus.
Prompt text and version must change together -- editing the text without bumping leaves
artifacts that claim a version they were not produced by."""

from __future__ import annotations

from typing import Any

from library_agent.reading.genre import GENRES

SYSTEM_LIBRARIAN = (
    "You are a careful research librarian building a searchable index of a personal "
    "library. You are concise, concrete, and you never invent facts that are not in the "
    "text you were given."
)

# ---------------------------------------------------------------- orientation

ORIENTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # Appended to every chunk's context prefix, so it must be short and situating.
        "one_liner": {"type": "string"},
        "about": {"type": "string"},
        "key_terms": {"type": "array", "items": {"type": "string"}},
        "document_kind": {"type": "string", "enum": list(GENRES)},
    },
    "required": ["one_liner", "about", "key_terms", "document_kind"],
}

ORIENTATION_PROMPT = """Here is the opening and closing of a document, plus its table of
contents. Build an orientation card for it — the context a reader needs before reading any
individual passage.

TITLE: {title}

SECTIONS:
{toc}

OPENING:
{opening}

CLOSING:
{closing}

Return:
- one_liner: at most 12 words naming what this document is about. It gets prepended to
  every excerpt of this document, so make it situating, not promotional. No title repeat.
- about: 2-3 sentences on the document's subject, approach, and contribution.
- key_terms: 5-10 distinctive technical terms, method names, or proper nouns.
- document_kind: what kind of writing this is. paper: a research paper. book. documentation:
  reference or user documentation for a system or product. runbook: an operational page --
  a procedure, a configuration, a how-to for one system. policy: rules and requirements.
  notes: working notes. report: a report with findings for an organisation.
  correspondence: email or letters. transcript: an interview, meeting or testimony.
  legal: a filing, contract or ruling. other."""

# ---------------------------------------------------------------- section pass

SECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "claims": {"type": "array", "items": {"type": "string"}},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "entities", "claims", "categories"],
}

SECTION_PROMPT = """Document: "{title}"
What it is: {orientation}
{previous}
Section: {section_path}

---
{text}
---

Summarise this section for a retrieval index.

- summary: 2-4 sentences. What does this section actually say? Be specific — name the
  methods, numbers, and conclusions rather than describing that the section "discusses" them.
- entities: distinctive names appearing here (methods, datasets, systems, people).
- claims: {claims_guidance}
- categories: 1-3 topic labels for this section specifically.
{category_guidance}"""

# ---------------------------------------------------------------- document summary

DOCUMENT_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "contributions": {"type": "array", "items": {"type": "string"}},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "contributions", "categories"],
}

DOCUMENT_SUMMARY_PROMPT = """Document: "{title}"
What it is: {orientation}

Section summaries, in order:
{sections}

Write the document-level entry for this library.

- summary: 4-8 sentences covering what the document argues or presents, how, and what it
  concludes. This is what gets retrieved when someone asks a broad question about the
  document, so it should stand alone.
- contributions: the 2-5 things this document actually adds.
- categories: 2-4 topic labels for the document as a whole.
{category_guidance}"""

PART_SUMMARY_PROMPT = """Document: "{title}"
What it is: {orientation}

This is one stretch of a long document ({span}), as the summaries of its sections, in order:
{sections}

In 5-8 sentences, say what this stretch covers and argues, in order, keeping its specific
names, figures and conclusions. Plain prose, no preamble. It will be read alongside the
same digest of the document's other stretches to write the entry for the whole."""

# ---------------------------------------------------------------- categories

CATEGORY_GUIDANCE_TEMPLATE = """
The library already uses these categories:
{existing}

Reuse an existing category whenever one reasonably fits — exact string match, please.
Propose a new one only when nothing existing is close. Prefer broad, durable topic names
over narrow ones."""

CATEGORY_GUIDANCE_EMPTY = """
The library has no categories yet. Propose broad, durable topic names that will still make
sense when the library has hundreds of documents."""


def category_guidance(existing: list[str]) -> str:
    """Steering taxonomy at the point of creation is far cheaper than merging duplicates
    afterwards, which is why the canonical list is passed into every tagging prompt."""
    if not existing:
        return CATEGORY_GUIDANCE_EMPTY
    return CATEGORY_GUIDANCE_TEMPLATE.format(existing="\n".join(f"- {c}" for c in sorted(existing)))


# ---------------------------------------------------------------- tier 2 reflections

REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reflections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "reflection": {"type": "string"},
                },
                "required": ["index", "reflection"],
            },
        }
    },
    "required": ["reflections"],
}

REFLECTION_PROMPT = """Document: "{title}"
What it is: {orientation}
Section: {section_path}
{previous}
Below are the numbered passages of this section.

{passages}

For each passage, write a reflection — what a thoughtful reader *thinks* while reading it,
not a restatement of it.

A reflection should do at least one of:
- name what the passage is really claiming, underneath how it is phrased
- connect it to something else in this document or to the wider field
- draw out an implication or consequence the passage leaves unstated
- note a limitation, assumption, or tension the passage glosses over
- explain why this matters, or when it would not

Do NOT summarise. A summary says what the passage contains; a reflection says what it
means. If a passage is genuinely inert — a table caption, a list of hyperparameters — say
so in one short line rather than inventing significance.

Two or three sentences each. Return one entry per passage, using the passage's index."""
