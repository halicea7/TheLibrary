"""What kind of writing a document is, and what that changes.

A paper makes findings; a runbook states settings and steps for one system; a policy
sets rules; a letter asserts things as one person to another on a date. Reading them all
as papers -- "name the methods, numbers and conclusions" -- gives a wiki of runbooks
threads that read like literature reviews and conflicts between two hosts. So genre is
a property of the document: guessed at orientation, settable by the maker of a
cartridge for the whole of it, and read by the section pass (what a claim is), the
cluster summary (how the entry is phrased) and the conflict judge (what a conflict is).
"""

from __future__ import annotations

GENRES = (
    "paper",
    "book",
    "documentation",
    "runbook",
    "policy",
    "notes",
    "report",
    "correspondence",
    "transcript",
    "legal",
    "other",
)

# The section pass: what a claim is, per genre. Papers keep the original wording.
CLAIMS_BY_GENRE = {
    "paper": "concrete factual assertions this section makes -- findings, measurements, "
    "definitions -- one per string.",
    "book": "concrete factual assertions or arguments this section makes, one per string.",
    "report": "findings, figures and recommendations this section states, one per string, "
    "each with the scope it applies to (which unit, period or system).",
    "documentation": "facts a user of this system would rely on: what a thing is, what it "
    "does, what it requires. Name the system, product or version each one is about; "
    "a claim about one system is not a claim about another.",
    "runbook": "settings, steps, owners, hosts and thresholds this page prescribes -- one "
    "per string, each naming the system or host it applies to. Procedures as steps "
    "('restart X before Y'), not as findings.",
    "policy": "rules and requirements this section sets, one per string, each with who "
    "it binds and when; distinguish must from should.",
    "notes": "the points actually recorded, one per string; leave out what is merely "
    "mentioned in passing.",
    "correspondence": "statements as attributed speech: 'who -> whom (date): what was "
    "asserted, asked or agreed'. One per string. Never merge two senders; never restate "
    "a quoted earlier message as the sender's own claim.",
    "transcript": "statements under their speaker, one per string ('Speaker: claim'); keep "
    "a question and its answer paired.",
    "legal": "the allegations, findings, rulings and obligations stated, one per string, "
    "each attributed (the court, a party, a witness) and dated where the text dates it.",
    "other": "concrete factual assertions this section makes, one per string.",
}

# How the library's entry for a theme is phrased. Keys are the dominant genre.
ENTRY_BY_GENRE = {
    "paper": "these are research findings: synthesise what the sources establish, and "
    "where their results or conclusions differ, say so",
    "documentation": "these are pages of documentation: say what the material covers "
    "and how the pages relate -- which system each describes -- rather than treating "
    "them as competing findings",
    "runbook": "these are operational runbooks: say what procedures and settings they "
    "cover and for which systems; note where two pages prescribe different things for "
    "the same system",
    "policy": "these are rules: say what is required, of whom, and where two documents "
    "set different requirements for the same case",
    "correspondence": "these are messages between people: say who is writing to whom "
    "about what, over which dates, and what was agreed or disputed -- attribute every "
    "statement to its sender; do not present a sender's view as the library's",
    "transcript": "this is testimony or discussion: say who said what, attributed, and "
    "where accounts of the same event differ",
    "legal": "these are legal documents: say what is alleged, found or ruled, by whom, "
    "attributed; a filing's allegation is not a finding",
}

# The conflict judge: what a conflict is, per genre, on top of the scope rule.
CONFLICT_BY_GENRE = {
    "documentation": "For documentation, a conflict is two pages giving incompatible facts "
    "about the SAME system or version -- not two systems described differently.",
    "runbook": "For runbooks, a conflict is two pages prescribing incompatible settings or "
    "steps for the SAME host or system; different hosts with different settings are not.",
    "policy": "For policies, a conflict is two documents setting incompatible requirements "
    "for the same case; a stricter rule in one place nests inside a looser one.",
    "correspondence": "For correspondence, a conflict is the SAME person asserting "
    "incompatible things, or two people giving incompatible accounts of the SAME event; "
    "two people holding different opinions is a disagreement of views, not a conflict "
    "of fact -- say which it is.",
    "transcript": "For testimony, a conflict is two accounts of the same event that cannot "
    "both be true, or a speaker contradicting their own earlier statement.",
    "legal": "For legal documents, an allegation and a finding are not in conflict; two "
    "findings on the same question are.",
    "paper": "For research, a conflict is opposite results for the same measurement of the "
    "same method under the same conditions; a research finding and a document's opinion "
    "are never in conflict.",
}


def normalise(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower()
    return v if v in GENRES else None


def dominant(genres: list[str | None]) -> str | None:
    """The genre most of a group's documents have, or None when nothing is known."""
    counts: dict[str, int] = {}
    for g in genres:
        if g:
            counts[g] = counts.get(g, 0) + 1
    return max(counts, key=counts.get) if counts else None
