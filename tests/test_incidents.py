"""Incidents: recording, dedupe, the documentation lookup and the command check."""

from __future__ import annotations

from library_agent.ops import incidents
from library_agent.ops.incidents import (
    _normalise,
    _scrub,
    command_is_documented,
    documented_commands,
    relevant_docs,
)


class TestNormalise:
    def test_ids_paths_numbers_collapse(self):
        a = _normalise(
            "tier1 failed for 6f9619ff-8b86-d011-b42d-00c04fc964ff at /Users/x/y.pdf after 3 tries"
        )
        b = _normalise(
            "tier1 failed for 0c12fa78-a707-4a7d-8450-f25649d44632 at /Users/z/w.pdf after 12 tries"
        )
        assert a == b

    def test_different_errors_stay_different(self):
        assert _normalise("connection refused") != _normalise("column does not exist")


class TestScrub:
    def test_home_paths_and_keys(self):
        s = _scrub("File /Users/alice/proj/x.py; LIBRARY_OLLAMA_API_KEY=abc123 token: zzz")
        assert "/Users/alice" not in s and "abc123" not in s and "zzz" not in s
        assert "~/proj/x.py" in s


class TestDocs:
    def test_tunnel_error_finds_the_tunnel_section(self):
        hits = relevant_docs("ConnectError All connection attempts failed 11434 ollama")
        assert hits and "Ollama is not reachable" in hits[0][0]

    def test_migration_error_finds_the_migration_section(self):
        hits = relevant_docs("ProgrammingError column shelf_id does not exist")
        assert any("Migration" in h for h, _, _ in hits[:3])

    def test_documented_commands(self):
        known = documented_commands()
        assert "./library status" in known
        assert command_is_documented("./library restart", known)
        assert command_is_documented("uv run alembic upgrade head", known)
        assert command_is_documented("redis-cli ping", known)
        assert not command_is_documented("./library cluster --dry-run", known)
        assert not command_is_documented("make deploy", known)


class TestRecording:
    async def test_dedupe_counts_up(self, db):
        a = await incidents.record(db, source="test", kind="RuntimeError", message="boom /tmp/a 1")
        b = await incidents.record(db, source="test", kind="RuntimeError", message="boom /tmp/b 2")
        assert a.id == b.id and b.count == 2
        c = await incidents.record(db, source="test", kind="ValueError", message="boom /tmp/a 1")
        assert c.id != a.id
        opened = [i for i in await incidents.list_incidents(db) if i.source == "test"]
        assert len(opened) == 2
        await incidents.resolve(db, a.id)
        opened = [i for i in await incidents.list_incidents(db) if i.source == "test"]
        assert len(opened) == 1
        # A resolved incident does not absorb a recurrence; that is a new one.
        d = await incidents.record(db, source="test", kind="RuntimeError", message="boom /tmp/c 3")
        assert d.id != a.id
