"""A scheduled rebuild waits for a quiet shelf and does not run twice for one burst."""

from library_agent.worker import tasks


class FakeRedis:
    def __init__(self, started=None):
        self.kv = {} if started is None else {tasks.REBUILD_STARTED_KEY: str(started)}
        self.enqueued = []

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v, **kw):
        self.kv[k] = v
        return True

    async def enqueue_job(self, name, *args, **kw):
        self.enqueued.append((name, args, kw))


async def test_deferred_while_reads_are_queued(monkeypatch):
    async def pending():
        return 5

    monkeypatch.setattr(tasks, "_reads_pending", pending)
    r = FakeRedis()
    out = await tasks.build_library_layer({"redis": r}, "all", requested_at=100.0)
    assert out["deferred"] and r.enqueued[0][0] == "build_library_layer"
    assert r.enqueued[0][2]["requested_at"] == 100.0


async def test_skipped_when_a_later_rebuild_already_ran(monkeypatch):
    async def pending():
        return 0

    monkeypatch.setattr(tasks, "_reads_pending", pending)
    r = FakeRedis(started=200.0)
    out = await tasks.build_library_layer({"redis": r}, "all", requested_at=100.0)
    assert "skipped" in out and not r.enqueued
