"""The hand-judged set: stored outside the repository, split so tuning never sees a test
volume, and scored per case."""

from __future__ import annotations

import pytest

from library_agent.eval import judged


@pytest.fixture
def tmp_set(tmp_path, monkeypatch):
    monkeypatch.setattr(judged, "eval_dir", lambda: tmp_path)
    (tmp_path / "runs").mkdir()
    return tmp_path


def test_cases_round_trip_privately(tmp_set):
    c = judged.Case(question="How does Raft elect a leader?", supporting=["a"], judged_at="x")
    judged.save([c])
    assert judged.load()[0].question == "How does Raft elect a leader?"
    assert oct(judged.set_path().stat().st_mode)[-3:] == "600"


def test_tune_never_holds_a_test_volume():
    vols = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(40)]
    tune = [v for v in vols if judged.volume_side(v) == "tune"]
    test = [v for v in vols if judged.volume_side(v) == "test"]
    assert tune and test
    assert judged.split_for([tune[0]]) == "tune"
    assert judged.split_for([tune[0], tune[1]]) == "tune"
    assert judged.split_for([tune[0], test[0]]) == "test"  # mixed: never tune
    assert judged.split_for([]) == "test"


def test_a_case_is_scored_by_its_first_supporting_passage():
    s = judged.score(["x", "a", "d", "b"], {"a", "b"}, {"d", "z"})
    assert s["first"] == 2 and s["rr"] == 0.5
    assert s["hit@1"] is False and s["hit@3"] is True
    assert s["recall@10"] == 1.0 and s["distractors@5"] == 1
    miss = judged.score(["x"], {"a"}, set())
    assert miss["first"] is None and miss["rr"] == 0.0 and miss["recall@10"] == 0.0


async def test_the_model_suggests_per_passage_and_a_failed_batch_stays_unmarked():
    passages = [
        {"chunk_id": f"c{i}", "title": "T", "page": i, "text": f"text {i}"} for i in range(10)
    ]
    calls = []

    class Client:
        async def structured(self, model, prompt, schema, **kw):
            calls.append(prompt)
            if len(calls) == 2:
                raise RuntimeError("down")  # the second batch (passages 8-9) fails
            return {
                "verdicts": [
                    {"n": 1, "verdict": "supports", "reason": "states it"},
                    {"n": 2, "verdict": "looks_relevant", "reason": "other protocol"},
                    {"n": 3, "verdict": "unrelated"},
                    {"n": 99, "verdict": "supports"},  # no such passage: ignored
                ]
            }

    m = await judged.suggest(Client(), "q", passages, model="m")
    assert m["supporting"] == ["c0"] and m["distractors"] == ["c1"]
    assert m["reasons"] == {"c0": "states it", "c1": "other protocol"}
    assert len(calls) == 2 and "[2] T, p.1" in calls[0]


def test_agreement_counts_every_pooled_passage_and_discounts_chance():
    def case(t, human, model, pooled):
        return judged.Case(
            question="q",
            type=t,
            supporting=human,
            judged_at="x",
            judge="human",
            pooled=pooled,
            model_marks={"supporting": model, "model": "m"},
        )

    pooled = [f"p{i}" for i in range(10)]
    same = case("exact", ["p0"], ["p0"], pooled)
    off = case("mechanism", ["p0", "p1"], ["p2"], pooled)
    a = judged.agreement([same, off])
    assert a["exact"]["agreement"] == 1.0 and a["exact"]["kappa"] == 1.0
    assert a["mechanism"]["precision"] == 0.0 and a["mechanism"]["recall"] == 0.0
    assert a["mechanism"]["agreement"] == 0.7 and a["mechanism"]["kappa"] < 0
    # a model-judged case has no owner's marks to agree with
    assert judged.agreement([case("exact", ["p0"], ["p0"], pooled)])["all"]["cases"] == 1
    model_only = judged.Case(
        question="q",
        judged_at="x",
        judge="model",
        pooled=pooled,
        model_marks={"supporting": ["p0"]},
        supporting=["p0"],
    )
    assert judged.agreement([model_only])["all"] == {"cases": 0, "passages": 0}


def test_model_judged_cases_are_scored_apart():
    rows = [judged.score(["a"], {"a"}, set()), judged.score(["b"], {"a"}, set())]
    s = judged._summarise(rows)
    assert s["n"] == 2 and s["mrr"] == 0.5 and s["hit@1"] == 0.5
    human = judged.Case(question="h", id="h", judge="human", type="exact")
    model = judged.Case(question="m", id="m", judge="model", type="exact")
    per = {"h": {"find": rows[0]}, "m": {"find": rows[1]}}
    assert [g for g, _ in judged._groups([human], per, "find")] == ["all", "exact"]
    only_model = dict(judged._groups([model], per, "find"))
    assert only_model["all"][0]["rr"] == 0.0


async def test_a_reviewer_judges_a_packet_and_it_comes_back_as_their_cases(tmp_set, monkeypatch):
    async def fake_seeds(limit=200):
        return [
            {"question": "How does Raft elect a leader?", "source": "ask"},
            {"question": "Which port does LDAPS use?", "source": "ask"},
        ]

    async def fake_pool(q):
        return [
            {
                "chunk_id": f"{q[:4]}-{i}",
                "document_id": "d",
                "title": "T",
                "section": "S",
                "page": i,
                "text": f"passage {i}",
            }
            for i in range(1, 4)
        ]

    async def fake_split(ids):
        return "tune"

    monkeypatch.setattr(judged, "seeds", fake_seeds)
    monkeypatch.setattr(judged, "pool", fake_pool)

    async def judge_no_db(case):  # the split needs the database; the rest is what we test
        case.split, case.judged_at = "tune", "now"
        case.answerable = bool(case.supporting) and case.type != "unanswerable"
        if not case.answerable:
            case.type = "unanswerable"
        judged.save([c for c in judged.load() if c.id != case.id] + [case])
        return case

    monkeypatch.setattr(judged, "judge", judge_no_db)
    packet = await judged.export_packet(5)
    assert len(packet["items"]) == 2 and packet["items"][0]["passages"][0]["label"] == "P1"
    assert "supporting" in packet["instructions"] and packet["answer_format"]["judgements"]
    raft, ldaps = packet["items"]
    answers = {
        "judge": "Astra",
        "judgements": [
            {
                "id": raft["id"],
                "type": "mechanism",
                "supporting": ["P2"],
                "distractors": ["P1", "P9"],
            },
            {"id": ldaps["id"], "unanswerable": True},
            {"id": "nope", "supporting": ["P1"]},
        ],
    }
    got = await judged.import_judgements(packet, answers)
    assert got == {"judge": "astra", "saved": 2, "skipped": 1, "unknown_labels": 1}
    cases = {c.question: c for c in judged.load()}
    r = cases["How does Raft elect a leader?"]
    assert r.judge == "astra" and r.type == "mechanism"
    assert r.supporting == [raft["passages"][1]["chunk_id"]] and r.distractors == [
        raft["passages"][0]["chunk_id"]
    ]
    assert len(r.pooled) == 3
    assert cases["Which port does LDAPS use?"].type == "unanswerable"
    # the owner's own judgement is never overwritten by a reviewer's
    mine = cases["How does Raft elect a leader?"]
    mine.judge = "human"
    judged.save(list(cases.values()))
    again = await judged.import_judgements(packet, answers)
    assert again["skipped"] == 2 and judged.load()[0].judge in ("human", "astra")
    assert next(c for c in judged.load() if c.question.startswith("How does Raft")).judge == "human"
