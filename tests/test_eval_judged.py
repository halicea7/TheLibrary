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
