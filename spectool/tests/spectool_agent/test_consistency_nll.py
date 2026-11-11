import pytest
import threading
from unittest.mock import ANY, MagicMock
from PIL import Image

from spectool_agent.model import compute_answer_nll_consistency
from spectool_agent.speculative_agent import SpeculativeAgentLoop
from spectool_agent.types import AgentConfig


def test_answer_nll_consistency_is_cross_model_gap_for_draft_answer():
    assert compute_answer_nll_consistency(1.2, 1.0) == pytest.approx(0.2)
    assert compute_answer_nll_consistency(0.8, 1.0) == pytest.approx(0.2)


def test_answer_nll_consistency_requires_both_scores():
    assert compute_answer_nll_consistency(None, 1.0) is None
    assert compute_answer_nll_consistency(1.0, None) is None


def test_verdict_scores_the_same_draft_answer_with_main_and_draft():
    config = AgentConfig(api_base="http://fake", model_name="fake",
                         consistency_threshold=0.5)
    main_model = MagicMock()
    barrier = threading.Barrier(2)

    def score(value):
        barrier.wait(timeout=2)
        return value

    main_model.score_answer_nll.side_effect = lambda *_: score(1.4)
    draft_model = MagicMock()
    draft_model.score_answer_nll.side_effect = lambda *_: score(1.0)
    loop = SpeculativeAgentLoop(main_model, draft_model, MagicMock(), config)
    pool = MagicMock()
    pool.get_answer.return_value = "B"

    try:
        accept, mode, score = loop._verdict(
            "A", 0.0, pool, image=Image.new("RGB", (2, 2)), user_text="Q")
    finally:
        loop.shutdown()

    assert accept
    assert mode == "consistency_nll"
    assert score == pytest.approx(0.4)
    main_model.score_answer_nll.assert_called_once_with(ANY, "B")
    draft_model.score_answer_nll.assert_called_once_with(ANY, "B")


def test_apply_review_records_consistency_nll_time():
    config = AgentConfig(api_base="http://fake", model_name="fake",
                         consistency_threshold=0.5)
    main_model = MagicMock()
    main_model.score_answer_nll.return_value = 1.4
    draft_model = MagicMock()
    draft_model.score_answer_nll.return_value = 1.0
    loop = SpeculativeAgentLoop(main_model, draft_model, MagicMock(), config)
    pool = MagicMock()
    pool.get_answer.return_value = "B"
    from spectool_agent.types import Trajectory
    traj = Trajectory()

    try:
        accepted = loop._apply_review_result(
            {
                "main_answer": "A",
                "main_reasoning_entropy": 0.0,
                "main_entropy": 0.0,
                "main_reasoning": "reasoning",
            },
            pool,
            traj,
            Image.new("RGB", (2, 2)),
            "Q",
            0,
        )
    finally:
        loop.shutdown()

    assert accepted
    assert traj.total_consistency_time >= 0.0


def test_consistency_mode_never_falls_back_to_entropy_when_unavailable():
    config = AgentConfig(api_base="http://fake", model_name="fake",
                         consistency_threshold=0.5, entropy_threshold=99.0)
    loop = SpeculativeAgentLoop(MagicMock(), MagicMock(), MagicMock(), config)
    pool = MagicMock()
    pool.get_answer.return_value = None

    accept, mode, score = loop._verdict("A", 0.0, pool)

    assert not accept
    assert mode == "consistency_unavailable"
    assert score == -1.0
    loop.shutdown()
