import math
import re
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from spectool_agent.model import (_answer_verification_regex,
                                  _verification_distribution, VLLMModel)
from spectool_agent.pool import DraftPool
from spectool_agent.speculative_agent import SpeculativeAgentLoop
from spectool_agent.types import AgentConfig, PoolEntry, Trajectory


def response(quality="NA", answer="B", decision="1"):
    return (f"{quality}|<evidence>1 target at x=100, beside 0 other targets.</evidence>"
            f"<answer>{answer}</answer><verified>{decision}</verified>")


def tokens(raw, alternative=-4.0):
    start = raw.index("<verified>") + len("<verified>")
    # Character tokens deliberately place many misleading 0/1 values earlier.
    result = [NS(token=c, logprob=-0.1, top_logprobs=[]) for c in raw]
    result[start].top_logprobs = [NS(token="0", logprob=alternative),
                                NS(token="1", logprob=-0.1)]
    return result


@pytest.mark.parametrize("quality", ["NA", "GOOD", "BAD"])
@pytest.mark.parametrize("answer", ["A", "B"])
@pytest.mark.parametrize("decision", ["0", "1"])
def test_regex_never_forces_a_candidate_to_pass(quality, answer, decision):
    pattern = _answer_verification_regex(["A", "B"], quality_enabled=quality != "NA")
    assert re.fullmatch(pattern, response(quality, answer, decision))


def test_score_reads_final_verdict_not_evidence_or_open_answer_numbers():
    raw = response(answer="1001")
    assert _verification_distribution(raw, tokens(raw)) == {"1": -0.1, "0": -4.0}


def test_vllm_terminal_token_is_not_part_of_message_content():
    raw = response()
    lp = tokens(raw) + [NS(token="<|im_end|>", logprob=0., top_logprobs=[])]
    assert _verification_distribution(raw, lp) == {"1": -0.1, "0": -4.0}
    lp[-1].token = "unexpected text"
    assert _verification_distribution(raw, lp) == {}


@pytest.mark.parametrize("alternative", [-9999, math.nan, math.inf])
def test_masked_or_invalid_alternative_is_not_replaced_with_fake_confidence(alternative):
    raw = response()
    assert _verification_distribution(raw, tokens(raw, alternative)) == {}


def test_missing_alternative_or_misaligned_tokens_fail_closed():
    raw = response()
    lp = tokens(raw)
    lp[raw.index("<verified>") + 10].top_logprobs = []
    assert _verification_distribution(raw, lp) == {}
    assert _verification_distribution(raw, tokens(raw)[1:]) == {}
    assert _verification_distribution("NA|1|<answer>B</answer>", tokens(raw)) == {}


def test_real_api_adapter_uses_answer_conditioned_score(monkeypatch):
    import spectool_agent.model as mod
    raw = response()
    client = MagicMock()
    client.chat.completions.create.return_value = NS(choices=[NS(
        message=NS(content=raw, model_dump=lambda: {}),
        logprobs=NS(content=tokens(raw)))])
    monkeypatch.setattr(mod, "OpenAI", lambda **kwargs: client)
    result = VLLMModel(AgentConfig(api_base="http://fake", model_name="fake")).fast_answer_gate(
        [{"role": "user", "content": "question"}], choices=["A", "B"])
    assert result["answer"] == "B"
    assert result["answerable"]
    assert result["answerability_score"] == pytest.approx(1 / (1 + math.exp(-3.9)))
    assert result["verification_log_0"] == -4
    assert result["verification_protocol"] == "answer_conditioned_v1"
    client.chat.completions.create.assert_called_once()
    assert client.chat.completions.create.call_args.kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("new_entry,status", [
    (PoolEntry(step_idx=1, reasoning="Correction: the dress is black."), "stale_evidence"),
    (PoolEntry(step_idx=1, answer="A"), "draft_answer_conflict"),
    (PoolEntry(step_idx=1, answer="B"), "accepted"),
])
def test_atomic_publication_catches_late_draft_updates(new_entry, status):
    pool, traj = DraftPool(), Trajectory()
    old = PoolEntry(step_idx=0, reasoning="first crop")
    pool.add(old)
    cursor = pool.position_after(old)
    pool.add(new_entry)
    assert SpeculativeAgentLoop._commit_early_answer(
        pool, traj, "B", cursor, "main_answer_from_draft_evidence") == status
    assert traj.final_answer == ("B" if status == "accepted" else None)


def test_pool_snapshot_cursor_does_not_skip_concurrent_adds():
    pool = DraftPool()
    first, second = PoolEntry(step_idx=0), PoolEntry(step_idx=1)
    pool.add(first)
    entries, cursor = pool.snapshot_since(0)
    pool.add(second)
    assert entries == [first]
    assert pool.snapshot_since(cursor) == ([second], 2)


def test_failed_early_verification_does_not_become_fallback():
    loop = SpeculativeAgentLoop(MagicMock(), MagicMock(), MagicMock(),
        AgentConfig(api_base="http://fake", model_name="fake", answerability_threshold=0.85))
    try:
        traj = Trajectory()
        assert not loop._apply_fast_answer_result(
            {"answer": "B", "answerable": True, "answerability_score": None},
            "Question\nA. first\nB. second", DraftPool(), traj)
        assert traj.final_answer is None
        assert "fallback" not in traj.stop_reason
    finally:
        loop.shutdown()
