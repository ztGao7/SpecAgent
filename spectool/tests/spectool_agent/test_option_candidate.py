"""Verifier sees an option-letter candidate together with its option text."""

from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent.pool import DraftPool
from spectool_agent.speculative_agent import (SpeculativeAgentLoop,
                                              _describe_option_candidate)
from spectool_agent.types import AgentConfig, PoolEntry

LABEL_QUESTION = ("Question: Which label contains poison?\nOptions:\nA. A\nB. C\n"
                  "C. B\nD. I\nPlease select the correct answer from the options above. \n")
COLOR_QUESTION = ("Question: What is the color of the cap?\nOptions:\nA. white\n"
                  "B. green\nC. black\nD. red\nPlease select the correct answer.\n")
MATHVISTA_QUESTION = "Question: What is angle EAD?\nChoices:\n(A) 10°\n(B) 15°\n(C) 20°"


@pytest.mark.parametrize("question,candidate,expected,label_note", [
    (LABEL_QUESTION, "B", 'B (option B: "C")', True),
    (LABEL_QUESTION, "A", 'A (option A: "A")', False),
    (COLOR_QUESTION, "B", 'B (option B: "green")', False),
    (MATHVISTA_QUESTION, "A", 'A (option A: "10°")', False),
])
def test_option_letter_is_shown_with_its_text(question, candidate, expected, label_note):
    described = _describe_option_candidate(candidate, question)
    assert described.startswith(expected)
    assert ('denotes the image label "C", not an image label "B"' in described) is label_note


@pytest.mark.parametrize("question,candidate", [
    ("Is there a cup in the image? Please answer yes or no.", "yes"),
    (COLOR_QUESTION, "E"),
    (COLOR_QUESTION, "green"),
    ("", "B"),
])
def test_non_option_candidates_are_unchanged(question, candidate):
    assert _describe_option_candidate(candidate, question) == candidate


def test_verifier_context_names_option_text_for_candidate_and_pool_entry():
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(),
        AgentConfig(api_base="http://fake", model_name="fake"))
    pool = DraftPool()
    pool.add(PoolEntry(step_idx=0, answer="B", reasoning="label C is the gaster"))
    try:
        messages = loop._build_main_context(
            Image.new("RGB", (4, 4)), LABEL_QUESTION, pool, 1)
    finally:
        loop.shutdown()

    texts = [part["text"] for msg in messages if isinstance(msg["content"], list)
             for part in msg["content"] if part.get("type") == "text"]
    pool_text = next(t for t in texts if "Candidate answer (not evidence)" in t)
    candidate_block = next(t for t in texts if "DRAFT CANDIDATE UNDER VERIFICATION" in t)
    for text in (pool_text, candidate_block):
        assert 'B (option B: "C")' in text
        assert 'denotes the image label "C"' in text
