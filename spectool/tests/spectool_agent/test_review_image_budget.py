from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent.pool import DraftPool
from spectool_agent.speculative_agent import SpeculativeAgentLoop
from spectool_agent.tools import ToolRegistry
from spectool_agent.types import AgentConfig, PoolEntry, ToolCall, Trajectory


def image_urls(messages):
    return [part['image_url']['url']
            for message in messages if isinstance(message['content'], list)
            for part in message['content'] if part.get('type') == 'image_url']


@pytest.mark.parametrize('budget,originals,expected', [
    (5, 1, 5), (5, 2, 5), (50, 1, 15), (1, 2, 2), (0, 1, 1),
])
def test_actual_review_request_obeys_budget_without_mutating_pool(
    monkeypatch, budget, originals, expected,
):
    calls = []

    def encode(image, **kwargs):
        calls.append(image)
        return f'image-{id(image)}'

    monkeypatch.setattr('spectool_agent.speculative_agent.pil_to_data_url', encode)
    main = MagicMock()
    main.judge_good_bad.return_value = {'score': .9, 'raw': 'GOOD'}
    loop = SpeculativeAgentLoop(main, MagicMock(), ToolRegistry(), AgentConfig(
        api_base='http://fake', model_name='fake', max_image_history=budget))
    inputs = [Image.new('RGB', (8, 8), (i, 0, 0)) for i in range(originals)]
    crops = [Image.new('RGB', (4, 4), (i, 1, 1)) for i in range(16)]
    pool = DraftPool()
    for step in range(2):
        pool.add(PoolEntry(step_idx=step, tool_observation=f'evidence-{step}',
                           tool_call=ToolCall('sandbox', {'code': 'print(1)'}),
                           tool_result_images=crops[step*8:(step+1)*8]))
    pool.add(PoolEntry(step_idx=2, answer='A', reasoning='candidate rationale'))
    shared = loop._shared_prompt_context(inputs, 'Question')
    shared_before = deepcopy(shared)
    original_urls = [f'image-{id(image)}' for image in inputs]
    try:
        result = loop._run_main_review(inputs, 'Question', pool, 3, Trajectory(),
                                       shared_ctx=shared)
        messages = main.judge_good_bad.call_args.args[0]
        retained = expected - originals
        assert image_urls(messages) == original_urls + [
            f'image-{id(image)}' for image in (crops[-retained:] if retained else [])]
        assert len(calls) == expected  # Omitted images are not encoded.
        assert result['candidate_answer'] == 'A'
        assert result['supported'] is True
        assert all(f'evidence-{i}' in repr(messages) for i in range(2))
        assert 'candidate rationale' in repr(messages)
        assert repr(messages).count('[之前图片已省略]') == 16 - retained
        assert shared == shared_before
        assert pool.get_all()[0].tool_result_images == crops[:8]
        assert pool.get_all()[1].tool_result_images == crops[8:]
        assert all(isinstance(url, str) for url in image_urls(messages))
    finally:
        loop.shutdown()


def test_review_under_budget_keeps_all_evidence(monkeypatch):
    monkeypatch.setattr('spectool_agent.speculative_agent.pil_to_data_url',
                        lambda image, **kwargs: f'image-{id(image)}')
    main = MagicMock()
    main.judge_good_bad.return_value = {'score': .8, 'raw': 'GOOD'}
    loop = SpeculativeAgentLoop(main, MagicMock(), ToolRegistry(), AgentConfig(
        api_base='http://fake', model_name='fake', max_image_history=5))
    original, crop = Image.new('RGB', (8, 8)), Image.new('RGB', (4, 4))
    pool = DraftPool()
    pool.add(PoolEntry(step_idx=0, tool_result_images=[crop], tool_observation='kept',
                       tool_call=ToolCall('sandbox', {'code': 'print(1)'})))
    pool.add(PoolEntry(step_idx=1, answer='B'))
    try:
        loop._run_main_review(original, 'Question', pool, 2, Trajectory())
        messages = main.judge_good_bad.call_args.args[0]
        assert image_urls(messages) == [f'image-{id(original)}', f'image-{id(crop)}']
        assert '[之前图片已省略]' not in repr(messages)
        assert 'kept' in repr(messages)
    finally:
        loop.shutdown()


def test_review_rejects_impossible_original_count_before_api_call():
    main = MagicMock()
    loop = SpeculativeAgentLoop(main, MagicMock(), ToolRegistry(), AgentConfig(
        api_base='http://fake', model_name='fake', max_image_history=50))
    pool = DraftPool()
    pool.add(PoolEntry(step_idx=0, answer='A'))
    shared = {'image_content': [
        {'type': 'image_url', 'image_url': {'url': f'original-{i}'}} for i in range(16)],
        'question_text': 'Q'}
    try:
        with pytest.raises(ValueError, match='15 original images'):
            loop._run_main_review(Image.new('RGB', (2, 2)), 'Q', pool, 1,
                                   Trajectory(), shared_ctx=shared)
        main.judge_good_bad.assert_not_called()
    finally:
        loop.shutdown()
