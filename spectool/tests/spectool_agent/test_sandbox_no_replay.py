from unittest.mock import MagicMock

import pytest
import requests
from PIL import Image

from spectool_agent import sandbox


@pytest.mark.parametrize('failure', ['timeout', 'http500', 'invalid_json', 'wrong_id'])
def test_unknown_outcome_is_never_replayed(monkeypatch, failure):
    post = MagicMock()
    response = post.return_value
    response.status_code, response.text = 500, 'Internal Server Error'
    if failure == 'timeout':
        post.side_effect = requests.ReadTimeout('reply lost')
    elif failure == 'http500':
        response.raise_for_status.side_effect = requests.HTTPError('500')
    elif failure == 'invalid_json':
        response.json.side_effect = ValueError('invalid JSON')
    else:
        response.json.return_value = {'request_id': 'wrong', 'status': 'success'}
    monkeypatch.setattr(sandbox.requests, 'post', post)
    payload, error = sandbox._post_code('session', 'counter += 1', 200, max_retries=3)
    assert payload is None and 'not replayed' in error
    assert post.call_count == 1
    assert post.call_args.kwargs['json']['request_id']
    assert post.call_args.kwargs['timeout'] == 240


@pytest.mark.parametrize('shared', ['0', '1'])
def test_unknown_session_is_quarantined_before_next_dispatch(tmp_path, monkeypatch, shared):
    monkeypatch.setenv('SPECTOOL_SANDBOX_SHARED_FS', shared)
    post = MagicMock(return_value=(None, 'ReadTimeout: unknown outcome'))
    monkeypatch.setattr(sandbox, '_post_code', post)
    image_path = tmp_path / 'input.png'
    Image.new('RGB', (4, 4)).save(image_path)
    first = sandbox.execute('write_file()', str(image_path), str(tmp_path))
    second = sandbox.execute('write_file_again()', str(image_path), str(tmp_path),
                             previous_context=first.execution_context)
    assert not first.success and not second.success
    assert 'quarantined' in second.error and post.call_count == 1


def test_confirmed_deadline_error_allows_next_corrective_step(tmp_path, monkeypatch):
    monkeypatch.setenv('SPECTOOL_SANDBOX_SHARED_FS', '1')
    post = MagicMock(side_effect=[(
        {'status': 'error', 'output': {}, 'error': 'SandboxDeadlineExceeded: stopped',
         'session_usable': True, 'state_committed': False}, None), (
        {'status': 'success', 'output': {'stdout': 'recovered', 'stderr': '', 'images': []},
         'session_usable': True, 'state_committed': True}, None)])
    monkeypatch.setattr(sandbox, '_post_code', post)
    first = sandbox.execute('slow()', 'input.png', str(tmp_path))
    second = sandbox.execute('print(image_1.size)', 'input.png', str(tmp_path),
                             previous_context=first.execution_context)
    assert not first.success and second.success and second.output == 'recovered'
    assert post.call_args_list[0].args[0] == post.call_args_list[1].args[0]


def test_explicit_uncertain_commit_quarantines(tmp_path):
    context = {'session_id': 'same'}
    result = sandbox._response_to_result({'status': 'error', 'output': {},
        'error': 'Redis reply lost', 'session_usable': False}, str(tmp_path), context)
    assert not result.success and result.execution_context['unusable_reason']
