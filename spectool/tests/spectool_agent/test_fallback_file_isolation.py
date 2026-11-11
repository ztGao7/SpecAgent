import contextlib
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent import sandbox
from spectool_agent.pool import DraftPool
from spectool_agent.speculative_agent import CancellableTask, SpeculativeAgentLoop
from spectool_agent.tools import ToolRegistry, register_default_tools
from spectool_agent.types import AgentConfig, Trajectory, ToolCall


@pytest.fixture(autouse=True)
def no_live_sandbox_cleanup(monkeypatch):
    monkeypatch.setattr(sandbox, 'clear_session', lambda context: None)


def make_registry(images):
    registry = ToolRegistry()
    register_default_tools(registry)
    registry.set_input_images(images)
    return registry


def test_clean_child_runs_real_collector_without_replaying_old_files(monkeypatch):
    sessions = {}
    monkeypatch.setenv('SPECTOOL_SANDBOX_SHARED_FS', '1')

    def run_test_code(session_id, code, timeout):
        namespace = sessions.setdefault(session_id, {})
        stdout = io.StringIO()
        # Only the fixed test snippets below are executed, never model output.
        with contextlib.redirect_stdout(stdout):
            exec(code, namespace)
        return {'status': 'success', 'output': {
            'stdout': stdout.getvalue(), 'stderr': '', 'result': '', 'images': []}}, None

    monkeypatch.setattr(sandbox, '_post_code', run_test_code)
    originals = [Image.new('RGB', (8, 8), 'red'), Image.new('RGB', (6, 9), 'blue')]
    parent = make_registry(originals)
    child = None
    old_root = parent._output_dir
    try:
        _, old_images = parent.dispatch(ToolCall('sandbox', {'code':
            "Image.new('RGB', (3, 3), 'green').save(os.path.join(temp_output_dir, 'draft.png'))"
        }), originals[0])
        assert len(old_images) == 1
        old_session = parent._previous_context['session_id']
        # Even a Draft overwrite of its input file must not become Main's input.
        Image.new('RGB', (8, 8), 'black').save(parent._input_image_path)
        child = parent.fork_clean_sandbox(originals)
        child_root = child._output_dir
        assert child_root != old_root
        assert child._tools['sandbox'].__self__ is child
        assert Path(old_root, 'draft.png').is_file()
        assert [Image.open(p).tobytes() for p in child._input_image_paths] == [
            image.tobytes() for image in originals]
        observation, images = child.dispatch(ToolCall('sandbox', {'code':
            'print(image_1.getpixel((0, 0)), image_2.size)'}), originals[0])
        assert images == []
        assert '(255, 0, 0)' in observation and '(6, 9)' in observation
        assert child._previous_context['session_id'] != old_session

        child.dispatch(ToolCall('sandbox', {'code': 'main_value = 42'}), originals[0])
        observation, images = child.dispatch(ToolCall('sandbox', {'code':
            "print(main_value)\nImage.new('RGB', (2, 4), 'yellow').save(os.path.join(temp_output_dir, 'main.png'))"
        }), originals[0])
        assert '42' in observation
        assert [image.size for image in images] == [(2, 4)]
        assert not Path(child_root, 'draft.png').exists()
        child.cleanup()
        assert not Path(child_root).exists()
        assert Path(old_root, 'draft.png').is_file()
    finally:
        if child is not None:
            child.cleanup()
        parent.cleanup()
    assert not Path(old_root).exists()


def test_late_draft_tool_keeps_its_own_directory_during_main_cleanup(monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = Image.new('RGB', (8, 8), 'red')
    parent = make_registry([original])
    child = None

    def execute(**kwargs):
        if kwargs['code'] == 'late_draft':
            started.set()
            assert release.wait(3)
        path = Path(kwargs['output_dir'], kwargs['code'] + '.png')
        Image.new('RGB', (3, 3), 'green').save(path)
        return sandbox.SandboxResult(success=True, output='ok', images=[str(path)],
                                     execution_context={'session_id': kwargs['code']})

    monkeypatch.setattr(sandbox, 'execute', execute)
    with ThreadPoolExecutor(max_workers=1) as executor:
        draft_job = executor.submit(parent.dispatch,
            ToolCall('sandbox', {'code': 'late_draft'}), original)
        try:
            assert started.wait(2)
            # Must not wait for or reset the still-executing Draft tool.
            child = parent.fork_clean_sandbox([original])
            root = child._output_dir
            _, images = child.dispatch(ToolCall('sandbox', {'code': 'main'}), original)
            assert len(images) == 1
            child.cleanup()
            assert not Path(root).exists()
            release.set()
            assert len(draft_job.result(timeout=2)[1]) == 1
            assert Path(parent._output_dir, 'late_draft.png').exists()
            assert not Path(parent._output_dir, 'main.png').exists()
        finally:
            release.set()
            if child is not None:
                child.cleanup()
            parent.cleanup()


def test_child_preserves_custom_tools_python_alias_and_sample_audit():
    original = Image.new('RGB', (3, 3))
    parent = make_registry([original])
    parent.register('PythonInterpreter', parent._sandbox)
    custom = lambda image, **kwargs: ('external evidence', [])
    parent.register('search', custom)
    parent._mmsearch_module = object()
    parent._tool_audit_context = {'sample_id': 'same-sample'}
    child = parent.fork_clean_sandbox([original])
    try:
        assert child.names() == parent.names()
        assert child._tools['PythonInterpreter'].__self__ is child
        assert child._tools['search'] is custom
        assert child._mmsearch_module is parent._mmsearch_module
        assert child._tool_audit_context is parent._tool_audit_context
    finally:
        child.cleanup()
        parent.cleanup()


def test_real_fallback_loop_dispatches_tools_only_in_child(monkeypatch):
    original = Image.new('RGB', (8, 8), 'red')
    parent = make_registry([original])
    parent_root = parent._output_dir
    parent._previous_context = {'session_id': 'draft-session'}
    Image.new('RGB', (2, 2), 'green').save(Path(parent_root, 'draft.png'))
    dispatch_dirs = []

    def execute(**kwargs):
        dispatch_dirs.append(kwargs['output_dir'])
        assert kwargs['previous_context'] is None
        path = Path(kwargs['output_dir'], 'main.png')
        Image.new('RGB', (2, 2), 'blue').save(path)
        images = [str(p) for p in Path(kwargs['output_dir']).glob('*.png')
                  if p.name != 'input.png']
        return sandbox.SandboxResult(success=True, output='main evidence', images=images,
                                     execution_context={'session_id': 'main-session'})

    monkeypatch.setattr(sandbox, 'execute', execute)
    main = MagicMock()
    requests = []

    def generate(messages, **kwargs):
        requests.append(messages)
        if len(requests) == 1:
            main.last_tool_calls = [ToolCall('sandbox', {'code': 'new crop'})]
            main.last_tool_call_ids = ['main-call']
            main.last_assistant_msg = {'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'main-call', 'type': 'function',
                    'function': {'name': 'sandbox', 'arguments': '{"code":"new crop"}'}}]}
            return ''
        main.last_tool_calls = []
        return '<answer>A</answer>'

    main.generate.side_effect = generate
    loop = SpeculativeAgentLoop(main, MagicMock(), parent, AgentConfig(
        api_base='http://fake', model_name='fake', max_steps=3))
    traj = Trajectory()
    try:
        loop._run_fallback_agent(original, 'Q', DraftPool(), traj, CancellableTask())
        assert traj.final_answer == 'A' and traj.fallback_steps_used == 2
        assert len(dispatch_dirs) == 1 and dispatch_dirs[0] != parent_root
        assert not Path(dispatch_dirs[0]).exists()
        images = [part for message in requests[-1]
                  if isinstance(message['content'], list)
                  for part in message['content'] if part.get('type') == 'image_url']
        assert len(images) == 2  # Original plus Main crop, never Draft crop.
        assert parent._previous_context == {'session_id': 'draft-session'}
        assert Path(parent_root, 'draft.png').exists()
    finally:
        loop.shutdown()
        parent.cleanup()


def test_failed_child_image_setup_removes_partial_directory(monkeypatch):
    original = Image.new('RGB', (4, 4))
    parent = make_registry([original])
    created = []
    real_setup = ToolRegistry.set_input_images

    def fail_after_setup(child, images):
        real_setup(child, images)
        created.append(child._output_dir)
        raise OSError('image setup failed')

    monkeypatch.setattr(ToolRegistry, 'set_input_images', fail_after_setup)
    try:
        with pytest.raises(OSError, match='image setup failed'):
            parent.fork_clean_sandbox([original])
        assert created and not Path(created[0]).exists()
        assert Path(parent._output_dir).exists()
    finally:
        parent.cleanup()


@pytest.mark.parametrize('fail', [False, True])
def test_fallback_child_is_cleaned_on_success_and_exception(monkeypatch, fail):
    original = Image.new('RGB', (4, 4))
    parent = make_registry([original])
    root = parent._output_dir
    parent._previous_context = {'session_id': 'draft'}
    child_dirs = []
    loop = SpeculativeAgentLoop(MagicMock(), MagicMock(), parent, AgentConfig(
        api_base='http://fake', model_name='fake'))

    def run_steps(*args, tools, **kwargs):
        child_dirs.append(tools._output_dir)
        assert tools is not parent
        assert tools._previous_context is None
        if fail:
            raise RuntimeError('model failed')

    monkeypatch.setattr(loop, '_run_fallback_steps', run_steps)
    try:
        if fail:
            with pytest.raises(RuntimeError, match='model failed'):
                loop._run_fallback_agent(original, 'Q', DraftPool(), Trajectory(), CancellableTask())
        else:
            loop._run_fallback_agent(original, 'Q', DraftPool(), Trajectory(), CancellableTask())
        assert child_dirs and all(not Path(path).exists() for path in child_dirs)
        assert Path(root).exists()
        assert parent._previous_context == {'session_id': 'draft'}
    finally:
        loop.shutdown()
        parent.cleanup()


@pytest.mark.parametrize('clean,trigger', [(True, 'fallback_only'), (False, 'quality_bad_streak')])
def test_baseline_and_explicit_legacy_takeover_do_not_fork(monkeypatch, clean, trigger):
    original = Image.new('RGB', (4, 4))
    parent = make_registry([original])
    root = parent._output_dir
    parent.fork_clean_sandbox = MagicMock(side_effect=AssertionError('unexpected copy'))
    loop = SpeculativeAgentLoop(MagicMock(), MagicMock(), parent, AgentConfig(
        api_base='http://fake', model_name='fake', fallback_clean_start=clean))
    captured = []
    monkeypatch.setattr(loop, '_run_fallback_steps',
                        lambda *args, tools, **kwargs: captured.append(tools))
    try:
        loop._run_fallback_agent(original, 'Q', DraftPool(), Trajectory(),
                                 CancellableTask(), trigger=trigger)
        assert captured == [parent]
        assert parent._output_dir == root and Path(root).exists()
        parent.fork_clean_sandbox.assert_not_called()
    finally:
        loop.shutdown()
        parent.cleanup()
