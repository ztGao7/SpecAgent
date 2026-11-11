"""Exercise published shell entry points with a local .env and external paths."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    # test.sh exports its own resolved paths; each fixture represents a new clone
    # opened in a clean terminal, rather than inheriting that parent's overrides.
    for key in list(os.environ):
        if key.startswith(('SPECAGENT_', 'MAIN_', 'DRAFT_', 'CLIENT_', 'SERVER_', 'CONDA_', 'PYTORCH_', 'VLLM_')) or key in {
            'RESOURCE_DIR', 'LMUData', 'CODE_ROOT', 'CONDA_INIT', 'SPECTOOL_ROOT', 'SANDBOX_ROOT',
        }:
            monkeypatch.delenv(key)
    repo = tmp_path / 'checkout with spaces'
    shutil.copytree(ROOT / 'scripts', repo / 'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT / 'start_test.sh', repo / 'start_test.sh')
    return repo


def run(repo, script, *args, env=None):
    return subprocess.run(['bash', str(repo / script), *args], cwd=repo.parent,
                          env=env or os.environ.copy(), capture_output=True, text=True)


def test_dotenv_drives_install_and_quick_plan_from_another_directory(checkout, tmp_path):
    work = tmp_path / 'data disk'
    client = work / 'clients/custom/bin/python'
    server = work / 'servers/custom/bin/python'
    (checkout / '.env').write_text(
        f'SPECAGENT_WORK_ROOT="{work}"\nCLIENT_PYTHON="{client}"\nSERVER_PYTHON="{server}"\n'
        f'LMUData="{work}/existing TSVs"\nMAIN_MODEL_PATH="{work}/weights main"\n'
        'MAIN_GPUS=4\nMAIN_TP=1\nDRAFT_GPUS=5\nDRAFT_TP=1\n'
        'PYTORCH_INDEX_URL=https://torch-mirror.example/cu129\n')
    result = run(checkout, 'scripts/setup_env.sh', '--dry-run')
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith('+ ')]
    creates = [cmd for cmd in commands if 'create' in cmd]
    assert {cmd[cmd.index('-p') + 1] for cmd in creates} == {str(client.parent.parent), str(server.parent.parent)}
    pip_installs = [cmd for cmd in commands if '-m' in cmd and 'pip' in cmd]
    assert pip_installs[0][0] == str(client)
    server_install = next(cmd for cmd in pip_installs if '--extra-index-url' in cmd)
    assert server_install[server_install.index('--extra-index-url') + 1] == 'https://torch-mirror.example/cu129'
    result = run(checkout, 'scripts/run_test.sh', '--method', 'spectool', '--dry-run')
    assert result.returncode == 0, result.stderr
    assert json.dumps(str(work / 'existing TSVs/VStarBench.tsv')) in result.stdout
    assert json.dumps(str(work / 'weights main')) in result.stdout
    assert json.dumps(str(client)) in result.stdout
    assert not work.exists()


def test_environment_then_cli_override_dotenv_through_child_launchers(checkout, tmp_path):
    (checkout / '.env').write_text('MAIN_GPUS=4\nMAIN_TP=1\n')
    env = dict(os.environ, MAIN_GPUS='6', MAIN_TP='1')
    result = run(checkout, 'scripts/run_test.sh', '--method', 'spectool', '--dry-run', '--main-gpus', '7', env=env)
    assert result.returncode == 0, result.stderr
    assert '"main_gpus": "7"' in result.stdout
    # Simulate the child's common.sh read after run_test.py exports the CLI value.
    env['MAIN_GPUS'] = '7'
    result = subprocess.run(['bash', '-c', 'source "$1/scripts/common.sh"; printf "%s" "$MAIN_GPUS"', '_', str(checkout)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == '7'


def test_explicit_missing_configuration_is_an_error(checkout, tmp_path):
    env = dict(os.environ, SPECAGENT_ENV_FILE=str(tmp_path / 'missing.env'))
    result = run(checkout, 'scripts/run_test.sh', '--dry-run', env=env)
    assert result.returncode != 0
    assert 'Configuration file not found' in result.stderr


def test_startup_activates_exact_prefixes_without_name_lookup(checkout, tmp_path):
    base = tmp_path / 'conda base'
    init = base / 'etc/profile.d/conda.sh'
    init.parent.mkdir(parents=True)
    init.write_text('conda() { printf "activate:%s\\n" "$2"; }\n')
    client = tmp_path / 'custom client'
    server = tmp_path / 'custom server'
    (checkout / '.env').write_text(f'CONDA_BASE="{base}"\nCLIENT_PYTHON="{client}/bin/python"\nSERVER_PYTHON="{server}/bin/python"\n')
    code = ('export SPECTOOL_START_TEST_SOURCE_ONLY=1; source "$1/start_test.sh"; '
            'source "$CONDA_INIT"; specagent_activate_client; specagent_activate_server')
    result = subprocess.run(['bash', '-c', code, '_', str(checkout)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f'activate:{client}', f'activate:{server}']


def test_download_and_offline_check_use_same_custom_model_paths(checkout, tmp_path):
    # Only the network boundary is fake: run both real download and weight-check entry points.
    stub = tmp_path / 'network stub'
    stub.mkdir()
    (stub / 'huggingface_hub.py').write_text('''
from pathlib import Path
import json
import numpy as np
from safetensors.numpy import save_file
class HfApi:
    def __init__(self, **kwargs): pass
    def model_info(self, repo): return type('Info', (), {'sha': 'fixed-revision'})()
def snapshot_download(local_dir, **kwargs):
    root = Path(local_dir)
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'preprocessor_config.json']:
        (root / name).write_text('{}')
    (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'model.safetensors'}}))
    save_file({'weight': np.ones((1,), dtype=np.float32)}, str(root / 'model.safetensors'))
''')
    main = tmp_path / 'existing models/main'
    draft = tmp_path / 'existing models/draft'
    work = tmp_path / 'isolated work'
    (checkout / '.env').write_text(
        f'SPECAGENT_WORK_ROOT="{work}"\nCLIENT_PYTHON="{sys.executable}"\n'
        f'MAIN_MODEL_PATH="{main}"\nDRAFT_MODEL_PATH="{draft}"\n'
        'MAIN_MODEL_ID=Qwen/Qwen3.8-27B-FP8\n')
    env = dict(os.environ, PYTHONPATH=str(stub))
    result = run(checkout, 'scripts/download_models.sh', env=env)
    assert result.returncode == 0, result.stderr
    assert f'OK {main.name}:' in result.stdout and f'OK {draft.name}:' in result.stdout
    assert json.loads((main / '.specagent-revision.json').read_text())['repo_id'] == 'Qwen/Qwen3.8-27B-FP8'
    assert not (work / 'resources/Qwen').exists()
    # --check-only must remain offline even without the network stub.
    result = run(checkout, 'scripts/download_models.sh', '--check-only')
    assert result.returncode == 0, result.stderr


def test_dependency_check_never_hides_other_errors(monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('check_dependencies', ROOT / 'scripts/check_dependencies.py')
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    monkeypatch.setattr(check.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(check.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(check, 'distribution', lambda name: SimpleNamespace(
        version='0.6.0', read_text=lambda _: 'Tag: cp36-cp36m-manylinux2010_x86_64'))
    monkeypatch.setattr(check.importlib, 'import_module', lambda name: SimpleNamespace(__version__='0.6.0'))
    diagnostic = 'decord 0.6.0 is not supported on this platform\n'
    monkeypatch.setattr(check.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1, stdout=diagnostic))
    check.main()
    diagnostic += 'example 1.0 requires missing-package, which is not installed.\n'
    with pytest.raises(SystemExit) as error:
        check.main()
    assert error.value.code == 1


def test_dependency_check_does_not_accept_a_broken_native_import(monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('check_dependencies', ROOT / 'scripts/check_dependencies.py')
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    monkeypatch.setattr(check.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(check.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(check, 'distribution', lambda name: SimpleNamespace(
        version='0.6.0', read_text=lambda _: 'Tag: cp36-cp36m-manylinux2010_x86_64'))
    monkeypatch.setattr(check.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout='decord 0.6.0 is not supported on this platform\n'))
    def fail(name):
        raise ImportError('native library missing')
    monkeypatch.setattr(check.importlib, 'import_module', fail)
    with pytest.raises(ImportError, match='native library missing'):
        check.main()


def test_evalkit_optional_env_never_overwrites_exported_configuration(tmp_path, monkeypatch, caplog):
    import ast
    from types import SimpleNamespace
    source = ROOT / 'spectool/VLMEvalKit/vlmeval/__init__.py'
    # Exercise the real loader without importing all optional ML backends.
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'load_env')
    namespace = {'__file__': str(tmp_path / 'vlmeval/__init__.py')}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    namespace['load_env']()
    assert not any(record.levelname == 'ERROR' for record in caplog.records)
    (tmp_path / '.env').write_text('')
    monkeypatch.setitem(sys.modules, 'dotenv', SimpleNamespace(dotenv_values=lambda _: {
        'SPECAGENT_TEST_EXISTING': 'stale-vendored-value', 'SPECAGENT_TEST_NEW': 'legacy-value'}))
    monkeypatch.setenv('SPECAGENT_TEST_EXISTING', 'root-config-value')
    monkeypatch.delenv('SPECAGENT_TEST_NEW', raising=False)
    namespace['load_env']()
    assert os.environ['SPECAGENT_TEST_EXISTING'] == 'root-config-value'
    assert os.environ['SPECAGENT_TEST_NEW'] == 'legacy-value'
    os.environ.pop('SPECAGENT_TEST_NEW')
