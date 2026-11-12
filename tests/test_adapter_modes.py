"""--dreamr / --specreason select their controller from the flag alone and prepare their runtime once."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

FREEZE_STUB = '''\
import argparse, pathlib, time
parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
args = parser.parse_args()
here = pathlib.Path(__file__).resolve().parent
with (here / "calls.log").open("a") as stream:
    stream.write(args.source + "\\n")
time.sleep(1)
(here / "runtime").mkdir()
(here / "runtime" / "manifest.json").write_text("{}")
'''


def clean_env(tmp_path, **extra):
    env = {key: value for key, value in os.environ.items() if not key.startswith(
        ('MAIN_', 'DRAFT_', 'SANDBOX_', 'SPECAGENT_', 'CLIENT_', 'SERVER_', 'CONDA_', 'SPECTOOL_', 'RUN_'))}
    (tmp_path / 'empty.env').touch()
    env.update(SPECAGENT_ENV_FILE=str(tmp_path / 'empty.env'), CLIENT_PYTHON=sys.executable,
               SPECTOOL_START_TEST_SOURCE_ONLY='1', SPECAGENT_WORK_ROOT=str(tmp_path / 'work'))
    env.update(extra)
    return env


def run_shell(code, repo, env, *args):
    return subprocess.run(['bash', '-c', code, '_', str(repo), *args], env=env, capture_output=True, text=True)


@pytest.fixture
def fake_repo(tmp_path):
    repo = tmp_path / 'repo'
    shutil.copytree(ROOT / 'scripts', repo / 'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT / 'start_test.sh', repo / 'start_test.sh')
    (repo / 'spectool').mkdir()
    for name in ('dreamr', 'specreason'):
        (repo / 'baselines' / name).mkdir(parents=True)
        (repo / 'baselines' / name / 'freeze_runtime.py').write_text(FREEZE_STUB)
    return repo


PARSE = 'source "$1/start_test.sh"; shift; parse_args "$@"; printf "%s|%s" "$RUN_MODE" "$SPECAGENT_EVAL_DRIVER"'


@pytest.mark.parametrize('flags,expected', [
    (['--dreamr'], 'spectool|dreamr'),
    (['--specreason'], 'spectool|specreason'),
    (['--spectool'], 'spectool|'),
    (['--speceyes'], 'speceyes|'),
    (['--baseline'], 'baseline|'),
    (['--direct'], 'direct|'),
    (['--draft-only'], 'draft-only|'),
    ([], 'spectool|'),
    (['--dreamr', '--baseline'], 'baseline|'),
    (['--baseline', '--specreason'], 'spectool|specreason'),
    (['--specreason', '--spectool'], 'spectool|'),
])
def test_mode_flags_select_the_controller_and_the_last_flag_wins(flags, expected, tmp_path):
    result = run_shell(PARSE, ROOT, clean_env(tmp_path), *flags)
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


@pytest.mark.parametrize('stale', ['dreamr', 'specreason', 'stale-driver'])
@pytest.mark.parametrize('flags,expected', [(['--baseline'], 'baseline|'), (['--spectool'], 'spectool|'), ([], 'spectool|')])
def test_an_inherited_driver_variable_cannot_hijack_other_methods(stale, flags, expected, tmp_path):
    result = run_shell(PARSE, ROOT, clean_env(tmp_path, SPECAGENT_EVAL_DRIVER=stale), *flags)
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


def test_help_lists_both_flags_with_examples(tmp_path):
    result = subprocess.run(['bash', str(ROOT / 'start_test.sh'), '--help'],
                            env=clean_env(tmp_path, SPECTOOL_START_TEST_SOURCE_ONLY='0'), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--specreason   ' in result.stdout and '--dreamr   ' in result.stdout
    assert 'start_test.sh --specreason --bench vstar' in result.stdout


PREPARE = 'source "$1/start_test.sh"; shift; parse_args "$@"; prepare_adapter_runtime'


@pytest.mark.parametrize('flag,name', [('--dreamr', 'dreamr'), ('--specreason', 'specreason')])
def test_runtime_is_prepared_once_from_the_shared_source(flag, name, fake_repo, tmp_path):
    env = clean_env(tmp_path)
    for _ in range(2):
        result = run_shell(PREPARE, fake_repo, env, flag)
        assert result.returncode == 0, result.stderr
    adapter = fake_repo / 'baselines' / name
    assert (adapter / 'runtime/manifest.json').is_file()
    assert (adapter / 'calls.log').read_text().splitlines() == [str(fake_repo / 'spectool')]
    assert not (adapter / '.freeze.lock').exists()
    other = fake_repo / 'baselines' / ('specreason' if name == 'dreamr' else 'dreamr')
    assert not (other / 'runtime').exists()


@pytest.mark.parametrize('flag', ['--spectool', '--speceyes', '--baseline', '--direct', '--draft-only'])
def test_other_methods_never_touch_adapter_runtimes(flag, fake_repo, tmp_path):
    result = run_shell(PREPARE, fake_repo, clean_env(tmp_path), flag)
    assert result.returncode == 0, result.stderr
    assert not list((fake_repo / 'baselines').rglob('calls.log'))


def test_parallel_launches_prepare_the_runtime_exactly_once(fake_repo, tmp_path):
    env = clean_env(tmp_path)
    procs = [subprocess.Popen(['bash', '-c', PREPARE, '_', str(fake_repo), '--specreason'], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
    results = [(p.wait(timeout=60), p.stderr.read()) for p in procs]
    assert [code for code, _ in results] == [0, 0, 0, 0], results
    adapter = fake_repo / 'baselines/specreason'
    assert len((adapter / 'calls.log').read_text().splitlines()) == 1
    assert (adapter / 'runtime/manifest.json').is_file()
    assert not (adapter / '.freeze.lock').exists()


def test_failed_preparation_stops_the_launch_and_releases_the_lock(fake_repo, tmp_path):
    adapter = fake_repo / 'baselines/dreamr'
    (adapter / 'freeze_runtime.py').write_text('raise SystemExit("source runtime is missing")\n')
    result = run_shell(PREPARE, fake_repo, clean_env(tmp_path), '--dreamr')
    assert result.returncode != 0
    assert 'Could not prepare' in result.stderr and 'source runtime is missing' in result.stderr
    assert not (adapter / '.freeze.lock').exists()
    assert not (adapter / 'runtime').exists()


def test_quick_entry_forwards_the_adapter_flag_instead_of_an_environment_variable(tmp_path):
    for method in ('specreason', 'dreamr'):
        result = subprocess.run(['bash', str(ROOT / 'scripts/run_test.sh'), '--method', method, '--dry-run'],
                                env=dict(os.environ, SPECAGENT_WORK_ROOT=str(tmp_path / 'w'), SPECAGENT_EVAL_DRIVER='stale-driver'),
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        command = result.stdout.strip().splitlines()[-1]
        assert f'start_test.sh --{method} ' in command
        assert '--spectool' not in command
        assert f'"eval_driver": "{method}"' in result.stdout
