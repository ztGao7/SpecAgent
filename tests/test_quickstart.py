"""Quickstart failures must be detected before any GPU service is started."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_subset_is_stable_and_deduplicated_without_answer_selection(tmp_path):
    table = tmp_path / "data.tsv"
    table.write_text("index\tquestion\tanswer\timage\n9\tQ\twrong\tb64\n9\tQ\tright\tb64\n2\tQ\tright\tb64\n4\tQ\twrong\tb64\n")
    assert module("make_smoke_indices").select(table, 2) == ["9", "2"]


def test_subset_refuses_silent_short_run(tmp_path):
    table = tmp_path / "data.tsv"
    table.write_text("index\tquestion\n1\tQ\n")
    with pytest.raises(ValueError, match="only 1 available"):
        module("make_smoke_indices").select(table, 2)


def test_html_error_cannot_be_used_as_dataset(tmp_path):
    table = tmp_path / "data.tsv"
    table.write_text("<html>download denied</html>")
    with pytest.raises(ValueError, match="Not a visual-evaluation TSV"):
        module("download_data").validate_table(table)


def test_model_run_without_dataset_fails_before_creating_run_directory(tmp_path):
    work = tmp_path / "data disk with spaces"
    env = dict(os.environ, SPECAGENT_WORK_ROOT=str(work), CLIENT_PYTHON=sys.executable,
               LMUData=str(work / "missing"))
    result = subprocess.run(["bash", str(ROOT / "scripts/run_test.sh"), "--method", "spectool"],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Dataset missing" in result.stderr
    assert not (work / "runs").exists()


@pytest.mark.parametrize("method", ["spectool", "speceyes", "specreason", "baseline", "direct", "draft-only"])
def test_each_method_has_side_effect_free_plan(method, tmp_path):
    work = tmp_path / "not-created"
    env = dict(os.environ, SPECAGENT_WORK_ROOT=str(work))
    result = subprocess.run(["bash", str(ROOT / "scripts/run_test.sh"), "--method", method, "--dry-run"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '"samples": 2' in result.stdout
    assert "--index-file" in result.stdout
    assert not work.exists()


def test_full_dataset_is_explicit_and_does_not_keep_smoke_subset():
    result = subprocess.run(["bash", str(ROOT / "scripts/run_test.sh"), "--method", "baseline", "--full", "--dry-run"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '"samples": "all"' in result.stdout
    assert "--index-file" not in result.stdout


def test_environment_plan_preserves_data_disk_path_and_pinned_cuda(tmp_path):
    work = tmp_path / "data disk"
    env = dict(os.environ, SPECAGENT_WORK_ROOT=str(work), CONDA_BASE=str(work / "miniconda3"),
               CLIENT_PYTHON=str(work / "miniconda3/envs/spectool/bin/python"),
               SERVER_PYTHON=str(work / "miniconda3/envs/vllm0251/bin/python"))
    result = subprocess.run(["bash", str(ROOT / "scripts/setup_env.sh"), "--dry-run"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "vllm-0.25.1+cu129" in result.stdout
    assert "python=3.10" in result.stdout and "python=3.12" in result.stdout
    assert not work.exists()
