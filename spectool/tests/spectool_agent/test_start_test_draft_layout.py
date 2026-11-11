import os
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(os.environ.get(
    "SPECTOOL_TEST_LAUNCHER",
    Path(__file__).resolve().parents[3] / "start_test.sh",
))
LAYOUT_COMMAND = r'''
source "$1"
resolve_draft_layout
printf 'groups=%s\n' "${DRAFT_GPU_GROUP_LIST[*]}"
printf 'ports=%s\n' "${DRAFT_PORT_LIST[*]}"
printf 'names=%s\n' "${DRAFT_SERVICE_NAMES[*]}"
draft_base_urls
'''
BASELINE_COMMAND = r'''
source "$1"
resolve_mode
printf 'draft_urls='
draft_base_urls
'''
DECISION_GATE_COMMAND = r'''
source "$1"
parse_args --answerability-threshold 0.7 --sample-wall-budget 180 --max-steps 9
resolve_mode
printf 'answerability=%s\n' "$ANSWERABILITY_THRESHOLD"
printf 'wall_budget=%s\n' "$SAMPLE_WALL_BUDGET_SECONDS"
printf 'max_steps=%s\n' "$MAX_STEPS"
'''

STEP_BUDGET_COMMAND = r'''
source "$1"
parse_args --max-steps 9
resolve_mode
export_step_budget
printf 'max_steps=%s\n' "$MAX_STEPS"
printf 'spectool=%s\n' "$SPECTOOL_MAX_STEPS"
printf 'mathvision=%s\n' "$SPECTOOL_MATHVISION_MAX_STEPS"
printf 'speceyes=%s\n' "$SPECEYE_MAX_STEPS"
printf 'mmsearch=%s\n' "$MMSEARCH_MAX_STEPS"
'''


def _run_layout(**overrides):
    env = os.environ.copy()
    env.update({
        "SPECTOOL_START_TEST_SOURCE_ONLY": "1",
        "RUN_MODE": "spectool",
        "MAIN_GPUS": "0,1",
    })
    env.update({key: str(value) for key, value in overrides.items()})
    return subprocess.run(
        ["bash", "-c", LAYOUT_COMMAND, "_", str(LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_default_draft_layout_is_single_gpu_tp1():
    result = _run_layout()

    assert result.returncode == 0, result.stderr
    assert "groups=4" in result.stdout
    assert "ports=8129" in result.stdout
    assert "names=draft" in result.stdout
    assert "http://127.0.0.1:8129/v1" in result.stdout


def test_two_tp1_replicas_get_distinct_gpus_ports_and_names():
    result = _run_layout(
        DRAFT_REPLICAS=2,
        DRAFT_TP=1,
        DRAFT_GPUS="4,5",
        DRAFT_PORT=8129,
    )

    assert result.returncode == 0, result.stderr
    assert "groups=4 5" in result.stdout
    assert "ports=8129 8130" in result.stdout
    assert "names=draft_1 draft_2" in result.stdout
    assert (
        "http://127.0.0.1:8129/v1,http://127.0.0.1:8130/v1"
        in result.stdout
    )


def test_duplicate_replica_gpu_is_rejected():
    result = _run_layout(
        DRAFT_REPLICAS=2,
        DRAFT_TP=1,
        DRAFT_GPUS="4,4",
    )

    assert result.returncode != 0
    assert "assigned to more than one replica" in result.stderr


def test_baseline_mode_does_not_require_a_draft_layout():
    env = os.environ.copy()
    env.update({
        "SPECTOOL_START_TEST_SOURCE_ONLY": "1",
        "RUN_MODE": "baseline",
    })
    result = subprocess.run(
        ["bash", "-c", BASELINE_COMMAND, "_", str(LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "draft_urls=\n"


def test_active_decision_gate_options_are_parsed():
    env = os.environ.copy()
    env.update({
        "SPECTOOL_START_TEST_SOURCE_ONLY": "1",
        "RUN_MODE": "spectool",
        "MAIN_GPUS": "0,1",
    })
    result = subprocess.run(
        ["bash", "-c", DECISION_GATE_COMMAND, "_", str(LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "answerability=0.7" in result.stdout
    assert "wall_budget=180" in result.stdout
    assert "max_steps=9" in result.stdout


def test_max_steps_is_the_single_launcher_step_budget():
    env = os.environ.copy()
    env.update({
        "SPECTOOL_START_TEST_SOURCE_ONLY": "1",
        "RUN_MODE": "spectool",
        "MAIN_GPUS": "0,1",
        "AGENT_MAX_STEPS": "3",
        "SPECEYE_MAX_STEPS": "4",
        "SPECTOOL_MATHVISION_MAX_STEPS": "5",
        "MMSEARCH_MAX_STEPS": "6",
    })
    result = subprocess.run(
        ["bash", "-c", STEP_BUDGET_COMMAND, "_", str(LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "max_steps=9",
        "spectool=9",
        "mathvision=9",
        "speceyes=9",
        "mmsearch=9",
    ]


def test_max_steps_must_be_positive():
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; parse_args --max-steps 0; resolve_mode',
         "_", str(LAUNCHER)],
        env={**os.environ, "SPECTOOL_START_TEST_SOURCE_ONLY": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "MAX_STEPS must be a positive integer" in result.stderr


@pytest.mark.parametrize("option", [
    "--skip-draft",
    "--fast-answer-threshold 0.7",
    "--fallback-shared-budget",
    "--no-fallback-shared-budget",
    "--fallback-min-steps 2",
])
def test_removed_legacy_options_are_rejected(option):
    result = subprocess.run(
        ["bash", "-c", f'source "$1"; parse_args {option}', "_", str(LAUNCHER)],
        env={**os.environ, "SPECTOOL_START_TEST_SOURCE_ONLY": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Unknown option" in result.stderr


def test_old_fast_answer_threshold_environment_does_not_override_new_gate():
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s\\n" "$ANSWERABILITY_THRESHOLD"',
         "_", str(LAUNCHER)],
        env={
            **os.environ,
            "SPECTOOL_START_TEST_SOURCE_ONLY": "1",
            "ANSWERABILITY_THRESHOLD": "",
            "SPECTOOL_FAST_ANSWER_THRESHOLD": "0.1",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "0.95\n"


def test_help_lists_current_decision_options_only():
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--help"],
        env={**os.environ, "SPECTOOL_START_TEST_SOURCE_ONLY": "0"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--answerability-threshold" in result.stdout
    assert "--max-steps" in result.stdout
    for removed in (
        "--skip-draft",
        "--fast-answer-threshold",
        "--fallback-shared-budget",
        "--no-fallback-shared-budget",
        "--fallback-min-steps",
    ):
        assert removed not in result.stdout
