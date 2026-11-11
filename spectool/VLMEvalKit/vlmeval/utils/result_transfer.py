"""Result transfer helpers for official benchmark submission formats."""

import json
import os


def MMMU_result_transfer(result_file):
    """Stub: return path unchanged for MMMU_TEST submission."""
    return os.path.abspath(result_file)


def MMTBench_result_transfer(result_file, **kwargs):
    """Stub: return path unchanged for MMT-Bench_ALL submission."""
    return os.path.abspath(result_file)
