"""Per-sample generate-call wall clock, separate from benchmark judge timing."""
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time

LOG = logging.getLogger(__name__)
SCOPE = 'model_generate_call_v1'
COLUMNS = (
    'sample_started_unix', 'sample_finished_unix', 'sample_wall_clock_s',
    'sample_timing_scope', 'sample_timing_status', 'sample_prediction_sha256',
)


def prediction_hash(value):
    if isinstance(value, dict) and 'prediction' in value and 'extra_records' in value:
        value = value['prediction']
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()


def timing_directory(work_dir, model_name, dataset_name):
    return Path(work_dir) / f'{model_name}_{dataset_name}_sample_timing'


def _record_path(directory, sample_index):
    key = hashlib.sha256(str(sample_index).encode('utf-8')).hexdigest()
    return Path(directory) / f'{key}.json'


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=True, allow_nan=False, indent=2)
            stream.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def timed_generate(*, generate, timing_dir, sample_index, model_name, message, dataset):
    # Called inside the worker, not while submitting to its work queue. The
    # returned result and exceptions are unchanged; timing I/O is outside the span.
    started_unix, started = time.time(), time.perf_counter()
    result, status = None, 'raised'
    try:
        result = generate(message=message, dataset=dataset)
        status = 'returned'
        return result
    finally:
        elapsed = time.perf_counter() - started
        finished_unix = time.time()
        record = {
            'index': str(sample_index), 'model_name': model_name, 'dataset_name': dataset,
            'sample_started_unix': started_unix,
            'sample_finished_unix': finished_unix,
            'sample_wall_clock_s': elapsed,
            'sample_timing_scope': SCOPE,
            'sample_timing_status': status,
            'sample_prediction_sha256': prediction_hash(result) if status == 'returned' else None,
        }
        try:
            _write_json(_record_path(timing_dir, sample_index), record)
        except Exception:
            LOG.warning('Could not persist sample wall clock for %s/%s; inference result unchanged',
                        dataset, sample_index, exc_info=True)


def _valid(record, prediction):
    if record.get('sample_timing_scope') != SCOPE:
        return False
    if record.get('sample_timing_status') != 'returned':
        return False
    if record.get('sample_prediction_sha256') != prediction_hash(prediction):
        return False
    for field in COLUMNS[:3]:
        value = record.get(field)
        if isinstance(value, bool):
            return False
        try:
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return float(record['sample_wall_clock_s']) >= 0


def attach_sample_timing(data, work_dir, model_name, dataset_name, previous=None):
    """Join only verified measurements, never internal Agent total_time."""
    directory = timing_directory(work_dir, model_name, dataset_name)
    old = {}
    if previous is not None and 'index' in previous:
        old = {str(row['index']): row for row in previous.to_dict('records')}
    measurements = []
    for row in data.to_dict('records'):
        index, prediction = str(row['index']), row['prediction']
        record = {}
        path = _record_path(directory, index)
        if path.exists():
            try:
                candidate = json.loads(path.read_text())
                if (candidate.get('index') == index
                        and candidate.get('model_name') == model_name
                        and candidate.get('dataset_name') == dataset_name):
                    record = candidate
            except (OSError, ValueError, AttributeError):
                LOG.warning('Unreadable sample timing: %s', path)
        if not _valid(record, prediction):
            record = old.get(index, {})
        if not _valid(record, prediction):
            record = {}
        measurements.append(record)
    for field in COLUMNS:
        data[field] = [record.get(field) for record in measurements]
    return data


def summarize_sample_wall_clock(data, result_file, logger=None):
    """Export all rows and report coverage so missing slow rows cannot disappear."""
    import pandas as pd

    logger = logger or LOG
    rows, measured = [], []
    for row in data.to_dict('records'):
        valid = _valid(row, row.get('prediction'))
        timing = {field: row.get(field) if valid else None for field in COLUMNS}
        rows.append({'index': row['index'], 'timing_available': valid, **timing})
        if valid:
            measured.append(float(row['sample_wall_clock_s']))
    total, count = len(rows), len(measured)
    mean = sum(measured) / count if count else None
    series = pd.Series(measured, dtype=float)
    summary = {
        'scope': SCOPE,
        'definition': 'Worker enters model.generate to return, measured with perf_counter.',
        'includes': ['per-call input preprocessing', 'internal waits and retries',
                     'model and tool requests', 'in-call cleanup'],
        'excludes': ['worker queue', 'upfront dataset/prompt construction', 'model deployment',
                     'result/checkpoint serialization', 'benchmark judge'],
        'total_samples': total, 'timed_samples': count, 'missing_samples': total - count,
        'coverage': count / total if total else 0.0,
        'mean_measured_sample_wall_clock_s': mean,
        'mean_all_sample_wall_clock_s': mean if count == total and total else None,
        'median_s': float(series.median()) if count else None,
        'p95_s': float(series.quantile(.95)) if count else None,
        'max_s': max(measured) if count else None,
    }
    base = Path(result_file).with_suffix('')
    csv_path = Path(str(base) + '_sample_wall_clock.csv')
    summary_path = Path(str(base) + '_sample_wall_clock_summary.json')
    pd.DataFrame(rows, columns=['index', 'timing_available', *COLUMNS]).to_csv(csv_path, index=False)
    _write_json(summary_path, summary)
    if count:
        label = 'all samples' if count == total else 'MEASURED SUBSET ONLY'
        logger.info('[sample-wall-clock] %s mean=%.3fs (%s); coverage=%d/%d; '
                    'missing=%d; excludes deployment and benchmark judge',
                    base.name, mean, label, count, total, total - count)
    else:
        logger.warning('[sample-wall-clock] %s mean=unavailable; coverage=0/%d. '
                       'No outer generate-call measurements; Agent total_time is not substituted.',
                       base.name, total)
    logger.info('[sample-wall-clock] Per-sample CSV: %s; summary: %s', csv_path, summary_path)
    return summary


def report_sample_wall_clock(result_file, logger=None):
    logger = logger or LOG
    try:
        from vlmeval.smp import load
        return summarize_sample_wall_clock(load(result_file), result_file, logger)
    except Exception:
        # Reporting must not prevent benchmark scoring or change predictions.
        logger.warning('Failed to report sample wall clock for %s', result_file, exc_info=True)
        return None
