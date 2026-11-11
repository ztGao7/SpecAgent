import importlib.util
import json
import logging
from pathlib import Path
import pickle
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sample_timing_test_target', ROOT / 'vlmeval/sample_timing.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def invoke(tmp_path, index='1', generate=None):
    return timing.timed_generate(
        generate=generate or (lambda **kwargs: 'A'), timing_dir=tmp_path,
        sample_index=index, model_name='model', message=['input'], dataset='bench')


def read_record(path, index='1'):
    return json.loads(timing._record_path(path, index).read_text())


def test_preserves_response_and_captures_complete_call(tmp_path, monkeypatch):
    clock = SimpleNamespace(wall=100.0, mono=20.0)
    monkeypatch.setattr(timing.time, 'time', lambda: clock.wall)
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: clock.mono)
    answer = {'prediction': 'A', 'extra_records': {'total_time': 2}}
    def generate(**kwargs):
        assert kwargs == {'message': ['input'], 'dataset': 'bench'}
        # The outer call also includes waits, retries and cleanup, not just 2s
        # of internal Agent time. A system-clock correction must not corrupt it.
        clock.mono += 9
        clock.wall -= 30
        return answer
    assert invoke(tmp_path, generate=generate) is answer
    record = read_record(tmp_path)
    assert record['sample_wall_clock_s'] == 9
    assert record['sample_started_unix'] == 100
    assert record['sample_finished_unix'] == 70
    assert record['sample_prediction_sha256'] == timing.prediction_hash('A')


def test_timing_does_not_include_its_own_persistence(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: clock[0])
    original = timing._write_json
    def write(path, record):
        clock[0] += 100
        original(path, record)
    monkeypatch.setattr(timing, '_write_json', write)
    def generate(**kwargs):
        clock[0] += 3
        return 'A'
    invoke(tmp_path, generate=generate)
    assert read_record(tmp_path)['sample_wall_clock_s'] == 3


def test_io_failure_never_retries_or_changes_prediction(tmp_path, monkeypatch, caplog):
    calls = []
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(timing, '_write_json', fail)
    assert invoke(tmp_path, generate=lambda **k: calls.append(k) or 'A') == 'A'
    assert len(calls) == 1 and 'inference result unchanged' in caplog.text


def test_exception_is_preserved_and_recorded(tmp_path):
    error = RuntimeError('original failure')
    def fail(**kwargs):
        raise error
    with pytest.raises(RuntimeError) as caught:
        invoke(tmp_path, generate=fail)
    assert caught.value is error
    assert read_record(tmp_path)['sample_timing_status'] == 'raised'


def test_worker_queue_is_excluded_and_parallel_spans_overlap(tmp_path):
    gate, running = threading.Event(), threading.Event()
    def hold():
        running.set()
        assert gate.wait(2)
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(hold)
        assert running.wait(1)
        task = executor.submit(invoke, tmp_path, 'queued')
        time.sleep(.04)
        released = time.time()
        gate.set()
        blocker.result()
        task.result()
    assert read_record(tmp_path, 'queued')['sample_started_unix'] >= released
    barrier = threading.Barrier(2)
    def simultaneous(**kwargs):
        barrier.wait(timeout=2)
        time.sleep(.02)
        return 'A'
    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = [executor.submit(invoke, tmp_path, str(i), simultaneous) for i in (2, 3)]
        assert [t.result() for t in tasks] == ['A', 'A']
    a, b = [read_record(tmp_path, str(i)) for i in (2, 3)]
    assert max(a['sample_started_unix'], b['sample_started_unix']) < min(
        a['sample_finished_unix'], b['sample_finished_unix'])
    assert len(list(tmp_path.glob('*.json'))) == 3


def measured_frame(tmp_path, predictions=('A', 'Failed to obtain answer via API.')):
    directory = timing.timing_directory(tmp_path, 'model', 'bench')
    for index, prediction in enumerate(predictions):
        invoke(directory, str(index), lambda prediction=prediction, **k: prediction)
    frame = pd.DataFrame({'index': list(range(len(predictions))), 'prediction': list(predictions)})
    return timing.attach_sample_timing(frame, tmp_path, 'model', 'bench')


def test_join_includes_failed_responses_but_never_internal_time(tmp_path):
    data = measured_frame(tmp_path)
    data['extra_records'] = [{'total_time': 100}, {'total_time': 200}]
    summary = timing.summarize_sample_wall_clock(data, tmp_path / 'pred.xlsx')
    assert summary['timed_samples'] == 2 and summary['missing_samples'] == 0
    assert summary['mean_all_sample_wall_clock_s'] == data.sample_wall_clock_s.mean()
    assert summary['mean_all_sample_wall_clock_s'] < 100
    exported = pd.read_csv(tmp_path / 'pred_sample_wall_clock.csv')
    assert len(exported) == 2 and all(exported.timing_available)


def test_partial_coverage_never_claims_full_mean(tmp_path, caplog):
    data = measured_frame(tmp_path)
    data.loc[1, 'sample_wall_clock_s'] = None
    with caplog.at_level(logging.INFO):
        summary = timing.summarize_sample_wall_clock(data, tmp_path / 'pred.xlsx')
    assert summary['mean_all_sample_wall_clock_s'] is None
    assert summary['timed_samples'] == 1 and summary['missing_samples'] == 1
    assert 'MEASURED SUBSET ONLY' in caplog.text
    assert len(pd.read_csv(tmp_path / 'pred_sample_wall_clock.csv')) == 2


def test_legacy_results_are_explicitly_unavailable(tmp_path, caplog):
    frame = pd.DataFrame({'index': [1], 'prediction': ['A'], 'extra_records': ['{"total_time": 99}']})
    summary = timing.summarize_sample_wall_clock(frame, tmp_path / 'legacy.xlsx')
    assert summary['mean_all_sample_wall_clock_s'] is None
    assert summary['mean_measured_sample_wall_clock_s'] is None
    assert summary['timed_samples'] == 0 and 'not substituted' in caplog.text


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True, 'bad'])
def test_invalid_durations_not_counted(tmp_path, value):
    frame = measured_frame(tmp_path, ('A',))
    frame['sample_wall_clock_s'] = [value]
    result = timing.summarize_sample_wall_clock(frame, tmp_path / 'invalid.xlsx')
    assert result['timed_samples'] == 0


def test_resume_preserves_verified_measurements_and_rejects_stale_ones(tmp_path):
    previous = measured_frame(tmp_path / 'old', ('A', 'B'))
    current = pd.DataFrame({'index': [0, 1], 'prediction': ['A', 'C']})
    joined = timing.attach_sample_timing(current, tmp_path / 'new', 'model', 'bench', previous)
    assert joined.loc[0, 'sample_wall_clock_s'] == previous.loc[0, 'sample_wall_clock_s']
    assert pd.isna(joined.loc[1, 'sample_wall_clock_s'])


def test_xlsx_numeric_columns_survive_long_trajectory(tmp_path):
    frame = measured_frame(tmp_path, ('A',))
    frame['extra_records'] = ['x' * 40000]
    path = tmp_path / 'long.xlsx'
    frame.to_excel(path, index=False)
    loaded = pd.read_excel(path)
    assert len(loaded.loc[0, 'extra_records']) == 32767
    result = timing.summarize_sample_wall_clock(loaded, path)
    assert result['timed_samples'] == 1


def load_inference(monkeypatch):
    def load(path):
        path = Path(path)
        if path.suffix == '.xlsx':
            return pd.read_excel(path)
        with path.open('rb') as stream:
            return pickle.load(stream)
    def dump(value, path):
        path = Path(path)
        if path.suffix == '.xlsx':
            value.to_excel(path, index=False)
        else:
            with path.open('wb') as stream:
                pickle.dump(value, stream)
    def module(name, **attrs):
        result = ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
        return result
    module('vlmeval')
    module('vlmeval.config', supported_VLM={})
    module('vlmeval.sample_timing', **{n: getattr(timing, n) for n in (
        'attach_sample_timing', 'timed_generate', 'timing_directory')})
    module('vlmeval.smp', dump=dump, load=load, get_logger=logging.getLogger,
           get_pred_file_format=lambda: 'xlsx', get_rank_and_world_size=lambda: (0, 1),
           get_pred_file_path=lambda work, model, dataset, **k: str(Path(work) / f'{model}_{dataset}.xlsx'))
    def track(func, tasks, nproc, save, keys, **kwargs):
        output = load(save) if Path(save).exists() else {}
        with ThreadPoolExecutor(max_workers=nproc) as executor:
            futures = [executor.submit(func, **task) for task in tasks]
            for key, future in zip(keys, futures):
                output[key] = future.result()
                dump(output, save)
    module('vlmeval.utils', track_progress_rich=track)
    spec = importlib.util.spec_from_file_location('inference_timing_test_target', ROOT / 'vlmeval/inference.py')
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    return inference


@pytest.mark.parametrize('structured', [False, True])
def test_actual_infer_entry_persists_timings_and_resume(tmp_path, monkeypatch, structured):
    monkeypatch.delenv('SPLIT_THINK', raising=False)
    inference = load_inference(monkeypatch)
    calls = []
    class Model:
        is_api = True
        def generate(self, **kwargs):
            calls.append(kwargs)
            return {'prediction': 'A', 'extra_records': {'total_time': 99}} if structured else 'A'
    class Dataset:
        dataset_name = 'bench'
        def __init__(self):
            self.data = pd.DataFrame({'index': [0, 1], 'question': ['q0', 'q1']})
        def __len__(self):
            return len(self.data)
        def build_prompt(self, item):
            return [{'type': 'text', 'value': item['question']}]
    model = Model()
    inference.infer_data_job(model, tmp_path, 'model', Dataset(), api_nproc=2)
    result = pd.read_excel(tmp_path / 'model_bench.xlsx')
    assert result.prediction.tolist() == ['A', 'A']
    assert all(result.sample_timing_scope == timing.SCOPE)
    assert len(calls) == 2
    assert all(set(call) == {'message', 'dataset'} for call in calls)
    original_times = result.sample_wall_clock_s.tolist()
    # EvalKit removed the transient checkpoints, but the timings remain in
    # numeric XLSX columns and per-sample JSON files.
    assert not (tmp_path / 'model_bench_checkpoint.pkl').exists()
    inference.infer_data_job(model, tmp_path, 'model', Dataset(), api_nproc=2)
    assert len(calls) == 2
    assert pd.read_excel(tmp_path / 'model_bench.xlsx').sample_wall_clock_s.tolist() == original_times


def test_eval_reporting_is_before_judge_and_after_inference():
    source = (ROOT / 'run.py').read_text()
    report = source.index('report_sample_wall_clock(result_file, logger=logger)')
    assert source.index('model = infer_data_job(') < report < source.index('eval_results = dataset.evaluate(')
    assert report < source.index("if args.mode == 'infer':", report)


def test_model_owned_batch_is_not_given_fake_individual_timings(tmp_path, monkeypatch):
    inference = load_inference(monkeypatch)
    class Model:
        is_api = True
        supports_batch_generation = True
        speceyes_batch_size = 2
        def generate_batch(self, requests):
            assert all(set(item) == {'message', 'dataset'} for item in requests)
            return ['A'] * len(requests)
    class Dataset:
        dataset_name = 'bench'
        data = pd.DataFrame({'index': [0, 1], 'question': ['q0', 'q1']})
        def build_prompt(self, item):
            return [{'type': 'text', 'value': item['question']}]
    result = inference.infer_data_api(Model(), tmp_path, 'model', Dataset(), api_nproc=1)
    assert result == {0: 'A', 1: 'A'}
    assert not timing.timing_directory(tmp_path, 'model', 'bench').exists()


def test_reporting_failure_does_not_abort_scoring(tmp_path, monkeypatch, caplog):
    load_inference(monkeypatch)
    assert timing.report_sample_wall_clock(tmp_path / 'missing.xlsx') is None
    assert 'Failed to report sample wall clock' in caplog.text
