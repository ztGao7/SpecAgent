#!/usr/bin/env python3
"""Save whole-benchmark token totals from vLLM's server-side counters."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request

COUNTERS = {'input_tokens': 'vllm:prompt_tokens_total',
            'output_tokens': 'vllm:generation_tokens_total'}
GAUGES = ('vllm:num_requests_running', 'vllm:num_requests_waiting')
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)(?:\s+\S+)?$')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def parse_metrics(text):
    series = {name: {} for name in COUNTERS.values()}
    created, gauges = {}, {name: [] for name in GAUGES}
    for line in text.splitlines():
        match = SAMPLE.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        if name not in series and name not in gauges and name not in {
            'vllm:prompt_tokens_created', 'vllm:generation_tokens_created', 'process_start_time_seconds'}:
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f'Invalid metric: {name}')
        if name in series:
            if not number.is_integer():
                raise ValueError(f'Non-integer token counter: {name}')
            series[name][labels or ''] = int(number)
        elif name in gauges:
            gauges[name].append(number)
        else:
            created[name + (labels or '')] = number
    if any(not values for values in series.values()):
        raise ValueError('vLLM input/output token counters missing from /metrics')
    return {'counters': series, 'created': created,
            'idle': all(values and sum(values) == 0 for values in gauges.values())}


def collect(service, metrics_dir):
    name, url = service
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=3) as response:
            text = response.read().decode('utf-8')
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f'{name}.prom').write_text(text)
        return name, dict(parse_metrics(text), url=url, captured_unix=time.time())
    except Exception as exc:
        return name, {'url': url, 'error': f'{type(exc).__name__}: {exc}', 'idle': False}


def snapshot(services, directory, settle=False, timeout=15):
    deadline, previous, stable = time.monotonic() + timeout, None, 0
    while True:
        with ThreadPoolExecutor(max_workers=len(services)) as pool:
            records = dict(pool.map(lambda service: collect(service, directory), services))
        if not settle or any('error' in record for record in records.values()):
            return records
        fingerprint = json.dumps({name: record.get('counters') for name, record in records.items()}, sort_keys=True)
        idle = all(record['idle'] for record in records.values())
        stable = stable + 1 if idle and fingerprint == previous else 0
        if stable >= 2 or time.monotonic() >= deadline:
            return records
        previous = fingerprint
        time.sleep(0.5)


def summarize(before, after, *, shared=False, exit_code=0):
    services, totals, errors = {}, {'input_tokens': 0, 'output_tokens': 0}, []
    for name, start in before.items():
        end = after.get(name, {})
        issue = start.get('error') or end.get('error')
        if not issue:
            if start.get('created') != end.get('created'):
                issue = 'Service restarted / metrics process changed during the benchmark'
            elif not start.get('idle') or not end.get('idle'):
                issue = 'Service was not idle at the accounting boundary'
        counts = {}
        for key, metric in COUNTERS.items():
            left, right = start.get('counters', {}).get(metric, {}), end.get('counters', {}).get(metric, {})
            if not left or left.keys() != right.keys() or any(right[k] < v for k, v in left.items()):
                issue = issue or 'Counter reset, disappeared or changed labels'
            counts[key] = sum(right.values()) - sum(left.values()) if not issue else None
        if issue:
            errors.append(f'{name}: {issue}')
            counts = dict.fromkeys(COUNTERS)
        else:
            for key in totals:
                totals[key] += counts[key]
        services[name] = dict(counts, url=start.get('url'), error=issue)
    if errors:
        totals = dict.fromkeys(COUNTERS)
    return {'source': 'vllm_metrics_counter_delta', 'scope': 'whole_benchmark_inference',
            'evaluation_exit_code': exit_code, 'benchmark_completed': exit_code == 0,
            'counts_available': not errors, 'exclusive_services': not shared,
            'bench_attribution_complete': not errors and not shared,
            **totals, 'services': services, 'errors': errors,
            'note': 'Server-counter totals for the evaluation interval; not GPU FLOPs. '
                    'Shared services include other clients. Missing/reset counters are null, never zero.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['before', 'after'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--service', action='append', required=True, help='name=http://host:port/metrics')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--exit-code', type=int, default=0)
    parser.add_argument('--bench', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--subset', action='store_true')
    args = parser.parse_args()
    services = [item.split('=', 1) for item in args.service]
    if len({url for _, url in services}) != len(services):
        parser.error('Main / Draft endpoints must be distinct to avoid double counting')
    records = snapshot(services, args.run_dir / 'metrics' / args.phase, settle=args.phase == 'after')
    write_json(args.run_dir / 'metrics' / f'{args.phase}.json', records)
    if args.phase == 'before':
        write_json(args.run_dir / 'run_info.json', {
            'bench': args.bench, 'method': args.method, 'started_unix': time.time(),
            'sample_scope': 'subset' if args.subset else 'full_dataset',
            'status': 'running', 'shared_services': args.shared,
            'predictions': 'predictions', 'logs': 'logs', 'metrics': 'metrics'})
        if any('error' in record for record in records.values()):
            raise SystemExit('Cannot collect vLLM token counters; see metrics/before.json')
    else:
        before = json.loads((args.run_dir / 'metrics/before.json').read_text())
        summary = summarize(before, records, shared=args.shared, exit_code=args.exit_code)
        summary.update(bench=args.bench, method=args.method,
                       sample_scope='subset' if args.subset else 'full_dataset')
        write_json(args.run_dir / 'token_usage.json', summary)
        info_path = args.run_dir / 'run_info.json'
        info = json.loads(info_path.read_text())
        info.update(finished_unix=time.time(), exit_code=args.exit_code,
                    status='completed' if args.exit_code == 0 else 'failed')
        write_json(info_path, info)
        print(f'Bench tokens: input={summary["input_tokens"]}, output={summary["output_tokens"]}; '
              f'saved {args.run_dir / "token_usage.json"}')


if __name__ == '__main__':
    main()
