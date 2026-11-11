#!/usr/bin/env python3
"""Download the two Spectool Qwen models and record immutable Hub revisions."""
import argparse
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-dir', required=True, type=Path)
    parser.add_argument('--model', choices=['27B', '9B', 'both'], default='both')
    args = parser.parse_args()
    root = args.resource_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
    api = HfApi(endpoint=endpoint)
    sizes = ['27B', '9B'] if args.model == 'both' else [args.model]
    for size in sizes:
        repo = f'Qwen/Qwen3.5-{size}'
        dest = root / repo
        record = root / f'Qwen3.5-{size}.revision.json'
        # Once resolved, retries and subsequent invocations keep the same revision.
        if record.exists():
            revision = json.loads(record.read_text())['revision']
        else:
            revision = api.model_info(repo).sha
            if not revision:
                raise RuntimeError(f'No immutable revision returned for {repo}')
            record.write_text(json.dumps({'repo_id': repo, 'revision': revision,
                                         'endpoint': endpoint}, indent=2) + '\n')
        for attempt in range(1, 6):
            try:
                snapshot_download(repo_id=repo, revision=revision, endpoint=endpoint,
                                  local_dir=str(dest), max_workers=1,
                                  ignore_patterns=['.DS_Store', '*/.DS_Store', '._*', '*/._*'])
                break
            except Exception:
                if attempt == 5:
                    raise
                print(f'{repo}: attempt {attempt} failed; retrying in 5 seconds', flush=True)
                time.sleep(5)
        print(f'Download completed: {repo}@{revision} -> {dest}', flush=True)


if __name__ == '__main__':
    main()
