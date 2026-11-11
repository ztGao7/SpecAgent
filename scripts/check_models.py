#!/usr/bin/env python3
"""Check model config/tokenizer files and all indexed safetensors shards offline."""
import argparse
import json
from pathlib import Path

from safetensors import safe_open
from model_paths import selected_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-dir', required=True, type=Path)
    parser.add_argument("--model", choices=["27B", "9B", "both"], default="both")
    args = parser.parse_args()
    for _, root in selected_models(args.resource_dir.expanduser().resolve(), args.model):
        for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json',
                     'preprocessor_config.json', 'model.safetensors.index.json']:
            with (root / name).open() as stream:
                json.load(stream)
        index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
        shards = {}
        for tensor, shard in index.items():
            shards.setdefault(shard, set()).add(tensor)
        for shard, expected in shards.items():
            with safe_open(str(root / shard), framework='numpy') as weights:
                missing = expected - set(weights.keys())
                if missing:
                    raise RuntimeError(f'{shard}: missing {len(missing)} indexed tensors')
        print(f'OK {root.name}: {len(shards)} shards, {len(index)} indexed tensors')


if __name__ == '__main__':
    main()
