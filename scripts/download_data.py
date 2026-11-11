#!/usr/bin/env python3
"""Download self-contained VLMEvalKit TSVs, validate, then publish atomically."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

csv.field_size_limit(sys.maxsize)
DATASETS = {
    "vstar": ("VStarBench", "https://huggingface.co/datasets/xjtupanda/VStar_Bench/resolve/main/VStarBench.tsv"),
    "mmstar": ("MMStar", "https://huggingface.co/datasets/Lin-Chen/MMStar/resolve/main/MMStar.tsv"),
    "hrbench4k": ("HRBench4K", "https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_4k.tsv"),
    "hrbench8k": ("HRBench8K", "https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_8k.tsv"),
    "pope": ("POPE", "https://opencompass.openxlab.space/utils/VLMEval/POPE.tsv"),
    "mathvista": ("MathVista_MINI", "https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv"),
}


def validate_table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        columns = set(reader.fieldnames or [])
        if not {"index", "question", "answer"} <= columns or not columns & {"image", "image_path"}:
            raise ValueError(f"Not a visual-evaluation TSV: {path}")
        rows = 0
        for row in reader:
            if not row.get("index") or not row.get("question"):
                raise ValueError(f"Missing index/question at record {rows + 1}: {path}")
            rows += 1
        if not rows:
            raise ValueError(f"Empty dataset: {path}")
    return rows


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=DATASETS, default="vstar")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dataset, official = DATASETS[args.bench]
    url = official
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    if url.startswith("https://huggingface.co/"):
        url = endpoint + url[len("https://huggingface.co"):]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    target = args.data_dir / f"{dataset}.tsv"
    if target.exists() and not args.force:
        rows = validate_table(target)
        print(f"Reusing {target}: {rows} rows, SHA256={digest(target)}")
        return
    partial = target.with_suffix(".tsv.part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SpecAgent-setup/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                expected = response.headers.get("Content-Length")
                copied = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    copied += len(block)
                if expected and copied != int(expected):
                    raise OSError(f"Incomplete download: {copied}/{expected} bytes")
            rows = validate_table(partial)
            checksum = digest(partial)
            partial.replace(target)
            record = {"dataset": dataset, "official_url": official, "download_url": url,
                      "rows": rows, "sha256": checksum}
            target.with_suffix(".source.json").write_text(json.dumps(record, indent=2) + "\n")
            print(f"Downloaded {target}: {rows} rows, SHA256={checksum}")
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


if __name__ == "__main__":
    main()
