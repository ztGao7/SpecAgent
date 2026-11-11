#!/usr/bin/env python3
"""Select the first N distinct dataset indices; never select by ground truth."""
import argparse
import csv
import json
from pathlib import Path
import sys

csv.field_size_limit(sys.maxsize)


def select(table, count):
    selected, seen = [], set()
    with Path(table).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"index", "question"} <= set(reader.fieldnames or []):
            raise ValueError(f"Missing index/question columns: {table}")
        for row in reader:
            index = row["index"]
            if index and index not in seen:
                seen.add(index)
                selected.append(index)
                if len(selected) == count:
                    return selected
    raise ValueError(f"Requested {count} unique indices but only {len(selected)} available")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    selected = select(args.table, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({args.dataset: selected}) + "\n")
    print(f"Selected {len(selected)} samples -> {args.output}")


if __name__ == "__main__":
    main()
