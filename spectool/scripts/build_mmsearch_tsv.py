"""Build an EvalKit TSV from the local MMSearch end2end Arrow dataset.

The visual subset is used because EvalKit's image-agent API requires at least
one image. The original MMSearch image_search_result is intentionally not
provided as an input image: it must be obtained through the aligned tools.
"""

import json
import os
from pathlib import Path

import pandas as pd
from datasets import load_from_disk


SRC_ROOT = Path(os.environ.get(
    "MMSEARCH_DATA_ROOT",
    str(Path(os.environ.get("RESOURCE_DIR", "resources")) / "benchmarks/MMSearch/end2end"),
))
LMU_ROOT = Path(os.environ.get("LMUData", str(Path(os.environ.get("RESOURCE_DIR", "resources")) / "LMUData")))
DATASET_NAME = os.environ.get("MMSEARCH_DATASET_NAME", "MMSearchMini")
LIMIT = int(os.environ.get("MMSEARCH_LIMIT", "10"))
START = int(os.environ.get("MMSEARCH_START", "0"))
OUT_FILE = LMU_ROOT / f"{DATASET_NAME}.tsv"
IMAGE_DIR = LMU_ROOT / "images" / DATASET_NAME


def main():
    dataset = load_from_disk(str(SRC_ROOT))
    candidates = [row for row in dataset if row.get("query_image") is not None]
    selected = candidates[START:START + LIMIT]
    if not selected:
        raise RuntimeError("MMSearch contains no visual rows in the requested range")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for row in selected:
        sample_id = str(row["sample_id"])
        image_path = IMAGE_DIR / f"{sample_id}.png"
        image = row["query_image"].convert("RGB")
        image.save(image_path, format="PNG")
        records.append({
            "index": sample_id,
            "question": str(row["query"]),
            "answer": str(row["gt_answer"]),
            "image_path": str(image_path),
            "sample_id": sample_id,
            "alternative_answers": json.dumps(
                row.get("alternative_gt_answers") or [], ensure_ascii=False),
            "gt_requery": str(row.get("gt_requery") or ""),
            "subfield": str(row.get("subfield") or ""),
        })

    LMU_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUT_FILE, sep="\t", index=False)
    print(f"Wrote {len(records)} visual MMSearch rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
