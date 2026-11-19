#!/usr/bin/env python3
"""Build LMUData/AgentVista.tsv from the official prepared AgentVista data.

Input: the official-format ``official_eval.json`` (question_id, problem with one
``<image>`` placeholder per image, images, solution, domain, subdomain) and the
folder with the exported images (hkust-nlp/AgentVista data 61de767, 209 rows,
307 images).  Output columns: index (row number 0-208), question_id, question
(problem without ``<image>`` placeholders), answer, image_path (absolute paths
in input order), category, subcategory, n_images.  Images are referenced by
path, not embedded, so the TSV stays small.

usage: build_agentvista_tsv.py --src-dir DIR --out LMUData/AgentVista.tsv
       (DIR holds official_eval.json and images/)
   or: build_agentvista_tsv.py --parquet-dir PARQUET --src-dir DIR --out LMUData/AgentVista.tsv
       (PARQUET holds the two pinned train-*.parquet files of Warrieryes/AgentVista
        @61de767; the images are exported to DIR/images/agentvista_NNN_KK.png, RGB PNG,
        with the same names and pixels as the official preparation)
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import sys


def rows_from_parquet(parquet_dir: str, img_dir: str) -> list[dict]:
    """Official-format rows from the dataset Parquet files, exporting each image once."""
    import pyarrow.parquet as pq
    from PIL import Image
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        sys.exit(f"no .parquet files in {parquet_dir}")
    records = [r for f in files for r in pq.read_table(f).to_pylist()]
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for i, rec in enumerate(records):
        names = []
        for j, image in enumerate(rec["images"]):
            name = f"agentvista_{i:03d}_{j + 1:02d}.png"
            path = os.path.join(img_dir, name)
            if not os.path.isfile(path):
                with Image.open(io.BytesIO(image["bytes"])) as opened:
                    opened.convert("RGB").save(path)
            names.append(name)
        rows.append({"question_id": f"agentvista_{i:03d}", "problem": rec["problem"], "images": names,
                     "solution": rec["answer"], "domain": rec["domain"], "subdomain": rec["subdomain"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=os.environ.get("SRC_DIR", ""))
    ap.add_argument("--input-file", default="")
    ap.add_argument("--image-folder", default="")
    ap.add_argument("--parquet-dir", default=os.environ.get("AGENTVISTA_PARQUET_DIR", ""))
    ap.add_argument("--out", default=os.path.join(os.environ.get("LMUData", ""), "AgentVista.tsv")
                    if os.environ.get("LMUData") else "")
    a = ap.parse_args()
    if not a.src_dir or not a.out:
        ap.error("--src-dir/--out (or SRC_DIR/LMUData) are required")
    src = os.path.abspath(a.src_dir)
    img_dir = os.path.abspath(a.image_folder or os.path.join(src, "images"))
    if a.parquet_dir:
        rows = rows_from_parquet(os.path.abspath(a.parquet_dir), img_dir)
    else:
        rows = json.load(open(a.input_file or os.path.join(src, "official_eval.json"), encoding="utf-8"))
    out_rows = []
    for row in rows:
        qid = row["question_id"]
        m = re.fullmatch(r"agentvista_(\d+)", qid)
        if not m:
            sys.exit(f"unexpected question_id {qid!r}")
        images = [os.path.join(img_dir, name) for name in row["images"]]
        missing = [p for p in images if not os.path.isfile(p)]
        if missing:
            sys.exit(f"{qid}: missing images {missing[:3]}")
        n_tags = len(re.findall(r"<image>", row["problem"], re.I))
        if n_tags != len(images):
            sys.exit(f"{qid}: {n_tags} <image> tags but {len(images)} images")
        question = re.sub(r"<image>", "", row["problem"], flags=re.I).strip()
        out_rows.append({"index": int(m.group(1)), "question_id": qid, "question": question,
                         "answer": str(row["solution"]), "image_path": json.dumps(images, ensure_ascii=False),
                         "category": row.get("domain", ""), "subcategory": row.get("subdomain", ""),
                         "n_images": len(images)})
    out_rows.sort(key=lambda r: r["index"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows, {sum(r['n_images'] for r in out_rows)} images -> {a.out}")


if __name__ == "__main__":
    main()
