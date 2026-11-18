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
"""
import argparse
import csv
import json
import os
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=os.environ.get("SRC_DIR", ""))
    ap.add_argument("--input-file", default="")
    ap.add_argument("--image-folder", default="")
    ap.add_argument("--out", default=os.path.join(os.environ.get("LMUData", ""), "AgentVista.tsv")
                    if os.environ.get("LMUData") else "")
    a = ap.parse_args()
    if not a.src_dir or not a.out:
        ap.error("--src-dir/--out (or SRC_DIR/LMUData) are required")
    src = os.path.abspath(a.src_dir)
    data_file = a.input_file or os.path.join(src, "official_eval.json")
    img_dir = os.path.abspath(a.image_folder or os.path.join(src, "images"))
    rows = json.load(open(data_file, encoding="utf-8"))
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
