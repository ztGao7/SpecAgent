"""
从本地解析的 VStar 数据集构建 VStarBench.tsv。

源目录（只读）:
    codebase/spectool/data/vstar/
        metadata.json
        images/<question_id>.jpg ...

输出: $LMUData/VStarBench.tsv  (VLMEvalKit MCQ 格式)
    列: index, image (base64 JPEG), question, A, B, C, D, answer, category

用法:
    SRC_DIR=/abs/path/to/codebase/spectool/data/vstar \
    LMUData=/abs/path/to/lmu_data \
        python scripts/build_vstar_tsv.py
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_default_src = str(_REPO_ROOT / "data" / "vstar")
if not Path(_default_src).is_dir():
    _data_root = os.environ.get("SPECTOOL_DATA_ROOT", "")
    if _data_root and (Path(_data_root) / "vstar").is_dir():
        _default_src = str(Path(_data_root) / "vstar")
SRC_DIR = Path(os.environ.get("SRC_DIR", _default_src)).resolve()
LMU_ROOT = Path(os.environ.get("LMUData", str(Path.home() / "LMUData"))).resolve()
OUT_TSV = LMU_ROOT / "VStarBench.tsv"

OPT_RE = re.compile(r"^\(([A-Z])\)\s*(.+?)\s*$")


def parse_text(text: str):
    """Split metadata.json `text` into (question, {A:..., B:..., C:..., D:...}).

    Format observed:
        <question stem possibly multi-line>
        (A) opt_a
        (B) opt_b
        (C) opt_c
        (D) opt_d
        Answer with the option's letter from the given choices directly.
    """
    question_lines, options = [], {}
    seen_option = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = OPT_RE.match(line)
        if m:
            options[m.group(1)] = m.group(2).strip()
            seen_option = True
        elif seen_option:
            continue  # trailing instruction line(s)
        else:
            question_lines.append(line)
    return " ".join(question_lines).strip(), options


def main():
    if not SRC_DIR.is_dir():
        sys.exit(f"SRC_DIR does not exist: {SRC_DIR}")
    meta_path = SRC_DIR / "metadata.json"
    if not meta_path.is_file():
        sys.exit(f"metadata.json missing: {meta_path}")

    with open(meta_path) as f:
        items = json.load(f)

    rows = []
    missing_imgs = []
    max_letter_idx = 0  # 0 → A, 1 → B, ...
    for row_idx, it in enumerate(items):
        raw_qid = str(it["question_id"])
        try:
            qid = int(raw_qid)
        except ValueError:
            qid = row_idx
        question, options = parse_text(it["text"])
        # Options must be a contiguous prefix A..Z with no gaps.
        letters = sorted(options.keys())
        expected = [chr(ord("A") + i) for i in range(len(letters))]
        if letters != expected:
            sys.exit(f"qid={raw_qid} parsed options={letters} are not a contiguous "
                     f"prefix starting at A. Raw text:\n{it['text']}")
        if it["label"] not in options:
            sys.exit(f"qid={raw_qid} label={it['label']!r} not in parsed options")
        max_letter_idx = max(max_letter_idx, len(letters) - 1)

        img_path = SRC_DIR / it["image"]
        if not img_path.is_file():
            missing_imgs.append(str(img_path))
            continue
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")

        row = {
            "index": qid,
            "image": img_b64,
            "question": question,
            "answer": it["label"],
            "category": it.get("category", ""),
        }
        row.update(options)  # 'A', 'B', ... (only those present)
        rows.append(row)

    if missing_imgs:
        sys.exit(f"{len(missing_imgs)} missing image(s), first: {missing_imgs[:3]}")

    option_cols = [chr(ord("A") + i) for i in range(max_letter_idx + 1)]
    cols = ["index", "image", "question"] + option_cols + ["answer", "category"]
    df = pd.DataFrame(rows, columns=cols)
    LMU_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {len(df)} rows to {OUT_TSV}")
    print(f"categories: {df['category'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
