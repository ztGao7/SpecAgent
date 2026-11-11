"""
从本地解析的 MathVision 数据集构建 MathVision_MINI.tsv。

源目录（只读）:
    codebase/spectool/data/mathvision_testmini/metadata.json   # 304 个样本
    codebase/spectool/data/mathvision/images/<id>.jpg ...       # 共享图片

输出: $LMUData/MathVision_MINI.tsv  (VLMEvalKit VQA 格式)
    列: index, image, question, choices, answer, category

Schema 说明:
  - `choices` 是字符串化的 Python 列表：
    - '[]' 表示自由输入数值题
    - '["A","B","C","D","E"]' 表示多选题
  - 去除 question 中的 <image\\d+> 标记（VLMEvalKit 单独附加图片）
  - `category` 统一为 'unknown'

用法:
    SRC_DIR=/abs/path/to/codebase/spectool/data/mathvision_testmini \
    IMG_ROOT=/abs/path/to/codebase/spectool/data/mathvision \
    LMUData=$HOME/LMUData \
        python scripts/build_mathvision_mini_tsv.py
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(os.environ.get(
    "SRC_DIR",
    str(_REPO_ROOT / "data" / "mathvision_testmini"))).resolve()
IMG_ROOT = Path(os.environ.get(
    "IMG_ROOT",
    str(_REPO_ROOT / "data" / "mathvision"))).resolve()
LMU_ROOT = Path(os.environ.get("LMUData", str(Path.home() / "LMUData"))).resolve()
OUT_TSV = LMU_ROOT / "MathVision_MINI.tsv"

IMG_MARKER_RE = re.compile(r"<image\d+>")


def main():
    meta_path = SRC_DIR / "metadata.json"
    if not meta_path.is_file():
        sys.exit(f"metadata.json missing: {meta_path}")

    with open(meta_path) as f:
        items = json.load(f)

    rows = []
    missing_imgs = []
    for it in items:
        qid = int(it["id"])
        # `it['image']` is like "../mathvision/images/4.jpg"; resolve relative
        # to SRC_DIR but also allow IMG_ROOT/images/<id>.jpg as a fallback.
        rel_img = SRC_DIR / it["image"]
        rel_img = rel_img.resolve()
        if not rel_img.is_file():
            rel_img = IMG_ROOT / "images" / f"{qid}.jpg"
        if not rel_img.is_file():
            missing_imgs.append(str(rel_img))
            continue

        with open(rel_img, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")

        question = IMG_MARKER_RE.sub("", it["question"]).strip()

        rows.append({
            "index": qid,
            "image": img_b64,
            "question": question,
            "choices": json.dumps(it.get("options", []),
                                  ensure_ascii=False),
            "answer": str(it["answer"]),
            "category": "unknown",
        })

    if missing_imgs:
        sys.exit(f"{len(missing_imgs)} missing image(s), first: {missing_imgs[:3]}")

    cols = ["index", "image", "question", "choices", "answer", "category"]
    df = pd.DataFrame(rows, columns=cols)
    LMU_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {len(df)} rows to {OUT_TSV}")
    print(f"  MCQ items   : {(df['choices'] != '[]').sum()}")
    print(f"  free-form   : {(df['choices'] == '[]').sum()}")


if __name__ == "__main__":
    main()
