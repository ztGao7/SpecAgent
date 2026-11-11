"""
将本地的 Hugging Face VDR-Bench parquet 文件转换为 VLMEvalKit TSV 格式。

从 VDR-Bench 的 parquet 文件读取数据，提取图像、问题和答案，
将图像保存到本地目录并生成 VLMEvalKit 兼容的 TSV 文件。

用法:
    SRC_FILE=/path/to/VDR-Bench-full.parquet \
    LMUData=/path/to/LMUData \
        python scripts/build_vdr_tsv.py
"""

import io
import os
from pathlib import Path

import pandas as pd
from PIL import Image


SRC_FILE = Path(os.environ.get(
    "SRC_FILE", str(Path(os.environ.get("RESOURCE_DIR", "resources")) / "Osilly__VDR-Bench/vision-deepresearch_VDR-Bench-full.parquet")))
LMU_ROOT = Path(os.environ.get("LMUData", str(Path(os.environ.get("RESOURCE_DIR", "resources")) / "LMUData")))
OUT_FILE = LMU_ROOT / "VDRBench.tsv"
IMAGE_DIR = LMU_ROOT / "images" / "VDRBench"


def pick_column(frame, names):
    by_lower = {str(column).lower(): column for column in frame.columns}
    for name in names:
        if name in by_lower:
            return by_lower[name]
    return None


def message_text(messages, role):
    if not isinstance(messages, (list, tuple)):
        return ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != role:
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, (list, tuple)):
            return "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") in ("text", None)
            ).strip()
    return ""


def as_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        value = value.tolist()
    return value if isinstance(value, (list, tuple)) else [value]


def materialize_images(value, row_index):
    paths = []
    for image_index, item in enumerate(as_list(value)):
        raw = None
        path = None
        if isinstance(item, dict):
            raw = item.get("bytes")
            path = item.get("path")
        elif isinstance(item, (bytes, bytearray)):
            raw = bytes(item)
        elif isinstance(item, str):
            path = item

        if raw:
            with Image.open(io.BytesIO(raw)) as image:
                suffix = "." + (image.format or "png").lower().replace("jpeg", "jpg")
            target = IMAGE_DIR / f"{row_index}_{image_index}{suffix}"
            target.write_bytes(raw)
            paths.append(str(target))
        elif path:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = SRC_FILE.parent / candidate
            paths.append(str(candidate))
    if not paths:
        return None
    return paths[0] if len(paths) == 1 else repr(paths)


def main():
    if not SRC_FILE.is_file():
        raise FileNotFoundError(f"VDR-Bench parquet not found: {SRC_FILE}")
    frame = pd.read_parquet(SRC_FILE)
    print(f"VDR-Bench rows={len(frame)} columns={list(frame.columns)}")

    question_col = pick_column(frame, ("question", "query", "prompt", "problem", "instruction"))
    answer_col = pick_column(frame, ("answer", "response", "ground_truth", "solution", "target"))
    image_col = pick_column(frame, ("image", "images", "image_path", "image_paths"))
    messages_col = pick_column(frame, ("messages", "conversation", "conversations"))
    if question_col is None and messages_col is None:
        raise ValueError(f"Cannot find a question/messages column in {list(frame.columns)}")
    if image_col is None:
        raise ValueError(f"Cannot find an image column in {list(frame.columns)}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for position, (_, row) in enumerate(frame.iterrows()):
        messages = row[messages_col] if messages_col else None
        question = row[question_col] if question_col else message_text(messages, "user")
        answer = row[answer_col] if answer_col else message_text(messages, "assistant")
        image_path = materialize_images(row[image_col], position)
        if image_path is None:
            continue
        records.append({
            "index": position,
            "question": str(question),
            "answer": str(answer) if answer is not None else "",
            "image_path": image_path,
        })

    if not records:
        raise ValueError("VDR-Bench parquet produced no usable image rows")
    LMU_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUT_FILE, sep="\t", index=False)
    print(f"Wrote {len(records)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
