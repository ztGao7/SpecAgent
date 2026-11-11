#!/usr/bin/env python3
"""固定随机种子地抽取 VLMEvalKit benchmark TSV 的 N 条子集。

用法:
    LMUData=/path/to/LMUData python3 make_subset.py <DATASET> [N] [SEED]
    python3 make_subset.py MMStar 400
    python3 make_subset.py HRBench8K 400
    python3 make_subset.py POPE 400
    python3 make_subset.py VStarBench 400        # VStar 只有 191 条，会保留全部

读取   $LMUData/<DATASET>.tsv
写出   $LMUData/<DATASET>_R<N>.tsv   （同样的列，随机 N 行，种子固定可复现）

子集名 <DATASET>_R<N> 不在 VLMEvalKit 注册表里，所以 build_dataset() 会走
"custom dataset" 分支：本地 TSV 有 A/B 列 + image → 自动当 CustomMCQDataset；
否则当 CustomVQADataset（POPE 这种 yes/no 走这条，infer 模式照常工作）。
不需要改任何 VLMEvalKit 注册代码，也不会触发下载。跑评测时 --data <DATASET>_R<N> 即可。

注意:
  - 同一 seed + 同一份源 TSV ⇒ 抽出的 400 条完全一致，保证不同模型评在同一子集上。
  - N >= 源数据总行数时，保留全部并告警（VStarBench 仅 191 条即属此情况）。
  - 整行原样复制（含 base64 image 列、原始 index），index 唯一性保持不变。
"""
import csv
import os
import sys
from collections import Counter

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    dataset = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    root = os.environ.get("LMUData", os.path.expanduser("~/LMUData"))
    src = os.path.join(root, f"{dataset}.tsv")
    if not os.path.isfile(src):
        sys.exit(f"ERROR: 源 TSV 不存在: {src}\n"
                 f"      (标准数据集需先让 VLMEvalKit 下载过一次，或先 build 出本地 TSV)")

    with open(src, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        rows = list(reader)
    total = len(rows)

    if total <= n:
        keep_idx = list(range(total))
        print(f"[warn] {dataset} 仅 {total} 行 (<= {n})，保留全部 {total} 行。")
    else:
        import random
        rng = random.Random(seed)
        keep_idx = sorted(rng.sample(range(total), n))  # 无放回，按原顺序保留

    out_name = f"{dataset}_R{n}"
    dst = os.path.join(root, f"{out_name}.tsv")
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(header)
        for i in keep_idx:
            w.writerow(rows[i])

    # 类别分布（若有 category 列）便于核对覆盖均衡度
    cat_line = ""
    cols = {h: i for i, h in enumerate(header)}
    for cat_col in ("category", "l2_category"):
        if cat_col in cols:
            dist = Counter(rows[i][cols[cat_col]] for i in keep_idx)
            cat_line = f"  {cat_col} 分布: " + ", ".join(
                f"{k}={v}" for k, v in sorted(dist.items()))
            break

    print(f"[ok] 写出 {dst}")
    print(f"     {len(keep_idx)} 行 (从 {total} 行随机抽，seed={seed})")
    if cat_line:
        print(cat_line)
    print(f"     评测时用: --data {out_name}")


if __name__ == "__main__":
    main()
