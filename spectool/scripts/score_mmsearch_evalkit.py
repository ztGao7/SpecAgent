"""Score EvalKit MMSearch predictions with the official-style token QA F1."""

import ast
import re
import sys
from collections import Counter

import pandas as pd


def normalize(text):
    text = str(text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def f1_score(prediction, reference):
    pred_tokens = normalize(prediction)
    ref_tokens = normalize(reference)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def references(row):
    result = [str(row.get("answer", ""))]
    try:
        values = jsonish = row.get("alternative_answers", "[]")
        values = ast.literal_eval(values) if isinstance(values, str) else values
        result.extend(str(value) for value in (values or []))
    except (ValueError, SyntaxError, TypeError):
        pass
    return result


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} PREDICTION_TSV_OR_XLSX")
    frame = pd.read_csv(sys.argv[1], sep="\t") if sys.argv[1].endswith(".tsv") else pd.read_excel(sys.argv[1])
    scores = []
    for _, row in frame.iterrows():
        score = max(f1_score(row.get("prediction", ""), ref) for ref in references(row))
        scores.append(score)
    frame["mmsearch_f1"] = scores
    print(f"samples={len(frame)}")
    print(f"qa_f1={sum(scores) / len(scores) * 100:.2f}%")
    print(f"exact_f1={sum(score == 1.0 for score in scores)}/{len(scores)}")
    print(frame[["index", "prediction", "answer", "mmsearch_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
