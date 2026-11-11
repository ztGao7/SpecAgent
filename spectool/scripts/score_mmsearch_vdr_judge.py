#!/usr/bin/env python3
"""Score EvalKit MMSearch predictions with Vision-DeepResearch's judge protocol.

The judge prompt, verdict parsing, exact-match fallback, and binary accuracy
aggregation mirror ``Vision-DeepResearch/rllm/eval/eval_runner.py``.  The only
adapter-specific work here is reading/writing EvalKit spreadsheets.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import AsyncOpenAI


# Verbatim canonical prompt from Vision-DeepResearch/rllm/eval/eval_runner.py.
JUDGE_PROMPT = """You are a impartial judge evaluating whether a deep research report contains the correct answer.

[Question]
{question}

[Correct Answer]
{reference_answer}

[Deep Research Report]
{assistant_answer}

Task: Determine if the deep research report contains the correct answer anywhere in its content.

Instructions:
1. Read through the entire research report carefully
2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)
3. Check if the information in the report is consistent with the correct answer
4. The answer does NOT need to be in a specific format or labeled as "final answer"
5. Provide your reasoning
6. Answer with "yes" if the report contains the correct answer, "no" if it doesn't or contradicts it

Output format:
correct: [yes/no]
reasoning: [your explanation]
"""


def _text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _parse_judge_response(text: str) -> Optional[bool]:
    if not text:
        return None
    visible = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    verdicts = re.findall(
        r"(?im)^\s*(?:[-*]\s*)?correct\s*:\s*(yes|no)\b", visible
    )
    if verdicts:
        return verdicts[-1].lower() == "yes"
    return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip(
        ".,;:!?\"'"
    )


def _extract_final_answer(response: str) -> str:
    """Strip thinking, prefer <answer>, then boxed text, else keep full report."""
    if not response:
        return ""
    cleaned = re.sub(
        r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    match = re.search(
        r"<answer>(.*?)</answer>", cleaned, flags=re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    match = re.search(r"\\boxed\{([^{}]+)\}", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


def _scan_quoted_value(text: str, start: int) -> str:
    """Read one Python/JSON quoted string even if later wrapper text is truncated."""
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] not in {"'", '"'}:
        return ""
    quote = text[start]
    escaped = False
    for position in range(start + 1, len(text)):
        char = text[position]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == quote:
            literal = text[start:position + 1]
            try:
                value = ast.literal_eval(literal)
            except (SyntaxError, ValueError):
                return ""
            return value if isinstance(value, str) else str(value)
    return ""


def _unwrap_evalkit_prediction(value) -> str:
    """Extract EvalKit's inner report without leaking trajectory into judging."""
    if isinstance(value, dict):
        return _text(value.get("prediction"))
    raw = _text(value).strip()
    if not raw:
        return ""
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(raw)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict) and "prediction" in parsed:
            return _text(parsed.get("prediction"))
    # Excel truncates cells at 32,767 characters.  The inner prediction is the
    # first field in EvalKit's repr, so scan only that quoted value and ignore
    # an incomplete trailing extra_records dictionary.
    match = re.search(r"(?s)(?:['\"]prediction['\"]\s*:\s*)", raw)
    if match:
        extracted = _scan_quoted_value(raw, match.end())
        if extracted:
            return extracted
    return raw


async def _judge_answer(
    client: AsyncOpenAI,
    judge_model: str,
    question: str,
    ground_truth: str,
    prediction: str,
) -> tuple[bool, str, bool]:
    """Return correctness, raw judge response, and whether fallback was used."""
    try:
        prompt = JUDGE_PROMPT.format(
            question=question,
            reference_answer=ground_truth,
            assistant_answer=prediction,
        )
        response = await client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        raw = (response.choices[0].message.content or "").strip()
        verdict = _parse_judge_response(raw)
        if verdict is None:
            return _normalize(ground_truth) == _normalize(prediction), raw, True
        return verdict, raw, False
    except Exception as exc:
        raw = f"judge_error: {type(exc).__name__}: {exc}"
        return _normalize(ground_truth) == _normalize(prediction), raw, True


def _episode(row: dict, prediction: str, correct: bool, judge_response: str) -> dict:
    reward = 1.0 if correct else 0.0
    sample_id = _text(row.get("index"))
    return {
        "id": sample_id,
        "task": {
            "id": sample_id,
            "question": _text(row.get("question")),
            "answer": _text(row.get("answer")),
        },
        "termination_reason": "evaluated_from_evalkit",
        "is_correct": bool(correct),
        "metrics": {"reward": reward},
        "info": {
            "prediction": prediction,
            "judge_response": judge_response,
        },
        "trajectories": [],
    }


async def _run(args: argparse.Namespace) -> None:
    frame = pd.read_excel(args.input_xlsx)
    missing = [
        column
        for column in ("question", "answer", "prediction")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    client = AsyncOpenAI(base_url=args.judge_base_url, api_key=args.judge_api_key)
    semaphore = asyncio.Semaphore(args.parallel)

    async def score(position: int, row: dict):
        prediction = _extract_final_answer(
            _unwrap_evalkit_prediction(row.get("prediction"))
        )
        async with semaphore:
            correct, raw, fallback = await _judge_answer(
                client=client,
                judge_model=args.judge_model,
                question=_text(row.get("question")),
                ground_truth=_text(row.get("answer")),
                prediction=prediction,
            )
        return position, prediction, correct, raw, fallback

    tasks = [
        score(position, row.to_dict())
        for position, (_, row) in enumerate(frame.iterrows())
    ]
    results = []
    for completed, task in enumerate(asyncio.as_completed(tasks), 1):
        results.append(await task)
        if completed % 10 == 0 or completed == len(tasks):
            print(f"progress={completed}/{len(tasks)}", flush=True)
    await client.close()

    results.sort(key=lambda item: item[0])
    predictions = [item[1] for item in results]
    correctness = [bool(item[2]) for item in results]
    responses = [item[3] for item in results]
    fallbacks = [bool(item[4]) for item in results]

    frame["vdr_judge_prediction"] = predictions
    frame["vdr_judge_correct"] = correctness
    frame["vdr_judge_reward"] = [1.0 if value else 0.0 for value in correctness]
    frame["vdr_judge_response"] = responses
    frame["vdr_judge_fallback"] = fallbacks
    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(args.output_xlsx, index=False)

    correct = sum(correctness)
    total = len(correctness)
    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "average_reward": correct / total if total else None,
        "judge_model": args.judge_model,
        "judge_parsed": total - sum(fallbacks),
        "judge_fallbacks": sum(fallbacks),
    }

    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if args.episodes_jsonl:
        args.episodes_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.episodes_jsonl.open("w", encoding="utf-8") as stream:
            for (_, row), prediction, is_correct, response in zip(
                frame.iterrows(), predictions, correctness, responses
            ):
                stream.write(
                    json.dumps(
                        _episode(row.to_dict(), prediction, is_correct, response),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"samples={total}")
    print(f"correct={correct}/{total}")
    print(f"accuracy={metrics['accuracy'] * 100:.2f}%")
    print(f"average_reward={metrics['average_reward']}")
    print(f"judge_parsed={metrics['judge_parsed']}/{total}")
    print(f"judge_fallbacks={metrics['judge_fallbacks']}/{total}")
    print(f"judge_model={args.judge_model}")
    print(f"output={args.output_xlsx}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score EvalKit MMSearch output with the VDR LLM judge."
    )
    parser.add_argument("input_xlsx", type=Path)
    parser.add_argument("output_xlsx", type=Path)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--episodes-jsonl", type=Path)
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    return args


if __name__ == "__main__":
    asyncio.run(_run(_parse_args()))
