"""AgentVista (hkust-nlp/AgentVista, data 61de767) as a VLMEvalKit dataset.

The TSV is built locally by ``spectool/scripts/build_agentvista_tsv.py`` from the
official prepared data: one row per question, ``image_path`` lists every input
image in order (1-10 images), ``answer`` is the reference answer and never part
of the prompt.

Scoring follows the official verifier (``utils/general_qa_tool.py``): the same
system/query prompt, temperature 0.2, ``Score: 1`` / ``Score: 0`` parsing and 5
attempts, against an OpenAI-compatible judge configured by ``VERIFIER_API_KEY``,
``VERIFIER_END_POINT`` (SDK base URL) and ``VERIFIER_MODEL_NAME``.  Unlike the
official code, a judge failure is reported separately (``judge_error``) instead
of silently counting as a wrong answer; it is still wrong in the accuracy.
"""
import os
import os.path as osp
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..smp import LMUDataRoot, dump, get_intermediate_file_path, load, toliststr
from .image_base import ImageBaseDataset

JUDGE_SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs "
    "for question-answer pairs. Compare the predicted answer with the correct answer and determine "
    "if they match meaningfully. Consider synonyms or paraphrases as valid matches."
)

JUDGE_QUERY_PROMPT = """
1. **Question**: {question}
2. **Ground Truth Answer**: {ground_truth}
3. **Model Predicted Answer**: {prediction}

Evaluate the model's prediction against the ground truth. Output an integer score: 1 for correct, 0 for incorrect.
Respond using exactly: Score: 1 or Score: 0
Explanation: <your explanation>"""

FAILED_PREDICTIONS = ("Failed to obtain answer via API.", "error", "")


def judge_one(question, ground_truth, prediction, client, model, max_retries=5, initial_delay=3):
    """Official verifier loop. Returns (score or None on judge failure, raw text)."""
    messages = [{"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": JUDGE_QUERY_PROMPT.format(
                    question=question, ground_truth=ground_truth, prediction=prediction)}]}]
    last = ""
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(model=model, messages=messages, temperature=0.2,
                                               max_tokens=8192, timeout=120)
            text = r.choices[0].message.content or ""
            if "score:" not in text.lower():
                raise ValueError("No score in response")
            part = text.lower().split("score:")[-1].strip().split("\n")[0].strip().split()[0]
            if "1" not in part and "0" not in part:
                raise ValueError("No 0/1 in response")
            return (1 if "1" in part else 0), text
        except Exception as exc:  # retried like the official client
            last = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries - 1:
                time.sleep(min(initial_delay * (attempt + 1), 10))
    return None, last


class AgentVista(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {"AgentVista": ""}
    DATASET_MD5 = {"AgentVista": None}

    @classmethod
    def supported_datasets(cls):
        return ["AgentVista"]

    def load_data(self, dataset):
        path = osp.join(LMUDataRoot(), f"{dataset}.tsv")
        if not osp.exists(path):
            raise FileNotFoundError(
                f"{path} not found; build it with spectool/scripts/build_agentvista_tsv.py")
        self.data_path = path
        return load(path)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        paths = toliststr(line["image_path"])
        msgs = [dict(type="image", value=p) for p in paths]
        msgs.append(dict(type="text", value=str(line["question"])))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        import openai
        data = load(eval_file)
        key, base = os.environ.get("VERIFIER_API_KEY"), os.environ.get("VERIFIER_END_POINT")
        model = os.environ.get("VERIFIER_MODEL_NAME", "gpt-4o")
        if not key or not base:
            raise RuntimeError("AgentVista scoring needs VERIFIER_API_KEY and VERIFIER_END_POINT (SDK base URL)")
        client = openai.OpenAI(base_url=base, api_key=key)
        nproc = int(judge_kwargs.get("nproc", 4))

        def score(row):
            pred = "" if pd.isna(row["prediction"]) else str(row["prediction"]).strip()
            if pred in FAILED_PREDICTIONS:
                return 0, "no prediction", False
            m = re.findall(r"<answer>\s*(.*?)\s*</answer>", pred, re.DOTALL)
            pred = m[-1].strip() if m else pred
            s, text = judge_one(str(row["question"]), str(row["answer"]), pred, client, model)
            return (0 if s is None else s), text, s is None

        rows = [r for _, r in data.iterrows()]
        with ThreadPoolExecutor(max_workers=max(1, nproc)) as pool:
            results = list(pool.map(score, rows))
        data["score"] = [r[0] for r in results]
        data["judge_response"] = [r[1] for r in results]
        data["judge_error"] = [r[2] for r in results]
        dump(data, get_intermediate_file_path(eval_file, f"_{model.replace('/', '_')}_judge", "xlsx"))

        out = [dict(split="Overall", n=len(data), accuracy=100 * data["score"].mean(),
                    judge_error=int(data["judge_error"].sum()))]
        for cat, g in data.groupby("category"):
            out.append(dict(split=cat, n=len(g), accuracy=100 * g["score"].mean(),
                            judge_error=int(g["judge_error"].sum())))
        acc = pd.DataFrame(out)
        dump(acc, get_intermediate_file_path(eval_file, "_acc", "csv"))
        return acc
