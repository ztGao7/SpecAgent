"""VLMEvalKit adapter for the SpecEyes + Spectool sandbox workflow."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import json
import os
import sys
import threading

from PIL import Image

from vlmeval.api.base import BaseAPI


class SpecEyesSpectoolAPI(BaseAPI):
    """Run the SpecEyes router/direct gate and Spectool sandbox fallback."""

    is_api: bool = True
    INTERLEAVE = True
    # ``infer_data_api`` detects this opt-in and dispatches one official-style
    # SpecEyes funnel at a time instead of treating every sample as an
    # unrelated API worker.
    supports_batch_generation = True

    @staticmethod
    def _draft_bases(value: str | None, fallback: str) -> list[str]:
        """Accept the comma-separated replica list emitted by start_test.sh."""
        candidates = str(value or fallback).split(",")
        bases = []
        for candidate in candidates:
            base = candidate.strip().rstrip("/")
            if base and base not in bases:
                bases.append(base)
        return bases or [fallback.rstrip("/")]

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "EMPTY",
        draft_model_name: str | None = None,
        draft_api_base: str | None = None,
        draft_api_key: str = "EMPTY",
        agent_repo_root: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: int = 300,
        max_steps: int = 6,
        max_image_history: int = 6,
        retry: int = 3,
        verbose: bool = True,
        benchmark_profile: str | None = None,
        **kwargs,
    ):
        if agent_repo_root is None:
            agent_repo_root = os.environ.get("SPECTOOL_AGENT_ROOT")
        assert agent_repo_root, "Set SPECTOOL_AGENT_ROOT to the Spectool repository root."
        if agent_repo_root not in sys.path:
            sys.path.insert(0, agent_repo_root)

        from spectool_agent.benchmarks import resolve_benchmark_profile
        from spectool_agent.model import VLLMModel
        from spectool_agent.types import AgentConfig, DEFAULT_MAX_TOKENS

        self.benchmark_profile = (
            resolve_benchmark_profile(benchmark_profile)
            if benchmark_profile is not None else None
        )
        # Share the ordinary Spectool request budget, not the short router or
        # direct-answer budgets. Keep existing environment override precedence.
        max_tokens = max(1, int(
            os.environ.get("SPECEYE_MAIN_MAX_TOKENS")
            or os.environ.get("SPECTOOL_MAX_TOKENS")
            or (max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS)))
        max_steps = max(1, int(os.environ.get(
            "SPECEYE_MAX_STEPS",
            os.environ.get("SPECTOOL_MAX_STEPS", str(max_steps)))))
        timeout = max(1, int(os.environ.get("SPECTOOL_API_TIMEOUT", str(timeout))))
        max_image_history = max(1, int(os.environ.get(
            "SPECTOOL_MAX_IMAGE_HISTORY", str(max_image_history))))
        self.cfg = AgentConfig(
            api_base=api_base, model_name=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens, timeout=timeout,
            max_steps=max_steps, max_image_history=max_image_history,
        )
        self.main_model = VLLMModel(self.cfg)
        draft_timeout = max(1, int(os.environ.get("SPECTOOL_DRAFT_TIMEOUT", str(timeout))))
        draft_name = draft_model_name or os.environ.get("SPECTOOL_DRAFT_MODEL", model)
        draft_key = draft_api_key or os.environ.get("SPECTOOL_DRAFT_KEY", api_key)
        draft_bases = self._draft_bases(
            draft_api_base or os.environ.get("SPECTOOL_DRAFT_BASE"), api_base)
        self.draft_models = [
            VLLMModel(replace(
                self.cfg, api_base=base, model_name=draft_name,
                api_key=draft_key, timeout=draft_timeout,
            ))
            for base in draft_bases
        ]
        self._draft_index = 0
        self._draft_lock = threading.Lock()
        self.speceyes_batch_size = max(1, int(os.environ.get("SPECEYE_BATCH_SIZE", "6")))
        self.speceyes_fallback_concurrency = max(1, int(os.environ.get("SPECEYE_FALLBACK_NPROC", "1")))
        self.print_traj = kwargs.pop("print_traj", False)
        self.fail_msg = "Unable to obtain an answer from SpecEyes Spectool."
        super().__init__(retry=max(1, int(retry)), verbose=verbose, **kwargs)

    @staticmethod
    def _parse_vlmeval_inputs(inputs):
        images, texts = [], []
        for item in inputs:
            if item["type"] == "image":
                images.append(Image.open(item["value"]).convert("RGB"))
            elif item["type"] == "text":
                texts.append(item["value"])
        if not images:
            raise ValueError("SpecEyes requires at least one image input")
        if len(images) > 1:
            texts.insert(0, (
                f"[Note] {len(images)} input images are available to the sandbox "
                "as image_1, image_2, and so on."))
        return images, "\n".join(texts)

    def _next_draft_model(self):
        """Round-robin independent TP=1 Draft replicas between EvalKit calls."""
        with self._draft_lock:
            model = self.draft_models[self._draft_index % len(self.draft_models)]
            self._draft_index += 1
            return model

    def generate_inner(self, inputs, **kwargs) -> tuple:
        from spectool_agent.benchmarks import resolve_benchmark_profile, resolve_sample_id
        from spectool_agent.prompts import MATHVISION_EXTRA_PROMPT, get_prompt_set
        from spectool_agent.speceyes import SpecEyesOrchestrator

        dataset = kwargs.get("dataset", None)
        profile = resolve_benchmark_profile(self.benchmark_profile, dataset)
        if profile == "mmsearch":
            return -1, self.fail_msg, json.dumps({
                "error": "SpecEyes intentionally exposes only the visual sandbox; use Spectool-MMSearch-VDR for MMSearch."
            })
        sample_id = resolve_sample_id(
            kwargs.get("sample_id", ""), inputs,
            benchmark_profile=profile, dataset=dataset)
        try:
            images, question = self._parse_vlmeval_inputs(inputs)
            extra_prompt = (
                MATHVISION_EXTRA_PROMPT
                if dataset and "mathvision" in dataset.lower() else ""
            )
            result = SpecEyesOrchestrator(
                self.main_model, self._next_draft_model(), self.cfg,
            ).run(
                images, question, sample_id=sample_id,
                extra_prompt=extra_prompt,
                prompt_set=get_prompt_set(benchmark_profile=profile, dataset=dataset),
            )
        except Exception as exc:
            return -1, self.fail_msg, json.dumps({
                "error": f"SpecEyes execution error: {type(exc).__name__}: {exc}"
            })

        metadata = result.metadata()
        trajectory = metadata.pop("trajectory", None)
        if trajectory is not None:
            metadata["trajectory"] = asdict(trajectory)
        metadata["dataset"] = dataset
        if self.print_traj:
            print(
                f"[SpecEyes] id={sample_id} route={result.route} "
                f"decision={result.decision} confidence={result.confidence} "
                f"tools={metadata['tool_call_count']} time={result.total_time:.2f}s"
            )
        if result.final_answer is None:
            return 1, self.fail_msg, metadata
        return 0, result.final_answer, metadata

    def _batch_failure(self, error: Exception | str, *, dataset: str | None = None):
        """Return the same structured shape as BaseAPI.generate on a failure."""
        return {
            "prediction": self.fail_msg,
            "extra_records": {
                "speceyes_error": str(error),
                "dataset": dataset,
            },
        }

    def _prepare_batch_request(self, struct: dict):
        """Mirror BaseAPI.generate preprocessing without per-item retries.

        ``infer_data_api`` already owns checkpointing and hands this adapter
        raw EvalKit message structures.  Mutating those structures would make
        later checkpoint retries unsafe, hence the deepcopy here.
        """
        from spectool_agent.benchmarks import resolve_benchmark_profile, resolve_sample_id
        from spectool_agent.prompts import MATHVISION_EXTRA_PROMPT, get_prompt_set
        from spectool_agent.speceyes import SpecEyesRequest

        dataset = struct.get("dataset")
        message = copy.deepcopy(struct["message"])
        if self.check_content(message) == "listdict":
            message = self.preprocess_message_with_role(message)
        assert self.check_content(message) in {"str", "dict", "liststr", "listdict"}
        inputs = self.preproc_content(message)
        assert inputs is not None and self.check_content(inputs) == "listdict"
        for item in inputs:
            assert item["type"] in self.allowed_types, f'Invalid input type: {item["type"]}'

        profile = resolve_benchmark_profile(self.benchmark_profile, dataset)
        if profile == "mmsearch":
            raise ValueError(
                "SpecEyes intentionally exposes only the visual sandbox; "
                "use Spectool-MMSearch-VDR for MMSearch.")
        images, question = self._parse_vlmeval_inputs(inputs)
        sample_id = resolve_sample_id(
            "", inputs, benchmark_profile=profile, dataset=dataset)
        extra_prompt = (
            MATHVISION_EXTRA_PROMPT
            if dataset and "mathvision" in dataset.lower() else ""
        )
        return SpecEyesRequest(
            image=images,
            question=question,
            sample_id=sample_id,
            extra_prompt=extra_prompt,
            prompt_set=get_prompt_set(benchmark_profile=profile, dataset=dataset),
        ), dataset, sample_id

    def generate_batch(self, structs: list[dict]):
        """Process one phase-barrier batch with bounded fallback concurrency.

        Results deliberately use the normal EvalKit structured-record format,
        so downstream scoring and ``extra_records`` handling are unchanged.
        """
        from spectool_agent.speceyes import SpecEyesBatchOrchestrator

        outputs = [None] * len(structs)
        requests = []
        request_slots = []
        for slot, struct in enumerate(structs):
            try:
                request, dataset, sample_id = self._prepare_batch_request(struct)
                requests.append(request)
                request_slots.append((slot, dataset, sample_id))
            except Exception as exc:
                outputs[slot] = self._batch_failure(exc, dataset=struct.get("dataset"))

        if requests:
            try:
                # One draft endpoint is selected for the complete batch.  This
                # gives a single 9B service the full Phase-II no-tool subset,
                # matching the official one-small-model batch more closely
                # than round-robin selection per individual sample.
                results = SpecEyesBatchOrchestrator(
                    self.main_model, self._next_draft_model(), self.cfg,
                    fallback_concurrency=self.speceyes_fallback_concurrency,
                ).run(requests)
                for result, (slot, dataset, sample_id) in zip(results, request_slots):
                    metadata = result.metadata()
                    trajectory = metadata.pop("trajectory", None)
                    if trajectory is not None:
                        metadata["trajectory"] = asdict(trajectory)
                    metadata["dataset"] = dataset
                    if self.print_traj:
                        print(
                            f"[SpecEyes] id={sample_id} route={result.route} "
                            f"decision={result.decision} confidence={result.confidence} "
                            f"tools={metadata['tool_call_count']} "
                            f"time={result.total_time:.2f}s "
                            f"batch={metadata['speceyes_batch_index']}/"
                            f"{metadata['speceyes_batch_size']}")
                    outputs[slot] = (
                        {"prediction": result.final_answer, "extra_records": metadata}
                        if result.final_answer is not None
                        else {"prediction": self.fail_msg, "extra_records": metadata}
                    )
            except Exception as exc:
                for slot, dataset, _sample_id in request_slots:
                    outputs[slot] = self._batch_failure(exc, dataset=dataset)

        assert all(output is not None for output in outputs)
        return outputs
