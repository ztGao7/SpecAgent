"""
VLMEvalKit BaseAPI 封装 — 推测性执行（Speculative Execution）模式。

将 spectool_agent 的 SpeculativeAgentLoop（Draft + Main 双模型并行）
封装为 VLMEvalKit 兼容的 API 接口，用于 Benchmark 评测。

使用方式与标准 Agent 相同，但内部使用 Draft 模型在后台进行工具探索，
Main 模型只做审阅和决策，大幅提升吞吐量。
"""
import hashlib
import json
import math
import os
import sys
import threading
import time
from dataclasses import asdict, replace

from PIL import Image

from vlmeval.api.base import BaseAPI

SEP = "=" * 72


def _runtime_parameter(value, env_name, default, cast=int, minimum=1):
    """Explicit constructor/CLI values win over environment and defaults."""
    raw = value if value is not None else os.environ.get(env_name, default)
    try:
        resolved = cast(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {env_name}: {raw!r}") from exc
    if not math.isfinite(resolved) or resolved < minimum:
        raise ValueError(f"{env_name} must be >= {minimum}; got {raw!r}")
    return resolved


def _normalize_draft_api_bases(value, fallback: str) -> list[str]:
    """Return ordered, de-duplicated Draft endpoints.

    ``SPECTOOL_DRAFT_BASE`` may contain a comma-separated list so independent
    TP=1 services can be used without placing another proxy in the hot path.
    """
    if value is None:
        candidates = [fallback]
    elif isinstance(value, (list, tuple)):
        candidates = list(value)
    else:
        candidates = str(value).split(",")

    bases = []
    for candidate in candidates:
        base = str(candidate or "").strip().rstrip("/")
        if base and base not in bases:
            bases.append(base)
    return bases or [fallback.rstrip("/")]


def _print_traj_step(step, sample_id: str = ""):
    """Print a single step with the same detail as get_trajectory.py."""
    si = step.step_idx
    times = f"infer: {step.inference_time:.2f}s"
    if step.tool_time > 0:
        times += f", tool: {step.tool_time:.2f}s"
    print(f"\n  {'─' * 66}")
    print(f"  STEP {si}  ({times})")
    print(f"  {'─' * 66}")

    if step.reasoning:
        print(f"\n  ┌── MODEL OUTPUT ──────────────────────────────")
        for line in step.reasoning.split("\n"):
            if line.strip():
                print(f"  │ {line}")
        print(f"  └──────────────────────────────────────────────")

    if step.tool_call is not None:
        code = step.tool_call.params.get("code", "")
        print(f"\n  ┌── TOOL CALL: {step.tool_call.tool_name}")
        print(f"  │ code ({len(code)} chars):")
        for line in code.split("\n"):
            print(f"  │   {line}")
        print(f"  └──────────────────────────────────────────────")

    if step.observation:
        print(f"\n  ┌── TOOL RESULT ───")
        for line in step.observation.split("\n"):
            if line.strip():
                print(f"  │ {line}")
        print(f"  └──────────────────────────────────────────────")

    if step.answer:
        print(f"\n  ★ FINAL ANSWER: {step.answer}")


def _print_traj_summary(traj, sample_id: str):
    print(f"\n  {'─' * 66}")
    print(f"  TRAJECTORY SUMMARY ({sample_id})")
    print(f"  {'─' * 66}")
    print(f"  Steps:            {len(traj.steps)}")
    print(f"  Tool calls:       {traj.tool_call_count}")
    print(f"  Stop reason:      {traj.stop_reason}")
    if traj.fallback_mode:
        print(
            "  Main takeover:    "
            f"trigger={traj.fallback_trigger} "
            f"validated={traj.fallback_validated_steps} "
            f"steps={traj.fallback_steps_used}/{traj.fallback_step_budget}"
        )
    print(f"  Final answer:     {traj.final_answer}")
    print(f"  Inference total:  {traj.total_inference_time:.2f}s")
    print(f"  Tool total:       {traj.total_tool_time:.2f}s")
    print(f"  Main requests:    {traj.main_request_time:.2f}s")
    print(f"  Draft requests:   {traj.draft_request_time:.2f}s")
    print(f"  Model active:     {traj.model_active_wall_time:.2f}s")
    print(f"  Model overlap:    {traj.model_overlap_time:.2f}s")
    print(f"  Wall-clock total: {traj.total_time:.2f}s")


class SpeculativeSpectoolAgentAPI(BaseAPI):
    """VLMEvalKit BaseAPI 封装 — 推测性执行（Speculative Execution）。

    使用 Draft 模型（如 Qwen3.5-9B）在后台进行工具探索，
    Main 模型（如 Qwen3.5-27B）审阅 Draft 产出的证据并最终决策。
    """
    is_api: bool = True
    INTERLEAVE = True
    SUPPORTS_RUNTIME_GENERATION_CONFIG = True

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "EMPTY",
        draft_model_name: str = None,
        draft_api_base: str = None,
        draft_api_key: str = "EMPTY",
        agent_repo_root: str = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_steps: int = 15,
        max_context_tokens: int = 0,
        max_total_output_tokens: int = 0,
        max_image_history: int = 6,
        retry: int | None = None,
        verbose: bool = True,
        benchmark_profile: str = None,
        fallback_only: bool = False,
        direct_only: bool = False,
        **kwargs,
    ):
        """
        参数:
            model: Main 模型名称（如 Qwen3.5-27B）
            api_base: Main 模型 API 基础 URL
            draft_model_name: Draft 模型名称（如 Qwen3.5-9B）
            draft_api_base: Draft 模型 API 基础 URL，或逗号分隔的副本 URL
            agent_repo_root: spectool_agent 仓库根目录
            max_steps: 最大推理步数
        """
        if agent_repo_root is None:
            agent_repo_root = os.environ.get("SPECTOOL_AGENT_ROOT")
        assert agent_repo_root, (
            "请设置 agent_repo_root 或环境变量 SPECTOOL_AGENT_ROOT 指向 "
            "包含 spectool_agent/ 的目录"
        )
        if agent_repo_root not in sys.path:
            sys.path.insert(0, agent_repo_root)

        from spectool_agent.model import VLLMModel, _bounded_max_tokens, _env_int
        from spectool_agent.draft_model import DraftModel
        from spectool_agent.benchmarks import resolve_benchmark_profile
        from spectool_agent.types import AgentConfig, DEFAULT_MAX_TOKENS

        max_tokens = _runtime_parameter(
            max_tokens, "SPECTOOL_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        timeout = _runtime_parameter(
            timeout, "SPECTOOL_API_TIMEOUT", 120, cast=float, minimum=0.001)
        retry = _runtime_parameter(retry, "SPECTOOL_API_RETRY", 5)

        self.benchmark_profile = (
            resolve_benchmark_profile(benchmark_profile)
            if benchmark_profile is not None else None
        )

        if self.benchmark_profile == "mmsearch":
            max_tokens = max(1, int(os.environ.get(
                "MMSEARCH_MAX_TOKENS", str(max_tokens))))
            max_steps = max(1, int(os.environ.get(
                "MMSEARCH_MAX_STEPS", str(max_steps))))
            max_context_tokens = max(0, int(os.environ.get(
                "MMSEARCH_MAX_CONTEXT_TOKENS", str(max_context_tokens))))
            max_total_output_tokens = max(0, int(os.environ.get(
                "MMSEARCH_MAX_TOTAL_OUTPUT_TOKENS",
                str(max_total_output_tokens))))

        draft_api_bases = _normalize_draft_api_bases(
            draft_api_base, api_base)
        draft_model_name = draft_model_name or model
        env_fallback_only = os.environ.get(
            "SPECTOOL_FALLBACK_ONLY", "false").strip().lower()
        self.direct_only = bool(direct_only)
        self.fallback_only = bool(fallback_only) or env_fallback_only in {
            "1", "true", "yes", "on"}

        quality_judge_enabled = (
            str(
                kwargs.pop(
                    "quality_judge_enabled",
                    os.environ.get("SPECTOOL_QUALITY_JUDGE_ENABLED", "false"),
                )
            ).strip().lower() in {"1", "true", "yes", "on"}
        )
        quality_judge_threshold = float(
            kwargs.pop(
                "quality_judge_threshold",
                os.environ.get("SPECTOOL_QUALITY_JUDGE_THRESHOLD", "0.6"),
            )
        )
        answerability_threshold = max(0.0, min(1.0, float(kwargs.pop(
            "answerability_threshold",
            os.environ.get(
                "SPECTOOL_ANSWERABILITY_THRESHOLD",
                # Keep hand-written launchers that still set the old name
                # working while they migrate to the shared threshold.
                os.environ.get("SPECTOOL_FAST_ANSWER_THRESHOLD", "0.95"),
            ),
        ))))
        fallback_clean_start = (
            str(kwargs.pop(
                "fallback_clean_start",
                os.environ.get("SPECTOOL_FALLBACK_CLEAN_START", "true"),
            )).strip().lower() in {"1", "true", "yes", "on"}
        )
        fallback_reuse_validated_evidence = (
            str(kwargs.pop(
                "fallback_reuse_validated_evidence",
                os.environ.get(
                    "SPECTOOL_FALLBACK_REUSE_VALIDATED_EVIDENCE", "false"),
            )).strip().lower() in {"1", "true", "yes", "on"}
        )
        fallback_shared_step_budget = (
            str(kwargs.pop(
                "fallback_shared_step_budget",
                os.environ.get(
                    "SPECTOOL_FALLBACK_SHARED_STEP_BUDGET", "true"),
            )).strip().lower() in {"1", "true", "yes", "on"}
        )
        fallback_min_steps = max(0, int(kwargs.pop(
            "fallback_min_steps",
            os.environ.get("SPECTOOL_FALLBACK_MIN_STEPS", "1"),
        )))
        sample_wall_time_budget = max(0.0, float(kwargs.pop(
            "sample_wall_time_budget",
            os.environ.get("SPECTOOL_SAMPLE_WALL_BUDGET_SECONDS", "0"),
        )))
        max_image_history = int(os.environ.get(
            "SPECTOOL_MAX_IMAGE_HISTORY", str(max_image_history)))
        max_image_history = max(1, max_image_history)
        self.cfg = AgentConfig(
            api_base=api_base, model_name=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens, timeout=timeout,
            max_steps=max_steps,
            draft_max_steps=max_steps,
            max_context_tokens=max_context_tokens,
            max_total_output_tokens=max_total_output_tokens,
            max_image_history=max_image_history,
            quality_judge_enabled=quality_judge_enabled,
            quality_judge_threshold=quality_judge_threshold,
            answerability_threshold=answerability_threshold,
            fallback_clean_start=fallback_clean_start,
            fallback_reuse_validated_evidence=fallback_reuse_validated_evidence,
            fallback_shared_step_budget=fallback_shared_step_budget,
            fallback_min_steps=fallback_min_steps,
            sample_wall_time_budget=sample_wall_time_budget,
        )

        self.main_model = VLLMModel(self.cfg)
        draft_timeout = _runtime_parameter(
            kwargs.pop("draft_timeout", None), "SPECTOOL_DRAFT_TIMEOUT", 200,
            cast=float, minimum=0.001)
        draft_max_image_size = kwargs.pop("draft_max_image_size", 0)
        self.draft_models = [] if self.fallback_only or self.direct_only else [
            DraftModel(
                api_base=base,
                model_name=draft_model_name,
                api_key=draft_api_key,
                timeout=draft_timeout,
                max_image_size=draft_max_image_size,
            )
            for base in draft_api_bases
        ]
        # Backward-compatible alias for callers that inspect the single-model
        # deployment. New sample routing always uses ``draft_models``.
        self.draft_model = self.draft_models[0] if self.draft_models else None
        self._draft_route_lock = threading.Lock()
        self._draft_route_counter = 0
        self.print_traj = kwargs.pop("print_traj", False)
        self.fail_msg = "无法通过 Spectool Speculative Agent 获得答案。"
        if self.direct_only:
            self.fail_msg = "Direct model did not return a valid <answer>."

        super().__init__(retry=retry, verbose=verbose, **kwargs)
        self.runtime_config = {
            "requested_max_tokens": max_tokens,
            "main_max_tokens": _bounded_max_tokens(max_tokens, default_cap=0),
            "draft_max_tokens": None if not self.draft_models else _bounded_max_tokens(
                max_tokens, cap_env="SPECTOOL_DRAFT_MAX_TOKENS_CAP", default_cap=0),
            "main_timeout_seconds": timeout,
            "draft_timeout_seconds": None if not self.draft_models else draft_timeout,
            "eval_attempts": retry,
            "openai_max_retries": max(0, _env_int("SPECTOOL_OPENAI_MAX_RETRIES", 0)),
        }
        print("[spectool-config] " + json.dumps(self.runtime_config, sort_keys=True), flush=True)

    def _select_draft_model(self, sample_id) -> tuple[int, object]:
        """Choose a sticky Draft replica for one sample and all of its retries."""
        if not self.draft_models:
            return -1, None
        if len(self.draft_models) == 1:
            return 0, self.draft_models[0]
        if sample_id is not None and str(sample_id).strip():
            digest = hashlib.blake2b(
                str(sample_id).encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % len(self.draft_models)
        else:
            with self._draft_route_lock:
                index = self._draft_route_counter % len(self.draft_models)
                self._draft_route_counter += 1
        return index, self.draft_models[index]

    def _parse_vlmeval_inputs(self, inputs):
        """解析 VLMEvalKit 的输入列表，提取图片和文本。"""
        images, texts = [], []
        for item in inputs:
            if item["type"] == "image":
                images.append(Image.open(item["value"]).convert("RGB"))
            elif item["type"] == "text":
                texts.append(item["value"])
        assert len(images) >= 1, (
            "Spectool agent 需要至少一张图片输入。")
        if len(images) > 1 and not getattr(self, "direct_only", False):
            texts.insert(0, (
                f"[注意] 提供了 {len(images)} 张图片。"
                "sandbox 中可通过 image_1、image_2 等变量访问。"))
        return images, "\n".join(texts)

    def _generate_direct(self, inputs, **kwargs) -> tuple:
        """One ordinary Main request, with no tools, gates or fallback loop."""
        from spectool_agent.benchmarks import resolve_benchmark_profile, resolve_sample_id
        from spectool_agent.model import parse_answer, parse_tool_call
        from spectool_agent.agent import _detect_mcq_choices
        from spectool_agent.prompts import DIRECT_SYSTEM_PROMPT, get_prompt_set
        from spectool_agent.speculative_agent import _as_input_images, _original_image_content

        started = time.perf_counter()
        record = {
            "runtime_config": self.runtime_config,
            "answer_path": "direct",
            "tool_call_count": 0,
            "total_tool_time": 0.0,
            "total_draft_time": 0.0,
            "fallback_mode": False,
            "draft_replica_count": 0,
            "trajectory": [],
        }
        request_started = None
        request_seconds = 0.0
        try:
            dataset = kwargs.get("dataset")
            profile = resolve_benchmark_profile(self.benchmark_profile, dataset)
            if profile != "default":
                raise ValueError("Direct evaluation currently supports visual QA, not research profiles")
            sample_id = resolve_sample_id(
                kwargs.get("sample_id", ""), inputs,
                benchmark_profile=profile, dataset=dataset)
            record["sample_id"] = sample_id
            images, question = self._parse_vlmeval_inputs(inputs)
            content = _original_image_content(_as_input_images(images))
            # Keep the image bytes/order identical; sandbox aliases are irrelevant here.
            for item in content:
                if item["type"] == "text":
                    item["text"] = item["text"].split(" / sandbox", 1)[0] + "]"
            content.append({"type": "text", "text": question})
            messages = [
                {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            request_started = time.perf_counter()
            response = self.main_model.generate(messages)
            request_seconds = time.perf_counter() - request_started
            prompt_set = get_prompt_set(benchmark_profile=profile, dataset=dataset)
            has_tool_call = bool(self.main_model.last_tool_calls) or parse_tool_call(response) is not None
            answer = None if has_tool_call else parse_answer(
                response, normalize_mcq=prompt_set.normalize_mcq_answers,
                choices=_detect_mcq_choices(question), question=question)
            record["trajectory"] = [{
                "step_idx": 0, "reasoning": response, "answer": answer,
                "inference_time": request_seconds, "tool_time": 0.0,
                "verdict_mode": "direct",
            }]
            record["stop_reason"] = (
                "unexpected_tool_call" if has_tool_call else
                "direct_answer" if answer else "missing_answer_tag")
        except Exception as exc:
            if request_started is not None and not request_seconds:
                request_seconds = time.perf_counter() - request_started
            answer = None
            record.update(stop_reason="exception", error=str(exc))
        record.update(
            total_time=time.perf_counter() - started,
            total_inference_time=request_seconds,
            total_main_time=request_seconds,
            main_request_time=request_seconds,
        )
        if not answer:
            return 1, self.fail_msg, json.dumps(record, ensure_ascii=False)
        return 0, answer, record

    def generate_inner(self, inputs, **kwargs) -> tuple:
        """生成单个样本的推测性执行推理结果（VLMEvalKit BaseAPI 接口）。"""
        if getattr(self, "direct_only", False):
            return self._generate_direct(inputs, **kwargs)

        from spectool_agent.speculative_agent import SpeculativeAgentLoop
        from spectool_agent.benchmarks import (resolve_benchmark_profile,
                                               resolve_sample_id)
        from spectool_agent.prompts import get_prompt_set
        from spectool_agent.tools import ToolRegistry, register_default_tools

        dataset = kwargs.get("dataset", None)
        benchmark_profile = resolve_benchmark_profile(
            self.benchmark_profile, dataset)
        profile_metadata = (
            {"benchmark_profile": benchmark_profile}
            if benchmark_profile != "default" else {}
        )
        sample_id = resolve_sample_id(
            kwargs.get("sample_id", ""), inputs,
            benchmark_profile=benchmark_profile, dataset=dataset)
        draft_replica_index, draft_model = self._select_draft_model(sample_id)
        try:
            images, question = self._parse_vlmeval_inputs(inputs)
        except Exception as e:
            return -1, self.fail_msg, __import__("json").dumps({"error": f"输入解析错误: {e}"})

        if self.print_traj:
            print(f"\n\n{SEP}")
            print(f"  SPECTool 样本  |  id={sample_id}  |  dataset={dataset}")
            if self.fallback_only:
                print("  Draft replica: disabled (fallback-only)")
            else:
                print(
                    f"  Draft replica: {draft_replica_index + 1}/"
                    f"{len(self.draft_models)}")
            image_sizes = ", ".join(
                f"{image.size[0]}x{image.size[1]}" for image in images)
            print(f"  图片 ({len(images)}): {image_sizes}")
            print(f"{SEP}")
            print(f"\n  ┌── 问题 ────────────────────────────")
            for line in question.split("\n"):
                print(f"  │ {line}")
            print(f"  └──────────────────────────────────────────────")

        is_mathvision = bool(dataset) and "mathvision" in dataset.lower()
        cfg_for_call = self.cfg
        if is_mathvision:
            mathvision_max_steps = int(
                os.environ.get("SPECTOOL_MATHVISION_MAX_STEPS", "12"))
            if mathvision_max_steps != self.cfg.max_steps:
                cfg_for_call = replace(self.cfg, max_steps=mathvision_max_steps)

        tools = ToolRegistry()
        tool_schemas = register_default_tools(
            tools, benchmark_profile=benchmark_profile, dataset=dataset)
        prompt_set = get_prompt_set(
            benchmark_profile=benchmark_profile, dataset=dataset)
        audit_token = tools.start_tool_audit(sample_id)
        tool_audit = {}
        agent = SpeculativeAgentLoop(
            self.main_model, draft_model, tools, cfg_for_call,
            prompt_set=prompt_set, tool_schemas=tool_schemas,
        )

        extra_prompt = ""
        if is_mathvision:
            from spectool_agent.prompts import MATHVISION_EXTRA_PROMPT
            extra_prompt = MATHVISION_EXTRA_PROMPT

        try:
            traj = agent.run(
                images, question,
                sample_id=sample_id,
                extra_prompt=extra_prompt,
                fallback_only=self.fallback_only,
            )
        except Exception as e:
            if tools.is_fatal_tool_error(e):
                raise
            print(f"  [WARN] AgentLoop 崩溃: {e}", file=sys.stderr)
            __import__("traceback").print_exc()
            agent.shutdown()
            return 1, self.fail_msg, __import__("json").dumps({
                "error": f"agent_loop: {e}",
                "stop_reason": "exception",
                **profile_metadata,
                "draft_replica_index": draft_replica_index,
                "draft_replica_count": len(self.draft_models),
            })
        finally:
            agent.shutdown()
            tool_audit = tools.finish_tool_audit(audit_token)
            tools.cleanup()

        if traj is None:
            return 1, self.fail_msg, __import__("json").dumps({
                "error": "agent_loop returned no trajectory",
                "stop_reason": "empty_trajectory",
                **profile_metadata,
                "draft_replica_index": draft_replica_index,
                "draft_replica_count": len(self.draft_models),
            })

        if self.print_traj:
            for step in traj.steps:
                _print_traj_step(step, sample_id)
            _print_traj_summary(traj, sample_id)

        if traj.final_answer is None:
            return 1, self.fail_msg, __import__("json").dumps({
                "stop_reason": traj.stop_reason,
                "runtime_config": self.runtime_config,
                **profile_metadata,
                "total_time": traj.total_time,
                "total_main_time": traj.total_main_time,
                "total_draft_time": traj.total_draft_time,
                "timing_version": traj.timing_version,
                "main_request_time": traj.main_request_time,
                "draft_request_time": traj.draft_request_time,
                "model_request_time": traj.model_request_time,
                "model_active_wall_time": traj.model_active_wall_time,
                "model_overlap_time": traj.model_overlap_time,
                "tool_request_time": traj.tool_request_time,
                "tool_active_wall_time": traj.tool_active_wall_time,
                "client_active_wall_time": traj.client_active_wall_time,
                "pipeline_active_wall_time": traj.pipeline_active_wall_time,
                "orchestration_wait_time": traj.orchestration_wait_time,
                "model_active_wall_ratio": traj.model_active_wall_ratio,
                "fallback_mode": traj.fallback_mode,
                "fallback_trigger": traj.fallback_trigger,
                "fallback_validated_steps": traj.fallback_validated_steps,
                "fallback_passive_entries": traj.fallback_passive_entries,
                "fallback_step_budget": traj.fallback_step_budget,
                "fallback_steps_used": traj.fallback_steps_used,
                "fallback_forced": traj.fallback_forced,
                "fallback_budget_exhausted": traj.fallback_budget_exhausted,
                "fallback_wall_budget": traj.fallback_wall_budget,
                "fallback_wall_elapsed_before": traj.fallback_wall_elapsed_before,
                "fallback_wall_time": traj.fallback_wall_time,
                "fallback_replay_dropped_steps": traj.fallback_replay_dropped_steps,
                "timing_events": traj.timing_events,
                "trajectory": [asdict(s) for s in traj.steps],
                "draft_replica_index": draft_replica_index,
                "draft_replica_count": len(self.draft_models),
            })

        return 0, traj.final_answer, {
            "runtime_config": self.runtime_config,
            "tool_audit": tool_audit,
            "tool_call_count": traj.tool_call_count,
            "stop_reason": traj.stop_reason,
            **profile_metadata,
            "total_inference_time": traj.total_inference_time,
            "total_tool_time": traj.total_tool_time,
            "total_time": traj.total_time,
            "total_main_time": traj.total_main_time,
            "total_draft_time": traj.total_draft_time,
            "timing_version": traj.timing_version,
            "main_request_time": traj.main_request_time,
            "draft_request_time": traj.draft_request_time,
            "model_request_time": traj.model_request_time,
            "model_active_wall_time": traj.model_active_wall_time,
            "model_overlap_time": traj.model_overlap_time,
            "tool_request_time": traj.tool_request_time,
            "tool_active_wall_time": traj.tool_active_wall_time,
            "client_active_wall_time": traj.client_active_wall_time,
            "pipeline_active_wall_time": traj.pipeline_active_wall_time,
            "orchestration_wait_time": traj.orchestration_wait_time,
            "model_active_wall_ratio": traj.model_active_wall_ratio,
            "fallback_mode": traj.fallback_mode,
            "fallback_trigger": traj.fallback_trigger,
            "fallback_validated_steps": traj.fallback_validated_steps,
            "fallback_passive_entries": traj.fallback_passive_entries,
            "fallback_step_budget": traj.fallback_step_budget,
            "fallback_steps_used": traj.fallback_steps_used,
            "fallback_forced": traj.fallback_forced,
            "fallback_budget_exhausted": traj.fallback_budget_exhausted,
            "fallback_wall_budget": traj.fallback_wall_budget,
            "fallback_wall_elapsed_before": traj.fallback_wall_elapsed_before,
            "fallback_wall_time": traj.fallback_wall_time,
            "fallback_replay_dropped_steps": traj.fallback_replay_dropped_steps,
            "timing_events": traj.timing_events,
            "trajectory": [asdict(s) for s in traj.steps],
            "dataset": dataset,
            "draft_replica_index": draft_replica_index,
            "draft_replica_count": len(self.draft_models),
        }
