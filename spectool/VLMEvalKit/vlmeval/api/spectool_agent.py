"""
VLMEvalKit BaseAPI 封装 — 标准 Agent 模式（非推测性执行）。

将 spectool_agent 的 AgentLoop 封装为 VLMEvalKit 兼容的 API，
使 spectool 可以通过 VLMEvalKit 的评测框架运行 Benchmark 评测。
"""
import os
import sys
import json
import time
from dataclasses import asdict, replace

from PIL import Image

from vlmeval.api.base import BaseAPI

SEP = "=" * 72


def _dump_raw_trajectory(raw_dir, sample_id, dataset, question, images,
                         traj, tool_audit):
    """Persist an untruncated trajectory outside Excel's cell-size limit."""
    os.makedirs(raw_dir, exist_ok=True)
    safe_id = "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in str(sample_id or "sample")
    )[:120]
    filename = f"{safe_id}_pid{os.getpid()}_{time.time_ns()}.json"
    path = os.path.join(raw_dir, filename)
    temp_path = path + ".tmp"
    payload = {
        "sample_id": sample_id,
        "dataset": dataset,
        "question": question,
        "image": {"width": images[0].width, "height": images[0].height},
        "images": [
            {"width": image.width, "height": image.height}
            for image in images
        ],
        "final_answer": traj.final_answer,
        "stop_reason": traj.stop_reason,
        "tool_audit": tool_audit,
        "trajectory": asdict(traj),
    }
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)
    return path


def _print_traj_step(step, sample_id: str = ""):
    """打印单步推理轨迹，包含推理时间、工具调用和答案。"""
    si = step.step_idx
    times = f"推理: {step.inference_time:.2f}s"
    if step.tool_time > 0:
        times += f", 工具: {step.tool_time:.2f}s"
    print(f"\n  {'─' * 66}")
    print(f"  步骤 {si}  ({times})")
    print(f"  {'─' * 66}")

    if step.reasoning:
        print(f"\n  ┌── 模型输出 ──────────────────────────────")
        for line in step.reasoning.split("\n"):
            if line.strip():
                print(f"  │ {line}")
        print(f"  └──────────────────────────────────────────────")

    if step.tool_call is not None:
        code = step.tool_call.params.get("code", "")
        print(f"\n  ┌── 工具调用: {step.tool_call.tool_name}")
        print(f"  │ 代码 ({len(code)} 字符):")
        for line in code.split("\n"):
            print(f"  │   {line}")
        print(f"  └──────────────────────────────────────────────")

    if step.observation:
        print(f"\n  ┌── 工具结果 ───")
        for line in step.observation.split("\n"):
            if line.strip():
                print(f"  │ {line}")
        print(f"  └──────────────────────────────────────────────")

    if step.answer:
        print(f"\n  ★ 最终答案: {step.answer}")


def _print_traj_summary(traj, sample_id: str):
    """打印推理轨迹汇总信息。"""
    print(f"\n  {'─' * 66}")
    print(f"  推理轨迹汇总 ({sample_id})")
    print(f"  {'─' * 66}")
    print(f"  步数:            {len(traj.steps)}")
    print(f"  工具调用:       {traj.tool_call_count}")
    print(f"  停止原因:       {traj.stop_reason}")
    print(f"  最终答案:       {traj.final_answer}")
    print(f"  推理总时间:     {traj.total_inference_time:.2f}s")
    print(f"  工具总时间:     {traj.total_tool_time:.2f}s")
    print(f"  墙钟总时间:     {traj.total_time:.2f}s")


class SpectoolAgentAPI(BaseAPI):
    """VLMEvalKit BaseAPI 封装 — 标准 Agent 推理模式。

    将 spectool_agent 的 AgentLoop（标准多轮工具调用）封装为
    VLMEvalKit 兼容的 API 接口，用于 Benchmark 评测。
    """
    is_api: bool = True
    INTERLEAVE = True

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "EMPTY",
        agent_repo_root: str = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 120,
        max_steps: int = 15,
        max_context_tokens: int = 0,
        max_total_output_tokens: int = 0,
        max_image_history: int = 6,
        retry: int = 5,
        verbose: bool = True,
        benchmark_profile: str = None,
        **kwargs,
    ):
        """
        参数:
            model: vLLM 模型名称
            api_base: OpenAI 兼容 API 基础 URL
            agent_repo_root: spectool_agent 仓库根目录（默认从环境变量读取）
            max_steps: Agent 最大推理步数
            max_image_history: 保留的历史图片最大数量
        """
        if agent_repo_root is None:
            agent_repo_root = os.environ.get("SPECTOOL_AGENT_ROOT")
        assert agent_repo_root, (
            "请设置 agent_repo_root 或环境变量 SPECTOOL_AGENT_ROOT 指向 "
            "包含 spectool_agent/ 的目录"
        )
        if agent_repo_root not in sys.path:
            sys.path.insert(0, agent_repo_root)

        from spectool_agent.model import VLLMModel
        from spectool_agent.benchmarks import resolve_benchmark_profile
        from spectool_agent.types import AgentConfig

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
        else:
            max_tokens = max(1, int(os.environ.get(
                "SPECTOOL_MAX_TOKENS", str(max_tokens))))
            max_steps = max(1, int(os.environ.get(
                "SPECTOOL_MAX_STEPS", str(max_steps))))
        timeout = max(1, int(os.environ.get(
            "SPECTOOL_API_TIMEOUT", str(timeout))))
        retry = max(1, int(os.environ.get(
            "SPECTOOL_API_RETRY", str(retry))))
        max_image_history = int(os.environ.get(
            "SPECTOOL_MAX_IMAGE_HISTORY", str(max_image_history)))
        max_image_history = max(1, max_image_history)
        self.cfg = AgentConfig(
            api_base=api_base, model_name=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens, timeout=timeout,
            max_steps=max_steps,
            max_context_tokens=max_context_tokens,
            max_total_output_tokens=max_total_output_tokens,
            max_image_history=max_image_history,
        )
        self.vllm_model = VLLMModel(self.cfg)
        self.print_traj = kwargs.pop("print_traj", False)
        self.raw_traj_dir = os.environ.get("SPECTOOL_RAW_TRAJ_DIR", "")
        self.fail_msg = "无法通过 Spectool Agent 获得答案。"

        super().__init__(retry=retry, verbose=verbose, **kwargs)

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
        if len(images) > 1:
            texts.insert(0, (
                f"[注意] 提供了 {len(images)} 张图片，按输入顺序对应 "
                "sandbox 中的 image_1、image_2 等变量。"))
        return images, "\n".join(texts)

    def generate_inner(self, inputs, **kwargs) -> tuple:
        """生成单个样本的推理结果（VLMEvalKit BaseAPI 接口）。"""
        from spectool_agent.agent import AgentLoop
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
        try:
            images, question = self._parse_vlmeval_inputs(inputs)
        except Exception as e:
            return -1, self.fail_msg, json.dumps({"error": f"输入解析错误: {e}"})

        if self.print_traj:
            print(f"\n\n{SEP}")
            print(f"  SPECTool 样本  |  id={sample_id}  |  dataset={dataset}")
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
        agent = AgentLoop(
            self.vllm_model, tools, cfg_for_call,
            prompt_set=prompt_set, tool_schemas=tool_schemas)

        extra_prompt = ""
        if is_mathvision:
            from spectool_agent.prompts import MATHVISION_EXTRA_PROMPT
            extra_prompt = MATHVISION_EXTRA_PROMPT

        try:
            traj = agent.run(
                images, question, sample_id=sample_id,
                extra_prompt=extra_prompt)
        finally:
            tool_audit = tools.finish_tool_audit(audit_token)
            tools.cleanup()

        raw_traj_path = ""
        if self.raw_traj_dir:
            try:
                raw_traj_path = _dump_raw_trajectory(
                    self.raw_traj_dir, sample_id, dataset, question, images,
                    traj, tool_audit)
                print(f"  原始轨迹: {raw_traj_path}")
            except Exception as exc:
                print(f"  [WARN] 原始轨迹写入失败: {exc}", file=sys.stderr)

        if self.print_traj:
            for step in traj.steps:
                _print_traj_step(step, sample_id)
            _print_traj_summary(traj, sample_id)

        if traj.final_answer is None:
            return 1, self.fail_msg, json.dumps({
                "stop_reason": traj.stop_reason,
                **profile_metadata,
                "total_time": traj.total_time,
                "trajectory": [asdict(s) for s in traj.steps],
                "raw_trajectory_path": raw_traj_path,
            })

        return 0, traj.final_answer, {
            "tool_audit": tool_audit,
            "tool_call_count": traj.tool_call_count,
            "stop_reason": traj.stop_reason,
            **profile_metadata,
            "total_inference_time": traj.total_inference_time,
            "total_tool_time": traj.total_tool_time,
            "total_time": traj.total_time,
            "trajectory": [asdict(s) for s in traj.steps],
            "dataset": dataset,
            "raw_trajectory_path": raw_traj_path,
        }
