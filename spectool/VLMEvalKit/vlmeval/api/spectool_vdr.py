"""
VLMEvalKit BaseAPI 封装 — VDR（Vision-DeepResearch）推测性执行模式。

与 Qwen3.5 的封装（spectool_speculative.py）隔离，
使用 spectool_agent.vdr.* 模块驱动 VDR-Bench 评测，
确保 VDR 的原生格式和开放题兜底逻辑不影响其他 Benchmark。
"""
import os
import sys
from dataclasses import asdict

from PIL import Image

from vlmeval.api.base import BaseAPI

SEP = "=" * 72


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
    print(f"  Final answer:     {traj.final_answer}")
    print(f"  Inference total:  {traj.total_inference_time:.2f}s")
    print(f"  Tool total:       {traj.total_tool_time:.2f}s")
    print(f"  Wall-clock total: {traj.total_time:.2f}s")


class SpeculativeSpectoolVDRAgentAPI(BaseAPI):
    """VLMEvalKit BaseAPI 封装 — VDR 推测性执行模式。

    使用 VDR（Vision-DeepResearch）专用组件：
      - VDRSpeculativeAgentLoop（VDR 版推测性执行循环）
      - DraftModelVDR（支持工具调用预测）
      - VDR 格式解析器（支持 XML 风格工具调用）
    """
    is_api: bool = True
    INTERLEAVE = True

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
        max_tokens: int = 2048,
        timeout: int = 120,
        max_steps: int = 6,
        max_image_history: int = 6,
        retry: int = 5,
        verbose: bool = True,
        **kwargs,
    ):
        """
        参数:
            model: Main 模型名称（如 VDR-30B）
            api_base: Main 模型 API 基础 URL
            draft_model_name: Draft 模型名称（如 VDR-8B）
            draft_api_base: Draft 模型 API 基础 URL
            agent_repo_root: spectool_agent 仓库根目录
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
        from spectool_agent.vdr.draft_model_vdr import DraftModelVDR
        from spectool_agent.types import AgentConfig

        draft_api_base = draft_api_base or api_base
        draft_model_name = draft_model_name or model

        quality_judge_enabled = kwargs.pop("quality_judge_enabled", None)
        if quality_judge_enabled is None:
            quality_judge_enabled = os.environ.get(
                "SPECTOOL_QUALITY_JUDGE_ENABLED", "false").lower() in (
                    "1", "true", "yes", "on")
        quality_judge_threshold = float(kwargs.pop(
            "quality_judge_threshold",
            os.environ.get("SPECTOOL_QUALITY_JUDGE_THRESHOLD", "0.6")))
        self.cfg = AgentConfig(
            api_base=api_base, model_name=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens, timeout=timeout,
            max_steps=max_steps, max_image_history=max_image_history,
            quality_judge_enabled=quality_judge_enabled,
            quality_judge_threshold=quality_judge_threshold,
        )

        self.main_model = VLLMModel(self.cfg)
        self.draft_model = DraftModelVDR(
            api_base=draft_api_base,
            model_name=draft_model_name,
            api_key=draft_api_key,
            timeout=kwargs.pop("draft_timeout", 200),
            max_image_size=kwargs.pop("draft_max_image_size", 0),
        )
        self.print_traj = kwargs.pop("print_traj", False)
        self.fail_msg = "无法通过 Spectool VDR Agent 获得答案。"

        super().__init__(retry=retry, verbose=verbose, **kwargs)

    def _parse_vlmeval_inputs(self, inputs):
        images, texts = [], []
        for item in inputs:
            if item["type"] == "image":
                images.append(Image.open(item["value"]).convert("RGB"))
            elif item["type"] == "text":
                texts.append(item["value"])
        assert len(images) >= 1, (
            "Spectool agent requires at least one image input.")
        if len(images) > 1:
            texts.insert(0, (
                f"[Note] {len(images)} images provided. "
                "Tools will operate on the first image."))
        return images[0], "\n".join(texts)

    def generate_inner(self, inputs, **kwargs) -> tuple:
        from spectool_agent.vdr.agent_vdr import VDRSpeculativeAgentLoop
        from spectool_agent.tools import ToolRegistry, register_default_tools

        dataset = kwargs.get("dataset", None)
        sample_id = kwargs.get("sample_id", "")
        try:
            image, question = self._parse_vlmeval_inputs(inputs)
        except Exception as e:
            return -1, self.fail_msg, __import__("json").dumps({"error": f"input parse: {e}"})

        if self.print_traj:
            print(f"\n\n{SEP}")
            print(f"  SPECTool SAMPLE  |  id={sample_id}  |  dataset={dataset}")
            print(f"  Image: {image.size[0]}x{image.size[1]}")
            print(f"{SEP}")
            print(f"\n  ┌── QUESTION ────────────────────────────")
            for line in question.split("\n"):
                print(f"  │ {line}")
            print(f"  └──────────────────────────────────────────────")

        tools = ToolRegistry()
        register_default_tools(tools)
        agent = VDRSpeculativeAgentLoop(
            self.main_model, self.draft_model, tools, self.cfg,
        )

        # VDR-Bench is open-ended; no MCQ/mathvision extra prompt applies here.
        try:
            traj = agent.run(
                image, question,
                sample_id=sample_id,
            )
        except Exception as e:
            print(f"  [WARN] AgentLoop crashed: {e}", file=sys.stderr)
            __import__("traceback").print_exc()
            agent.shutdown()
            return 1, self.fail_msg, __import__("json").dumps({
                "error": f"agent_loop: {e}",
                "stop_reason": "exception",
            })
        finally:
            agent.shutdown()
            tools.cleanup()

        if self.print_traj:
            for step in traj.steps:
                _print_traj_step(step, sample_id)
            _print_traj_summary(traj, sample_id)

        if traj.final_answer is None:
            return 1, self.fail_msg, __import__("json").dumps({
                "stop_reason": traj.stop_reason,
                "total_time": traj.total_time,
                "trajectory": [asdict(s) for s in traj.steps],
            })

        return 0, traj.final_answer, {
            "tool_call_count": traj.tool_call_count,
            "stop_reason": traj.stop_reason,
            "total_inference_time": traj.total_inference_time,
            "total_tool_time": traj.total_tool_time,
            "total_time": traj.total_time,
            "trajectory": [asdict(s) for s in traj.steps],
            "dataset": dataset,
        }
