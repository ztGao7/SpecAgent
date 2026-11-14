"""Generated from the current EvalKit adapter; only the loop class changes."""
import os
import sys
from dataclasses import asdict, replace
from vlmeval.api.spectool_speculative import (
    SpeculativeSpectoolAgentAPI, SEP, _print_traj_step, _print_traj_summary,
)
from .sync_audit import save_record


class SynchronousSpectoolAgentAPI(SpeculativeSpectoolAgentAPI):
    def generate_inner(self, inputs, **kwargs):
        status, answer, payload = self._generate_sync(inputs, **kwargs)
        if isinstance(payload, dict):
            payload = dict(payload, synchronization_protocol="evidence_handshake_v1",
                sync_config=dict(initial_fast_answer="before_first_draft",
                    tool_actions_per_generation="first_only_regenerate_after_continue",
                    quality_judge_threshold=self.cfg.quality_judge_threshold,
                    terminal_candidate_threshold=self.cfg.quality_judge_threshold,
                    terminal_candidate_comparison=">=",
                    answerability_threshold=float(os.environ["SPECTOOL_ANSWERABILITY_THRESHOLD"])))
        result = status, answer, payload
        save_record(result, inputs, kwargs)
        return result

    def _generate_sync(self, inputs, **kwargs) -> tuple:
        """生成单个样本的推测性执行推理结果（VLMEvalKit BaseAPI 接口）。"""
        if getattr(self, "direct_only", False):
            return self._generate_direct(inputs, **kwargs)

        from spectool_agent.sync_testing.synchronous_agent import SynchronousSpeculativeAgentLoop as SpeculativeAgentLoop
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
            "timing_events": traj.timing_events,
            "trajectory": [asdict(s) for s in traj.steps],
            "dataset": dataset,
            "draft_replica_index": draft_replica_index,
            "draft_replica_count": len(self.draft_models),
        }
