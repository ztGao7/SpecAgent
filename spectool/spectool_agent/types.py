"""
纯数据类定义 — 不含 IO 或业务逻辑。

定义整个 spectool 系统中所有核心数据结构：
  - ToolCall:     工具调用（如 sandbox 代码执行、web_search）
  - StepResult:   单步推理结果（含推理时间、工具调用时间等）
  - Trajectory:   完整样本的推理轨迹
  - PoolEntry:    Draft 模型的中间产出（供 Main 模型审阅）
  - AgentConfig:  Agent 的配置参数
"""
from dataclasses import dataclass, field
from typing import Optional


# Shared default for ordinary Main/Draft requests, not short judge requests.
DEFAULT_MAX_TOKENS = 4096


@dataclass
class ToolCall:
    """工具调用描述"""
    tool_name: str       # 工具名称：sandbox / web_search
    params: dict         # 参数：sandbox 的 code, web_search 的 query


@dataclass
class StepResult:
    """单步推理结果，记录每一步的推理过程、工具调用和耗时"""
    step_idx: int                              # 步序号
    reasoning: str = ""                        # 模型的推理文本
    raw_response: str = ""                     # 未经解析/去标签的模型原始响应
    tool_call: Optional[ToolCall] = None       # 工具调用（如果有）
    observation: Optional[str] = None          # 工具执行结果观察
    answer: Optional[str] = None               # 答案（如果有）
    inference_time: float = 0.0                # 推理耗时（秒）
    tool_time: float = 0.0                     # 工具执行耗时（秒）
    draft_time: float = 0.0                    # Draft 模型耗时（秒）
    verdict_mode: str = ""                     # 本步 verdict 使用的策略
    judge_score: float = -1.0                  # QJ/answerability 概率（-1=未计算）


@dataclass
class Trajectory:
    """完整样本的推理轨迹，汇总所有步骤的信息"""
    sample_id: str = ""                        # 样本唯一标识
    steps: list = field(default_factory=list)  # 所有步骤的列表
    final_answer: Optional[str] = None         # 最终答案
    tool_call_count: int = 0                   # 工具调用次数（Agent 路径用；投机路径见 draft_tool_call_count）
    stop_reason: str = ""                      # 停止原因（answer / max_steps / error 等）
    budget_trigger: str = ""                   # MMSearch 提前收口原因（max_steps/context/output）
    estimated_output_tokens: int = 0           # 无 tokenizer 时的累计生成 token 估算
    total_inference_time: float = 0.0          # 总推理时间（Agent / fallback 路径用）
    total_tool_time: float = 0.0               # 总工具执行时间（Agent / fallback 路径用）
    total_time: float = 0.0                    # 总耗时（墙钟时间）
    total_main_time: float = 0.0               # 投机路径：Main 模型审阅/决策的累计推理时间
    insufficient_count: int = 0                # Main 声明「证据不足」的总次数（轮询+终局，每次 _is_insufficient 命中 +1）
    # Timing v2 records request intervals so parallel Main/Draft work is not
    # incorrectly treated as serial wall-clock latency.
    timing_version: int = 2
    timing_origin: float = field(default=0.0, repr=False, compare=False)
    timing_events: list = field(default_factory=list)
    main_request_time: float = 0.0              # Main 请求耗时之和（计算负载）
    draft_request_time: float = 0.0             # Draft 请求耗时之和（计算负载）
    total_draft_time: float = 0.0               # draft_request_time 的兼容别名
    model_request_time: float = 0.0             # Main + Draft 请求耗时之和
    model_active_wall_time: float = 0.0         # Main/Draft 区间并集（样本墙钟占用）
    model_overlap_time: float = 0.0             # 并行重叠掉的模型请求时间
    tool_request_time: float = 0.0              # Main/Draft 所有工具调用耗时之和
    tool_active_wall_time: float = 0.0           # 所有工具区间并集
    client_active_wall_time: float = 0.0         # 图片编码/上下文构造区间并集
    pipeline_active_wall_time: float = 0.0       # 模型、工具、客户端准备区间并集
    orchestration_wait_time: float = 0.0         # 轮询、同步及未归类墙钟时间
    model_active_wall_ratio: float = 0.0         # model_active_wall_time / total_time
    # Main takeover / fallback accounting.  Keep the historical fallback stop
    # reasons for result compatibility, while exposing whether the fallback
    # resumed validated work and how much shared budget remained.
    fallback_mode: str = ""                      # "main_takeover" when fallback ran
    fallback_trigger: str = ""                   # quality/tool/exhaustion trigger
    fallback_validated_steps: int = 0             # canonical Draft tool turns replayed
    fallback_passive_entries: int = 0             # unvalidated entries quoted as context
    fallback_step_budget: int = 0                 # iterative Main steps granted
    fallback_steps_used: int = 0                  # iterative Main requests attempted
    fallback_forced: bool = False                 # emergency answer-only request used
    fallback_budget_exhausted: bool = False       # step/wall budget ended before answer
    fallback_wall_budget: float = 0.0             # whole-sample wall budget; 0=disabled
    fallback_wall_elapsed_before: float = 0.0     # elapsed before Main takeover
    fallback_wall_time: float = 0.0               # takeover wall-clock duration
    fallback_replay_dropped_steps: int = 0        # reused turns cut by a failed sandbox replay

    def record_timing_event(
        self, component: str, phase: str, started_at: float, ended_at: float,
        *, step_idx: int = -1, status: str = "ok", thread_id: int = 0,
    ) -> None:
        """Record one real operation as an interval relative to this sample."""
        origin = self.timing_origin or 0.0
        start_s = max(0.0, float(started_at) - origin)
        end_s = max(start_s, float(ended_at) - origin)
        self.timing_events.append({
            "component": str(component),
            "phase": str(phase),
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "step_idx": int(step_idx),
            "status": str(status),
            "thread_id": int(thread_id),
        })

    @staticmethod
    def _interval_union_duration(events: list) -> float:
        intervals = sorted(
            (float(event["start_s"]), float(event["end_s"]))
            for event in events
            if float(event.get("end_s", 0.0)) > float(event.get("start_s", 0.0))
        )
        if not intervals:
            return 0.0
        total = 0.0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                total += current_end - current_start
                current_start, current_end = start, end
        return total + current_end - current_start

    def finalize_timing(self) -> None:
        """Derive cumulative-load and overlap-safe wall-clock timing metrics."""
        main_events = [
            event for event in self.timing_events
            if event.get("component") == "main"
        ]
        draft_events = [
            event for event in self.timing_events
            if event.get("component") == "draft"
        ]
        model_events = main_events + draft_events
        tool_events = [
            event for event in self.timing_events
            if event.get("component") == "tool"
        ]
        client_events = [
            event for event in self.timing_events
            if event.get("component") == "client"
        ]

        self.main_request_time = sum(
            float(event.get("duration_s", 0.0)) for event in main_events)
        self.draft_request_time = sum(
            float(event.get("duration_s", 0.0)) for event in draft_events)
        self.total_draft_time = self.draft_request_time
        self.model_request_time = (
            self.main_request_time + self.draft_request_time)
        self.model_active_wall_time = self._interval_union_duration(model_events)
        self.model_overlap_time = max(
            0.0, self.model_request_time - self.model_active_wall_time)
        self.tool_request_time = sum(
            float(event.get("duration_s", 0.0)) for event in tool_events)
        self.tool_active_wall_time = self._interval_union_duration(tool_events)
        self.client_active_wall_time = self._interval_union_duration(client_events)
        self.pipeline_active_wall_time = self._interval_union_duration(
            model_events + tool_events + client_events)
        self.orchestration_wait_time = max(
            0.0, self.total_time - self.pipeline_active_wall_time)
        self.model_active_wall_ratio = (
            self.model_active_wall_time / self.total_time
            if self.total_time > 0 else 0.0)


@dataclass
class PoolEntry:
    """Draft 模型的中间产出，供 Main 模型审阅使用

    包含 Draft 模型每一步的推理文本、工具调用、工具观察结果和 tentative answer。
    """
    step_idx: int                                          # 步序号
    response_text: str = ""                                # Draft 模型的原始响应文本
    reasoning: str = ""                                    # 剥离标签后的推理文本
    tool_call: Optional[ToolCall] = None                   # Draft 调用的工具
    tool_observation: Optional[str] = None                 # 工具执行的观察结果
    tool_result_images: list = field(default_factory=list)  # 工具执行生成的图片
    answer: Optional[str] = None                           # Draft 的候选答案（未验证）
    draft_time: float = 0.0                                # Draft 模型推理耗时
    # Preserve the native tool-call turn so a fallback can replay a valid
    # assistant -> tool conversation instead of inventing user messages.
    assistant_message: Optional[dict] = None
    tool_call_id: Optional[str] = None
    # None means this entry was never quality judged (for example step 0).
    quality_passed: Optional[bool] = None


@dataclass
class AgentConfig:
    """Agent 配置参数

    控制 Agent 的行为，包括模型连接信息、推理参数、步数限制、图片历史和审核阈值等。
    """
    api_base: str                    # API 基础 URL（如 http://localhost:8123/v1）
    model_name: str                  # 模型名称（如 Qwen3.5-27B）
    api_key: str = "EMPTY"           # API 密钥（默认 EMPTY）
    temperature: float = 0.0         # 采样温度（0 为 greedy）
    max_tokens: int = DEFAULT_MAX_TOKENS  # 单次生成的最大 token 数
    timeout: int = 120               # 请求超时（秒）
    max_steps: int = 15              # unified agent/fallback step budget
    # Benchmark-specific safety budgets.  Zero keeps the historical behavior.
    # MMSearch enables both so long search/visit transcripts are finalized
    # before they overflow the model context and become invalid samples.
    max_context_tokens: int = 0       # 估算上下文上限；0=不限制
    max_total_output_tokens: int = 0  # 单样本累计生成上限；0=不限制
    draft_max_steps: int = 15        # unified with max_steps by the adapters
    max_image_history: int = 6       # 保留的历史图片最大数量
    quality_judge_enabled: bool = False  # True 时先让 Main 判断 draft 探索质量
    quality_judge_threshold: float = 0.6  # P(GOOD) 低于该值则 Main clean takeover
    # Both the initial direct-answer check and the evidence-based
    # answerability check use P(answerable) from the same 0/1 token gate.
    # This is deliberately separate from P(GOOD): useful evidence need not
    # yet be sufficient to answer the task.
    answerability_threshold: float = 0.95
    # A clean restart keeps Main independent of every Draft message and its
    # persisted sandbox state.  Disable only for explicit legacy experiments
    # that replay QJ-approved native tool turns.
    fallback_clean_start: bool = True
    # With a clean restart, optionally replay only complete Draft tool turns
    # whose every call passed QJ.  The Main sandbox remains clean and no
    # unverified Draft rationale or answer is injected.
    fallback_reuse_validated_evidence: bool = False
    # Legacy takeover controls used only when fallback_clean_start is False.
    fallback_shared_step_budget: bool = True
    fallback_min_steps: int = 1
    sample_wall_time_budget: float = 0.0
