"""VLMEvalKit integration using frozen task/format/image/tool helpers."""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from vlmeval.api.base import BaseAPI
from vlmeval.api.spectool_agent import SpectoolAgentAPI

from .core import Config, InfrastructureError, SpecReasonAgent
from .model import Endpoint
from .prompts import GENERATION_PROMPT, prompt_manifest


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


class SpecReasonAPI(BaseAPI):
    is_api = True
    INTERLEAVE = True
    requires_sample_id = True

    def __init__(self, *, main_model, main_url, draft_model, draft_url,
                 config=None, raw_dir, main_thinking=False, draft_thinking=True,
                 main_timeout=180, draft_timeout=200, **kwargs):
        self.config = config or Config()
        self.raw_dir = Path(raw_dir)
        self.failed = threading.Event()
        self.main = Endpoint(role="main", model=main_model, base_url=main_url,
            timeout=main_timeout, thinking=main_thinking,
            max_tokens=self.config.max_tokens, api_key=os.environ.get("SPECTOOL_VLLM_KEY", "EMPTY"))
        self.draft = Endpoint(role="draft", model=draft_model, base_url=draft_url,
            timeout=draft_timeout, thinking=draft_thinking,
            max_tokens=self.config.max_tokens, api_key=os.environ.get("SPECTOOL_DRAFT_KEY", "EMPTY"))
        # Exactly one outer attempt, including format errors; no random sleep.
        super().__init__(retry=1, wait=0, verbose=False, **kwargs)

    _parse_vlmeval_inputs = SpectoolAgentAPI._parse_vlmeval_inputs

    def generate_inner(self, inputs, **kwargs):
        from spectool_agent.model import pil_to_data_url, build_tool_result_messages
        from spectool_agent.agent import _truncate_image_history, _diagnose_tool_call
        from spectool_agent.tools import ToolRegistry, register_default_tools
        from spectool_agent.benchmarks import resolve_sample_id, resolve_benchmark_profile

        if self.failed.is_set():
            raise SystemExit("SpecReason halted after a service failure; partial JSON records retained")
        dataset = kwargs.get("dataset", "")
        if resolve_benchmark_profile(dataset=dataset) != "default":
            raise ValueError("This adapter currently supports the default visual/sandbox profile only")
        sample_id = resolve_sample_id(kwargs.get("sample_id", ""), inputs, dataset=dataset)
        images, question = self._parse_vlmeval_inputs(inputs)
        content = []
        for index, image in enumerate(images, start=1):
            if len(images) > 1:
                content.append({"type": "text", "text": f"[输入图片 {index} / sandbox image_{index}]"})
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(image)}})
        content.append({"type": "text", "text": question})
        history = [{"role": "system", "content": GENERATION_PROMPT}, {"role": "user", "content": content}]
        tools = ToolRegistry()
        schemas = register_default_tools(tools, dataset=dataset)
        agent = SpecReasonAgent(self.main, self.draft, tools, schemas, self.config,
            build_tool_result_messages, _truncate_image_history,
            format_feedback=lambda turn: _diagnose_tool_call(turn.message.get('content', '')))
        audit_token = tools.start_tool_audit(sample_id)
        result, fatal = None, None
        try:
            result = agent.run(history, images, sample_id)
        except InfrastructureError as exc:
            self.failed.set()
            fatal = exc
            result = agent.last_result
        except Exception as exc:
            self.failed.set()
            fatal = exc
            result = getattr(agent, "last_result", {})
            result.update(stop_reason="adapter_error", error=f"{type(exc).__name__}: {exc}")
        finally:
            tool_audit = tools.finish_tool_audit(audit_token)
            started = time.perf_counter()
            try:
                tools.cleanup()
            finally:
                if result is not None:
                    result.update(dataset=dataset, question=question, tool_audit=tool_audit,
                        cleanup_seconds=time.perf_counter() - started,
                        prompts=prompt_manifest(),
                        input_images=[{"size": list(im.size), "rgb_sha256":
                            hashlib.sha256(im.tobytes()).hexdigest()} for im in images])
                    safe = hashlib.sha256(f"{dataset}:{sample_id}".encode()).hexdigest()[:24]
                    path = self.raw_dir / f"{safe}_{os.getpid()}_{time.time_ns()}.json"
                    atomic_json(path, result)
        if fatal is not None:
            # BaseAPI retries catch Exception. SystemExit deliberately stops the
            # evaluation worker/main loop rather than querying a dead service.
            raise SystemExit(f"SpecReason service failure; saved {path}: {fatal}") from fatal
        extra = {"raw_trajectory_path": str(path), "stop_reason": result["stop_reason"],
            "total_time": result["total_time"], "tool_call_count": result["accounting"]["tool_calls"],
            "accounting": result["accounting"], "method": "SpecReason-Agent-pre-action"}
        if result["final_answer"] is None:
            return 1, "Failed to obtain answer via API.", extra
        return 0, result["final_answer"], extra
