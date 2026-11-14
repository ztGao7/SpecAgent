"""Persist lossless per-sample records and check the synchronous protocol."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def input_signature(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def audit_events(events):
    sync = [e for e in events if e.get("component") == "sync"]
    published = [e for e in sync if e.get("phase") == "evidence_published"]
    generations = [e for e in sync if e.get("phase") == "draft_generation_start"]
    fast_gates = [e for e in sync if e.get("phase") == "fast_answer_gate_complete"]
    actions = [e for e in events if e.get("phase") in {"draft_step", "draft_tool"}]
    violations = []
    decisions = {}
    completed = {}
    for event in sync:
        eid = event.get("evidence_id")
        if event.get("phase") == "decision" and eid:
            if eid in decisions:
                violations.append(f"duplicate decision for evidence {eid}")
            decisions[eid] = event
        elif event.get("phase") == "assessment_complete" and eid:
            completed[eid] = event
    if fast_gates and generations:
        first_generation = min(generations, key=lambda event: event["start_s"])
        first_gate = min(fast_gates, key=lambda event: event["start_s"])
        if first_generation["start_s"] < first_gate["end_s"]:
            violations.append("first Draft generation preceded initial fast-answer completion")
    for index, entry in enumerate(published):
        eid, start = entry["evidence_id"], entry["start_s"]
        decision = decisions.get(eid)
        if decision is None:
            violations.append(f"evidence {eid} has no terminal or continue decision")
            continue
        decided = decision["start_s"]
        if decided < start:
            violations.append(f"decision precedes publication for evidence {eid}")
        completion = completed.get(eid)
        if decision.get("decision") == "continue":
            if completion is None:
                violations.append(f"continue without assessment for evidence {eid}")
            elif not start <= completion["end_s"] <= decided:
                violations.append(f"assessment completion order for evidence {eid}")
            elif "quality_passed" not in completion or "answerability_status" not in completion:
                violations.append(f"continue without both assessment outcomes for evidence {eid}")
        for generation in generations:
            when = generation["start_s"]
            if start < when < decided:
                violations.append(f"generation while evidence {eid} is pending")
            if when > decided and decision.get("decision") != "continue":
                violations.append(f"generation after terminal decision for evidence {eid}")
        for action in actions:
            when = action["start_s"]
            if start < when < decided:
                violations.append(f"{action['phase']} request while evidence {eid} is pending")
            if when > decided and decision.get("decision") != "continue":
                violations.append(f"{action['phase']} request after terminal decision for evidence {eid}")
        if index + 1 < len(published):
            following = published[index + 1]
            if following["start_s"] < decided:
                violations.append(f"more than one pending evidence entry after {eid}")
    wait_seconds = sum(e.get("duration_s", 0) for e in sync if e.get("phase") == "draft_wait")
    return {
        "passed": not violations,
        "violations": violations,
        "evidence_entries": len(published),
        "main_assessments": sum(e.get("status") != "skipped_tool_error_policy" for e in completed.values()),
        "continue_decisions": sum(e.get("decision") == "continue" for e in decisions.values()),
        "generation_requests": len(generations),
        "initial_fast_answer_checks": len(fast_gates),
        "discarded_parallel_actions": sum(e.get("discarded_count", 0) for e in sync
                                          if e.get("phase") == "discarded_parallel_actions"),
        "draft_wait_seconds": wait_seconds,
    }


def save_record(result, inputs, kwargs):
    directory = os.environ.get("SPECTOOL_SYNC_RECORD_DIR")
    if not directory:
        return
    status, answer, payload = result
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {"raw": payload}
    signature = input_signature(inputs)
    sample_id = str(kwargs.get("sample_id", ""))
    map_path = Path(directory).parent / "input_index.json"
    if not sample_id and map_path.exists():
        sample_id = str(json.loads(map_path.read_text()).get(signature, ""))
    record = dict(payload)
    record.update(api_status=status, prediction=answer,
                  sample_id=sample_id, input_signature=signature,
                  input_text=[i["value"] for i in inputs if i["type"] == "text"],
                  input_images=[i["value"] for i in inputs if i["type"] == "image"],
                  dataset=kwargs.get("dataset"), finished_unix=time.time())
    record["sync_audit"] = audit_events(record.get("timing_events", []))
    if not record.get("timing_events"):
        record["sync_audit"].update(passed=None, status="unverifiable")
    else:
        record["sync_audit"]["status"] = "pass" if record["sync_audit"]["passed"] else "violation"
    target = Path(directory) / f"{signature}_{uuid.uuid4().hex[:8]}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    temporary.replace(target)
    if record["sync_audit"]["status"] == "violation":
        marker = Path(directory).parent / "sync_audit_violation.json"
        if not marker.exists():
            marker.write_text(json.dumps({"record": str(target),
                "violations": record["sync_audit"]["violations"]}, indent=2))
        raise RuntimeError(f"Synchronous protocol violation: {record['sync_audit']}")
